#!/usr/bin/env bash
set -euo pipefail

# Resumable long-context max-quality GPTQ-Pro recipe for Qwen3.8-27B.
#
# Strategy:
# - quantize in deterministic four-layer chunks matching the model's repeating
#   3x linear-attention + 1x full-attention schedule;
# - save every chunk as an independent checkpoint artifact;
# - mark successful chunks atomically so interrupted runs restart from the last
#   completed boundary rather than from layer zero;
# - use the same fixed calibration JSONL and max-quality g64 settings for every
#   chunk;
# - finish with an explicit assembly step handled by
#   assemble_qwen38_resumable.py, which refuses missing or incompatible chunks.
#
# This is crash-resumable at four-layer boundaries. A process killed while a
# chunk is in progress reruns only that chunk.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:-Qwen/Qwen3.8-27B}"
CALIBRATION_JSONL="${CALIBRATION_JSONL:-/data/qwen38-calibration.jsonl}"
WORKDIR="${WORKDIR:-/models/qwen38-gptq-pro-resume}"
FINAL_OUT="${FINAL_OUT:-/models/Qwen3.8-27B-GPTQ-Pro-INT4-g64-longctx}"
GPU_LIST="${GPU_LIST:-0,1,2}"
NSAMPLE="${NSAMPLE:-128}"
GROUP_SIZE="${GROUP_SIZE:-64}"
CHUNK_LAYERS="${CHUNK_LAYERS:-4}"
TOTAL_LAYERS=64

mkdir -p "$WORKDIR/chunks" "$WORKDIR/logs" "$WORKDIR/state"

if [[ ! -s "$CALIBRATION_JSONL" ]]; then
  echo "Calibration JSONL missing or empty: $CALIBRATION_JSONL" >&2
  exit 2
fi

python scripts/validate_qwen38_long_context_calibration.py \
  --input "$CALIBRATION_JSONL" \
  --nsample "$NSAMPLE" \
  --require-long-context

python scripts/quant_qwen3_8_27b_gptqpro.py \
  --model "$MODEL" \
  --preflight-only

for ((start=0; start<TOTAL_LAYERS; start+=CHUNK_LAYERS)); do
  end=$((start + CHUNK_LAYERS))
  tag=$(printf "%02d-%02d" "$start" "$((end - 1))")
  out="$WORKDIR/chunks/layers-$tag"
  done_marker="$WORKDIR/state/layers-$tag.done.json"
  log="$WORKDIR/logs/layers-$tag.log"

  if [[ -s "$done_marker" ]]; then
    echo "[resume] layers $tag already complete"
    continue
  fi

  rm -rf "$out.tmp"
  mkdir -p "$out.tmp"

  echo "[run] quantizing layers $start..$((end - 1))"
  CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  python scripts/quant_qwen3_8_27b_gptqpro.py \
    --model "$MODEL" \
    --out "$out.tmp" \
    --calib text \
    --calibration-jsonl "$CALIBRATION_JSONL" \
    --nsample "$NSAMPLE" \
    --layer-start "$start" \
    --layer-count "$CHUNK_LAYERS" \
    --group-size "$GROUP_SIZE" \
    --preset max_quality \
    --calib-device cuda:0 \
    --offload-disk \
    2>&1 | tee "$log"

  rm -rf "$out"
  mv "$out.tmp" "$out"

  python - "$done_marker" "$MODEL" "$CALIBRATION_JSONL" "$NSAMPLE" "$GROUP_SIZE" "$start" "$end" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

marker, model, calib, nsample, group_size, start, end = sys.argv[1:]
calib_path = Path(calib)
h = hashlib.sha256()
with calib_path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        h.update(block)
payload = {
    "schema": "qwen38-gptq-pro-resume/v1",
    "model": model,
    "calibration_sha256": h.hexdigest(),
    "nsample": int(nsample),
    "group_size": int(group_size),
    "preset": "max_quality",
    "layer_start": int(start),
    "layer_end_exclusive": int(end),
}
path = Path(marker)
path.parent.mkdir(parents=True, exist_ok=True)
fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_name, path)
finally:
    if os.path.exists(temp_name):
        os.unlink(temp_name)
PY

done

python scripts/assemble_qwen38_resumable.py \
  --source "$MODEL" \
  --chunks "$WORKDIR/chunks" \
  --state "$WORKDIR/state" \
  --out "$FINAL_OUT" \
  --expected-layers "$TOTAL_LAYERS" \
  --chunk-layers "$CHUNK_LAYERS"

echo "[done] resumable long-context artifact: $FINAL_OUT"
