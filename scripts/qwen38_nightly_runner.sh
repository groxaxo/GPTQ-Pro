#!/usr/bin/env bash
set -euo pipefail

# Run the resumable Qwen3.8 long-context quantizer only during the local
# 00:00-07:00 window. The hard wall-clock deadline is enforced even if a
# persistent systemd timer catches up after midnight.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:-Qwen/Qwen3.8-27B}"
CALIBRATION_JSONL="${CALIBRATION_JSONL:-/data/qwen38-calibration.jsonl}"
WORKDIR="${WORKDIR:-/models/qwen38-gptq-pro-resume}"
FINAL_OUT="${FINAL_OUT:-/models/Qwen3.8-27B-GPTQ-Pro-INT4-g64-longctx}"
GPU_LIST="${GPU_LIST:-0,1,2}"
NSAMPLE="${NSAMPLE:-128}"
GROUP_SIZE="${GROUP_SIZE:-64}"
BENCH_GPU="${BENCH_GPU:-0}"
BENCH_CONTEXTS="${BENCH_CONTEXTS:-2048,8192,32768}"
BENCH_NEW_TOKENS="${BENCH_NEW_TOKENS:-128}"
STOP_GRACE_SECONDS="${STOP_GRACE_SECONDS:-120}"
MIN_RUN_SECONDS="${MIN_RUN_SECONDS:-300}"

mkdir -p "$WORKDIR/logs" "$WORKDIR/state"
BENCH_OUT="${BENCH_OUT:-$WORKDIR/benchmark-qwen38-gptqpro.json}"
BENCH_DONE="$WORKDIR/state/benchmark.done.json"
FINAL_MANIFEST="$FINAL_OUT/qwen38_resumable_manifest.json"
NIGHTLY_LOG="$WORKDIR/logs/nightly-$(date +%F).log"

exec > >(tee -a "$NIGHTLY_LOG") 2>&1

echo "[nightly] started $(date --iso-8601=seconds)"

remaining_until_0700() {
  python - <<'PY'
from datetime import datetime
now = datetime.now().astimezone()
deadline = now.replace(hour=7, minute=0, second=0, microsecond=0)
allowed = now.hour < 7
seconds = max(0, int((deadline - now).total_seconds())) if allowed else 0
print(1 if allowed else 0, seconds)
PY
}

read -r allowed seconds_to_deadline < <(remaining_until_0700)
if [[ "$allowed" != "1" ]]; then
  echo "[nightly] outside 00:00-07:00 local window; exiting without work"
  exit 0
fi

run_seconds=$((seconds_to_deadline - STOP_GRACE_SECONDS))
if (( run_seconds < MIN_RUN_SECONDS )); then
  echo "[nightly] only ${seconds_to_deadline}s remain before 07:00; not starting work"
  exit 0
fi

# A completed benchmark marker means the whole workflow is done and future timer
# invocations become no-ops.
if [[ -s "$BENCH_DONE" ]]; then
  echo "[nightly] quantization and benchmark already complete"
  exit 0
fi

export MODEL CALIBRATION_JSONL WORKDIR FINAL_OUT GPU_LIST NSAMPLE GROUP_SIZE

# The assembler promotes qwen38_resumable_manifest.json together with the final
# model in one directory rename. That manifest, rather than an early metadata
# file, is the authoritative completion signal.
if [[ ! -s "$FINAL_MANIFEST" ]]; then
  echo "[nightly] quantization window: ${run_seconds}s before graceful deadline"
  set +e
  timeout --foreground --signal=TERM --kill-after="${STOP_GRACE_SECONDS}s" \
    "${run_seconds}s" bash scripts/qwen38_long_context_recipe.sh
  rc=$?
  set -e

  case "$rc" in
    0)
      if [[ ! -s "$FINAL_MANIFEST" ]]; then
        echo "[nightly] recipe returned success but final assembly manifest is missing" >&2
        exit 3
      fi
      echo "[nightly] quantization/assembly completed"
      ;;
    75|124|137|143)
      echo "[nightly] scheduled pause reached; next midnight will resume"
      exit 0
      ;;
    *)
      echo "[nightly] quantization failed with exit code $rc" >&2
      exit "$rc"
      ;;
  esac
fi

# Recompute the remaining wall-clock budget after assembly. If completion occurs
# near 07:00 the benchmark is deferred to the next midnight rather than crossing
# the user's stop boundary.
read -r allowed seconds_to_deadline < <(remaining_until_0700)
run_seconds=$((seconds_to_deadline - STOP_GRACE_SECONDS))
if [[ "$allowed" != "1" || "$run_seconds" -lt "$MIN_RUN_SECONDS" ]]; then
  echo "[nightly] final artifact is ready; benchmark deferred to next midnight"
  exit 0
fi

echo "[nightly] final artifact ready; starting benchmark"
set +e
CUDA_VISIBLE_DEVICES="$BENCH_GPU" \
timeout --foreground --signal=TERM --kill-after="${STOP_GRACE_SECONDS}s" \
  "${run_seconds}s" \
  python scripts/benchmark_qwen38_gptqpro.py \
    --model "$FINAL_OUT" \
    --contexts "$BENCH_CONTEXTS" \
    --new-tokens "$BENCH_NEW_TOKENS" \
    --output "$BENCH_OUT"
rc=$?
set -e

if [[ "$rc" == "124" || "$rc" == "137" || "$rc" == "143" ]]; then
  echo "[nightly] benchmark hit 07:00 boundary; it will rerun next midnight"
  exit 0
fi
if (( rc != 0 )); then
  echo "[nightly] benchmark failed with exit code $rc" >&2
  exit "$rc"
fi

python - "$BENCH_DONE" "$BENCH_OUT" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

marker = Path(sys.argv[1])
report = Path(sys.argv[2])
h = hashlib.sha256(report.read_bytes()).hexdigest()
payload = {
    "schema": "qwen38-nightly-complete/v1",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "benchmark_report": str(report),
    "benchmark_sha256": h,
}
marker.parent.mkdir(parents=True, exist_ok=True)
fd, tmp = tempfile.mkstemp(prefix=marker.name + ".", dir=marker.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, marker)
finally:
    if os.path.exists(tmp):
        os.unlink(tmp)
PY

echo "[nightly] COMPLETE: quantization + benchmark finished at $(date --iso-8601=seconds)"
echo "[nightly] benchmark report: $BENCH_OUT"
