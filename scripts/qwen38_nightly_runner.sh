#!/usr/bin/env bash
set -euo pipefail

# Run the resumable Qwen3.8 long-context workflow only inside the configured
# local 00:00-stop-time window. Four-layer chunk markers make interruption safe;
# the next invocation repeats only the active, incomplete chunk.

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
NIGHTLY_TIMEZONE="${NIGHTLY_TIMEZONE:-Pacific/Auckland}"
NIGHTLY_STOP_TIME="${NIGHTLY_STOP_TIME:-07:00:00}"

for numeric_name in NSAMPLE GROUP_SIZE BENCH_NEW_TOKENS STOP_GRACE_SECONDS MIN_RUN_SECONDS; do
  numeric_value="${!numeric_name}"
  if [[ ! "$numeric_value" =~ ^[0-9]+$ ]]; then
    echo "$numeric_name must be a non-negative integer, got: $numeric_value" >&2
    exit 2
  fi
done
if (( NSAMPLE <= 0 || GROUP_SIZE <= 0 || BENCH_NEW_TOKENS <= 0 )); then
  echo "NSAMPLE, GROUP_SIZE, and BENCH_NEW_TOKENS must be positive" >&2
  exit 2
fi

export TZ="$NIGHTLY_TIMEZONE"
mkdir -p "$WORKDIR/logs" "$WORKDIR/state"
BENCH_OUT="${BENCH_OUT:-$WORKDIR/benchmark-qwen38-gptqpro.json}"
BENCH_DONE="$WORKDIR/state/benchmark.done.json"
FINAL_MANIFEST="$FINAL_OUT/qwen38_resumable_manifest.json"
NIGHTLY_LOG="$WORKDIR/logs/nightly-$(date +%F).log"
LOCK_FILE="$WORKDIR/state/nightly.lock"

exec > >(tee -a "$NIGHTLY_LOG") 2>&1

echo "[nightly] started $(date --iso-8601=seconds) timezone=$NIGHTLY_TIMEZONE stop=$NIGHTLY_STOP_TIME"

if ! command -v flock >/dev/null 2>&1; then
  echo "GNU flock is required (provided by util-linux on Ubuntu)" >&2
  exit 2
fi
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[nightly] another nightly workflow owns $LOCK_FILE; exiting"
  exit 0
fi

remaining_until_stop() {
  python scripts/qwen38_nightly_state.py remaining \
    --timezone "$NIGHTLY_TIMEZONE" \
    --stop-time "$NIGHTLY_STOP_TIME"
}

read -r allowed seconds_to_deadline < <(remaining_until_stop)
if [[ "$allowed" != "1" ]]; then
  echo "[nightly] outside 00:00-${NIGHTLY_STOP_TIME} $NIGHTLY_TIMEZONE; exiting without work"
  exit 0
fi

run_seconds=$((seconds_to_deadline - STOP_GRACE_SECONDS))
if (( run_seconds < MIN_RUN_SECONDS )); then
  echo "[nightly] only ${seconds_to_deadline}s remain before the stop boundary; not starting work"
  exit 0
fi

# Do not trust marker existence alone. It is valid only when it still binds to
# the current model directory, final manifest, benchmark report, and both hashes.
if [[ -s "$BENCH_DONE" ]]; then
  if python scripts/qwen38_nightly_state.py check-complete \
      --marker "$BENCH_DONE" \
      --report "$BENCH_OUT" \
      --manifest "$FINAL_MANIFEST" \
      --model "$FINAL_OUT"; then
    echo "[nightly] quantization and strict benchmark already complete"
    exit 0
  fi
  echo "[nightly] completion marker is stale; benchmark will be regenerated"
fi

export MODEL CALIBRATION_JSONL WORKDIR FINAL_OUT GPU_LIST NSAMPLE GROUP_SIZE

# The assembler atomically promotes the final manifest with the model. That
# manifest, rather than an early chunk report, is the authoritative build signal.
if [[ ! -s "$FINAL_MANIFEST" ]]; then
  echo "[nightly] quantization budget: ${run_seconds}s before graceful TERM"
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
      echo "[nightly] quantization and assembly completed"
      ;;
    75|124|137|143)
      echo "[nightly] scheduled pause reached; the next midnight will resume"
      exit 0
      ;;
    *)
      echo "[nightly] quantization failed with exit code $rc" >&2
      exit "$rc"
      ;;
  esac
fi

# Recompute the real elapsed-time budget after assembly. Timestamp arithmetic in
# qwen38_nightly_state.py keeps this correct across Auckland DST transitions.
read -r allowed seconds_to_deadline < <(remaining_until_stop)
run_seconds=$((seconds_to_deadline - STOP_GRACE_SECONDS))
if [[ "$allowed" != "1" || "$run_seconds" -lt "$MIN_RUN_SECONDS" ]]; then
  echo "[nightly] final artifact is ready; strict benchmark deferred to next midnight"
  exit 0
fi

echo "[nightly] final artifact ready; starting strict benchmark"
set +e
CUDA_VISIBLE_DEVICES="$BENCH_GPU" \
timeout --foreground --signal=TERM --kill-after="${STOP_GRACE_SECONDS}s" \
  "${run_seconds}s" \
  python scripts/benchmark_qwen38_gptqpro.py \
    --model "$FINAL_OUT" \
    --contexts "$BENCH_CONTEXTS" \
    --new-tokens "$BENCH_NEW_TOKENS" \
    --expected-group-size "$GROUP_SIZE" \
    --expected-preset max_quality \
    --output "$BENCH_OUT"
rc=$?
set -e

if [[ "$rc" == "124" || "$rc" == "137" || "$rc" == "143" ]]; then
  echo "[nightly] benchmark hit the stop boundary; it will rerun next midnight"
  exit 0
fi
if (( rc != 0 )); then
  echo "[nightly] strict benchmark failed with exit code $rc; no completion marker written" >&2
  exit "$rc"
fi

python scripts/qwen38_nightly_state.py mark-complete \
  --marker "$BENCH_DONE" \
  --report "$BENCH_OUT" \
  --manifest "$FINAL_MANIFEST" \
  --model "$FINAL_OUT"

echo "[nightly] COMPLETE: quantization + strict benchmark finished at $(date --iso-8601=seconds)"
echo "[nightly] benchmark report: $BENCH_OUT"
