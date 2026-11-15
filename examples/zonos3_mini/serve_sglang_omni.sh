#!/usr/bin/env bash
# Opt-in real SGLang-Omni ZONOS3-Mini server using native GPTQ-Pro INT4 MoE.
# The original Zyphra checkpoint and ZAC codec are required.
set -eo pipefail
for var in ZONOS3_SGLANG_OMNI_CHECKOUT ZONOS3_SOURCE_CHECKOUT ZONOS3_BASE_MODEL_DIR ZONOS3_SGLANG_BIN; do
    if [ -z "$(printenv "$var")" ]; then
        echo "Required environment variable missing: $var" >&2
        exit 2
    fi
done
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
if [ -z "$ZONOS3_GPTQPRO_BASE" ]; then
    export ZONOS3_GPTQPRO_BASE="$ZONOS3_BASE_MODEL_DIR/lm/zonos3_mini"
fi
if [ -z "$ZONOS3_GPTQPRO_OVERLAY" ]; then
    export ZONOS3_GPTQPRO_OVERLAY=groxaxo/ZONOS3-Mini-GPTQ-Pro-INT4
fi
export PYTHONPATH="$ZONOS3_SGLANG_OMNI_CHECKOUT:$repo_root:$ZONOS3_SOURCE_CHECKOUT/model/src:$PYTHONPATH"
if [ -z "$CUDA_HOME" ]; then export CUDA_HOME=/usr/local/cuda-12.8; fi
export PATH="$CUDA_HOME/bin:$PATH"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then export CUDA_VISIBLE_DEVICES=1; fi
if [ -z "$ZONOS3_SGLANG_HOST" ]; then export ZONOS3_SGLANG_HOST=127.0.0.1; fi
if [ -z "$ZONOS3_SGLANG_PORT" ]; then export ZONOS3_SGLANG_PORT=18082; fi
exec "$ZONOS3_SGLANG_BIN" serve \
    --model-path "$ZONOS3_BASE_MODEL_DIR" \
    --host "$ZONOS3_SGLANG_HOST" --port "$ZONOS3_SGLANG_PORT" \
    --tts_engine.gpu_memory_fraction 0.65 \
    --tts_engine.factory.max_running_requests 2 \
    --tts_engine.engine.cuda_graph_max_bs 2 "$@"
