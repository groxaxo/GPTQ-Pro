#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_ID=0
MODEL_PATH=""
TOKENIZER_PATH=""
SYSTEM_PROMPT=""
MAX_NEW_TOKENS=4096
TRUST_REMOTE_CODE=0

print_help() {
  cat <<'EOF'
Usage:
  ./run.sh --gpu_id <GPU_ID> --model_path <MODEL_PATH> [options]

Options:
  --tokenizer_path <TOKENIZER_PATH>  Optional tokenizer path.
  --system_prompt <SYSTEM_PROMPT>    Optional system prompt.
  --max_new_tokens <MAX_NEW_TOKENS>  Maximum generated tokens per reply.
  --trust_remote_code                Allow custom Hugging Face model/tokenizer code.
  --help                             Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --help|-h)
      print_help
      exit 0
      ;;
    --gpu_id)
      GPU_ID="$2"
      shift
      shift
      ;;
    --model_path)
      MODEL_PATH="$2"
      shift
      shift
      ;;
    --tokenizer_path)
      TOKENIZER_PATH="$2"
      shift
      shift
      ;;
    --system_prompt)
      SYSTEM_PROMPT="$2"
      shift
      shift
      ;;
    --max_new_tokens)
      MAX_NEW_TOKENS="$2"
      shift
      shift
      ;;
    --trust_remote_code)
      TRUST_REMOTE_CODE=1
      shift
      ;;
    *)
      echo "Unknown $1"
      exit 1
      ;;
  esac
done

if [[ -z "$MODEL_PATH" ]]; then
  echo "--model_path REQUIRED！"
  exit 1
fi

CMD=(python "$SCRIPT_DIR/chat.py" --model_path "$MODEL_PATH" --max_new_tokens "$MAX_NEW_TOKENS")

if [[ -n "$TOKENIZER_PATH" ]]; then
  CMD+=(--tokenizer_path "$TOKENIZER_PATH")
fi

if [[ -n "$SYSTEM_PROMPT" ]]; then
  CMD+=(--system_prompt "$SYSTEM_PROMPT")
fi

if [[ "$TRUST_REMOTE_CODE" -eq 1 ]]; then
  CMD+=(--trust_remote_code)
fi

env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_ID" "${CMD[@]}"
