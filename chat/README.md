## Chat CLI

## Usage
```bash
./run.sh --gpu_id <GPU_ID> --model_path <MODEL_PATH>
```

--gpu_id: Specifies the GPU ID to use. This maps to the CUDA_VISIBLE_DEVICES environment variable.

--model_path: The file path to the chat model. This must point to a valid model directory or file.

### Useful options

```bash
./run.sh \
  --gpu_id 0 \
  --model_path /models/Qwen3.5-4B-abliterated-GPTQ-Pro-4bit \
  --tokenizer_path wangzhang/Qwen3.5-4B-abliterated \
  --max_new_tokens 1024 \
  --trust_remote_code
```

- `--tokenizer_path`: Reuse the source tokenizer when a quantized checkpoint should not use its local tokenizer files.
- `--system_prompt`: Override the default system prompt. Pass an empty string to disable it.
- `--max_new_tokens`: Cap the number of generated tokens per assistant reply.
- `--trust_remote_code`: Required for some newer Hugging Face model families such as Qwen 3.5.

### Qwen 3.5 / GPTQ-Pro notes

- For GPTQ-Pro checkpoints quantized from Qwen 3.5 models, `GPTQModel.load()` is the serving path used by this CLI.
- If you are benchmarking with `vLLM`, keep that workflow separate from `chat.py`; the chat CLI is intended as a lightweight local frontend for quick manual checks.
- The main repository `README.md` contains the full replication guide, Docker/conda setup, and Qwen 3.5 vLLM compatibility notes.
