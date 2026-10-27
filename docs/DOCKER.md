# Docker development environment

[Documentation index](README.md) · [Runtime contract](STATUS.md#native-runtime-contract)

The Dockerfile installs GPTQ-Pro from this checkout into `/opt/venv` and places
that environment on `PATH`. Both the default shell and an overridden command,
such as `docker run ... python`, use the same interpreter. Conda activation is
not required; `environment.yml` remains a separate Conda setup option.

## Image configuration

| Component | Docker default |
| --- | --- |
| Base | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| Python | Ubuntu 24.04 Python 3.12 in `/opt/venv` |
| PyTorch | `2.10.0+cu128`, from the official CUDA 12.8 wheel index |
| TorchAO | `0.16.0`, matching PyTorch 2.10's extension ABI |
| Build tools | CUDA toolkit, C++ compiler, CMake, Ninja and setuptools `>=77.0.1,<83` |
| Source | Editable install at `/workspace/GPTQ-Pro` |
| Compiler concurrency | `MAX_JOBS=4`; override at runtime if needed |

[Docker constraints](../docker/constraints.txt) pin the CUDA/PyTorch compatibility
pair without changing the package's general dependency requirements. Other
dependencies follow [requirements.txt](../requirements.txt); this is not a fully
locked dependency set. The CUDA base tag is versioned, not pinned to a digest.
The final image build runs `pip check` and a CPU-only package import/recipe check.
It does not download model weights or require a GPU.

## Build and run

Build from the repository root:

```bash
docker build --progress=plain -t gptq-pro:local .
```

GPU execution requires a compatible host NVIDIA driver and NVIDIA Container
Toolkit configured for Docker. On a shared machine, select an admitted GPU.
For a standalone host where GPU 0 is available:

```bash
docker run --rm -it --gpus '"device=0"' --shm-size=8g \
  -v gptq-pro-cache:/workspace/.cache \
  -v "$PWD/models:/models" \
  gptq-pro:local
```

Mount model inputs/outputs at `/models` and the cache at `/workspace/.cache`.
Mounting over `/workspace/GPTQ-Pro` hides the installed editable source.
The image runs as root by default; files written to bind mounts have the
container user's ownership.

The CUDA kernel is lazy-built on first native use. Keep the development toolkit
in the image and persist the cache to retain build artifacts between runs.
The native loader still applies the [architecture and precision gates](STATUS.md).

## Smoke checks

Direct-command import and dependency checks need no GPU:

```bash
docker run --rm --network none gptq-pro:local python -m pip check
docker run --rm --network none gptq-pro:local python -c \
  'import sys, torch, gptqmodel; assert sys.prefix == "/opt/venv"; assert torch.version.cuda == "12.8"; print(torch.__version__, gptqmodel.__version__)'
```

Check GPU visibility only on a GPU allocated for this operation:

```bash
docker run --rm --gpus '"device=0"' gptq-pro:local python -c \
  'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

An import or device-visibility pass does not compile/test the native kernel,
load a checkpoint or qualify a model workload. Use the [validation guide](VALIDATION.md)
for those separate checks.
