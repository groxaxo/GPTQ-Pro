# syntax=docker/dockerfile:1

# Match the toolkit used for first-use CUDA extension compilation to the wheel.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PIP_CONSTRAINT=/opt/gptq-pro/constraints.txt \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    HF_HOME=/workspace/.cache/huggingface \
    GPTQMODEL_EXT_BUILD=/workspace/.cache/gptqmodel_extensions \
    TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions \
    MAX_JOBS=4 \
    PATH=/opt/venv/bin:${PATH}

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        cmake \
        git \
        git-lfs \
        libglib2.0-0t64 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        pkg-config \
        python3-dev \
        python3-venv && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install --system && \
    python3 -m venv "${VIRTUAL_ENV}"

# Constraints apply to every pip operation, including downstream installs.
# Install torch from its CUDA-specific index, rather than an extra index that
# can select a newer PyPI wheel with a different CUDA ABI.
COPY docker/constraints.txt /opt/gptq-pro/constraints.txt
RUN python -m pip install --upgrade pip wheel 'setuptools>=77.0.1,<83' && \
    python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
        'torch==2.10.0+cu128'

# Keep dependency installation cached when only repository source changes.
COPY requirements.txt /tmp/gptq-pro-requirements.txt
RUN python -m pip install -r /tmp/gptq-pro-requirements.txt && \
    rm /tmp/gptq-pro-requirements.txt

WORKDIR /workspace/GPTQ-Pro
COPY . .

# Import/metadata checks need no GPU and do not compile the lazy CUDA kernel.
RUN python -m pip install --no-build-isolation --no-deps -e . && \
    python -m pip check && \
    python -c "import torch; from gptqmodel import GPTQModel, QuantizeConfig; assert torch.version.cuda == '12.8'; assert QuantizeConfig.quality_4bit().bits == 4; print('GPTQ-Pro import OK:', torch.__version__)"

# PATH selects the same environment for the default shell and overridden CMDs.
CMD ["/bin/bash"]
