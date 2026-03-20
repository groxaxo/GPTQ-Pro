FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ARG CONDA_DIR=/opt/conda
ARG ENV_NAME=gptq-pro-vllm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    PATH=${CONDA_DIR}/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    git \
    git-lfs \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    pkg-config \
    wget && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install

RUN wget -qO /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm -f /tmp/miniforge.sh && \
    conda config --system --set auto_activate_base false

WORKDIR /workspace/GPTQ-Pro

COPY environment.yml ./environment.yml

RUN conda env create -f environment.yml && \
    conda clean -afy

SHELL ["/bin/bash", "-lc"]

COPY . .

RUN source ${CONDA_DIR}/etc/profile.d/conda.sh && \
    conda activate ${ENV_NAME} && \
    python -m pip install --upgrade pip && \
    python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 "torch>=2.8.0" && \
    python -m pip install -v --no-build-isolation -e ".[vllm,eval,openai]"

CMD ["/bin/bash", "-lc", "source /opt/conda/etc/profile.d/conda.sh && conda activate gptq-pro-vllm && exec bash"]
