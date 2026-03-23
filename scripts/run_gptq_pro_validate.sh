#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

PRIMARY_GPU="${1:-${GPTQ_PRO_VALIDATE_GPU:-2}}"
FALLBACK_GPU="${GPTQ_PRO_VALIDATE_FALLBACK_GPU:-3}"
VALIDATE_SRC="${REPO_ROOT}/gptqmodel_ext/gptq_pro/gptq_pro_validate.cu"
KERNEL_SRC="${REPO_ROOT}/gptqmodel_ext/gptq_pro/gptq_pro_kernel.cu"
OUT_BIN="${TMPDIR:-/tmp}/gptq_pro_validate_$(date +%s)"

cleanup() {
    rm -f "${OUT_BIN}"
}
trap cleanup EXIT

run_on_gpu() {
    local gpu_index="$1"
    local gpu_name
    gpu_name=$(nvidia-smi -i "${gpu_index}" --query-gpu=name --format=csv,noheader | tr -d '[:space:]')

    echo "==> Building standalone validator"
    nvcc -arch=sm_80 -std=c++17 "${VALIDATE_SRC}" "${KERNEL_SRC}" -o "${OUT_BIN}"

    echo "==> Selected GPU ${gpu_index}"
    nvidia-smi -i "${gpu_index}" \
        --query-gpu=index,name,uuid,memory.total,memory.used,memory.free \
        --format=csv,noheader

    if [[ "${gpu_name}" != *"RTX3060"* && "${gpu_name}" != *"RTX 3060"* ]]; then
        echo "warning: GPU ${gpu_index} is not an RTX 3060 (${gpu_name})" >&2
    fi

    echo "==> Running validator"
    CUDA_VISIBLE_DEVICES="${gpu_index}" "${OUT_BIN}"
}

if ! run_on_gpu "${PRIMARY_GPU}"; then
    if [[ $# -eq 0 && "${PRIMARY_GPU}" != "${FALLBACK_GPU}" ]]; then
        echo "==> Primary GPU ${PRIMARY_GPU} failed; retrying on fallback GPU ${FALLBACK_GPU}" >&2
        run_on_gpu "${FALLBACK_GPU}"
    else
        exit 1
    fi
fi
