/*
 * PyTorch extension entrypoint for the standalone gptq_pro CUDA scaffold.
 */

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

cudaError_t gptq_pro_gemm(
    const half*    A,
    const uint8_t* B_packed,
    const half*    S,
    half*          C,
    int M, int N, int K, int group_size,
    cudaStream_t stream);

namespace {

void check_inputs(const torch::Tensor& a,
                  const torch::Tensor& b_packed,
                  const torch::Tensor& scales,
                  int64_t group_size) {
    TORCH_CHECK(a.is_cuda(), "gptq_pro_gemm: activations must be CUDA tensors.");
    TORCH_CHECK(b_packed.is_cuda(), "gptq_pro_gemm: packed weights must be CUDA tensors.");
    TORCH_CHECK(scales.is_cuda(), "gptq_pro_gemm: scales must be CUDA tensors.");
    TORCH_CHECK(a.scalar_type() == torch::kFloat16, "gptq_pro_gemm: activations must be float16.");
    TORCH_CHECK(b_packed.scalar_type() == torch::kUInt8, "gptq_pro_gemm: packed weights must be uint8.");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat16, "gptq_pro_gemm: scales must be float16.");
    TORCH_CHECK(a.dim() == 2, "gptq_pro_gemm: activations must be 2D [M, K].");
    TORCH_CHECK(b_packed.dim() == 2, "gptq_pro_gemm: packed weights must be 2D [(K+1)/2, N].");
    TORCH_CHECK(scales.dim() == 2, "gptq_pro_gemm: scales must be 2D [groups, N].");
    TORCH_CHECK(a.is_contiguous(), "gptq_pro_gemm: activations must be contiguous.");
    TORCH_CHECK(b_packed.is_contiguous(), "gptq_pro_gemm: packed weights must be contiguous.");
    TORCH_CHECK(scales.is_contiguous(), "gptq_pro_gemm: scales must be contiguous.");
    TORCH_CHECK(a.device() == b_packed.device() && a.device() == scales.device(),
                "gptq_pro_gemm: all tensors must live on the same CUDA device.");
    TORCH_CHECK(group_size > 0 && (group_size % 16) == 0,
                "gptq_pro_gemm: group_size must be a positive multiple of 16.");

    const auto k = a.size(1);
    const auto packed_rows = b_packed.size(0);
    TORCH_CHECK(packed_rows == (k + 1) / 2,
                "gptq_pro_gemm: packed weights shape does not match activation K dimension.");
    TORCH_CHECK(scales.size(1) == b_packed.size(1),
                "gptq_pro_gemm: scales second dimension must equal packed weight N dimension.");
    TORCH_CHECK(scales.size(0) == (k + group_size - 1) / group_size,
                "gptq_pro_gemm: scales first dimension must equal ceil(K / group_size).");
}

}  // namespace

torch::Tensor gptq_pro_gemm_torch(torch::Tensor a,
                                  torch::Tensor b_packed,
                                  torch::Tensor scales,
                                  int64_t group_size) {
    check_inputs(a, b_packed, scales, group_size);

    auto out = torch::empty({a.size(0), b_packed.size(1)}, a.options());

    const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.device().index());

    const auto status = gptq_pro_gemm(
        reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
        b_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        static_cast<int>(a.size(0)),
        static_cast<int>(b_packed.size(1)),
        static_cast<int>(a.size(1)),
        static_cast<int>(group_size),
        stream);

    TORCH_CHECK(status == cudaSuccess,
                "gptq_pro_gemm launch failed: ",
                cudaGetErrorString(status));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gptq_pro_gemm", &gptq_pro_gemm_torch, "GPTQ-Pro FP16xINT4 matmul.");
}
