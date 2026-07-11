/*
 * PyTorch extension entrypoint for GPTQ-Pro's Ampere INT4 kernels.
 */

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <limits>
#include <string>

cudaError_t gptq_pro_gemm(
    const half* A,
    const int32_t* Q,
    const half* S,
    half* C,
    int M,
    int N,
    int K,
    int group_size,
    cudaStream_t stream,
    int kernel_mode);

namespace {

constexpr int kKernelAuto = 0;
constexpr int kKernelGemv = 1;
constexpr int kKernelAmpere = 2;
constexpr int kKernelLegacy = 3;

int parse_kernel_mode(const std::string& kernel_mode) {
    if (kernel_mode == "auto") {
        return kKernelAuto;
    }
    if (kernel_mode == "gemv") {
        return kKernelGemv;
    }
    if (kernel_mode == "ampere") {
        return kKernelAmpere;
    }
    if (kernel_mode == "legacy") {
        return kKernelLegacy;
    }
    TORCH_CHECK(
        false,
        "gptq_pro_gemm: kernel_mode must be one of auto, gemv, ampere, or legacy; got `",
        kernel_mode,
        "`.");
    return kKernelAuto;
}

void check_inputs(
    const torch::Tensor& a,
    const torch::Tensor& qweight,
    const torch::Tensor& scales,
    int64_t group_size) {
    TORCH_CHECK(a.is_cuda(), "gptq_pro_gemm: activations must be CUDA tensors.");
    TORCH_CHECK(
        qweight.is_cuda(), "gptq_pro_gemm: qweight must be a CUDA tensor.");
    TORCH_CHECK(scales.is_cuda(), "gptq_pro_gemm: scales must be CUDA tensors.");
    TORCH_CHECK(
        a.scalar_type() == torch::kFloat16,
        "gptq_pro_gemm: activations must be float16.");
    TORCH_CHECK(
        qweight.scalar_type() == torch::kInt32,
        "gptq_pro_gemm: qweight must use GPTQ int32 packing.");
    TORCH_CHECK(
        scales.scalar_type() == torch::kFloat16,
        "gptq_pro_gemm: scales must be float16.");
    TORCH_CHECK(a.dim() == 2, "gptq_pro_gemm: activations must be 2D [M, K].");
    TORCH_CHECK(
        qweight.dim() == 2,
        "gptq_pro_gemm: qweight must be 2D [ceil(K / 8), N].");
    TORCH_CHECK(
        scales.dim() == 2,
        "gptq_pro_gemm: scales must be 2D [groups, N].");
    TORCH_CHECK(a.is_contiguous(), "gptq_pro_gemm: activations must be contiguous.");
    TORCH_CHECK(qweight.is_contiguous(), "gptq_pro_gemm: qweight must be contiguous.");
    TORCH_CHECK(scales.is_contiguous(), "gptq_pro_gemm: scales must be contiguous.");
    TORCH_CHECK(
        a.device() == qweight.device() && a.device() == scales.device(),
        "gptq_pro_gemm: all tensors must live on the same CUDA device.");
    TORCH_CHECK(
        group_size > 0 && (group_size % 16) == 0,
        "gptq_pro_gemm: group_size must be a positive multiple of 16.");

    const auto m = a.size(0);
    const auto k = a.size(1);
    const auto n = qweight.size(1);
    const auto int_max = static_cast<int64_t>(std::numeric_limits<int>::max());
    TORCH_CHECK(
        m <= int_max && n <= int_max && k <= int_max && group_size <= int_max,
        "gptq_pro_gemm: dimensions exceed the CUDA launcher's 32-bit range.");

    TORCH_CHECK(
        qweight.size(0) == (k + 7) / 8,
        "gptq_pro_gemm: qweight first dimension must equal ceil(K / 8).");
    TORCH_CHECK(
        scales.size(1) == n,
        "gptq_pro_gemm: scales second dimension must equal qweight N dimension.");
    TORCH_CHECK(
        scales.size(0) == (k + group_size - 1) / group_size,
        "gptq_pro_gemm: scales first dimension must equal ceil(K / group_size).");
}

}  // namespace

torch::Tensor gptq_pro_gemm_torch(
    torch::Tensor a,
    torch::Tensor qweight,
    torch::Tensor scales,
    int64_t group_size,
    const std::string& kernel_mode) {
    check_inputs(a, qweight, scales, group_size);
    const int mode = parse_kernel_mode(kernel_mode);

    auto out = torch::empty({a.size(0), qweight.size(1)}, a.options());

    const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(a.device().index());

    const auto status = gptq_pro_gemm(
        reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
        qweight.data_ptr<int32_t>(),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        static_cast<int>(a.size(0)),
        static_cast<int>(qweight.size(1)),
        static_cast<int>(a.size(1)),
        static_cast<int>(group_size),
        stream,
        mode);

    TORCH_CHECK(
        status == cudaSuccess,
        "gptq_pro_gemm launch failed in `",
        kernel_mode,
        "` mode: ",
        cudaGetErrorString(status));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "gptq_pro_gemm",
        &gptq_pro_gemm_torch,
        "GPTQ-Pro FP16 x native GPTQ INT4 matmul.",
        pybind11::arg("activations"),
        pybind11::arg("qweight"),
        pybind11::arg("scales"),
        pybind11::arg("group_size"),
        pybind11::arg("kernel_mode") = "auto");
}
