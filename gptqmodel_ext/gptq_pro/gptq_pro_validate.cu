/*
 * Standalone GPTQ-Pro CUDA validation harness.
 *
 * Build without PyTorch:
 *   nvcc -O3 -std=c++17 -arch=sm_80 \
 *     gptq_pro_validate.cu gptq_pro_kernel.cu -o gptq_pro_validate
 *
 * Run on an Ampere-or-newer CUDA GPU. The harness checks the LOP3 fragment
 * decoder and compares every dispatch path with a scalar FP16-dequant/FP32-
 * accumulation reference, including edge shapes handled by the legacy path.
 */

#include "gptq_pro_kernel.cuh"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

#define CHECK_CUDA(expr)                                                        \
    do {                                                                        \
        const cudaError_t error = (expr);                                       \
        if (error != cudaSuccess) {                                             \
            std::fprintf(                                                       \
                stderr,                                                         \
                "CUDA error at %s:%d: %s\n",                                  \
                __FILE__,                                                       \
                __LINE__,                                                       \
                cudaGetErrorString(error));                                     \
            return false;                                                       \
        }                                                                       \
    } while (0)

namespace {

const char* mode_name(int mode) {
    switch (mode) {
        case GPTQ_PRO_KERNEL_AUTO:
            return "auto";
        case GPTQ_PRO_KERNEL_GEMV:
            return "gemv";
        case GPTQ_PRO_KERNEL_AMPERE:
            return "ampere";
        case GPTQ_PRO_KERNEL_LEGACY:
            return "legacy";
        default:
            return "unknown";
    }
}

__global__ void validate_decode_kernel(
    const uint16_t* __restrict__ packed,
    const half* __restrict__ scales,
    float* __restrict__ decoded) {
    const int lane = threadIdx.x;
    if (lane >= GPTQ_PRO_WARP_SIZE) {
        return;
    }

    uint32_t registers[2];
    decode_bfrag_to_rb(
        packed[lane], scales[lane], __float2half(8.0f), registers);
    Half2Reg first;
    Half2Reg second;
    first.u32 = registers[0];
    second.u32 = registers[1];
    decoded[lane * 4 + 0] = __half2float(__low2half(first.h2));
    decoded[lane * 4 + 1] = __half2float(__high2half(first.h2));
    decoded[lane * 4 + 2] = __half2float(__low2half(second.h2));
    decoded[lane * 4 + 3] = __half2float(__high2half(second.h2));
}

bool validate_fragment_decode() {
    std::vector<uint16_t> packed(GPTQ_PRO_WARP_SIZE);
    std::vector<half> scales(GPTQ_PRO_WARP_SIZE);
    std::vector<float> decoded(GPTQ_PRO_WARP_SIZE * 4);

    for (int lane = 0; lane < GPTQ_PRO_WARP_SIZE; ++lane) {
        const uint16_t w0 = static_cast<uint16_t>((lane + 0) & 0xF);
        const uint16_t w1 = static_cast<uint16_t>((lane + 3) & 0xF);
        const uint16_t w2 = static_cast<uint16_t>((lane + 7) & 0xF);
        const uint16_t w3 = static_cast<uint16_t>((lane + 11) & 0xF);
        packed[lane] = static_cast<uint16_t>(
            w0 | (w1 << 4) | (w2 << 8) | (w3 << 12));
        scales[lane] = __float2half(0.015625f * static_cast<float>(1 + lane % 4));
    }

    uint16_t* device_packed = nullptr;
    half* device_scales = nullptr;
    float* device_decoded = nullptr;
    CHECK_CUDA(cudaMalloc(&device_packed, packed.size() * sizeof(uint16_t)));
    CHECK_CUDA(cudaMalloc(&device_scales, scales.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&device_decoded, decoded.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(
        device_packed,
        packed.data(),
        packed.size() * sizeof(uint16_t),
        cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        device_scales,
        scales.data(),
        scales.size() * sizeof(half),
        cudaMemcpyHostToDevice));

    validate_decode_kernel<<<1, GPTQ_PRO_WARP_SIZE>>>(
        device_packed, device_scales, device_decoded);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(
        decoded.data(),
        device_decoded,
        decoded.size() * sizeof(float),
        cudaMemcpyDeviceToHost));

    cudaFree(device_packed);
    cudaFree(device_scales);
    cudaFree(device_decoded);

    int mismatches = 0;
    for (int lane = 0; lane < GPTQ_PRO_WARP_SIZE; ++lane) {
        const float scale = __half2float(scales[lane]);
        for (int index = 0; index < 4; ++index) {
            const int nibble = (packed[lane] >> (index * 4)) & 0xF;
            const float expected = __half2float(__float2half(
                scale * static_cast<float>(nibble - 8)));
            const float actual = decoded[lane * 4 + index];
            if (std::fabs(actual - expected) > 1e-4f) {
                ++mismatches;
            }
        }
    }

    if (mismatches != 0) {
        std::fprintf(stderr, "  decode mismatches: %d\n", mismatches);
        return false;
    }
    std::printf("  PASS LOP3 fragment decode\n");
    return true;
}

void fill_activations(std::vector<half>& activations, int M, int K) {
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            const float value =
                0.03125f * static_cast<float>((m % 11) - 5) +
                0.0078125f * static_cast<float>((k % 13) - 6);
            activations[m * K + k] = __float2half(value);
        }
    }
}

void fill_packed_weights(std::vector<uint8_t>& packed, int K, int N) {
    const int packed_rows = (K + 1) / 2;
    for (int pair = 0; pair < packed_rows; ++pair) {
        const int k0 = pair * 2;
        for (int n = 0; n < N; ++n) {
            const uint8_t low = static_cast<uint8_t>((k0 * 3 + n * 5 + 1) & 0xF);
            uint8_t high = static_cast<uint8_t>(((k0 + 1) * 3 + n * 5 + 1) & 0xF);
            if (k0 + 1 >= K) {
                high = 8;
            }
            packed[pair * N + n] = static_cast<uint8_t>(low | (high << 4));
        }
    }
}

void fill_scales(std::vector<half>& scales, int groups, int N) {
    for (int group = 0; group < groups; ++group) {
        for (int n = 0; n < N; ++n) {
            const float value =
                0.0078125f * static_cast<float>(1 + ((group * 3 + n) % 8));
            scales[group * N + n] = __float2half(value);
        }
    }
}

float reference_weight(
    const std::vector<uint8_t>& packed,
    const std::vector<half>& scales,
    int N,
    int group_size,
    int k,
    int n) {
    const uint8_t pair = packed[(k >> 1) * N + n];
    const int nibble = (k & 1) ? ((pair >> 4) & 0xF) : (pair & 0xF);
    const float scale = __half2float(scales[(k / group_size) * N + n]);
    return __half2float(
        __float2half(scale * static_cast<float>(nibble - 8)));
}

bool run_case(
    int M,
    int N,
    int K,
    int group_size,
    int mode,
    const char* label) {
    const int packed_rows = (K + 1) / 2;
    const int groups = (K + group_size - 1) / group_size;

    std::vector<half> activations(M * K);
    std::vector<uint8_t> packed(packed_rows * N);
    std::vector<half> scales(groups * N);
    std::vector<half> output(M * N, __float2half(0.0f));
    fill_activations(activations, M, K);
    fill_packed_weights(packed, K, N);
    fill_scales(scales, groups, N);

    half* device_activations = nullptr;
    uint8_t* device_packed = nullptr;
    half* device_scales = nullptr;
    half* device_output = nullptr;
    CHECK_CUDA(cudaMalloc(
        &device_activations, activations.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&device_packed, packed.size() * sizeof(uint8_t)));
    CHECK_CUDA(cudaMalloc(&device_scales, scales.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&device_output, output.size() * sizeof(half)));
    CHECK_CUDA(cudaMemcpy(
        device_activations,
        activations.data(),
        activations.size() * sizeof(half),
        cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        device_packed,
        packed.data(),
        packed.size() * sizeof(uint8_t),
        cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        device_scales,
        scales.data(),
        scales.size() * sizeof(half),
        cudaMemcpyHostToDevice));

    const cudaError_t launch_status = gptq_pro_gemm(
        device_activations,
        device_packed,
        device_scales,
        device_output,
        M,
        N,
        K,
        group_size,
        0,
        mode);
    if (launch_status != cudaSuccess) {
        std::fprintf(
            stderr,
            "  FAIL %s (%s) launch: %s\n",
            label,
            mode_name(mode),
            cudaGetErrorString(launch_status));
        cudaFree(device_activations);
        cudaFree(device_packed);
        cudaFree(device_scales);
        cudaFree(device_output);
        return false;
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(
        output.data(),
        device_output,
        output.size() * sizeof(half),
        cudaMemcpyDeviceToHost));

    cudaFree(device_activations);
    cudaFree(device_packed);
    cudaFree(device_scales);
    cudaFree(device_output);

    int mismatches = 0;
    float maximum_error = 0.0f;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float reference = 0.0f;
            for (int k = 0; k < K; ++k) {
                const float activation =
                    __half2float(activations[m * K + k]);
                const float weight = reference_weight(
                    packed, scales, N, group_size, k, n);
                reference = std::fma(activation, weight, reference);
            }
            const float expected = __half2float(__float2half(reference));
            const float actual = __half2float(output[m * N + n]);
            const float error = std::fabs(actual - expected);
            maximum_error = error > maximum_error ? error : maximum_error;
            const float tolerance = 0.02f + 0.002f * std::fabs(expected);
            if (!std::isfinite(actual) || error > tolerance) {
                if (mismatches < 8) {
                    std::fprintf(
                        stderr,
                        "  FAIL %s (%s) at (%d,%d): got=%f expected=%f error=%f\n",
                        label,
                        mode_name(mode),
                        m,
                        n,
                        actual,
                        expected,
                        error);
                }
                ++mismatches;
            }
        }
    }

    if (mismatches != 0) {
        std::fprintf(
            stderr,
            "  %s (%s) mismatches=%d max_error=%f\n",
            label,
            mode_name(mode),
            mismatches,
            maximum_error);
        return false;
    }

    std::printf(
        "  PASS %-24s mode=%-7s shape=%dx%dx%d g=%d max_error=%f\n",
        label,
        mode_name(mode),
        M,
        N,
        K,
        group_size,
        maximum_error);
    return true;
}

}  // namespace

int main() {
    int passed = 0;
    int total = 0;

    ++total;
    passed += validate_fragment_decode() ? 1 : 0;

    struct TestCase {
        int M;
        int N;
        int K;
        int group_size;
        int mode;
        const char* label;
    };

    const TestCase cases[] = {
        {1, 128, 128, 32, GPTQ_PRO_KERNEL_GEMV, "decode-m1"},
        {4, 80, 128, 32, GPTQ_PRO_KERNEL_GEMV, "decode-m4"},
        {1, 128, 128, 32, GPTQ_PRO_KERNEL_AUTO, "auto-decode"},
        {16, 64, 128, 32, GPTQ_PRO_KERNEL_AMPERE, "ampere-single-warp-n"},
        {31, 320, 256, 64, GPTQ_PRO_KERNEL_AMPERE, "ampere-tail-mn"},
        {32, 320, 256, 64, GPTQ_PRO_KERNEL_AUTO, "auto-ampere"},
        {13, 41, 29, 16, GPTQ_PRO_KERNEL_LEGACY, "legacy-edge"},
        {13, 41, 29, 16, GPTQ_PRO_KERNEL_AUTO, "auto-legacy"},
    };

    for (const auto& test_case : cases) {
        ++total;
        passed += run_case(
                      test_case.M,
                      test_case.N,
                      test_case.K,
                      test_case.group_size,
                      test_case.mode,
                      test_case.label)
                      ? 1
                      : 0;
    }

    std::printf("\n=== GPTQ-Pro CUDA validation: %d / %d passed ===\n", passed, total);
    return passed == total ? 0 : 1;
}
