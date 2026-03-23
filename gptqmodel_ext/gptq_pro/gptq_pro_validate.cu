/*
 * GPTQ-Pro kernel validation harness
 *
 * Implements the two validation milestones from the design TODO list:
 *
 *  TODO 1 — Validate decode-only against a scalar host/device reference for
 *            one warp fragment.
 *
 *  TODO 2 — Validate one full ks/j MMA step against a reference with FP32
 *            accumulation semantics and FP16 inputs.
 *
 * Each milestone produces a per-thread pass/fail flag stored in a result
 * buffer that can be inspected from the host.  Both kernels target sm80+.
 *
 * Build (standalone, no PyTorch):
 *   nvcc -arch=sm_80 -std=c++17 gptq_pro_validate.cu gptq_pro_kernel.cu -o gptq_pro_validate
 */

#include "gptq_pro_kernel.cuh"

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

cudaError_t gptq_pro_gemm(
    const half*    A,
    const uint8_t* B_packed,
    const half*    S,
    half*          C,
    int M, int N, int K, int group_size,
    cudaStream_t stream);

#define CHECK_CUDA(expr)                                                         \
    do {                                                                         \
        cudaError_t _err = (expr);                                               \
        if (_err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_err));               \
            return 1;                                                            \
        }                                                                        \
    } while (0)

// ============================================================
// TODO 1  — Decode-only scalar reference
// ============================================================

/// Scalar (lane-independent) decode of one lane-local 4-nibble B fragment.
/// Mirrors exactly what decode_bfrag_to_rb() does on-device, using only
/// host-visible arithmetic so the result can be used as ground truth.
///
/// @param w0          unsigned 4-bit integer [0, 15] for RB[0] low half
/// @param w1          unsigned 4-bit integer [0, 15] for RB[0] high half
/// @param w2          unsigned 4-bit integer [0, 15] for RB[1] low half
/// @param w3          unsigned 4-bit integer [0, 15] for RB[1] high half
/// @param scale_f     FP32 version of the per-group scale
/// @param zp_f        FP32 version of the per-group zero-point
/// @param out_rb_f    [out] dequantized values in order {w0, w1, w2, w3}
inline void scalar_decode_bfrag(uint32_t w0, uint32_t w1,
                                uint32_t w2, uint32_t w3,
                                float scale_f, float zp_f,
                                float (&out_rb_f)[4]) {
    out_rb_f[0] = scale_f * (static_cast<float>(w0) - zp_f);
    out_rb_f[1] = scale_f * (static_cast<float>(w1) - zp_f);
    out_rb_f[2] = scale_f * (static_cast<float>(w2) - zp_f);
    out_rb_f[3] = scale_f * (static_cast<float>(w3) - zp_f);
}

// ---------------------------------------------------------------------------
// Device-side TODO 1 validation kernel
//
// Each thread (= one warp lane):
//   1. Reads its packed INT4 data from a pre-filled Bfrag smem region.
//   2. Calls fetch_bfrag_packed16() (the real ld.shared.u32 path).
//   3. Calls decode_bfrag_to_rb() to obtain RB[0], RB[1].
//   4. Compares against a pre-computed float reference stored in ref_rb
//      (4 floats per lane, computed by scalar_decode_bfrag on host).
//   5. Sets result[lane] = 1 if all four decoded values match.
// ---------------------------------------------------------------------------
__global__ void validate_decode_kernel(
    const uint32_t* __restrict__ bfrag_smem_src,  // GPTQ_PRO_BFRAG_WORDS_PER_BUF words (global, will be copied to smem)
    float           scale_f,
    float           zp_f,
    const float*    ref_rb,    // [WARP_SIZE * 4] ground-truth per lane
    int*            result)    // [WARP_SIZE] 1=pass, 0=fail
{
    // One warp handles one tile at ks=0, j=0.
    const int lane = threadIdx.x & (GPTQ_PRO_WARP_SIZE - 1);

    // ---- Stage Bfrag into shared memory ----
    extern __shared__ uint32_t smem_bfrag[];
    // Each thread copies one word.
    if (lane < GPTQ_PRO_BFRAG_WORDS_PER_BUF) {
        smem_bfrag[lane] = bfrag_smem_src[lane];
    }
    __syncthreads();

    // ---- TODO 1: fetch via ld.shared.u32 path ----
    const int ks = 0, j = 0, buf = 0;
    uint16_t packed_16 = fetch_bfrag_packed16(smem_bfrag, buf, ks, j, lane);

    // ---- Decode nibbles ----
    const half scale = __float2half(scale_f);
    const half zp    = __float2half(zp_f);
    uint32_t RB[2];
    decode_bfrag_to_rb(packed_16, scale, zp, RB);

    // ---- Extract decoded FP16 values and compare with reference ----
    Half2Reg rb0_r, rb1_r;
    rb0_r.u32 = RB[0];
    rb1_r.u32 = RB[1];

    float got_rb0_lo = __half2float(__low2half(rb0_r.h2));
    float got_rb0_hi = __half2float(__high2half(rb0_r.h2));
    float got_rb1_lo = __half2float(__low2half(rb1_r.h2));
    float got_rb1_hi = __half2float(__high2half(rb1_r.h2));

    // FP16 has ~1e-3 relative error; use 2 ULP tolerance in FP16 space.
    const float tol = 2.0f * __half2float(__float2half(1.0f)) * 1e-3f;

    bool ok = (fabsf(got_rb0_lo - ref_rb[lane * 4 + 0]) <= tol + 1e-5f) &&
              (fabsf(got_rb0_hi - ref_rb[lane * 4 + 1]) <= tol + 1e-5f) &&
              (fabsf(got_rb1_lo - ref_rb[lane * 4 + 2]) <= tol + 1e-5f) &&
              (fabsf(got_rb1_hi - ref_rb[lane * 4 + 3]) <= tol + 1e-5f);
    result[lane] = ok ? 1 : 0;
}

// ============================================================
// TODO 2  — Scalar FP32-accumulating MMA reference
// ============================================================

/// Scalar reference for mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32.
///
/// Interprets A and B as FP16 inputs and accumulates into FP32, matching the
/// hardware accumulation semantics of the tensor-core instruction.
///
/// PTX fragment ownership for mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32:
///   A: row = groupID for ai in {0,1,4,5}, else groupID + 8
///      col = 2 * threadID + (i & 1) [+8 for i >= 4]
///   B: row = 2 * threadID + (i & 1) [+8 for i >= 2]
///      col = groupID
///   C/D: row = groupID for ci in {0,1}, else groupID + 8
///        col = 2 * threadID + (i & 1)
/// where groupID = lane >> 2 and threadID = lane & 3.
///
/// This scalar function operates on the unpacked FP32 equivalents; the caller
/// is responsible for unpacking and repacking.
inline float fma_f32_from_f16_inputs_proxy(float a, float b, float c) {
    const half ha = __float2half(a);
    const half hb = __float2half(b);
    return c + __half2float(ha) * __half2float(hb);
}

inline float mma_ref_a_value(int m_row, int k_col) {
    return 0.03125f * static_cast<float>(m_row + 1)
         + 0.001953125f * static_cast<float>(k_col + 1);
}

inline float mma_ref_b_value(int k_row, int group_id) {
    return 0.0625f * static_cast<float>(k_row + 1)
         + 0.0078125f * static_cast<float>(group_id);
}

inline uint32_t pack_half2(float lo, float hi) {
    Half2Reg reg;
    const half hlo = __float2half(lo);
    const half hhi = __float2half(hi);
    memcpy(&reg.u16[0], &hlo, sizeof(uint16_t));
    memcpy(&reg.u16[1], &hhi, sizeof(uint16_t));
    return reg.u32;
}

// ---------------------------------------------------------------------------
// Device-side TODO 2 validation kernel
//
// One thread block = one warp.  Validates a single (ks=0, j=0) MMA step.
// Steps:
//   1. Loads pre-dequantized RA[4] and RB[2] from global memory.
//   2. Zeroes RC[4].
//   3. Calls mma_f32_m16n8k16() (the real tensor-core MMA).
//   4. Computes a scalar reference using fma_f32_from_f16_inputs_proxy().
//   5. Compares RC output against reference; writes 1=pass / 0=fail.
// ---------------------------------------------------------------------------
__global__ void validate_mma_step_kernel(
    const uint32_t* __restrict__ ra_global,   // [4] per lane
    const uint32_t* __restrict__ rb_global,   // [2] per lane
    const float*    __restrict__ ref_rc,      // [4] per lane (float proxy)
    int*            result)                   // [WARP_SIZE] 1=pass, 0=fail
{
    const int lane = threadIdx.x & (GPTQ_PRO_WARP_SIZE - 1);

    // Load fragment registers for this lane.
    uint32_t RA[4], RB[2];
    float RC[4];
    RA[0] = ra_global[lane * 4 + 0];
    RA[1] = ra_global[lane * 4 + 1];
    RA[2] = ra_global[lane * 4 + 2];
    RA[3] = ra_global[lane * 4 + 3];
    RB[0] = rb_global[lane * 2 + 0];
    RB[1] = rb_global[lane * 2 + 1];
    RC[0] = 0.0f;
    RC[1] = 0.0f;
    RC[2] = 0.0f;
    RC[3] = 0.0f;

    // ---- TODO 2: real tensor-core MMA ----
    mma_f32_m16n8k16(RA, RB, RC);

    // FP32 accumulation with exact FP16 inputs should agree closely with the
    // scalar proxy. Allow a tiny epsilon for instruction-order differences.
    const float tol = 1e-6f;
    bool ok = true;
    for (int i = 0; i < 4; ++i) {
        if (fabsf(RC[i] - ref_rc[lane * 4 + i]) > tol) {
            ok = false;
        }
    }
    result[lane] = ok ? 1 : 0;
}

// ============================================================
// Host-side driver — fills test data, launches kernels, checks results
// ============================================================

/// Fill Bfrag shared-memory image with deterministic lane-local INT4 payloads.
/// packed_16 for lane `l` in tile (0,0) packs four distinct nibbles:
///   bits [3:0]   = (4*l + 0) & 0xF
///   bits [7:4]   = (4*l + 1) & 0xF
///   bits [11:8]  = (4*l + 2) & 0xF
///   bits [15:12] = (4*l + 3) & 0xF
static void fill_bfrag_test_image(uint32_t* words) {
    for (int w = 0; w < GPTQ_PRO_BFRAG_WORDS_PER_BUF; ++w) {
        words[w] = 0u;
    }
    // Tile ks=0, j=0, buf=0 starts at word offset 0.
    for (int lane = 0; lane < GPTQ_PRO_WARP_SIZE; ++lane) {
        uint32_t w0 = (4u * lane + 0u) & 0xFu;
        uint32_t w1 = (4u * lane + 1u) & 0xFu;
        uint32_t w2 = (4u * lane + 2u) & 0xFu;
        uint32_t w3 = (4u * lane + 3u) & 0xFu;
        uint16_t p16 = static_cast<uint16_t>(
            w0 | (w1 << 4) | (w2 << 8) | (w3 << 12));
        int word_idx = lane >> 1;   // lane pair
        if (lane & 1) {
            words[word_idx] = (words[word_idx] & 0x0000FFFFu)
                            | (static_cast<uint32_t>(p16) << 16);
        } else {
            words[word_idx] = (words[word_idx] & 0xFFFF0000u)
                            | static_cast<uint32_t>(p16);
        }
    }
}

static void fill_end_to_end_a(std::vector<half>& a, int M, int K) {
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            const float value = 0.125f * static_cast<float>(m + 1)
                              + 0.03125f * static_cast<float>((k % 7) + 1);
            a[m * K + k] = __float2half(value);
        }
    }
}

static void fill_end_to_end_b(std::vector<uint8_t>& b_packed, int K, int N) {
    const int packed_rows = (K + 1) / 2;
    for (int kp = 0; kp < packed_rows; ++kp) {
        const int k0 = kp * 2;
        for (int n = 0; n < N; ++n) {
            const uint8_t lo = static_cast<uint8_t>(8 + ((k0 + 2 * n) % 3));
            uint8_t hi = static_cast<uint8_t>(8 + (((k0 + 1) + 2 * n) % 3));
            if (k0 + 1 >= K) {
                hi = 8;
            }
            b_packed[kp * N + n] = static_cast<uint8_t>(lo | (hi << 4));
        }
    }
}

static void fill_end_to_end_s(std::vector<half>& scales, int groups, int N) {
    for (int g = 0; g < groups; ++g) {
        for (int n = 0; n < N; ++n) {
            const float scale = 0.125f * static_cast<float>(1 + ((g + n) % 4));
            scales[g * N + n] = __float2half(scale);
        }
    }
}

static float dequant_weight_ref(const std::vector<uint8_t>& b_packed,
                                const std::vector<half>& scales,
                                int N, int group_size,
                                int k, int n) {
    const uint8_t byte = b_packed[(k >> 1) * N + n];
    const uint32_t nibble = (k & 1) ? ((byte >> 4) & 0xFu) : (byte & 0xFu);
    const float scale = __half2float(scales[(k / group_size) * N + n]);
    return scale * (static_cast<float>(nibble) - 8.0f);
}

static bool run_end_to_end_case(int M, int N, int K, int group_size, const char* label) {
    const int packed_rows = (K + 1) / 2;
    const int groups = (K + group_size - 1) / group_size;

    std::vector<half> h_a(M * K);
    std::vector<uint8_t> h_b(packed_rows * N);
    std::vector<half> h_s(groups * N);
    std::vector<half> h_c(M * N, __float2half(0.0f));

    fill_end_to_end_a(h_a, M, K);
    fill_end_to_end_b(h_b, K, N);
    fill_end_to_end_s(h_s, groups, N);

    half* d_a = nullptr;
    half* d_s = nullptr;
    half* d_c = nullptr;
    uint8_t* d_b = nullptr;

    auto fail = [&](const char* what, cudaError_t err) {
        std::fprintf(stderr, "  FAIL %s: %s\n", what, cudaGetErrorString(err));
        if (d_a) cudaFree(d_a);
        if (d_s) cudaFree(d_s);
        if (d_c) cudaFree(d_c);
        if (d_b) cudaFree(d_b);
        return false;
    };

    cudaError_t err = cudaMalloc(&d_a, h_a.size() * sizeof(half));
    if (err != cudaSuccess) return fail("cudaMalloc(d_a)", err);
    err = cudaMalloc(&d_s, h_s.size() * sizeof(half));
    if (err != cudaSuccess) return fail("cudaMalloc(d_s)", err);
    err = cudaMalloc(&d_c, h_c.size() * sizeof(half));
    if (err != cudaSuccess) return fail("cudaMalloc(d_c)", err);
    err = cudaMalloc(&d_b, h_b.size() * sizeof(uint8_t));
    if (err != cudaSuccess) return fail("cudaMalloc(d_b)", err);

    err = cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(half), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return fail("cudaMemcpy(d_a)", err);
    err = cudaMemcpy(d_s, h_s.data(), h_s.size() * sizeof(half), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return fail("cudaMemcpy(d_s)", err);
    err = cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return fail("cudaMemcpy(d_b)", err);
    err = cudaMemset(d_c, 0, h_c.size() * sizeof(half));
    if (err != cudaSuccess) return fail("cudaMemset(d_c)", err);

    err = gptq_pro_gemm(d_a, d_b, d_s, d_c, M, N, K, group_size, 0);
    if (err != cudaSuccess) return fail("gptq_pro_gemm launch", err);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return fail("cudaDeviceSynchronize()", err);
    err = cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(half), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return fail("cudaMemcpy(h_c)", err);

    int mismatches = 0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                const float a = __half2float(h_a[m * K + k]);
                const float w = dequant_weight_ref(h_b, h_s, N, group_size, k, n);
                acc = fma_f32_from_f16_inputs_proxy(a, w, acc);
            }
            const float expect = __half2float(__float2half(acc));
            const float got = __half2float(h_c[m * N + n]);
            if (fabsf(got - expect) > 1e-3f) {
                if (mismatches < 8) {
                    std::fprintf(stderr,
                                 "  FAIL %s at (%d, %d): got=%f expect=%f\n",
                                 label, m, n, got, expect);
                }
                ++mismatches;
            }
        }
    }

    cudaFree(d_a);
    cudaFree(d_s);
    cudaFree(d_c);
    cudaFree(d_b);

    if (mismatches != 0) {
        std::fprintf(stderr, "  %s mismatches: %d\n", label, mismatches);
        return false;
    }

    std::printf("  PASS %s\n", label);
    return true;
}

#ifndef GPTQ_PRO_VALIDATE_SKIP_MAIN

int main() {
    const float scale_f = 0.015625f;  // 2^-6, exact in FP16
    const float zp_f    = 8.0f;       // unsigned-to-signed shift

    // -----------------------------------------------------------------------
    // TODO 1 — Decode validation
    // -----------------------------------------------------------------------
    printf("=== TODO 1: decode-only validation ===\n");

    // Build host Bfrag image.
    const int bfrag_words = GPTQ_PRO_BFRAG_WORDS_PER_BUF;
    uint32_t h_bfrag[bfrag_words];
    fill_bfrag_test_image(h_bfrag);

    // Compute scalar reference for every lane.
    float h_ref_rb[GPTQ_PRO_WARP_SIZE * 4];
    for (int lane = 0; lane < GPTQ_PRO_WARP_SIZE; ++lane) {
        uint32_t w0 = (4u * lane + 0u) & 0xFu;
        uint32_t w1 = (4u * lane + 1u) & 0xFu;
        uint32_t w2 = (4u * lane + 2u) & 0xFu;
        uint32_t w3 = (4u * lane + 3u) & 0xFu;
        float decoded[4];
        scalar_decode_bfrag(w0, w1, w2, w3, scale_f, zp_f, decoded);
        for (int i = 0; i < 4; ++i) {
            h_ref_rb[lane * 4 + i] = decoded[i];
        }
    }

    // Allocate device memory.
    uint32_t* d_bfrag;
    float*    d_ref_rb;
    int*      d_result1;
    CHECK_CUDA(cudaMalloc(&d_bfrag,   bfrag_words * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_ref_rb,  GPTQ_PRO_WARP_SIZE * 4 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_result1, GPTQ_PRO_WARP_SIZE * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(
        d_bfrag, h_bfrag, bfrag_words * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        d_ref_rb, h_ref_rb, GPTQ_PRO_WARP_SIZE * 4 * sizeof(float),
        cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_result1, 0, GPTQ_PRO_WARP_SIZE * sizeof(int)));

    size_t smem_bytes = GPTQ_PRO_BFRAG_WORDS_PER_BUF * sizeof(uint32_t);
    validate_decode_kernel<<<1, GPTQ_PRO_WARP_SIZE, smem_bytes>>>(
        d_bfrag, scale_f, zp_f, d_ref_rb, d_result1);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    int h_result1[GPTQ_PRO_WARP_SIZE];
    CHECK_CUDA(cudaMemcpy(
        h_result1, d_result1, GPTQ_PRO_WARP_SIZE * sizeof(int),
        cudaMemcpyDeviceToHost));

    int pass1 = 0;
    for (int i = 0; i < GPTQ_PRO_WARP_SIZE; ++i) {
        const bool ok = (h_result1[i] == 1);
        pass1 += ok ? 1 : 0;
        if (!ok) printf("  FAIL lane %d\n", i);
    }
    printf("  %d / %d lanes passed\n", pass1, GPTQ_PRO_WARP_SIZE);

    CHECK_CUDA(cudaFree(d_bfrag));
    CHECK_CUDA(cudaFree(d_ref_rb));
    CHECK_CUDA(cudaFree(d_result1));

    // -----------------------------------------------------------------------
    // TODO 2 — MMA step validation
    // -----------------------------------------------------------------------
    printf("=== TODO 2: ks/j MMA step validation ===\n");

    // Build synthetic RA, RB fragments from the PTX-defined fragment ownership.
    // Both A and B vary with their logical coordinates, so row/column mix-ups in
    // either fragment contract now perturb the final D fragment immediately.
    float h_a_tile[GPTQ_PRO_M_PER_WARP][GPTQ_PRO_K_PER_WARP];
    float h_b_tile[GPTQ_PRO_K_PER_WARP][8];
    for (int m = 0; m < GPTQ_PRO_M_PER_WARP; ++m) {
        for (int k = 0; k < GPTQ_PRO_K_PER_WARP; ++k) {
            h_a_tile[m][k] = mma_ref_a_value(m, k);
        }
    }
    for (int k = 0; k < GPTQ_PRO_K_PER_WARP; ++k) {
        for (int n = 0; n < 8; ++n) {
            h_b_tile[k][n] = mma_ref_b_value(k, n);
        }
    }

    uint32_t h_ra[GPTQ_PRO_WARP_SIZE * 4];
    uint32_t h_rb[GPTQ_PRO_WARP_SIZE * 2];
    for (int lane = 0; lane < GPTQ_PRO_WARP_SIZE; ++lane) {
        const int group_id = lane >> 2;
        const int thread_id = lane & 3;
        const int a_col_lo = 2 * thread_id;
        const int a_col_hi = a_col_lo + 8;
        const int row0 = 2 * thread_id + 0;
        const int row1 = 2 * thread_id + 1;

        h_ra[lane * 4 + 0] = pack_half2(
            h_a_tile[group_id + 0][a_col_lo + 0],
            h_a_tile[group_id + 0][a_col_lo + 1]);
        h_ra[lane * 4 + 1] = pack_half2(
            h_a_tile[group_id + 8][a_col_lo + 0],
            h_a_tile[group_id + 8][a_col_lo + 1]);
        h_ra[lane * 4 + 2] = pack_half2(
            h_a_tile[group_id + 0][a_col_hi + 0],
            h_a_tile[group_id + 0][a_col_hi + 1]);
        h_ra[lane * 4 + 3] = pack_half2(
            h_a_tile[group_id + 8][a_col_hi + 0],
            h_a_tile[group_id + 8][a_col_hi + 1]);

        h_rb[lane * 2 + 0] = pack_half2(
            h_b_tile[row0 + 0][group_id],
            h_b_tile[row1 + 0][group_id]);
        h_rb[lane * 2 + 1] = pack_half2(
            h_b_tile[row0 + 8][group_id],
            h_b_tile[row1 + 8][group_id]);
    }

    // Scalar reference: each D[m][n] = sum_{k=0}^{15} A[m][k] * B[k][n].
    // For m16n8k16.row.col.f32 the lane-local D fragment is a 2-row x 2-column
    // tile, so compare against the full 16x8 GEMM reference rather than a
    // collapsed per-column reduction.
    //   rows = {lane >> 2, lane >> 2 + 8}
    //   cols = {2 * (lane & 3), 2 * (lane & 3) + 1}
    float h_ref_rc[GPTQ_PRO_WARP_SIZE * 4];
    {
        float h_d_tile[GPTQ_PRO_M_PER_WARP][8];
        for (int m = 0; m < GPTQ_PRO_M_PER_WARP; ++m) {
            for (int n = 0; n < 8; ++n) {
                float acc = 0.0f;
                for (int k = 0; k < GPTQ_PRO_K_PER_WARP; ++k) {
                    acc = fma_f32_from_f16_inputs_proxy(
                        h_a_tile[m][k], h_b_tile[k][n], acc);
                }
                h_d_tile[m][n] = acc;
            }
        }
        for (int lane = 0; lane < GPTQ_PRO_WARP_SIZE; ++lane) {
            const int row_base = lane >> 2;
            const int col_base = 2 * (lane & 3);
            h_ref_rc[lane * 4 + 0] = h_d_tile[row_base + 0][col_base + 0];
            h_ref_rc[lane * 4 + 1] = h_d_tile[row_base + 0][col_base + 1];
            h_ref_rc[lane * 4 + 2] = h_d_tile[row_base + 8][col_base + 0];
            h_ref_rc[lane * 4 + 3] = h_d_tile[row_base + 8][col_base + 1];
        }
    }

    uint32_t* d_ra;
    uint32_t* d_rb;
    float*    d_ref_rc;
    int*      d_result2;
    CHECK_CUDA(cudaMalloc(&d_ra,     GPTQ_PRO_WARP_SIZE * 4 * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_rb,     GPTQ_PRO_WARP_SIZE * 2 * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_ref_rc, GPTQ_PRO_WARP_SIZE * 4 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_result2, GPTQ_PRO_WARP_SIZE * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(
        d_ra, h_ra, GPTQ_PRO_WARP_SIZE * 4 * sizeof(uint32_t),
        cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        d_rb, h_rb, GPTQ_PRO_WARP_SIZE * 2 * sizeof(uint32_t),
        cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        d_ref_rc, h_ref_rc, GPTQ_PRO_WARP_SIZE * 4 * sizeof(float),
        cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_result2, 0, GPTQ_PRO_WARP_SIZE * sizeof(int)));

    validate_mma_step_kernel<<<1, GPTQ_PRO_WARP_SIZE>>>(
        d_ra, d_rb, d_ref_rc, d_result2);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    int h_result2[GPTQ_PRO_WARP_SIZE];
    CHECK_CUDA(cudaMemcpy(
        h_result2, d_result2, GPTQ_PRO_WARP_SIZE * sizeof(int),
        cudaMemcpyDeviceToHost));

    int pass2 = 0;
    for (int i = 0; i < GPTQ_PRO_WARP_SIZE; ++i) {
        const bool ok = (h_result2[i] == 1);
        pass2 += ok ? 1 : 0;
        if (!ok) printf("  FAIL lane %d\n", i);
    }
    printf("  %d / %d lanes passed\n", pass2, GPTQ_PRO_WARP_SIZE);

    CHECK_CUDA(cudaFree(d_ra));
    CHECK_CUDA(cudaFree(d_rb));
    CHECK_CUDA(cudaFree(d_ref_rc));
    CHECK_CUDA(cudaFree(d_result2));

    // -----------------------------------------------------------------------
    // TODO 3 — End-to-end kernel validation
    // -----------------------------------------------------------------------
    printf("=== TODO 3: end-to-end kernel validation ===\n");
    int pass3 = 0;
    pass3 += run_end_to_end_case(16, 64, 16, 16, "aligned-16x64x16") ? 1 : 0;
    pass3 += run_end_to_end_case(13, 41, 29, 16, "edge-13x41x29") ? 1 : 0;
    printf("  %d / %d cases passed\n", pass3, 2);

    // -----------------------------------------------------------------------
    int total = pass1 + pass2 + pass3;
    int total_max = 2 * GPTQ_PRO_WARP_SIZE + 2;
    printf("\n=== Overall: %d / %d checks passed ===\n", total, total_max);
    return (total == total_max) ? 0 : 1;
}

#endif  // GPTQ_PRO_VALIDATE_SKIP_MAIN
