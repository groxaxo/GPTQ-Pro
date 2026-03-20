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
 *   nvcc -arch=sm_80 -std=c++17 -I.. gptq_pro_validate.cu -o gptq_pro_validate
 */

#include "gptq_pro_kernel.cuh"

#include <cstdio>
#include <cmath>
#include <cstring>

// ============================================================
// TODO 1  — Decode-only scalar reference
// ============================================================

/// Scalar (lane-independent) decode of one INT4 nibble pair.
/// Mirrors exactly what decode_bfrag_to_rb() does on-device, using only
/// host-visible arithmetic so the result can be used as ground truth.
///
/// @param w0          unsigned 4-bit integer [0, 15] for RB[0]
/// @param w1          unsigned 4-bit integer [0, 15] for RB[1]
/// @param scale_f     FP32 version of the per-group scale
/// @param zp_f        FP32 version of the per-group zero-point
/// @param out_rb0_f   [out] dequantized value that goes into RB[0]
/// @param out_rb1_f   [out] dequantized value that goes into RB[1]
inline void scalar_decode_nibble(uint32_t w0, uint32_t w1,
                                  float scale_f, float zp_f,
                                  float& out_rb0_f, float& out_rb1_f) {
    out_rb0_f = scale_f * (static_cast<float>(w0) - zp_f);
    out_rb1_f = scale_f * (static_cast<float>(w1) - zp_f);
}

// ---------------------------------------------------------------------------
// Device-side TODO 1 validation kernel
//
// Each thread (= one warp lane):
//   1. Reads its packed INT4 data from a pre-filled Bfrag smem region.
//   2. Calls fetch_bfrag_packed16() (the real ld.shared.u32 path).
//   3. Calls decode_bfrag_to_rb() to obtain RB[0], RB[1].
//   4. Compares against a pre-computed float reference stored in ref_rb0 and
//      ref_rb1 (one float per lane, computed by scalar_decode_nibble on host).
//   5. Sets result[lane] = 1 if both half2 lanes match within FP16 precision.
// ---------------------------------------------------------------------------
__global__ void validate_decode_kernel(
    const uint32_t* __restrict__ bfrag_smem_src,  // GPTQ_PRO_BFRAG_WORDS_PER_BUF words (global, will be copied to smem)
    float           scale_f,
    float           zp_f,
    const float*    ref_rb0,   // [WARP_SIZE] ground-truth for RB[0] per lane
    const float*    ref_rb1,   // [WARP_SIZE] ground-truth for RB[1] per lane
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

    bool ok = (fabsf(got_rb0_lo - ref_rb0[lane]) <= tol + 1e-5f) &&
              (fabsf(got_rb0_hi - ref_rb0[lane]) <= tol + 1e-5f) &&
              (fabsf(got_rb1_lo - ref_rb1[lane]) <= tol + 1e-5f) &&
              (fabsf(got_rb1_hi - ref_rb1[lane]) <= tol + 1e-5f);
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
/// Thread p (0..31) owns:
///   RA: A[(p>>2) + {0,8}][(p&3)*2 + {0,1}]  → a[0..3] (4 fp16 in 2 u32)
///   RB: B[(p>>2)*2 + {0,1}][(p&3)*2 + {0,1}] → b[0..1] (4 fp16 in 2 u32 via
///         half2 packing)
///   RC[j][0..3]: output accumulator fragment (4 fp32 values per tile j)
///
/// This scalar function operates on the unpacked FP32 equivalents; the caller
/// is responsible for unpacking and repacking.
inline float fma_f32_from_f16_inputs_proxy(float a, float b, float c) {
    const half ha = __float2half(a);
    const half hb = __float2half(b);
    return c + __half2float(ha) * __half2float(hb);
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

/// Fill Bfrag shared-memory image with pseudo-random INT4 nibbles.
/// packed_16 for lane `l` in tile (0,0) = (l & 0xF) | ((l+1 & 0xF) << 8).
static void fill_bfrag_test_image(uint32_t* words) {
    for (int w = 0; w < GPTQ_PRO_BFRAG_WORDS_PER_BUF; ++w) {
        words[w] = 0u;
    }
    // Tile ks=0, j=0, buf=0 starts at word offset 0.
    for (int lane = 0; lane < GPTQ_PRO_WARP_SIZE; ++lane) {
        uint32_t w0 = lane & 0xFu;
        uint32_t w1 = (lane + 1u) & 0xFu;
        // packed_16 = w0 | (w1 << 8) in the appropriate byte position.
        uint16_t p16 = static_cast<uint16_t>(w0 | (w1 << 8));
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
    float h_ref_rb0[GPTQ_PRO_WARP_SIZE];
    float h_ref_rb1[GPTQ_PRO_WARP_SIZE];
    for (int lane = 0; lane < GPTQ_PRO_WARP_SIZE; ++lane) {
        uint32_t w0 = lane & 0xFu;
        uint32_t w1 = (lane + 1u) & 0xFu;
        scalar_decode_nibble(w0, w1, scale_f, zp_f,
                             h_ref_rb0[lane], h_ref_rb1[lane]);
    }

    // Allocate device memory.
    uint32_t* d_bfrag;    cudaMalloc(&d_bfrag,    bfrag_words * sizeof(uint32_t));
    float*    d_ref_rb0;  cudaMalloc(&d_ref_rb0,  GPTQ_PRO_WARP_SIZE * sizeof(float));
    float*    d_ref_rb1;  cudaMalloc(&d_ref_rb1,  GPTQ_PRO_WARP_SIZE * sizeof(float));
    int*      d_result1;  cudaMalloc(&d_result1,  GPTQ_PRO_WARP_SIZE * sizeof(int));

    cudaMemcpy(d_bfrag,   h_bfrag,   bfrag_words * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref_rb0, h_ref_rb0, GPTQ_PRO_WARP_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref_rb1, h_ref_rb1, GPTQ_PRO_WARP_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    size_t smem_bytes = GPTQ_PRO_BFRAG_WORDS_PER_BUF * sizeof(uint32_t);
    validate_decode_kernel<<<1, GPTQ_PRO_WARP_SIZE, smem_bytes>>>(
        d_bfrag, scale_f, zp_f, d_ref_rb0, d_ref_rb1, d_result1);
    cudaDeviceSynchronize();

    int h_result1[GPTQ_PRO_WARP_SIZE];
    cudaMemcpy(h_result1, d_result1, GPTQ_PRO_WARP_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    int pass1 = 0;
    for (int i = 0; i < GPTQ_PRO_WARP_SIZE; ++i) {
        pass1 += h_result1[i];
        if (!h_result1[i]) printf("  FAIL lane %d\n", i);
    }
    printf("  %d / %d lanes passed\n", pass1, GPTQ_PRO_WARP_SIZE);

    cudaFree(d_bfrag); cudaFree(d_ref_rb0); cudaFree(d_ref_rb1); cudaFree(d_result1);

    // -----------------------------------------------------------------------
    // TODO 2 — MMA step validation
    // -----------------------------------------------------------------------
    printf("=== TODO 2: ks/j MMA step validation ===\n");

    // Build synthetic RA, RB fragments.
    // All elements = a small FP16-representable constant so the scalar path
    // and the hardware path agree exactly.
    const float a_val = 0.5f;
    const float b_val = 0.25f;

    uint32_t h_ra[GPTQ_PRO_WARP_SIZE * 4];
    uint32_t h_rb[GPTQ_PRO_WARP_SIZE * 2];
    for (int lane = 0; lane < GPTQ_PRO_WARP_SIZE; ++lane) {
        half2 ah2 = __float2half2_rn(a_val);
        half2 bh2 = __float2half2_rn(b_val);
        uint32_t av, bv;
        memcpy(&av, &ah2, 4);
        memcpy(&bv, &bh2, 4);
        h_ra[lane*4+0] = h_ra[lane*4+1] = h_ra[lane*4+2] = h_ra[lane*4+3] = av;
        h_rb[lane*2+0] = h_rb[lane*2+1] = bv;
    }

    // Scalar reference: each D[m][n] = sum_{k=0}^{15} A[m][k] * B[k][n].
    // With uniform inputs, D[m][n] = 16 * a_val * b_val for all (m,n).
    // In FP32 accumulation with FP16 inputs: simulate the 16-step chain in order.
    // For m16n8k16, each thread's RA[0..3] contributes 8 A elements and RB[0..1]
    // contributes 4 B elements; across 4 threads per output element, each
    // thread contributes 4 of the 16 K multiplications.
    // With uniform inputs every lane computes the same reference value.
    float h_ref_rc[GPTQ_PRO_WARP_SIZE * 4];
    {
        // Simulate k=16 FP32-precision accumulations with FP16-rounded inputs.
        float acc = 0.0f;
        for (int k = 0; k < 16; ++k) {
            acc = fma_f32_from_f16_inputs_proxy(a_val, b_val, acc);
        }
        for (int lane = 0; lane < GPTQ_PRO_WARP_SIZE; ++lane) {
            for (int c = 0; c < 4; ++c) {
                h_ref_rc[lane*4 + c] = acc;
            }
        }
    }

    uint32_t* d_ra;     cudaMalloc(&d_ra,    GPTQ_PRO_WARP_SIZE * 4 * sizeof(uint32_t));
    uint32_t* d_rb;     cudaMalloc(&d_rb,    GPTQ_PRO_WARP_SIZE * 2 * sizeof(uint32_t));
    float*    d_ref_rc; cudaMalloc(&d_ref_rc,GPTQ_PRO_WARP_SIZE * 4 * sizeof(float));
    int*      d_result2;cudaMalloc(&d_result2,GPTQ_PRO_WARP_SIZE * sizeof(int));

    cudaMemcpy(d_ra,     h_ra,     GPTQ_PRO_WARP_SIZE * 4 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rb,     h_rb,     GPTQ_PRO_WARP_SIZE * 2 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref_rc, h_ref_rc, GPTQ_PRO_WARP_SIZE * 4 * sizeof(float),    cudaMemcpyHostToDevice);

    validate_mma_step_kernel<<<1, GPTQ_PRO_WARP_SIZE>>>(
        d_ra, d_rb, d_ref_rc, d_result2);
    cudaDeviceSynchronize();

    int h_result2[GPTQ_PRO_WARP_SIZE];
    cudaMemcpy(h_result2, d_result2, GPTQ_PRO_WARP_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    int pass2 = 0;
    for (int i = 0; i < GPTQ_PRO_WARP_SIZE; ++i) {
        pass2 += h_result2[i];
        if (!h_result2[i]) printf("  FAIL lane %d\n", i);
    }
    printf("  %d / %d lanes passed\n", pass2, GPTQ_PRO_WARP_SIZE);

    cudaFree(d_ra); cudaFree(d_rb); cudaFree(d_ref_rc); cudaFree(d_result2);

    // -----------------------------------------------------------------------
    int total = pass1 + pass2;
    int total_max = 2 * GPTQ_PRO_WARP_SIZE;
    printf("\n=== Overall: %d / %d checks passed ===\n", total, total_max);
    return (total == total_max) ? 0 : 1;
}

#endif  // GPTQ_PRO_VALIDATE_SKIP_MAIN
