/*
 * GPTQ-Pro Ampere INT4 kernel primitives.
 *
 * The runtime has three execution paths:
 *   - a dedicated coalesced GEMV kernel for very small M;
 *   - a four-warp, double-buffered cp.async Tensor Core kernel for aligned GEMM;
 *   - the original general-shape kernel as a correctness fallback.
 *
 * Symmetric INT4 weights are stored as unsigned nibbles with an implicit
 * zero-point of 8. Tensor Core paths dequantize to FP16 and accumulate in FP32.
 */

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Tile and dispatch constants
// ---------------------------------------------------------------------------
static constexpr int GPTQ_PRO_PIPE = 2;
static constexpr int GPTQ_PRO_KS_TILES = 1;
static constexpr int GPTQ_PRO_J_TILES = 8;
static constexpr int GPTQ_PRO_WARP_SIZE = 32;

static constexpr int GPTQ_PRO_M_PER_WARP = 16;
static constexpr int GPTQ_PRO_N_PER_WARP = GPTQ_PRO_J_TILES * 8;    // 64
static constexpr int GPTQ_PRO_K_PER_WARP = GPTQ_PRO_KS_TILES * 16;  // 16

static constexpr int GPTQ_PRO_WARPS_PER_CTA = 4;
static constexpr int GPTQ_PRO_THREADS_PER_CTA =
    GPTQ_PRO_WARPS_PER_CTA * GPTQ_PRO_WARP_SIZE;
static constexpr int GPTQ_PRO_N_PER_CTA =
    GPTQ_PRO_WARPS_PER_CTA * GPTQ_PRO_N_PER_WARP;  // 256

static constexpr int GPTQ_PRO_B_PACKED_ROWS_PER_K_TILE =
    GPTQ_PRO_K_PER_WARP / 2;  // 8 packed rows
static constexpr int GPTQ_PRO_B_BYTES_PER_WARP_TILE =
    GPTQ_PRO_B_PACKED_ROWS_PER_K_TILE * GPTQ_PRO_N_PER_WARP;  // 512

static constexpr int GPTQ_PRO_GEMV_THREADS = 128;
static constexpr int GPTQ_PRO_GEMV_MAX_M = 4;

// Number of uint32_t words per legacy (ks,j) tile in Bfrag shared memory.
static constexpr int GPTQ_PRO_BFRAG_WORDS_PER_TILE =
    GPTQ_PRO_WARP_SIZE / 2;
static constexpr int GPTQ_PRO_BFRAG_WORDS_PER_BUF =
    GPTQ_PRO_KS_TILES * GPTQ_PRO_J_TILES *
    GPTQ_PRO_BFRAG_WORDS_PER_TILE;

enum GptqProKernelMode : int {
    GPTQ_PRO_KERNEL_AUTO = 0,
    GPTQ_PRO_KERNEL_GEMV = 1,
    GPTQ_PRO_KERNEL_AMPERE = 2,
    GPTQ_PRO_KERNEL_LEGACY = 3,
};

// ---------------------------------------------------------------------------
// Helper types
// ---------------------------------------------------------------------------
union Half2Reg {
    half2 h2;
    uint32_t u32;
    uint16_t u16[2];
};

__device__ __forceinline__ uint32_t pack_half2_reg(half lo, half hi) {
    Half2Reg reg;
    reg.h2 = __halves2half2(lo, hi);
    return reg.u32;
}

// ---------------------------------------------------------------------------
// Ampere asynchronous copy helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ void cp_async_ca_16(
    void* smem_ptr, const void* global_ptr, int source_bytes = 16) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t smem =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
        :
        : "r"(smem), "l"(global_ptr), "r"(source_bytes));
#else
    (void)smem_ptr;
    (void)global_ptr;
    (void)source_bytes;
#endif
}

__device__ __forceinline__ void cp_async_cg_16(
    void* smem_ptr, const void* global_ptr, int source_bytes = 16) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t smem =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :
        : "r"(smem), "l"(global_ptr), "r"(source_bytes));
#else
    (void)smem_ptr;
    (void)global_ptr;
    (void)source_bytes;
#endif
}

__device__ __forceinline__ void cp_async_commit_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int PendingGroups>
__device__ __forceinline__ void cp_async_wait_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" : : "n"(PendingGroups));
#endif
}

// ---------------------------------------------------------------------------
// LOP3-assisted INT4 decode
// ---------------------------------------------------------------------------
template <int Lut>
__device__ __forceinline__ uint32_t lop3_u32(
    uint32_t a, uint32_t b, uint32_t c) {
    uint32_t result;
    asm volatile(
        "lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(result)
        : "r"(a), "r"(b), "r"(c), "n"(Lut));
    return result;
}

// Convert four packed nibbles to the two FP16 B-fragment registers expected by
// mma.sync. The bit trick follows the Apache-2.0 Marlin/FasterTransformer-style
// conversion and fuses the symmetric -8 offset before scale multiplication.
__device__ __forceinline__ void decode_bfrag_to_rb(
    uint16_t packed_16, half scale, half zero_point, uint32_t (&RB)[2]) {
    (void)zero_point;  // GPTQ-Pro's runtime contract always uses zero-point 8.

    const uint32_t p = static_cast<uint32_t>(packed_16);
    // Reorder [w0,w1,w2,w3] so LOP3 creates half2(w0,w1) and
    // half2(w2,w3), matching the PTX B-fragment register order.
    const uint32_t q =
        (p & 0x0000000Fu) | ((p >> 4) & 0x000000F0u) |
        ((p & 0x000000F0u) << 12) | ((p & 0x0000F000u) << 8);

    constexpr uint32_t LO = 0x000f000f;
    constexpr uint32_t HI = 0x00f000f0;
    constexpr uint32_t EX = 0x64006400;
    constexpr uint32_t SUB = 0x64086408;
    constexpr uint32_t MUL = 0x2c002c00;
    constexpr uint32_t ADD = 0xd480d480;

    constexpr int AND_OR_LUT = (0xf0 & 0xcc) | 0xaa;
    Half2Reg lo_reg;
    Half2Reg hi_reg;
    Half2Reg sub_reg;
    Half2Reg mul_reg;
    Half2Reg add_reg;
    Half2Reg out0;
    Half2Reg out1;

    lo_reg.u32 = lop3_u32<AND_OR_LUT>(q, LO, EX);
    hi_reg.u32 = lop3_u32<AND_OR_LUT>(q, HI, EX);
    sub_reg.u32 = SUB;
    mul_reg.u32 = MUL;
    add_reg.u32 = ADD;

    const half2 scale2 = __half2half2(scale);
    out0.h2 = __hmul2(scale2, __hsub2(lo_reg.h2, sub_reg.h2));
    out1.h2 = __hmul2(
        scale2, __hfma2(hi_reg.h2, mul_reg.h2, add_reg.h2));
    RB[0] = out0.u32;
    RB[1] = out1.u32;
}

// ---------------------------------------------------------------------------
// Raw packed-B shared-memory layout used by the optimized pipeline
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint16_t load_raw_bfrag_packed16(
    const uint8_t* __restrict__ smem_b, int j, int lane) {
    const int group_id = lane >> 2;
    const int thread_id = lane & 3;
    const int n_local = j * 8 + group_id;
    const uint8_t byte01 =
        smem_b[thread_id * GPTQ_PRO_N_PER_WARP + n_local];
    const uint8_t byte89 =
        smem_b[(thread_id + 4) * GPTQ_PRO_N_PER_WARP + n_local];
    return static_cast<uint16_t>(byte01) |
           (static_cast<uint16_t>(byte89) << 8);
}

// ---------------------------------------------------------------------------
// Legacy B-fragment helpers retained for the general-shape fallback and
// standalone fragment validator.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t bfrag_smem_addr(
    const uint32_t* __restrict__ smem_bfrag_base,
    int buf,
    int ks,
    int j,
    int lane) {
    const int tile_idx = ks * GPTQ_PRO_J_TILES + j;
    const int buf_words = GPTQ_PRO_BFRAG_WORDS_PER_BUF;
    const int word_idx =
        buf * buf_words + tile_idx * GPTQ_PRO_BFRAG_WORDS_PER_TILE +
        (lane >> 1);
    return static_cast<uint32_t>(
               __cvta_generic_to_shared(smem_bfrag_base)) +
           static_cast<uint32_t>(word_idx * sizeof(uint32_t));
}

__device__ __forceinline__ uint32_t ld_shared_u32(uint32_t smem_addr) {
    uint32_t value;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(value) : "r"(smem_addr));
    return value;
}

__device__ __forceinline__ uint16_t fetch_bfrag_packed16(
    const uint32_t* __restrict__ smem_bfrag,
    int buf,
    int ks,
    int j,
    int lane) {
    const uint32_t address =
        bfrag_smem_addr(smem_bfrag, buf, ks, j, lane);
    const uint32_t word = ld_shared_u32(address);
    return static_cast<uint16_t>(
        (lane & 1) ? (word >> 16) : (word & 0xFFFFu));
}

// ---------------------------------------------------------------------------
// A-fragment packing for mma.sync.aligned.m16n8k16.row.col
// ---------------------------------------------------------------------------
__device__ __forceinline__ void load_a_fragment_rowmajor(
    const half* __restrict__ smem_a, int lane, uint32_t (&RA)[4]) {
    const int group_id = lane >> 2;
    const int thread_id = lane & 3;
    const int a_col_lo = 2 * thread_id;
    const int a_col_hi = a_col_lo + 8;

    RA[0] = pack_half2_reg(
        smem_a[(group_id + 0) * GPTQ_PRO_K_PER_WARP + a_col_lo + 0],
        smem_a[(group_id + 0) * GPTQ_PRO_K_PER_WARP + a_col_lo + 1]);
    RA[1] = pack_half2_reg(
        smem_a[(group_id + 8) * GPTQ_PRO_K_PER_WARP + a_col_lo + 0],
        smem_a[(group_id + 8) * GPTQ_PRO_K_PER_WARP + a_col_lo + 1]);
    RA[2] = pack_half2_reg(
        smem_a[(group_id + 0) * GPTQ_PRO_K_PER_WARP + a_col_hi + 0],
        smem_a[(group_id + 0) * GPTQ_PRO_K_PER_WARP + a_col_hi + 1]);
    RA[3] = pack_half2_reg(
        smem_a[(group_id + 8) * GPTQ_PRO_K_PER_WARP + a_col_hi + 0],
        smem_a[(group_id + 8) * GPTQ_PRO_K_PER_WARP + a_col_hi + 1]);
}

// ---------------------------------------------------------------------------
// FP32 accumulating Tensor Core MMA
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mma_f32_m16n8k16(
    const uint32_t RA[4], const uint32_t RB[2], float RC[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(RC[0]), "+f"(RC[1]), "+f"(RC[2]), "+f"(RC[3])
        : "r"(RA[0]),
          "r"(RA[1]),
          "r"(RA[2]),
          "r"(RA[3]),
          "r"(RB[0]),
          "r"(RB[1]));
}

cudaError_t gptq_pro_gemm(
    const half* A,
    const uint8_t* B_packed,
    const half* S,
    half* C,
    int M,
    int N,
    int K,
    int group_size,
    cudaStream_t stream,
    int kernel_mode = GPTQ_PRO_KERNEL_AUTO);
