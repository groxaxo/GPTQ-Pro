/*
 * GPTQ-Pro custom INT4 dequantized GEMM kernel
 *
 * Architecture: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
 *   A: FP16 activations  (M×K, row-major)
 *   B: INT4 weights      (K×N, col-major, packed as nibbles)
 *   C: FP16 output       (M×N, row-major, FP32 accumulation with FP16 store)
 *
 * Key design:
 *   - PIPE=2 double-buffered shared memory with CP_ASYNC
 *   - KS_TILES=4 K-dimension inner stages (each mma.sync covers k=4 elements)
 *   - J_TILES=8  N-dimension inner tiles  (each mma.sync covers n=8 elements)
 *   - B weights stored as packed INT4 nibbles; decoded in-register to FP16
 *   - Per-group scale and offset (GPTQ asymmetric quantization)
 *   - Paro transform (pairwise rotation) fused before MMA
 *
 * Physical Bfrag shared-memory layout (TODO 4 — implemented):
 *   Each (ks, j) tile uses 16 uint32_t words (one word per lane-pair).
 *   Word index = lane >> 1 (4-byte aligned).
 *   Within a word:
 *     bits [15:0]  → two INT4 nibbles for even lane (2k):
 *                      bits [3:0]  = w0 nibble  (RB[0], low half)
 *                      bits [7:4]  = reserved in the current scaffold
 *                      bits [11:8] = w1 nibble  (RB[1], low half)
 *                      bits [15:12]= reserved in the current scaffold
 *     bits [31:16] → same pattern for odd lane (2k+1)
 */

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Tile dimensions
// ---------------------------------------------------------------------------
static constexpr int GPTQ_PRO_PIPE      = 2;   // double-buffered smem
static constexpr int GPTQ_PRO_KS_TILES  = 4;   // K inner iterations
static constexpr int GPTQ_PRO_J_TILES   = 8;   // N inner iterations
static constexpr int GPTQ_PRO_WARP_SIZE = 32;

// Warp-tile covers (m=16, n=64, k=16).
// The n=64 comes from J_TILES=8 x n8 per mma.sync.
// The k=16 comes from KS_TILES=4 x k4 per mma.sync.
static constexpr int GPTQ_PRO_M_PER_WARP = 16;
static constexpr int GPTQ_PRO_N_PER_WARP = GPTQ_PRO_J_TILES  * 8;   // 64
static constexpr int GPTQ_PRO_K_PER_WARP = GPTQ_PRO_KS_TILES * 4;   // 16

// Number of uint32_t words per (ks,j) tile in Bfrag smem (lane-pair packing).
static constexpr int GPTQ_PRO_BFRAG_WORDS_PER_TILE =
    GPTQ_PRO_WARP_SIZE / 2;  // 16 words

// Total uint32_t words for all (ks,j) tiles in one smem buffer.
static constexpr int GPTQ_PRO_BFRAG_WORDS_PER_BUF =
    GPTQ_PRO_KS_TILES * GPTQ_PRO_J_TILES * GPTQ_PRO_BFRAG_WORDS_PER_TILE;

// ---------------------------------------------------------------------------
// Helper types
// ---------------------------------------------------------------------------
union Half2Reg {
    half2   h2;
    uint32_t u32;
    uint16_t u16[2];
};

// ---------------------------------------------------------------------------
// TODO 4 — Physical ld.shared.u32 layout helpers for Bfrag
//
// Shared-memory address for lane `lane` in tile (ks, j), double buffer `buf`.
// The layout packs consecutive lane-pairs into one 4-byte word:
//   lane pair (2k, 2k+1) → word at offset (2k >> 1) = k.
// A single ld.shared.u32 at that word gives both lane's nibble data.
// ---------------------------------------------------------------------------

/// Return the shared-memory uint32_t* base for one smem buffer of Bfrag.
/// `smem_bfrag_base` must be 4-byte aligned.
__device__ __forceinline__
uint32_t bfrag_smem_addr(const uint32_t* __restrict__ smem_bfrag_base,
                         int buf, int ks, int j, int lane) {
    // Word index within the whole buffer.
    int tile_idx  = ks * GPTQ_PRO_J_TILES + j;
    int buf_words = GPTQ_PRO_BFRAG_WORDS_PER_BUF;
    int word_idx  = buf * buf_words
                  + tile_idx * GPTQ_PRO_BFRAG_WORDS_PER_TILE
                  + (lane >> 1);
    // Convert to byte address for PTX ld.shared.u32.
    uint32_t generic_ptr =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem_bfrag_base))
        + static_cast<uint32_t>(word_idx * sizeof(uint32_t));
    return generic_ptr;
}

/// Load one uint32_t from shared memory using PTX ld.shared.u32.
__device__ __forceinline__
uint32_t ld_shared_u32(uint32_t smem_addr) {
    uint32_t val;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(val) : "r"(smem_addr));
    return val;
}

/// Store one uint32_t to shared memory using PTX st.shared.u32.
__device__ __forceinline__
void st_shared_u32(uint32_t smem_addr, uint32_t val) {
    asm volatile("st.shared.u32 [%0], %1;" :: "r"(smem_addr), "r"(val));
}

// ---------------------------------------------------------------------------
// TODO 3 — Real shared-memory packed fetch for a B tile (ks, j).
//
// Replaces the old flat-smem access:
//   uint16_t packed_16 = Bfrag[(ks * 8 + j) * 32 + lane];
//
// The new path uses a single ld.shared.u32 per lane-pair, then selects the
// correct uint16_t half based on the lane's parity.  This is 4-byte aligned
// and produces no bank conflicts (each pair hits the same bank exactly once).
// ---------------------------------------------------------------------------

/// Fetch the packed 16-bit nibble word for this lane from tile (ks, j).
/// Returns a uint16_t holding 4 nibbles:
///   bits [3:0]   = weight nibble 0  (will become RB[0] low FP16)
///   bits [7:4]   = reserved in the current scaffold
///   bits [11:8]  = weight nibble 1  (will become RB[1] low FP16)
///   bits [15:12] = reserved in the current scaffold
__device__ __forceinline__
uint16_t fetch_bfrag_packed16(const uint32_t* __restrict__ smem_bfrag,
                              int buf, int ks, int j, int lane) {
    uint32_t addr   = bfrag_smem_addr(smem_bfrag, buf, ks, j, lane);
    uint32_t word   = ld_shared_u32(addr);
    // Even lanes take the low half; odd lanes take the high half.
    return static_cast<uint16_t>((lane & 1) ? (word >> 16) : (word & 0xFFFFu));
}

// ---------------------------------------------------------------------------
// INT4 nibble decode → FP16 (with scale & asymmetric offset)
//
// packed_16 layout (as produced by fetch_bfrag_packed16):
//   bits [3:0]  → w0: INT4 weight for RB[0] half 0
//   bits [11:8] → w1: INT4 weight for RB[1] half 0
// (The high nibble in each byte is currently unused by the scaffold.)
//
// Decode: fp16_w = scale * (nibble - zero_point)
// The scaffold uses direct int→half conversion because the focus here is the
// fragment contract and accumulation semantics, not the final decode fast path.
// ---------------------------------------------------------------------------
__device__ __forceinline__
void decode_bfrag_to_rb(uint16_t packed_16,
                        half scale, half zero_point,
                        uint32_t (&RB)[2]) {
    // Extract the two 4-bit weights (unsigned, range 0–15).
    const uint32_t p = static_cast<uint32_t>(packed_16);
    const uint32_t w0 = (p >> 0) & 0xFu;
    const uint32_t w1 = (p >> 8) & 0xFu;

    // Direct integer → half conversion (range 0–15, exact in FP16).
    half2 vals = __halves2half2(__int2half_rn(static_cast<int>(w0)),
                                __int2half_rn(static_cast<int>(w1)));

    // Apply asymmetric dequantization: result = scale * (w - zero_point).
    const half2 zp_h2 = __halves2half2(zero_point, zero_point);
    const half2 sc_h2 = __halves2half2(scale, scale);
    half2 dq = __hmul2(sc_h2, __hsub2(vals, zp_h2));

    // Pack back into two uint32_t RB registers (each holds one FP16 pair).
    // RB[0]: both halves of the n-column 0 (same weight, different k rows).
    // RB[1]: both halves of the n-column 1.
    Half2Reg rb0, rb1;
    rb0.h2 = __halves2half2(__low2half(dq), __low2half(dq));   // w0, w0
    rb1.h2 = __halves2half2(__high2half(dq), __high2half(dq)); // w1, w1
    RB[0] = rb0.u32;
    RB[1] = rb1.u32;
}

// ---------------------------------------------------------------------------
// FP32 accumulating MMA: RC[j] += RA × RB  (m16n8k16.row.col.f32.f16.f16.f32)
// ---------------------------------------------------------------------------
__device__ __forceinline__
void mma_f32_m16n8k16(const uint32_t RA[4],
                      const uint32_t RB[2],
                      float          RC[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(RC[0]), "+f"(RC[1]), "+f"(RC[2]), "+f"(RC[3])
        :  "r"(RA[0]),  "r"(RA[1]),  "r"(RA[2]),  "r"(RA[3]),
           "r"(RB[0]),  "r"(RB[1]));
}

// ---------------------------------------------------------------------------
// CP_ASYNC helpers (sm80+)
// ---------------------------------------------------------------------------
__device__ __forceinline__
void cp_async16(void* __restrict__ dst, const void* __restrict__ src) {
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], %2;\n"
        :: "r"(smem), "l"(src), "n"(16));
}

__device__ __forceinline__ void cp_async_fence() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}
