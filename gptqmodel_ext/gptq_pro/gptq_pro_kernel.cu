/*
 * GPTQ-Pro INT4 dequantized GEMM kernel
 *
 * Implements the standalone scaffold for the pipelined kernel discussed in the
 * design review. The shared-memory ring and fragment math are real; the async
 * global-to-shared staging and Paro metadata path are still placeholders.
 *
 * Tile shape (one warp):  M=16,  N=64 (8 × n8),  K=16 (4 × k4)
 * Shared memory:          PIPE=2 double-buffered stages
 * Pipeline:               CP_ASYNC + cp_async_wait<PIPE-2>
 * Accumulator:            float RC[J_TILES][4]     (FP32 outputs)
 *
 * Completed TODOs (relative to the research scaffold discussed in review):
 *   [TODO 3] Replaced flat smem access `uint16_t packed_16 = Bfrag[...]`
 *            with the real ld.shared.u32 packed fetch via fetch_bfrag_packed16().
 *   [TODO 4] Physical Bfrag shared-memory layout with 4-byte aligned lane-pair
 *            packing is implemented in bfrag_smem_addr() in gptq_pro_kernel.cuh.
 *
 * Validation of decode and MMA correctness is in gptq_pro_validate.cu
 * (TODO 1 and TODO 2).
 */

#include "gptq_pro_kernel.cuh"

// ---------------------------------------------------------------------------
// Shared memory layout
//
//  smem_A:      PIPE × (M_PER_WARP × K_PER_WARP) half elements  (A tiles)
//  smem_S:      PIPE × (J_TILES × group_size)   half elements  (scales)
//  smem_Bfrag:  PIPE × BFRAG_WORDS_PER_BUF      uint32_t words  (INT4 B)
//
// smem_Bfrag is 4-byte aligned and uses the lane-pair layout from TODO 4.
// ---------------------------------------------------------------------------
struct GptqProSmem {
    half     A[GPTQ_PRO_PIPE][GPTQ_PRO_M_PER_WARP * GPTQ_PRO_K_PER_WARP];
    half     S[GPTQ_PRO_PIPE][GPTQ_PRO_J_TILES];   // one scale per N-tile per stage
    uint32_t Bfrag[GPTQ_PRO_PIPE * GPTQ_PRO_BFRAG_WORDS_PER_BUF];
};

// ---------------------------------------------------------------------------
// apply_paro_transform: fused pairwise rotation on two adjacent A-fragment
// halves.  This is the lightweight linear transform that must be applied to
// every A row before the MMA, in shared/register space.
//
// The actual rotation matrix is supplied as a 2×2 FP16 parameter pair.
// For the research scaffold we use an identity rotation.
// ---------------------------------------------------------------------------
__device__ __forceinline__
void apply_paro_transform(half2& a0, half2& a1,
                          half cos_th, half sin_th) {
    // a0_new = cos*a0 - sin*a1
    // a1_new = sin*a0 + cos*a1
    half2 c2 = __halves2half2(cos_th, cos_th);
    half2 s2 = __halves2half2(sin_th, sin_th);
    half2 a0_new = __hsub2(__hmul2(c2, a0), __hmul2(s2, a1));
    half2 a1_new = __hadd2(__hmul2(s2, a0), __hmul2(c2, a1));
    a0 = a0_new;
    a1 = a1_new;
}

// ---------------------------------------------------------------------------
// do_mma_inner_loop: executes KS_TILES × J_TILES MMA steps for one warp.
//
// Each step:
//   1. Load A fragment from smem using ldmatrix (x4).
//   2. For each j-tile:
//      a. [TODO 3] Fetch B packed nibbles using ld.shared.u32 path.
//      b. Decode INT4 → FP16 with scale & zero-point.
//      c. Execute mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32.
// ---------------------------------------------------------------------------
__device__ __forceinline__
void do_mma_inner_loop(
    const GptqProSmem* __restrict__ smem,
    int                smem_load_buf,  // which PIPE buffer to read
    int                col_base,       // global N offset for this warp
    float              RC[GPTQ_PRO_J_TILES][4])
{
    const int lane = threadIdx.x & (GPTQ_PRO_WARP_SIZE - 1);

    #pragma unroll
    for (int ks = 0; ks < GPTQ_PRO_KS_TILES; ++ks) {
        // ---- Load A fragment with ldmatrix ----
        uint32_t RA[4];
        {
            // Pointer to A tile for this ks stage within the loaded smem buf.
            const half* A_tile = smem->A[smem_load_buf]
                               + ks * (GPTQ_PRO_M_PER_WARP * 4);
            uint32_t smem_a_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(A_tile))
                + static_cast<uint32_t>(
                    ((lane >> 2) * GPTQ_PRO_K_PER_WARP + (lane & 3) * 2)
                    * sizeof(half));
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                "{%0, %1, %2, %3}, [%4];\n"
                : "=r"(RA[0]), "=r"(RA[1]), "=r"(RA[2]), "=r"(RA[3])
                : "r"(smem_a_addr));
        }

        // ---- For each N-tile, decode B and execute MMA ----
        #pragma unroll
        for (int j = 0; j < GPTQ_PRO_J_TILES; ++j) {
            // ---- Scale for this N tile ----
            half scale = smem->S[smem_load_buf][j];
            half zp    = __float2half(8.0f);   // GPTQ-Pro default zero-point

            // ---- [TODO 3] Fetch packed nibbles via ld.shared.u32 ----
            //  Old (flat smem):
            //    uint16_t packed_16 = Bfrag[(ks * 8 + j) * 32 + lane];
            //  New (physical ld.shared.u32 with 4-byte aligned lane-pair packing):
            uint16_t packed_16 = fetch_bfrag_packed16(
                smem->Bfrag,    // uint32_t* base of the Bfrag smem region
                smem_load_buf,  // double-buffer index
                ks, j, lane);

            // ---- Decode INT4 nibbles → FP16 registers ----
            uint32_t RB[2];
            decode_bfrag_to_rb(packed_16, scale, zp, RB);

            // ---- MMA accumulate (FP32 accumulation) ----
            mma_f32_m16n8k16(RA, RB, RC[j]);
        }
    }
}

// ---------------------------------------------------------------------------
// Main GPTQ-Pro GEMM kernel
//
// Grid:  (M / M_PER_WARP, N / N_PER_WARP, 1)
// Block: (WARP_SIZE, 1, 1)  — one warp per CTA for clarity; extend as needed.
//
// Parameters:
//   A           FP16 activation matrix,    [M, K] row-major
//   B_packed    INT4 weight matrix packed, [K/2, N] byte-packed
//   S           FP16 scale matrix,         [K/group_size, N]
//   C           FP16 output matrix,        [M, N] row-major
//   M, N, K     problem dimensions
//   group_size  quantization group size along K
// ---------------------------------------------------------------------------
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
__global__ void gptq_pro_gemm_kernel(
    const half*    __restrict__ A,
    const uint8_t* __restrict__ B_packed,
    const half*    __restrict__ S,
    half*          __restrict__ C,
    int M, int N, int K, int group_size)
{
    extern __shared__ uint8_t raw_smem[];
    GptqProSmem* smem = reinterpret_cast<GptqProSmem*>(raw_smem);

    const int warp_m = blockIdx.x;
    const int warp_n = blockIdx.y;
    const int lane   = threadIdx.x & (GPTQ_PRO_WARP_SIZE - 1);

    const int m_base  = warp_m * GPTQ_PRO_M_PER_WARP;
    const int n_base  = warp_n * GPTQ_PRO_N_PER_WARP;
    const int num_k_stages = K / GPTQ_PRO_K_PER_WARP;

    // ---- Zero accumulators ----
    float RC[GPTQ_PRO_J_TILES][4];
    #pragma unroll
    for (int j = 0; j < GPTQ_PRO_J_TILES; ++j) {
        RC[j][0] = 0.0f;
        RC[j][1] = 0.0f;
        RC[j][2] = 0.0f;
        RC[j][3] = 0.0f;
    }

    int smem_load_idx  = 0;
    int smem_store_idx = 0;

    // ---- Prefill pipeline (PIPE-1 stages) ----
    #pragma unroll
    for (int pre = 0; pre < GPTQ_PRO_PIPE - 1 && pre < num_k_stages; ++pre) {
        // Issue CP_ASYNC for A tile and B tile at stage `pre`.
        // (Omitted for brevity: address calculation and cp_async16 calls here.)
        cp_async_fence();
        smem_store_idx = (smem_store_idx + 1) % GPTQ_PRO_PIPE;
    }

    // ---- Main K loop ----
    for (int t = 0; t < num_k_stages; ++t) {
        // Prefetch the next stage if it exists.
        if (t + GPTQ_PRO_PIPE - 1 < num_k_stages) {
            // Issue CP_ASYNC for A and B at stage (t + PIPE - 1).
            // Address calculation for B uses the INT4 packed layout.
            // (Full address calculation omitted; real integration sets these up.)
            cp_async_fence();
            smem_store_idx = (smem_store_idx + 1) % GPTQ_PRO_PIPE;
        }

        // Wait until the load-buffer stage is resident (PIPE-2 outstanding).
        cp_async_wait<GPTQ_PRO_PIPE - 2>();
        __syncthreads();   // A: stage resident + CTA-visible

        // ---- Apply paro (pairwise rotation) transform ----
        // Each pair of adjacent A-fragment halves is rotated in-smem.
        // Using identity (cos=1, sin=0) for the scaffold; real kernel loads
        // the rotation coefficients from a per-group parameter buffer.
        if (lane < GPTQ_PRO_M_PER_WARP / 2) {
            half* row0 = smem->A[smem_load_idx] + (lane * 2)     * GPTQ_PRO_K_PER_WARP;
            half* row1 = smem->A[smem_load_idx] + (lane * 2 + 1) * GPTQ_PRO_K_PER_WARP;
            for (int k = 0; k < GPTQ_PRO_K_PER_WARP; k += 2) {
                half2 a0 = *reinterpret_cast<half2*>(row0 + k);
                half2 a1 = *reinterpret_cast<half2*>(row1 + k);
                apply_paro_transform(a0, a1,
                                     __float2half(1.0f),   // cos θ
                                     __float2half(0.0f));  // sin θ
                *reinterpret_cast<half2*>(row0 + k) = a0;
                *reinterpret_cast<half2*>(row1 + k) = a1;
            }
        }
        __syncthreads();   // B: barrier required before MMA reads smem

        // ---- Execute MMA inner loop ----
        do_mma_inner_loop(smem, smem_load_idx, n_base, RC);

        __syncthreads();   // C: stage no longer in use before ring-buffer reuse

        smem_load_idx  = (smem_load_idx  + 1) % GPTQ_PRO_PIPE;
    }

    // ---- Store FP16 results to C ----
    //
    // For m16n8k16.row.col with FP32 accumulators, each lane owns 4 output
    // elements for a given j tile:
    //   rows = {2*tid4 + 0, 2*tid4 + 1, 2*tid4 + 8, 2*tid4 + 9}
    //   col  = groupID
    const int groupID = lane >> 2;
    const int tid4    = lane & 3;
    #pragma unroll
    for (int j = 0; j < GPTQ_PRO_J_TILES; ++j) {
        const int n  = n_base + j * 8 + groupID;
        const int m0 = m_base + 2 * tid4 + 0;
        const int m1 = m_base + 2 * tid4 + 1;
        const int m2 = m_base + 2 * tid4 + 8;
        const int m3 = m_base + 2 * tid4 + 9;

        if (n < N) {
            if (m0 < M) C[m0 * N + n] = __float2half_rn(RC[j][0]);
            if (m1 < M) C[m1 * N + n] = __float2half_rn(RC[j][1]);
            if (m2 < M) C[m2 * N + n] = __float2half_rn(RC[j][2]);
            if (m3 < M) C[m3 * N + n] = __float2half_rn(RC[j][3]);
        }
    }
}

#else  // sm80 stub
__global__ void gptq_pro_gemm_kernel(
    const half*, const uint8_t*, const half*, half*, int, int, int, int) {}
#define GPTQ_PRO_SM80_STUB 1
#endif

// ---------------------------------------------------------------------------
// Host launcher (always compiled)
// ---------------------------------------------------------------------------
cudaError_t gptq_pro_gemm(
    const half*    A,
    const uint8_t* B_packed,
    const half*    S,
    half*          C,
    int M, int N, int K, int group_size,
    cudaStream_t stream)
{
    dim3 grid(
        (M + GPTQ_PRO_M_PER_WARP - 1) / GPTQ_PRO_M_PER_WARP,
        (N + GPTQ_PRO_N_PER_WARP - 1) / GPTQ_PRO_N_PER_WARP,
        1);
    dim3 block(GPTQ_PRO_WARP_SIZE, 1, 1);
    size_t smem_bytes = sizeof(GptqProSmem);

    gptq_pro_gemm_kernel<<<grid, block, smem_bytes, stream>>>(
        A, B_packed, S, C, M, N, K, group_size);
    return cudaGetLastError();
}
