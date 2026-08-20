/*
 * GPTQ-Pro Ampere V4 kernel overlay.
 *
 * V4 keeps V3 as the exact behavioral baseline while introducing the first
 * quality-preserving data-movement optimization: per-lane scale values are
 * promoted from shared memory into registers once per quantization group and
 * reused across all K16 MMA tiles in that group.
 *
 * Arithmetic invariants are intentionally unchanged:
 *   - native GPTQ INT4 qweight layout;
 *   - FP16 activations and dequantized B fragments;
 *   - FP32 Tensor Core accumulation;
 *   - identical K16 MMA ordering;
 *   - FP16 round-to-nearest output conversion.
 */

// Overlay V3 under renamed symbols so its validated GEMV path and fragment
// helpers remain the exact behavioral baseline inside this translation unit.
// The prefix macro makes V3 rename its own externally-linkable definitions
// (and its V2-baseline dispatch) instead of leaving them under literal names,
// which previously collided with the V4 definitions below.
#define GPTQ_PRO_JOIN_IMPL(a, b) a##b
#define GPTQ_PRO_JOIN(a, b) GPTQ_PRO_JOIN_IMPL(a, b)
#define GPTQ_PRO_V3_KERNEL_ALIAS_PREFIX v3_
#include "gptq_pro_kernel_v3.cu"
#undef GPTQ_PRO_V3_KERNEL_ALIAS_PREFIX
#undef GPTQ_PRO_JOIN
#undef GPTQ_PRO_JOIN_IMPL

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

__device__ __forceinline__ void load_scale_registers_v4(
    const half* __restrict__ smem_s,
    int lane,
    half (&scale_regs)[GPTQ_PRO_J_TILES]) {
    const int group_id = lane >> 2;
#pragma unroll
    for (int j = 0; j < GPTQ_PRO_J_TILES; ++j) {
        scale_regs[j] = smem_s[j * 8 + group_id];
    }
}

__device__ __forceinline__ void do_mma_ampere_v4(
    const half* __restrict__ smem_a,
    const uint32_t* __restrict__ smem_qweight,
    const half (&scale_regs)[GPTQ_PRO_J_TILES],
    int lane,
    float RC[GPTQ_PRO_J_TILES][4]) {
    const half zero_point = __float2half(8.0f);

    uint32_t RA[4];
    load_a_fragment_rowmajor(smem_a, lane, RA);

#pragma unroll
    for (int j = 0; j < GPTQ_PRO_J_TILES; ++j) {
        const uint16_t packed_16 =
            load_qweight_bfrag_packed16(smem_qweight, j, lane);
        uint32_t RB[2];
        decode_bfrag_to_rb(packed_16, scale_regs[j], zero_point, RB);
        mma_f32_m16n8k16(RA, RB, RC[j]);
    }
}

__global__ __launch_bounds__(GPTQ_PRO_THREADS_PER_CTA, 4)
void gptq_pro_gemm_kernel_ampere(
    const half* __restrict__ A,
    const int32_t* __restrict__ Q,
    const half* __restrict__ S,
    half* __restrict__ C,
    int M,
    int N,
    int K,
    int group_size) {
    extern __shared__ uint8_t raw_smem[];
    auto* smem = reinterpret_cast<GptqProAmpereSmemV3*>(raw_smem);

    const int warp_id = threadIdx.x / GPTQ_PRO_WARP_SIZE;
    const int lane = threadIdx.x & (GPTQ_PRO_WARP_SIZE - 1);
    const int m_base = blockIdx.x * GPTQ_PRO_M_PER_WARP;
    const int n_base =
        blockIdx.y * GPTQ_PRO_N_PER_CTA + warp_id * GPTQ_PRO_N_PER_WARP;

    float RC[GPTQ_PRO_J_TILES][4];
#pragma unroll
    for (int j = 0; j < GPTQ_PRO_J_TILES; ++j) {
        RC[j][0] = 0.0f;
        RC[j][1] = 0.0f;
        RC[j][2] = 0.0f;
        RC[j][3] = 0.0f;
    }

    half scale_regs[GPTQ_PRO_J_TILES];

    const int num_k_tiles = K / GPTQ_PRO_K_PER_WARP;
    const int group_tiles = group_size / GPTQ_PRO_K_PER_WARP;
    int read_buffer = 0;
    int scale_read_buffer = 0;

    prefetch_ampere_tile_v3(
        smem,
        A,
        Q,
        S,
        M,
        N,
        K,
        group_size,
        m_base,
        n_base,
        0,
        read_buffer,
        scale_read_buffer,
        true,
        warp_id,
        lane);
    cp_async_commit_group();

    for (int tile = 0; tile < num_k_tiles; ++tile) {
        const int next_tile = tile + 1;
        const int write_buffer = read_buffer ^ 1;
        const bool starts_group = (tile % group_tiles) == 0;
        const bool next_starts_group =
            next_tile < num_k_tiles && (next_tile % group_tiles) == 0;
        const int scale_write_buffer =
            next_starts_group ? (scale_read_buffer ^ 1) : scale_read_buffer;

        if (next_tile < num_k_tiles) {
            prefetch_ampere_tile_v3(
                smem,
                A,
                Q,
                S,
                M,
                N,
                K,
                group_size,
                m_base,
                n_base,
                next_tile * GPTQ_PRO_K_PER_WARP,
                write_buffer,
                scale_write_buffer,
                next_starts_group,
                warp_id,
                lane);
            cp_async_commit_group();
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }

        __syncthreads();
        if (starts_group) {
            load_scale_registers_v4(
                smem->S[warp_id][scale_read_buffer], lane, scale_regs);
        }
        do_mma_ampere_v4(
            smem->A[read_buffer],
            smem->Q[warp_id][read_buffer],
            scale_regs,
            lane,
            RC);
        __syncthreads();

        read_buffer = write_buffer;
        scale_read_buffer = scale_write_buffer;
    }

    store_mma_output(C, M, N, m_base, n_base, lane, RC);
}

#else

__global__ void gptq_pro_gemm_kernel_ampere(
    const half*, const int32_t*, const half*, half*, int, int, int, int) {}

#endif

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
    int kernel_mode) {
    if (M <= 0 || N <= 0 || K <= 0) {
        return cudaSuccess;
    }
    if (group_size <= 0) {
        group_size = K;
    }
    if ((group_size % GPTQ_PRO_K_PER_WARP) != 0) {
        return cudaErrorInvalidValue;
    }

    const bool gemv_compatible =
        M <= GPTQ_PRO_GEMV_MAX_M &&
        (K % GPTQ_PRO_QWORD_VALUES_PER_WORD) == 0;
    const bool ampere_compatible =
        (K % GPTQ_PRO_K_PER_WARP) == 0 && (N % 8) == 0;

    int selected_mode = kernel_mode;
    if (selected_mode == GPTQ_PRO_KERNEL_AUTO) {
        if (gemv_compatible) {
            selected_mode = GPTQ_PRO_KERNEL_GEMV;
        } else if (ampere_compatible) {
            selected_mode = GPTQ_PRO_KERNEL_AMPERE;
        } else {
            selected_mode = GPTQ_PRO_KERNEL_LEGACY;
        }
    }

    if (selected_mode == GPTQ_PRO_KERNEL_GEMV) {
        if (!gemv_compatible) {
            return cudaErrorInvalidValue;
        }
        return v3_gptq_pro_gemm(
            A, Q, S, C, M, N, K, group_size, stream, GPTQ_PRO_KERNEL_GEMV);
    }

    if (selected_mode == GPTQ_PRO_KERNEL_AMPERE) {
        if (!ampere_compatible) {
            return cudaErrorInvalidValue;
        }
        const dim3 grid(
            (M + GPTQ_PRO_M_PER_WARP - 1) / GPTQ_PRO_M_PER_WARP,
            (N + GPTQ_PRO_N_PER_CTA - 1) / GPTQ_PRO_N_PER_CTA,
            1);
        const dim3 block(GPTQ_PRO_THREADS_PER_CTA, 1, 1);
        const size_t smem_bytes = sizeof(GptqProAmpereSmemV3);
        gptq_pro_gemm_kernel_ampere<<<grid, block, smem_bytes, stream>>>(
            A, Q, S, C, M, N, K, group_size);
        return cudaGetLastError();
    }

    return v3_gptq_pro_gemm(
        A, Q, S, C, M, N, K, group_size, stream, selected_mode);
}
