/*
 * GPTQ-Pro Ampere V3 kernel overlay.
 *
 * The V2 translation unit is included under renamed symbols so its validated
 * legacy path and low-level fragment helpers remain the correctness baseline.
 * V3 replaces AUTO's decode and aligned Tensor Core launches with:
 *
 *   - fused M<=4 decode with four-way intra-CTA split-K;
 *   - group-resident scale staging in the cp.async GEMM pipeline;
 *   - optimized Tensor Core dispatch for every runtime-valid N % 8 shape.
 */

#define gptq_pro_gemv_kernel gptq_pro_gemv_kernel_v2_baseline
#define gptq_pro_gemm_kernel_ampere gptq_pro_gemm_kernel_ampere_v2_baseline
#define gptq_pro_gemm_kernel_legacy gptq_pro_gemm_kernel_legacy_v2
#define gptq_pro_gemm gptq_pro_gemm_v2_baseline
#include "gptq_pro_kernel.cu"
#undef gptq_pro_gemv_kernel
#undef gptq_pro_gemm_kernel_ampere
#undef gptq_pro_gemm_kernel_legacy
#undef gptq_pro_gemm

static constexpr int GPTQ_PRO_GEMV_SPLIT_K = GPTQ_PRO_WARPS_PER_CTA;
static constexpr int GPTQ_PRO_GEMV_N_PER_CTA = GPTQ_PRO_WARP_SIZE;

struct __align__(16) GptqProGemvSmemV3 {
    float partial[GPTQ_PRO_GEMV_SPLIT_K][GPTQ_PRO_WARP_SIZE]
                 [GPTQ_PRO_GEMV_MAX_M];
};

struct __align__(16) GptqProAmpereSmemV3 {
    half A[GPTQ_PRO_PIPE][GPTQ_PRO_M_PER_WARP * GPTQ_PRO_K_PER_WARP];
    half S[GPTQ_PRO_WARPS_PER_CTA][GPTQ_PRO_PIPE][GPTQ_PRO_N_PER_WARP];
    uint32_t
        Q[GPTQ_PRO_WARPS_PER_CTA][GPTQ_PRO_PIPE]
         [GPTQ_PRO_QWORD_ROWS_PER_K_TILE * GPTQ_PRO_N_PER_WARP];
};

__device__ __forceinline__ void prefetch_ampere_tile_v3(
    GptqProAmpereSmemV3* __restrict__ smem,
    const half* __restrict__ A,
    const int32_t* __restrict__ Q,
    const half* __restrict__ S,
    int M,
    int N,
    int K,
    int group_size,
    int m_base,
    int n_base,
    int k_base,
    int data_buffer,
    int scale_buffer,
    bool stage_scales,
    int warp_id,
    int lane) {
    if (warp_id == 0) {
        const int row = lane >> 1;
        const int segment = lane & 1;
        const int global_m = m_base + row;
        half* destination =
            &smem->A[data_buffer][row * GPTQ_PRO_K_PER_WARP + segment * 8];
        const bool valid = global_m < M;
        const half* source =
            valid ? &A[global_m * K + k_base + segment * 8] : A;
        cp_async_ca_16(destination, source, valid ? 16 : 0);
    }

    const int qword_row_local = lane >> 4;
    const int n_segment = lane & 15;
    const int global_n = n_base + n_segment * 4;
    uint32_t* q_destination =
        &smem->Q[warp_id][data_buffer]
                [qword_row_local * GPTQ_PRO_N_PER_WARP + n_segment * 4];
    const bool q_valid = global_n + 4 <= N;
    const int32_t* q_source =
        q_valid
            ? &Q[((k_base >> 3) + qword_row_local) * N + global_n]
            : Q;
    cp_async_cg_16(q_destination, q_source, q_valid ? 16 : 0);

    // Scales live for group_size / 16 K tiles. The A/Q pipe advances every
    // tile, while the scale buffer advances only at a group boundary.
    if (stage_scales && lane < 8) {
        const int scale_n = n_base + lane * 8;
        half* scale_destination = &smem->S[warp_id][scale_buffer][lane * 8];
        const bool valid = scale_n + 8 <= N;
        const int group_idx = k_base / group_size;
        const half* scale_source =
            valid ? &S[group_idx * N + scale_n] : S;
        cp_async_ca_16(scale_destination, scale_source, valid ? 16 : 0);
    }
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

__global__ __launch_bounds__(GPTQ_PRO_GEMV_THREADS, 4)
void gptq_pro_gemv_kernel(
    const half* __restrict__ A,
    const int32_t* __restrict__ Q,
    const half* __restrict__ S,
    half* __restrict__ C,
    int M,
    int N,
    int K,
    int group_size) {
    __shared__ GptqProGemvSmemV3 smem;

    const int warp_id = threadIdx.x / GPTQ_PRO_WARP_SIZE;
    const int lane = threadIdx.x & (GPTQ_PRO_WARP_SIZE - 1);
    const int n = blockIdx.x * GPTQ_PRO_GEMV_N_PER_CTA + lane;
    const int qword_rows = K / GPTQ_PRO_QWORD_VALUES_PER_WORD;
    const half zero_point = __float2half(8.0f);

    float accumulators[GPTQ_PRO_GEMV_MAX_M];
#pragma unroll
    for (int m = 0; m < GPTQ_PRO_GEMV_MAX_M; ++m) {
        accumulators[m] = 0.0f;
    }

    int cached_group = -1;
    half cached_scale = zero_half();
    if (n < N) {
        // Every warp handles an interleaved quarter of K for the same 32 output
        // columns. Each decoded weight is reused across all active M rows.
        for (int qword_row = warp_id; qword_row < qword_rows;
             qword_row += GPTQ_PRO_GEMV_SPLIT_K) {
            const int group_idx =
                (qword_row * GPTQ_PRO_QWORD_VALUES_PER_WORD) / group_size;
            if (group_idx != cached_group) {
                cached_scale = S[group_idx * N + n];
                cached_group = group_idx;
            }

            const uint32_t word =
                static_cast<uint32_t>(Q[qword_row * N + n]);
            uint32_t lower_rb[2];
            uint32_t upper_rb[2];
            decode_bfrag_to_rb(
                static_cast<uint16_t>(word & 0xFFFFu),
                cached_scale,
                zero_point,
                lower_rb);
            decode_bfrag_to_rb(
                static_cast<uint16_t>(word >> 16),
                cached_scale,
                zero_point,
                upper_rb);

            Half2Reg weights[4];
            weights[0].u32 = lower_rb[0];
            weights[1].u32 = lower_rb[1];
            weights[2].u32 = upper_rb[0];
            weights[3].u32 = upper_rb[1];

#pragma unroll
            for (int byte_index = 0; byte_index < 4; ++byte_index) {
                const int pair_index = qword_row * 4 + byte_index;
                const float2 dequantized = __half22float2(weights[byte_index].h2);
#pragma unroll
                for (int m = 0; m < GPTQ_PRO_GEMV_MAX_M; ++m) {
                    if (m < M) {
                        const half2 activation_pair =
                            reinterpret_cast<const half2*>(&A[m * K])[pair_index];
                        const float2 activation = __half22float2(activation_pair);
                        accumulators[m] =
                            fmaf(activation.x, dequantized.x, accumulators[m]);
                        accumulators[m] =
                            fmaf(activation.y, dequantized.y, accumulators[m]);
                    }
                }
            }
        }
    }

#pragma unroll
    for (int m = 0; m < GPTQ_PRO_GEMV_MAX_M; ++m) {
        smem.partial[warp_id][lane][m] = accumulators[m];
    }
    __syncthreads();

    if (warp_id == 0 && n < N) {
#pragma unroll
        for (int m = 0; m < GPTQ_PRO_GEMV_MAX_M; ++m) {
            if (m < M) {
                float total = smem.partial[0][lane][m];
#pragma unroll
                for (int split = 1; split < GPTQ_PRO_GEMV_SPLIT_K; ++split) {
                    total += smem.partial[split][lane][m];
                }
                C[m * N + n] = __float2half_rn(total);
            }
        }
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
        do_mma_ampere(
            smem->A[read_buffer],
            smem->S[warp_id][scale_read_buffer],
            smem->Q[warp_id][read_buffer],
            lane,
            RC);
        __syncthreads();
        read_buffer = write_buffer;
        scale_read_buffer = scale_write_buffer;
    }

    store_mma_output(C, M, N, m_base, n_base, lane, RC);
}

#else

__global__ void gptq_pro_gemv_kernel(
    const half*, const int32_t*, const half*, half*, int, int, int, int) {}
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
        const dim3 grid(
            (N + GPTQ_PRO_GEMV_N_PER_CTA - 1) / GPTQ_PRO_GEMV_N_PER_CTA,
            1,
            1);
        const dim3 block(GPTQ_PRO_GEMV_THREADS, 1, 1);
        gptq_pro_gemv_kernel<<<grid, block, 0, stream>>>(
            A, Q, S, C, M, N, K, group_size);
        return cudaGetLastError();
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

    if (selected_mode != GPTQ_PRO_KERNEL_LEGACY) {
        return cudaErrorInvalidValue;
    }

    const dim3 grid(
        (M + GPTQ_PRO_M_PER_WARP - 1) / GPTQ_PRO_M_PER_WARP,
        (N + GPTQ_PRO_N_PER_WARP - 1) / GPTQ_PRO_N_PER_WARP,
        1);
    const dim3 block(GPTQ_PRO_WARP_SIZE, 1, 1);
    const size_t smem_bytes = sizeof(GptqProLegacySmem);
    gptq_pro_gemm_kernel_legacy_v2<<<grid, block, smem_bytes, stream>>>(
        A, Q, S, C, M, N, K, group_size);
    return cudaGetLastError();
}
