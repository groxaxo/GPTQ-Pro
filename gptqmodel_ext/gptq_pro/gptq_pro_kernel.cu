/*
 * GPTQ-Pro INT4 dequantized matrix multiplication for Ampere-class GPUs.
 *
 * AUTO dispatch uses:
 *   1. a coalesced FP32-accumulating GEMV kernel for M <= 4;
 *   2. a four-warp, double-buffered cp.async Tensor Core kernel for aligned
 *      production shapes;
 *   3. the original one-warp general-shape implementation as a fallback.
 */

#include "gptq_pro_kernel.cuh"

struct __align__(16) GptqProLegacySmem {
    half A[GPTQ_PRO_M_PER_WARP * GPTQ_PRO_K_PER_WARP];
    half S[GPTQ_PRO_N_PER_WARP];
    uint32_t Bfrag[GPTQ_PRO_BFRAG_WORDS_PER_BUF];
};

struct __align__(16) GptqProAmpereSmem {
    // A is shared by the four N-warps in a CTA. B and scales are warp-private.
    half A[GPTQ_PRO_PIPE][GPTQ_PRO_M_PER_WARP * GPTQ_PRO_K_PER_WARP];
    half S[GPTQ_PRO_WARPS_PER_CTA][GPTQ_PRO_PIPE][GPTQ_PRO_N_PER_WARP];
    uint8_t
        B[GPTQ_PRO_WARPS_PER_CTA][GPTQ_PRO_PIPE]
         [GPTQ_PRO_B_BYTES_PER_WARP_TILE];
};

__device__ __forceinline__ half zero_half() {
    return __float2half(0.0f);
}

// ---------------------------------------------------------------------------
// General-shape fallback helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint8_t load_b_pair_byte(
    const uint8_t* __restrict__ B_packed,
    int K,
    int N,
    int k_even,
    int n_col) {
    if (n_col >= N || k_even >= K) {
        return 0x88u;
    }

    const int packed_row = k_even >> 1;
    uint8_t byte = B_packed[packed_row * N + n_col];
    if (k_even + 1 >= K) {
        byte = static_cast<uint8_t>((byte & 0x0Fu) | 0x80u);
    }
    return byte;
}

__device__ __forceinline__ uint16_t pack_lane_bfrag(
    const uint8_t* __restrict__ B_packed,
    int N,
    int K,
    int k_base,
    int n_base,
    int j,
    int lane) {
    const int group_id = lane >> 2;
    const int thread_id = lane & 3;
    const int n_col = n_base + j * 8 + group_id;
    if (n_col >= N) {
        return 0x8888u;
    }

    const int k01 = k_base + 2 * thread_id;
    const int k89 = k01 + 8;
    const uint8_t byte01 =
        load_b_pair_byte(B_packed, K, N, k01, n_col);
    const uint8_t byte89 =
        load_b_pair_byte(B_packed, K, N, k89, n_col);
    return static_cast<uint16_t>(byte01) |
           (static_cast<uint16_t>(byte89) << 8);
}

__device__ __forceinline__ void stage_a_tile_legacy(
    GptqProLegacySmem* __restrict__ smem,
    const half* __restrict__ A,
    int M,
    int K,
    int m_base,
    int k_base) {
    for (int index = threadIdx.x;
         index < GPTQ_PRO_M_PER_WARP * GPTQ_PRO_K_PER_WARP;
         index += blockDim.x) {
        const int row = index / GPTQ_PRO_K_PER_WARP;
        const int col = index % GPTQ_PRO_K_PER_WARP;
        const int global_m = m_base + row;
        const int global_k = k_base + col;
        smem->A[index] =
            (global_m < M && global_k < K)
                ? A[global_m * K + global_k]
                : zero_half();
    }
}

__device__ __forceinline__ void stage_scale_row_legacy(
    GptqProLegacySmem* __restrict__ smem,
    const half* __restrict__ S,
    int N,
    int k_base,
    int group_size,
    int n_base) {
    const int group_idx = k_base / group_size;
    for (int index = threadIdx.x; index < GPTQ_PRO_N_PER_WARP;
         index += blockDim.x) {
        const int global_n = n_base + index;
        smem->S[index] =
            (global_n < N) ? S[group_idx * N + global_n] : zero_half();
    }
}

__device__ __forceinline__ void stage_bfrag_tiles_legacy(
    GptqProLegacySmem* __restrict__ smem,
    const uint8_t* __restrict__ B_packed,
    int N,
    int K,
    int k_base,
    int n_base) {
    for (int index = threadIdx.x;
         index < GPTQ_PRO_BFRAG_WORDS_PER_BUF;
         index += blockDim.x) {
        const int j = index / GPTQ_PRO_BFRAG_WORDS_PER_TILE;
        const int lane_pair = index % GPTQ_PRO_BFRAG_WORDS_PER_TILE;
        const int even_lane = lane_pair * 2;
        const uint16_t even_p16 = pack_lane_bfrag(
            B_packed, N, K, k_base, n_base, j, even_lane);
        const uint16_t odd_p16 = pack_lane_bfrag(
            B_packed, N, K, k_base, n_base, j, even_lane + 1);
        smem->Bfrag[index] = static_cast<uint32_t>(even_p16) |
                             (static_cast<uint32_t>(odd_p16) << 16);
    }
}

__device__ __forceinline__ void do_mma_legacy(
    const GptqProLegacySmem* __restrict__ smem,
    float RC[GPTQ_PRO_J_TILES][4]) {
    const int lane = threadIdx.x & (GPTQ_PRO_WARP_SIZE - 1);
    const int group_id = lane >> 2;
    const half zero_point = __float2half(8.0f);

    uint32_t RA[4];
    load_a_fragment_rowmajor(smem->A, lane, RA);

#pragma unroll
    for (int j = 0; j < GPTQ_PRO_J_TILES; ++j) {
        const half scale = smem->S[j * 8 + group_id];
        const uint16_t packed_16 =
            fetch_bfrag_packed16(smem->Bfrag, 0, 0, j, lane);

        uint32_t RB[2];
        decode_bfrag_to_rb(packed_16, scale, zero_point, RB);
        mma_f32_m16n8k16(RA, RB, RC[j]);
    }
}

// ---------------------------------------------------------------------------
// Ampere cp.async pipeline helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ void prefetch_ampere_tile(
    GptqProAmpereSmem* __restrict__ smem,
    const half* __restrict__ A,
    const uint8_t* __restrict__ B_packed,
    const half* __restrict__ S,
    int M,
    int N,
    int K,
    int group_size,
    int m_base,
    int n_base,
    int k_base,
    int buffer,
    int warp_id,
    int lane) {
    // One warp cooperatively stages the common 16x16 A tile. Each lane moves a
    // naturally aligned 16-byte segment; invalid M rows are zero-filled by
    // cp.async's source-size operand.
    if (warp_id == 0) {
        const int row = lane >> 1;
        const int segment = lane & 1;
        const int global_m = m_base + row;
        half* destination =
            &smem->A[buffer][row * GPTQ_PRO_K_PER_WARP + segment * 8];
        const bool valid = global_m < M;
        const half* source =
            valid ? &A[global_m * K + k_base + segment * 8] : A;
        cp_async_ca_16(destination, source, valid ? 16 : 0);
    }

    // Each warp stages an 8x64 packed-weight tile using 32 coalesced 16-byte
    // copies: one packed K row and one N segment per lane.
    const int packed_row_local = lane >> 2;
    const int n_segment = lane & 3;
    const int global_n = n_base + n_segment * 16;
    uint8_t* b_destination =
        &smem->B[warp_id][buffer]
                [packed_row_local * GPTQ_PRO_N_PER_WARP + n_segment * 16];
    const int remaining_n = N - global_n;
    const int b_source_bytes =
        remaining_n >= 16 ? 16 : (remaining_n >= 8 ? 8 : 0);
    const uint8_t* b_source =
        b_source_bytes > 0
            ? &B_packed[((k_base >> 1) + packed_row_local) * N + global_n]
            : B_packed;
    cp_async_cg_16(b_destination, b_source, b_source_bytes);

    // Eight lanes per warp stage the 64 FP16 scales in 16-byte vectors.
    if (lane < 8) {
        const int scale_n = n_base + lane * 8;
        half* scale_destination = &smem->S[warp_id][buffer][lane * 8];
        const bool valid = scale_n < N;
        const int group_idx = k_base / group_size;
        const half* scale_source =
            valid ? &S[group_idx * N + scale_n] : S;
        cp_async_ca_16(scale_destination, scale_source, valid ? 16 : 0);
    }
}

__device__ __forceinline__ void do_mma_ampere(
    const half* __restrict__ smem_a,
    const half* __restrict__ smem_s,
    const uint8_t* __restrict__ smem_b,
    int lane,
    float RC[GPTQ_PRO_J_TILES][4]) {
    const int group_id = lane >> 2;
    const half zero_point = __float2half(8.0f);

    uint32_t RA[4];
    load_a_fragment_rowmajor(smem_a, lane, RA);

#pragma unroll
    for (int j = 0; j < GPTQ_PRO_J_TILES; ++j) {
        const half scale = smem_s[j * 8 + group_id];
        const uint16_t packed_16 =
            load_raw_bfrag_packed16(smem_b, j, lane);
        uint32_t RB[2];
        decode_bfrag_to_rb(packed_16, scale, zero_point, RB);
        mma_f32_m16n8k16(RA, RB, RC[j]);
    }
}

__device__ __forceinline__ void store_mma_output(
    half* __restrict__ C,
    int M,
    int N,
    int m_base,
    int n_base,
    int lane,
    float RC[GPTQ_PRO_J_TILES][4]) {
    const int row_base = lane >> 2;
    const int col_pair = 2 * (lane & 3);

#pragma unroll
    for (int j = 0; j < GPTQ_PRO_J_TILES; ++j) {
        const int m0 = m_base + row_base;
        const int m1 = m_base + row_base + 8;
        const int n0 = n_base + j * 8 + col_pair;
        const int n1 = n0 + 1;

        if (n1 < N) {
            if (m0 < M) {
                *reinterpret_cast<half2*>(&C[m0 * N + n0]) =
                    __floats2half2_rn(RC[j][0], RC[j][1]);
            }
            if (m1 < M) {
                *reinterpret_cast<half2*>(&C[m1 * N + n0]) =
                    __floats2half2_rn(RC[j][2], RC[j][3]);
            }
        } else if (n0 < N) {
            // The optimized path always has even N. This scalar tail keeps the
            // general-shape fallback correct for standalone odd-N validation.
            if (m0 < M) {
                C[m0 * N + n0] = __float2half_rn(RC[j][0]);
            }
            if (m1 < M) {
                C[m1 * N + n0] = __float2half_rn(RC[j][2]);
            }
        }
    }
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

// ---------------------------------------------------------------------------
// Dedicated small-M decode path
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(GPTQ_PRO_GEMV_THREADS, 4)
void gptq_pro_gemv_kernel(
    const half* __restrict__ A,
    const uint8_t* __restrict__ B_packed,
    const half* __restrict__ S,
    half* __restrict__ C,
    int M,
    int N,
    int K,
    int group_size) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (n >= N || m >= M) {
        return;
    }

    const half2* a_pairs = reinterpret_cast<const half2*>(&A[m * K]);
    const int groups = (K + group_size - 1) / group_size;
    float accumulator = 0.0f;

    for (int group = 0; group < groups; ++group) {
        const int k_begin = group * group_size;
        const int k_end_candidate = k_begin + group_size;
        const int k_end = k_end_candidate < K ? k_end_candidate : K;
        const half2 scale2 = __half2half2(S[group * N + n]);
        const int pair_begin = k_begin >> 1;
        const int pair_end = k_end >> 1;

        for (int pair = pair_begin; pair < pair_end; ++pair) {
            const float2 activation = __half22float2(a_pairs[pair]);
            const uint8_t packed = B_packed[pair * N + n];
            const int weight0 = static_cast<int>(packed & 0x0Fu) - 8;
            const int weight1 = static_cast<int>((packed >> 4) & 0x0Fu) - 8;
            const half2 signed_weights = __floats2half2_rn(
                static_cast<float>(weight0), static_cast<float>(weight1));
            const float2 dequantized =
                __half22float2(__hmul2(scale2, signed_weights));
            accumulator =
                fmaf(activation.x, dequantized.x, accumulator);
            accumulator =
                fmaf(activation.y, dequantized.y, accumulator);
        }
    }

    C[m * N + n] = __float2half_rn(accumulator);
}

// ---------------------------------------------------------------------------
// Four-warp, double-buffered Ampere Tensor Core path
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(GPTQ_PRO_THREADS_PER_CTA, 4)
void gptq_pro_gemm_kernel_ampere(
    const half* __restrict__ A,
    const uint8_t* __restrict__ B_packed,
    const half* __restrict__ S,
    half* __restrict__ C,
    int M,
    int N,
    int K,
    int group_size) {
    extern __shared__ uint8_t raw_smem[];
    auto* smem = reinterpret_cast<GptqProAmpereSmem*>(raw_smem);

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
    int read_buffer = 0;
    prefetch_ampere_tile(
        smem,
        A,
        B_packed,
        S,
        M,
        N,
        K,
        group_size,
        m_base,
        n_base,
        0,
        read_buffer,
        warp_id,
        lane);
    cp_async_commit_group();

    for (int tile = 0; tile < num_k_tiles; ++tile) {
        const int next_tile = tile + 1;
        const int write_buffer = read_buffer ^ 1;

        if (next_tile < num_k_tiles) {
            prefetch_ampere_tile(
                smem,
                A,
                B_packed,
                S,
                M,
                N,
                K,
                group_size,
                m_base,
                n_base,
                next_tile * GPTQ_PRO_K_PER_WARP,
                write_buffer,
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
            smem->S[warp_id][read_buffer],
            smem->B[warp_id][read_buffer],
            lane,
            RC);
        // No thread may overwrite this buffer until every warp has finished
        // reading the common A tile.
        __syncthreads();
        read_buffer = write_buffer;
    }

    store_mma_output(C, M, N, m_base, n_base, lane, RC);
}

// ---------------------------------------------------------------------------
// Original general-shape fallback
// ---------------------------------------------------------------------------
__global__ void gptq_pro_gemm_kernel_legacy(
    const half* __restrict__ A,
    const uint8_t* __restrict__ B_packed,
    const half* __restrict__ S,
    half* __restrict__ C,
    int M,
    int N,
    int K,
    int group_size) {
    extern __shared__ uint8_t raw_smem[];
    auto* smem = reinterpret_cast<GptqProLegacySmem*>(raw_smem);

    const int m_base = blockIdx.x * GPTQ_PRO_M_PER_WARP;
    const int n_base = blockIdx.y * GPTQ_PRO_N_PER_WARP;
    const int lane = threadIdx.x & (GPTQ_PRO_WARP_SIZE - 1);

    float RC[GPTQ_PRO_J_TILES][4];
#pragma unroll
    for (int j = 0; j < GPTQ_PRO_J_TILES; ++j) {
        RC[j][0] = 0.0f;
        RC[j][1] = 0.0f;
        RC[j][2] = 0.0f;
        RC[j][3] = 0.0f;
    }

    const int num_k_tiles =
        (K + GPTQ_PRO_K_PER_WARP - 1) / GPTQ_PRO_K_PER_WARP;
    for (int tile = 0; tile < num_k_tiles; ++tile) {
        const int k_base = tile * GPTQ_PRO_K_PER_WARP;
        stage_a_tile_legacy(smem, A, M, K, m_base, k_base);
        stage_scale_row_legacy(
            smem, S, N, k_base, group_size, n_base);
        stage_bfrag_tiles_legacy(
            smem, B_packed, N, K, k_base, n_base);
        __syncthreads();
        do_mma_legacy(smem, RC);
        __syncthreads();
    }

    store_mma_output(C, M, N, m_base, n_base, lane, RC);
}

#else

__global__ void gptq_pro_gemv_kernel(
    const half*, const uint8_t*, const half*, half*, int, int, int, int) {}
__global__ void gptq_pro_gemm_kernel_ampere(
    const half*, const uint8_t*, const half*, half*, int, int, int, int) {}
__global__ void gptq_pro_gemm_kernel_legacy(
    const half*, const uint8_t*, const half*, half*, int, int, int, int) {}

#endif

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

    const bool gemv_compatible = (K % 2) == 0;
    const bool ampere_compatible =
        (K % GPTQ_PRO_K_PER_WARP) == 0 && (N % 16) == 0;

    int selected_mode = kernel_mode;
    if (selected_mode == GPTQ_PRO_KERNEL_AUTO) {
        if (M <= GPTQ_PRO_GEMV_MAX_M && gemv_compatible) {
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
            (N + GPTQ_PRO_GEMV_THREADS - 1) / GPTQ_PRO_GEMV_THREADS,
            M,
            1);
        const dim3 block(GPTQ_PRO_GEMV_THREADS, 1, 1);
        gptq_pro_gemv_kernel<<<grid, block, 0, stream>>>(
            A, B_packed, S, C, M, N, K, group_size);
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
        const size_t smem_bytes = sizeof(GptqProAmpereSmem);
        gptq_pro_gemm_kernel_ampere<<<grid, block, smem_bytes, stream>>>(
            A, B_packed, S, C, M, N, K, group_size);
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
    gptq_pro_gemm_kernel_legacy<<<grid, block, smem_bytes, stream>>>(
        A, B_packed, S, C, M, N, K, group_size);
    return cudaGetLastError();
}
