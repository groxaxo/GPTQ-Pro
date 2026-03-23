# TRAM-Quant: Mixed-Precision Ampere Kernel — Progress

## Peer Review

**Reviewer:** Copilot (Beta)  
**Scope:** Full review of the TRAM-Quant kernel design as documented in `Project.md` lines 7393–9396.  
**Verdict: REJECT — conditional on 5 must-fix items below. The architecture is strong but the current skeleton has a precision-killing accumulator choice, an unfinished transform path that will bottleneck Tensor Core throughput, and zero benchmarking infrastructure.**

---

### 1. CRITICAL — FP16 Accumulation Will Ruin PPL

The single most damaging decision in the current skeleton is this instruction:

```
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
```

The last `.f16` means the accumulator is FP16 (10-bit mantissa, ~3.3 decimal digits). For a model with K=4096, each output element accumulates over `K/16 = 256` MMA operations. FP16 accumulation over 256 steps introduces catastrophic precision loss — partial sums overflow and small contributions vanish entirely. This will measurably degrade PPL, especially on models with wide hidden dimensions (≥4096).

**The fix:** Switch to the FP32 accumulator variant. Ampere Tensor Cores support this at identical throughput:

```cpp
// CORRECTED: FP32 accumulator, FP16 inputs
#define MMA_SYNC_M16N8K16_F32(RC, RA, RB)                                  \
    asm volatile(                                                           \
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "               \
        "{%0, %1, %2, %3}, "                                               \
        "{%4, %5, %6, %7}, "                                               \
        "{%8, %9}, "                                                        \
        "{%0, %1, %2, %3};\n"                                              \
        : "+r"((RC)[0]), "+r"((RC)[1]), "+r"((RC)[2]), "+r"((RC)[3])       \
        : "r"((RA)[0]), "r"((RA)[1]), "r"((RA)[2]), "r"((RA)[3]),          \
          "r"((RB)[0]), "r"((RB)[1])                                        \
    )

// Accumulators revert to 4 regs (4x FP32) per m16n8k16 slice:
uint32_t RC[8][4];
```

This restores 4 accumulator registers per slice (32 total for 8 column slices), which was ironically the *original* layout before it was incorrectly "fixed" to 2 regs. The PTX `m16n8k16` with `.f32` C/D uses 4 × 32-bit registers holding 4 FP32 values. The register pressure increase from 16 → 32 regs for accumulators is well within Ampere's budget (see §5 below).

Convert accumulators to FP16 only in the epilogue when writing to global memory. This is what Marlin does and is what you should do.

---

### 2. CRITICAL — Paro Transform Is Underspecified and Will Stall the Pipeline

The `apply_paro_transform()` is still a pseudocode placeholder. This is not just an implementation gap — it's a **design risk** because the transform happens on the critical path between Barrier A and the MMA loop, directly gating Tensor Core utilization.

**Quantified cost per stage (INT4 kernel, 8 warps):**
- Channel scaling: 16 rows × 64 cols = 1024 FP16 multiplies per warp
- Givens rotations (8 pairs): 16 rows × 8 pairs × 4 FMAs = 512 FMAs per warp
- Total: ~1536 FP16 ops per warp per K_STAGE

These operations hit shared memory for both reads and writes, creating a read-modify-write dependency chain. The `__syncthreads()` (Barrier B) after the transform further serializes this with the MMA phase.

**Specific concerns:**

(a) **Shared memory bank conflicts during transform:** Each warp's 16-row slab has 64 columns of FP16 = 128 bytes per row. When warp 0 and warp 4 both apply Givens rotations that touch the same column pair within their respective row slabs, the shared memory addresses differ by `16 * 128 = 2048 bytes = 64 banks`. On Ampere (32 banks, 4-byte banking), this wraps to `64 mod 32 = 0` — **same bank**. If the Givens pair indices `(a, b)` are identical across warps executing simultaneously, you get 2-way bank conflicts.

**Fix:** Pad the A tile row stride from 128 bytes to 132 bytes (add 2 FP16 padding per row). This breaks the bank alignment pattern:

```cpp
// Padded activation tile to avoid cross-warp bank conflicts during transform
constexpr int A_ROW_STRIDE_BYTES = 132; // 64 FP16 + 2 padding = 66 half values
constexpr int A_STAGE_BYTES_PADDED = M_TILE * A_ROW_STRIDE_BYTES; // 64 * 132 = 8448
```

(b) **Transform as a throughput bottleneck:** With 8 Givens rotations per K_STAGE=64 block, the rotation pass consumes ~30 cycles per row (2 FP16 loads + 4 FMAs + 2 stores, assuming ~4-cycle FMA latency with ILP). Over 16 rows, that's ~480 cycles per warp. The MMA inner loop (4 ks × 8 j × ~8 cycles per mma.sync) is ~256 cycles. **The transform is nearly 2× the cost of the MMA phase.** This means Tensor Cores are idle nearly half the time waiting for the transform to complete.

**Fix options:**
- Reduce rotation count. 4 rotations per block instead of 8 may be the quality-speed sweet spot. Benchmark quality with `ROT_COUNT ∈ {2, 4, 6, 8}`.
- Overlap transform with MMA from the *previous* stage by restructuring the pipeline:
  ```
  Stage N:   [MMA(N)] + [Transform(N+1)]   // concurrent if on disjoint smem
  ```
  This requires keeping 2 A tiles live simultaneously (doubling A-tile shared memory), but the total smem footprint stays under 64KB.

---

### 3. MAJOR — Validation Layout Has Hidden Bank Conflicts

The current `uint16_t Bfrag[(ks * 8 + j) * 32 + lane]` validation layout has a 2-way bank conflict pattern.

Ampere shared memory has 32 banks with 4-byte stride. When 32 lanes read consecutive `uint16_t` addresses:
- Lane 0 reads byte offset `0` → bank 0
- Lane 1 reads byte offset `2` → bank 0 (same bank!)
- Lane 2 reads byte offset `4` → bank 1
- Lane 3 reads byte offset `6` → bank 1

This is a **2-way bank conflict on every B fragment load.**

Yes, this is a "validation layout" meant to be replaced — but you should fix it now rather than baking in a performance anti-pattern that masks real bottlenecks during validation. The production `ld.shared.u32` path (4 bytes per lane) will resolve this naturally, but you should move to it immediately rather than treating it as a future optimization.

**Immediate fix for validation:**

```cpp
// Use u32 fetch even for validation — kills bank conflicts and matches production path
uint32_t packed_pair = *reinterpret_cast<uint32_t*>(
    &Bfrag_u16[(ks * 8 + j) * 32 + (lane & ~1)]);
uint16_t packed_16 = (lane & 1) ? (packed_pair >> 16) : (packed_pair & 0xFFFF);
```

---

### 4. MAJOR — Missing Edge-Case Handling Throughout

The skeleton has zero bounds checking. This will produce garbage results for real-world matrix dimensions:

**(a) M-edge tiles:** When `M % 64 != 0`, the last CTA row-block reads out-of-bounds from the A matrix. cp.async will happily fetch garbage from global memory. Fix: add `cp.async` predication or zero-fill the tail.

```cpp
// Predicated cp.async for M-edge tiles
int global_m = block_m + local_m;
if (global_m < M) {
    // issue cp.async
} else {
    // zero-fill this smem slot
    *(uint4*)(smem + offset) = make_uint4(0, 0, 0, 0);
}
```

**(b) N-edge tiles:** When `N % 128 != 0` (INT4) or `N % 64 != 0` (INT8), the last CTA column-block's weight loads go out of bounds. Same fix needed.

**(c) Short-K drain:** The prologue fix `min(PIPE - 1, num_k_stages)` was identified but never integrated into the actual skeleton code. The drain loop body after the main loop is still just a comment (`// ... wait_group(2) -> math ...`). This must be written out — it's not trivial with the transform + barrier sequence.

**(d) Odd batch sizes at inference time:** For autoregressive decoding with batch size 1, `M_TILE = 64` wastes 63/64 rows. The kernel needs a separate small-M path or dynamic M_TILE selection. This is an inference-critical case.

---

### 5. MAJOR — No Benchmarking Strategy Exists

The entire design conversation contains zero profiling methodology. The proposal cannot be approved without a concrete benchmarking plan.

**Required benchmarking deliverables:**

**(a) Microbenchmark suite (before full kernel integration):**

| Test | What it measures | Tool |
|------|-----------------|------|
| Decode-only kernel | INT4→FP16 throughput per SM, cycles/element | Nsight Compute, `sm__inst_executed` |
| Transform-only kernel | Paro transform throughput, bank conflict rate | Nsight Compute, `l1tex__data_bank_conflicts_pipe_lsu` |
| MMA-only kernel (no decode) | Theoretical TC peak for this tile size | Nsight Compute, `sm__pipe_tensor_op_hmma_cycles_active` |
| Full inner loop (1 CTA) | Combined throughput, stall reasons | Nsight Compute warp stall analysis |

**(b) Matrix dimension sweep:**

```
M ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}  # batch sizes
K ∈ {4096, 8192, 11008, 14336}  # common hidden dims
N ∈ {4096, 8192, 11008, 14336}  # common output dims
```

Include at least 3 non-power-of-2 edge cases: `M=7, K=5120, N=13824`.

**(c) End-to-end perplexity comparison:**

| Configuration | Model | Metric |
|--------------|-------|--------|
| FP16 baseline | Qwen3.5-4B / Llama-3-8B | WikiText-2 PPL |
| GPTQ 4-bit (Marlin kernel) | same | WikiText-2 PPL |
| TRAM-Quant INT4-only | same | WikiText-2 PPL |
| TRAM-Quant INT4+INT8 mixed | same | WikiText-2 PPL |

**(d) Nsight Systems timeline:**
- Capture at least 100 token generation steps
- Verify no gaps between kernel launches (kernel fusion is working)
- Verify SM occupancy ≥ 50% sustained
- Verify memory throughput ≥ 70% of Ampere's theoretical bandwidth (A100: 2039 GB/s HBM)

**(e) Roofline target:**
The INT4 kernel is memory-bound for small M. Compute the theoretical arithmetic intensity:
```
AI = (2 * M * N * K) / (M*K*2 + K*N/2 + N*2 + transform_meta) FLOP/byte
```
For M=1, K=4096, N=4096:
```
AI = (2*1*4096*4096) / (1*4096*2 + 4096*4096/2 + 4096*2 + 256*64) ≈ 3.9 FLOP/byte
```
At 3.9 FLOP/byte, you're memory-bound on A100 (312 TFLOPS / 2039 GB/s = 153 FLOP/byte crossover). Target: ≥ 80% of HBM bandwidth for M ≤ 16.

---

### 6. MODERATE — Scale Recomputation in the Decode Double-Buffer

Inside the `j` loop, the decode path recomputes `scale2_nxt`, `c2_nxt`, `s2_reg_nxt`, `c2_reg_nxt` for every column slice. Since `scale = S_smem[col_base + j*8 + groupID]`, and `groupID` is constant per thread, there are only 8 unique scales per thread across the 8 `j` iterations.

**Fix:** Preload all 8 scales into registers before the `j` loop:

```cpp
uint32_t s2_regs[8], c2_regs[8];
#pragma unroll
for (int j = 0; j < 8; ++j) {
    int global_col = col_base + j * 8 + groupID;
    half scale = S_smem[global_col];
    Half2Reg s2, c2;
    s2.h2 = __halves2half2(scale, scale);
    c2.h2 = __hmul2(s2.h2, offset_base);
    s2_regs[j] = s2.u32;
    c2_regs[j] = c2.u32;
}
```

Cost: 16 extra registers. Saves: 8 shared memory loads + 8 half2 constructions + 8 multiplications per `ks` iteration. Net win.

---

### 7. MODERATE — Epilogue Is Completely Absent

The accumulator-to-global-memory store path is the most common place to introduce coalescing disasters. The current skeleton has only `// Write accum[] to global memory C`.

For 8 warps each holding `RC[8][4]` (with FP32 accumulators as recommended in §1), the epilogue must:
1. Convert FP32 accumulators to FP16
2. Write to global memory with 128-byte coalesced transactions

**Skeleton for the epilogue:**

```cpp
// Convert FP32 accums -> FP16 and store
// Each warp owns a 16x64 output tile
// Row = row_base + D-fragment row mapping
// Col = col_base + j*8 + D-fragment col mapping
//
// For m16n8k16 FP32 D-fragment:
//   groupID = lane >> 2, tid4 = lane & 3
//   d0 -> row = 2*tid4+0,  col = groupID (already known)
//   d1 -> row = 2*tid4+1,  col = groupID
//   d2 -> row = 2*tid4+8,  col = groupID
//   d3 -> row = 2*tid4+9,  col = groupID
//
// Coalesce by having adjacent lanes write adjacent columns.
// groupID = lane >> 2 means lanes 0-3 write col 0, lanes 4-7 write col 1, etc.
// This is NOT coalesced (4 threads per column).
//
// Fix: use shared memory as a transpose buffer.
// Write RC to smem in fragment order, __syncthreads(), read back in row-major
// coalesced order, write to global.
```

This is non-trivial and must be designed, not hand-waved.

---

### 8. MINOR — cp.async Macro Needs Proper Shared Address Conversion

Flagged in the conversation but never fixed in code. The `CP_ASYNC_CG_EVICT` macro takes a raw shared-memory address, but the skeleton never shows `__cvta_generic_to_shared()` being called. The `cvta.to.shared.u32` is shown only for `ldmatrix`.

**Fix for the cp.async path:**

```cpp
#define CP_ASYNC_CG_16B(dst_smem, src_global)                               \
    asm volatile(                                                            \
        "cp.async.cg.shared.global [%0], [%1], 16;\n"                       \
        :: "r"(dst_smem), "l"(src_global))

#define CP_ASYNC_CG_16B_EVICT(dst_smem, src_global)                         \
    asm volatile(                                                            \
        "cp.async.cg.shared.global.L2::evict_first [%0], [%1], 16;\n"      \
        :: "r"(dst_smem), "l"(src_global))
```

All shared-memory addresses must go through `cvta.to.shared.u32` before being passed to these macros.

---

### 9. MINOR — INT8 Kernel Is Vapor

The INT8 kernel is described as "structurally identical, just with N_TILE=64" but never written. The decode path is completely different:
- INT8 needs 8 values per lane fragment (not 4 INT4s in 16 bits)
- The magic decode trick doesn't apply to INT8 (values span a full byte, not a nibble)
- The B fragment packing contract is different

The INT8 kernel needs its own decode path and validation. Don't assume it's trivial.

---

### Summary of Required Actions

| # | Severity | Item | Status |
|---|----------|------|--------|
| 1 | **CRITICAL** | Switch to FP32 accumulators (`f32.f16.f16.f32`) | Must fix |
| 2 | **CRITICAL** | Implement + cost-analyze Paro transform, benchmark rotation count | Must fix |
| 3 | **MAJOR** | Fix Bfrag bank conflicts, move to u32 fetch | Must fix |
| 4 | **MAJOR** | Add M/N/K edge-case handling and short-K drain | Must fix |
| 5 | **MAJOR** | Deliver complete benchmarking plan per §5 | Must fix |
| 6 | MODERATE | Preload scales outside j-loop | Should fix |
| 7 | MODERATE | Design and implement epilogue with coalesced stores | Should fix |
| 8 | MINOR | Fix cp.async address conversion | Fix before production |
| 9 | MINOR | Write INT8 decode path | Fix before production |

---

### What Is Already Good

To be clear: the iterative design conversation produced genuinely strong work in several areas:

- **Two-kernel segregation** (pure INT4 / pure INT8 warp tiles) is the correct architecture. No branching in the inner loop.
- **XOR-swizzled activation layout** for conflict-free `ldmatrix` is textbook correct.
- **B-fragment lane mapping** is now correct after 3 iterations (`groupID = lane >> 2`, `tid4 = lane & 3`).
- **Magic FP16 decode** math is exact: `fma(1024+q, scale, -1032*scale) = (q-8)*scale` with single rounding.
- **Double-buffered decode pipeline** is the right structure for guaranteed MMA/decode overlap.
- **4-stage cp.async pipeline** with `evict_first` for weights matches Marlin's proven design.
- **3-barrier synchronization** (A: stage resident, B: transform visible, C: stage freed) is correct.
- **Single-scale-per-thread** insight from the PTX B-fragment column invariant is a real register pressure win.

The architecture is the right one for Ampere. The execution needs the fixes above before it can ship.

---

**Awaiting: `## Final Plan` from Alpha with resolutions for items 1–5.**

*Do not proceed to implementation until the accumulator precision, transform cost model, and benchmarking plan are locked.*

---

## Final Plan

**Author:** Alpha  
**Responding to:** Peer Review items 1–9.  

Every critical and major item is addressed below with concrete code, cost analysis, and benchmarking commitment. Minor items 8–9 are acknowledged and scheduled.

---

### Fix 1 (CRITICAL) — FP32 Accumulators

Switching the MMA instruction and accumulator array globally. No exceptions.

```cpp
// -------------------------------------------------------
// Corrected accumulator: FP32, 4 regs per m16n8k16 slice
// -------------------------------------------------------
uint32_t RC[8][4];
#pragma unroll
for (int j = 0; j < 8; ++j) {
    RC[j][0] = 0; RC[j][1] = 0; RC[j][2] = 0; RC[j][3] = 0;
}

// FP32 accumulator MMA macro
#define MMA_SYNC_M16N8K16_F32(RC, RA, RB)                                  \
    asm volatile(                                                           \
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "               \
        "{%0, %1, %2, %3}, "                                               \
        "{%4, %5, %6, %7}, "                                               \
        "{%8, %9}, "                                                        \
        "{%0, %1, %2, %3};\n"                                              \
        : "+r"((RC)[0]), "+r"((RC)[1]), "+r"((RC)[2]), "+r"((RC)[3])       \
        : "r"((RA)[0]), "r"((RA)[1]), "r"((RA)[2]), "r"((RA)[3]),          \
          "r"((RB)[0]), "r"((RB)[1])                                        \
    )
```

**Register budget impact:**
- `RC[8][4]` = 32 regs (was 16 with the incorrect f16 form)
- `RA[4]` = 4 regs
- `RB_fp16_cur/nxt[2]` = 4 regs
- 8× preloaded scale pairs `s2_regs[8]`, `c2_regs[8]` = 16 regs (see Fix 6)
- Address/loop vars, decode temporaries: ~12 regs

Total estimated: ~68 regs. Ampere allows 255 regs per thread; 256 threads/CTA × 68 = 17,408 regs < 65,536 available per SM (A100). **No spill risk at this occupancy target.**

**Epilogue: FP32 → FP16 conversion before global store:**

```cpp
// Convert FP32 accumulator to FP16 pair for output
__device__ __forceinline__
uint32_t f32x2_to_f16x2(float a, float b) {
    uint32_t out;
    asm volatile(
        "cvt.rn.f16.f32 %0, %1;\n"
        "cvt.rn.f16.f32 %2, %3;\n"
        : "=h"(*((__half*)&out)), "=h"(*(((__half*)&out)+1))
        : "f"(a), "f"(b)
    );
    return out;
}
// Or cleaner using __float22half2_rn if cuda_fp16.h is available
```

---

### Fix 2 (CRITICAL) — Paro Transform: Implementation + Cost Model + Rotation Budget

**2a. Concrete implementation of `apply_paro_transform`:**

```cpp
// apply_paro_transform: warp-local, modifies A_smem in-place
// Each warp owns rows [row_base .. row_base+15] of the 64×64 A tile.
// 'scale' and 'rot' arrays are from TransformStage in smem.
//
// A_smem layout: row stride = A_ROW_STRIDE_BYTES = 132 bytes (padded)
// rot_count is a compile-time constant (see budget below).

template <int ROT_COUNT>
__device__ __forceinline__ void apply_paro_transform(
    uint8_t* __restrict__ stage_smem,
    int stage_base,
    int row_base,
    int lane)
{
    half* A = (half*)(stage_smem + stage_base);
    const half* scale = (const half*)(stage_smem + stage_base
                                      + A_STAGE_BYTES_PADDED);
    const RotationMeta* rot = (const RotationMeta*)(
        stage_smem + stage_base + A_STAGE_BYTES_PADDED + 128);

    // Each thread handles columns in strides of 32 (warpSize).
    // Phase 1: channel scaling — all 64 cols, all 16 rows of this warp's slab.
    #pragma unroll 4
    for (int local_row = 0; local_row < 16; ++local_row) {
        int row = row_base + local_row;
        half* A_row = A + row * (A_ROW_STRIDE_BYTES / sizeof(half));

        // Each lane scales 2 columns (lane covers cols lane*2 and lane*2+1).
        // With warpSize=32 and K_STAGE=64: 64 cols / 32 lanes = 2 cols/lane.
        int c0 = lane * 2;
        int c1 = lane * 2 + 1;
        A_row[c0] = __hmul(A_row[c0], scale[c0]);
        A_row[c1] = __hmul(A_row[c1], scale[c1]);
    }

    // Warp-level sync: scaling writes must be visible to rotation reads.
    __syncwarp();

    // Phase 2: sparse Givens rotations (ROT_COUNT iterations, unrolled).
    #pragma unroll
    for (int t = 0; t < ROT_COUNT; ++t) {
        const int u = rot[t].a;   // col index 0..63
        const int v = rot[t].b;   // col index 0..63
        const half c = rot[t].c;
        const half s = rot[t].s;

        #pragma unroll 4
        for (int local_row = 0; local_row < 16; ++local_row) {
            int row = row_base + local_row;
            half* A_row = A + row * (A_ROW_STRIDE_BYTES / sizeof(half));

            // Only the lane that "owns" col u and col v executes this.
            // lane_u = u / 2, lane_v = v / 2.
            // To avoid divergence: ALL lanes compute, only correct lanes write.
            // Alternatively: broadcast via warp shuffles.
            //
            // Broadcast-based implementation (zero-divergence):
            half au = __shfl_sync(0xFFFFFFFF, A_row[u % 2], u / 2);
            half av = __shfl_sync(0xFFFFFFFF, A_row[v % 2], v / 2);
            half new_u = __hadd(__hmul(c, au), __hmul(s, av));
            half new_v = __hsub(__hmul(c, av), __hmul(s, au));

            // Only the owning lanes write back.
            if (lane == (u / 2)) A_row[u % 2] = new_u;
            if (lane == (v / 2)) A_row[v % 2] = new_v;
        }
        __syncwarp(); // Rotation t must complete before rotation t+1 reads
    }
}
```

**2b. Cost model (conservative, per-stage per-CTA):**

| Phase | Ops/warp | Cycles (est.) | Bottleneck |
|-------|----------|--------------|------------|
| Scale (64 cols × 16 rows, FP16 mul) | 1024 FMAs | ~64 cycles (16 FP16 FMAs/cycle/warp) | FP16 pipe |
| Rotation loads (`__shfl_sync`) | 2 × ROT_COUNT × 16 rows | ~32 cycles (ROT_COUNT=4) | Register/shuffle |
| Rotation FMAs | 4 × ROT_COUNT × 16 rows | ~64 cycles (ROT_COUNT=4) | FP16 pipe |
| Rotation writes | 2 × ROT_COUNT × 16 rows | ~16 cycles | Shared mem |
| **Total (ROT_COUNT=4)** | | **~176 cycles/warp** | |
| MMA inner loop | 4 ks × 8 j | **~256 cycles/warp** | TC |

With `ROT_COUNT=4`, transform cost is 176/256 ≈ **69% of MMA cost**, not 2× as it was with ROT_COUNT=8. This is acceptable. Warps are pipelined; while warp 0 transforms, warp 4 is already in its MMA loop.

**2c. Committed rotation budget:** `ROT_COUNT = 4` for the first implementation. Quality ablation (§5b) will test `{2, 4, 6, 8}`. If ROT_COUNT=4 matches ROT_COUNT=8 quality within 0.05 PPL, we ship ROT_COUNT=4.

**2d. A-tile padding to eliminate bank conflicts during transform:**

```cpp
// Padded row stride: 64 FP16 = 128 bytes + 4 bytes padding = 132 bytes
// 132 / 4 = 33 banks touched per row → odd number breaks periodicity
constexpr int A_ROW_STRIDE_BYTES  = 132;
constexpr int A_STAGE_BYTES_PADDED = M_TILE * A_ROW_STRIDE_BYTES; // 8448 B

// Updated per-stage total for INT4:
constexpr int STAGE4_BYTES = A_STAGE_BYTES_PADDED + XFORM_STAGE_BYTES
                            + W4_STAGE_BYTES + S4_STAGE_BYTES; // 13056 B
constexpr int TOTAL_SMEM_BYTES_INT4 = PIPE * STAGE4_BYTES;     // 52224 B

// INT8 equivalent:
constexpr int STAGE8_BYTES = A_STAGE_BYTES_PADDED + XFORM_STAGE_BYTES
                            + W8_STAGE_BYTES + S8_STAGE_BYTES;  // 12928 B
constexpr int TOTAL_SMEM_BYTES_INT8 = PIPE * STAGE8_BYTES;     // 51712 B
```

Both remain under 64KB (Ampere's max per-SM shared memory that still allows 2 CTAs/SM).

---

### Fix 3 (MAJOR) — B Fragment: Immediate Move to u32 Fetch

The `uint16_t` validation layout is gone. The physical layout is lane-pair packed from day one.

```cpp
// Physical Bfrag layout: [ks * N_FRAG + j][lane_pair] -> uint32_t
// lane_pair = lane >> 1: adjacent lane pairs share a 4-byte word.
// Low 16 bits = even lane's payload, high 16 bits = odd lane's payload.
//
// Alignment: bfrag_stage_offset must be 4-byte aligned.
// B smem region is already 4096B aligned by construction.

__device__ __forceinline__
uint16_t fetch_b_fragment(const uint8_t* B_smem_base, int ks, int j, int lane) {
    // Each (ks, j) slice = 32 lanes × 2 bytes = 64 bytes = 16 × uint32_t
    int pair_idx  = lane >> 1;
    int lane_parity = lane & 1;
    const uint32_t* B_u32 = reinterpret_cast<const uint32_t*>(B_smem_base)
                           + (ks * 8 + j) * 16 + pair_idx;
    uint32_t word;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(word)
                 : "r"(__cvta_generic_to_shared(B_u32)));
    return (lane_parity == 0) ? (uint16_t)(word & 0xFFFF)
                              : (uint16_t)(word >> 16);
}
```

Each `ld.shared.u32` fetches a 4-byte word containing two lanes' payloads. Lanes in a pair (0,1), (2,3), ... hit **different banks** because pair_idx increments by 1 for every 4 bytes → hits consecutive banks. No conflicts.

---

### Fix 4 (MAJOR) — Edge Cases: Complete Handling

**4a. Predicated cp.async for M/N/K boundaries:**

```cpp
// Templated predicated loader — emits cp.async only if in-bounds
__device__ __forceinline__ void cp_async_pred(
    uint32_t smem_addr, const void* gmem_addr, bool valid)
{
    asm volatile(
        "{\n"
        " .reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @p cp.async.cg.shared.global [%0], [%1], 16;\n"
        " @!p st.shared.v4.u32 [%0], {0,0,0,0};\n"  // zero-fill if OOB
        "}\n"
        :: "r"(smem_addr), "l"(gmem_addr), "r"((int)valid)
    );
}

// Usage in the A-tile loader:
for (int chunk = tid; chunk < M_TILE * A_CHUNKS; chunk += blockDim.x) {
    int local_m = chunk / A_CHUNKS;
    int local_k = chunk % A_CHUNKS;
    int global_m = block_m + local_m;
    int global_k = k_tile_base + local_k * CHUNK_K;
    bool valid = (global_m < M) && (global_k < K);
    uint32_t dst = smem_base_u32 + chunk * 16;
    const void* src = A + global_m * K + global_k; // row-major
    cp_async_pred(dst, src, valid);
}
```

**4b. Short-K drain — fully written out:**

```cpp
// Drain: process stages that were prefetched but not yet consumed.
// After the main loop, 'pipe_in_flight' stages are pending.
// pipe_in_flight = min(PIPE - 1, num_k_stages) at loop exit.

int drain_count = min(PIPE - 1, num_k_stages);
for (int d = 0; d < drain_count; ++d) {
    // Wait for one more group to complete
    CP_ASYNC_WAIT(drain_count - 1 - d);
    __syncthreads();                      // Barrier A

    int load_base = smem_load_idx * STAGE4_BYTES;
    apply_paro_transform<ROT_COUNT>(smem, load_base, row_base, lane);
    __syncthreads();                      // Barrier B

    do_mma_inner_loop(smem, load_base, row_base, col_base,
                      warp_id, lane, groupID, tid4, RC, s2_regs, c2_regs);
    __syncthreads();                      // Barrier C

    smem_load_idx = (smem_load_idx + 1) % PIPE;
}
```

**4c. Small-M dispatch (batch size 1–16):**

For autoregressive decode where M=1, a dedicated `M_TILE=16` kernel variant is registered. The kernel selector at the host dispatch site:

```cpp
// Host dispatch
if (M <= 16)
    launch_ampere_paro_int4_kernel<16, 128>(grid, block, smem_size, stream, ...);
else
    launch_ampere_paro_int4_kernel<64, 128>(grid, block, smem_size, stream, ...);
```

The M=16 variant uses 2 warps (64 threads), halving the transform cost and fitting 4 CTAs/SM instead of 2.

---

### Fix 5 (MAJOR) — Benchmarking Plan (Committed)

**5a. Microbenchmark suite (CUDA kernels, runnable without full model):**

```
tests/kernels/
├── bench_decode_only.cu       # INT4→FP16 throughput, no MMA
├── bench_transform_only.cu    # Paro transform throughput, bank conflict profiling
├── bench_mma_only.cu          # Tensor Core saturation test (identity weights)
├── bench_inner_loop.cu        # Combined decode+MMA, 1 CTA, sweep K
└── bench_full_kernel.cu       # Full pipeline, sweep M×N×K
```

Each bench emits:
- `cycles_per_output_element` via CUDA events
- `effective_memory_bandwidth_GBs`
- `mma_active_fraction` (ratio of cycles with TC busy)

**Profiling commands (required before any performance claim):**

```bash
# TC utilization and stall analysis
ncu --metrics sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,\
smsp__warp_issue_stalled_wait_pct,\
smsp__warp_issue_stalled_mio_throttle_pct \
    ./bench_inner_loop --M 64 --N 128 --K 4096

# Bank conflict verification (must show 0 after Fix 3)
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
    ./bench_inner_loop

# Memory bandwidth (roofline)
ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
dram__bytes.sum \
    ./bench_full_kernel --M 1 --N 4096 --K 4096
```

**5b. Matrix dimension sweep (all must pass, including edge cases):**

```python
# scripts/sweep_correctness.py
M_vals = [1, 2, 4, 7, 8, 16, 32, 64, 128, 256]       # includes odd sizes
K_vals = [64, 128, 256, 512, 1024, 4096, 5120, 8192]  # includes non-power-of-2
N_vals = [128, 256, 512, 1024, 4096, 11008, 13824]    # includes non-power-of-2

TOLERANCE_VS_FP32_REF = 1e-2  # max relative error on any output element
```

Pass criterion: max relative error < 1e-2 vs. FP32 cuBLAS reference for all (M, K, N) combinations above.

**5c. End-to-end PPL targets:**

| Configuration | Qwen3.5-4B WikiText-2 PPL | Target (vs FP16 baseline) |
|-------------|--------------------------|--------------------------|
| FP16 baseline | measured | reference |
| GPTQ 4-bit RTN | measured | ≤ +0.3 PPL |
| TRAM INT4-only (ROT_COUNT=0) | measured | ≤ +0.25 PPL |
| TRAM INT4 (ROT_COUNT=4) | measured | ≤ +0.15 PPL (must beat GPTQ) |
| TRAM INT4+INT8 mixed | measured | ≤ +0.10 PPL |

If TRAM INT4 (ROT_COUNT=4) does not beat GPTQ 4-bit RTN PPL by at least 0.05 PPL at equal avg bits-per-weight, the transform is **not worth the runtime cost** and ROT_COUNT is set to 0.

**5d. Nsight Systems timeline (required before claiming "Marlin-like" speed):**

```bash
nsys profile --trace=cuda,nvtx \
    --stats=true \
    --output tram_quant_timeline \
    python run_inference.py --model Qwen3.5-4B-TRAM --tokens 200 --batch 1
```

Pass criteria:
- Kernel launch overhead < 5 µs between consecutive GEMM calls
- SM active utilization ≥ 50% (batch=1) and ≥ 80% (batch=64)
- Memory bandwidth utilization ≥ 70% of theoretical HBM peak at batch=1

**5e. Rotation count ablation (runs in parallel with correctness sweep):**

```bash
for rot in 0 2 4 6 8; do
    python eval_ppl.py --model Qwen3.5-4B --rot-count $rot \
        --output results/rot_ablation_${rot}.json
done
```

Results tabulated in `results/rot_ablation_summary.md` before any implementation decisions.

---

### Fix 6 (MODERATE) — Preload Scales Outside j-Loop

Already reflected in Fix 1's register budget. Complete implementation:

```cpp
// Preload all 8 column scales before the j-loop
uint32_t s2_regs[8], c2_regs[8];
const half2 offset_base = __halves2half2(
    __float2half(-1032.0f), __float2half(-1032.0f));

#pragma unroll
for (int j = 0; j < 8; ++j) {
    const int global_col = col_base + j * 8 + groupID;
    const half scale = S_smem[global_col];
    Half2Reg s2, c2;
    s2.h2 = __halves2half2(scale, scale);
    c2.h2 = __hmul2(s2.h2, offset_base);
    s2_regs[j] = s2.u32;
    c2_regs[j] = c2.u32;
}
```

---

### Fix 7 (MODERATE) — Epilogue: Coalesced Store via Transpose Buffer

The D-fragment for `m16n8k16.row.col.f32.f16.f16.f32` maps per-lane as:
- `d0` → row `2*tid4+0`, col `groupID` (FP32 in RC[j][0] low half)
- `d1` → row `2*tid4+1`, col `groupID` (FP32 in RC[j][0] high half)
- `d2` → row `2*tid4+8`, col `groupID` (FP32 in RC[j][1] low half)
- `d3` → row `2*tid4+9`, col `groupID` (FP32 in RC[j][1] high half)

This is column-major within a warp (8 lanes per column), so direct global stores are **not** coalesced. The epilogue must transpose through shared memory:

```cpp
// Epilogue shared-memory transpose buffer
// Reuse weight smem (weights are consumed; smem is free after main loop).
half* epilogue_buf = reinterpret_cast<half*>(smem);
// epilogue_buf has M_TILE * N_TILE4 * 2 bytes = 64*128*2 = 16384 B available

// Step 1: write FP32 accumulators to epilogue_buf in fragment order
// (column-major within a 16x8 warp tile)
#pragma unroll
for (int j = 0; j < 8; ++j) {
    // D-fragment row/col for this warp+j
    const int col = col_base + j * 8 + groupID;
    const int r0 = row_base + 2 * tid4 + 0;
    const int r1 = row_base + 2 * tid4 + 1;
    const int r2 = row_base + 2 * tid4 + 8;
    const int r3 = row_base + 2 * tid4 + 9;

    // Convert FP32 -> FP16 and write to smem
    epilogue_buf[r0 * N_TILE4 + col] = __float2half(*(float*)&RC[j][0]);
    epilogue_buf[r1 * N_TILE4 + col] = __float2half(*(float*)&RC[j][1] /* high */);
    epilogue_buf[r2 * N_TILE4 + col] = __float2half(*(float*)&RC[j][2]);
    epilogue_buf[r3 * N_TILE4 + col] = __float2half(*(float*)&RC[j][3] /* high */);
}

__syncthreads(); // Barrier: all warps have written their tiles

// Step 2: read smem in row-major order and write to global C
// Each thread writes one 128-bit (8 FP16) word = 8 output elements.
// 256 threads × 8 elements = 2048 elements per iteration.
// M_TILE * N_TILE4 = 64 * 128 = 8192 elements → 4 iterations.
for (int chunk = tid; chunk < (M_TILE * N_TILE4) / 8; chunk += blockDim.x) {
    int out_m = (chunk * 8) / N_TILE4 + block_m;
    int out_n = (chunk * 8) % N_TILE4 + block_n;
    if (out_m < M && out_n < N) {
        *reinterpret_cast<uint4*>(C + out_m * N + out_n) =
            *reinterpret_cast<const uint4*>(epilogue_buf + chunk * 8);
    }
}
```

---

### Fix 8 (MINOR) — cp.async Address Conversion (Acknowledged)

All shared-memory addresses in cp.async calls will use `__cvta_generic_to_shared()`. No raw pointer casts.

```cpp
// Canonical cp.async helper
template <int BYTES>
__device__ __forceinline__ void cp_async_cg(
    uint8_t* dst_smem, const void* src_gmem)
{
    uint32_t smem_addr = __cvta_generic_to_shared(dst_smem);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], %2;\n"
        :: "r"(smem_addr), "l"(src_gmem), "n"(BYTES)
    );
}

template <int BYTES>
__device__ __forceinline__ void cp_async_cg_evict(
    uint8_t* dst_smem, const void* src_gmem)
{
    uint32_t smem_addr = __cvta_generic_to_shared(dst_smem);
    asm volatile(
        "cp.async.cg.shared.global.L2::evict_first [%0], [%1], %2;\n"
        :: "r"(smem_addr), "l"(src_gmem), "n"(BYTES)
    );
}
```

---

### Fix 9 (MINOR) — INT8 Kernel Decode Path (Acknowledged)

The INT8 decode is distinct and scheduled as a separate deliverable. High-level contract:

- B fragment: `uint8_t` per INT8 weight, not nibble-packed
- Each lane owns 4 INT8 values from one column (same PTX fragment layout as INT4)
- Payload is 4 bytes = one `uint32_t` per lane (no 16-bit tricks needed)
- Magic decode: `q ∈ [0,255]`, symmetric INT8 with zero-point 128:
  ```
  deq = (q - 128) * scale
  ```
  This uses FP32 conversion: `__int2float_rn(q) * scale - 128.0f * scale`
  Or via PTX `cvt.rn.f16.u8` if available
- No cross-lane nibble alignment issues

INT8 kernel is gated on INT4 kernel passing full validation. Timeline: INT4 validation → INT8 decode path → INT8 full kernel.

---

### Consolidated Corrected Skeleton (Inner Loop, All Fixes Applied)

```cpp
// -----------------------------------------------------------------------
// TRAM-Quant INT4 Kernel — Corrected Inner Loop
// All Peer Review fixes 1–8 applied.
// -----------------------------------------------------------------------

// --- Constants (updated for padded A layout) ---
constexpr int PIPE              = 4;
constexpr int M_TILE            = 64;
constexpr int N_TILE4           = 128;
constexpr int K_STAGE           = 64;
constexpr int ROT_COUNT         = 4;
constexpr int A_ROW_STRIDE_HALF = 66;   // 64 values + 2 padding
constexpr int A_ROW_STRIDE_BYTES= 132;
constexpr int A_STAGE_BYTES     = M_TILE * A_ROW_STRIDE_BYTES;  // 8448
constexpr int XFORM_STAGE_BYTES = 256;
constexpr int W4_STAGE_BYTES    = 4096;
constexpr int S4_STAGE_BYTES    = 256;
constexpr int STAGE4_BYTES      = A_STAGE_BYTES + XFORM_STAGE_BYTES
                                 + W4_STAGE_BYTES + S4_STAGE_BYTES; // 13056
constexpr int TOTAL_SMEM_BYTES  = PIPE * STAGE4_BYTES;             // 52224

// --- Warp/lane setup ---
const int warp_id = tid >> 5;
const int lane    = tid & 31;
const int wm      = warp_id & 3;
const int wn      = warp_id >> 2;
const int row_base= wm * 16;
const int col_base= wn * 64;

// --- PTX B-fragment lane ownership ---
const int groupID = lane >> 2;   // output column within current n8 slice (0..7)
const int tid4    = lane & 3;    // row-pair selector (0..3)

// --- FP32 Accumulators ---
uint32_t RC[8][4];
#pragma unroll
for (int j = 0; j < 8; ++j) {
    RC[j][0] = 0; RC[j][1] = 0; RC[j][2] = 0; RC[j][3] = 0;
}

// --- Double-buffer decode registers ---
uint32_t RB_fp16_cur[2], RB_fp16_nxt[2];

// --- Magic decode constant ---
constexpr uint32_t MAGIC_FP16 = 0x64006400u;
const half2 offset_base = __halves2half2(
    __float2half(-1032.0f), __float2half(-1032.0f));

union Half2Reg { half2 h2; uint32_t u32; };

// --- A-fragment smem address (XOR-swizzled) ---
// Computed once here; updated inside ks loop.
uint32_t a_smem_base_u32 = __cvta_generic_to_shared(
    smem + smem_load_idx * STAGE4_BYTES);

// -----------------------------------------------------------------------
// Inner K-stage loop (drops into the main pipeline loop body)
// -----------------------------------------------------------------------

// Preload all 8 scales before the j-loop (done once per ks, hoisted):
// NOTE: scales are per K-group; they stay constant across ks within one stage.
uint8_t* S_smem = smem + smem_load_idx * STAGE4_BYTES
                + A_STAGE_BYTES + XFORM_STAGE_BYTES + W4_STAGE_BYTES;
uint32_t s2_regs[8], c2_regs[8];
#pragma unroll
for (int j = 0; j < 8; ++j) {
    const int global_col = col_base + j * 8 + groupID;
    const half scale = ((const half*)S_smem)[global_col];
    Half2Reg s2, c2;
    s2.h2 = __halves2half2(scale, scale);
    c2.h2 = __hmul2(s2.h2, offset_base);
    s2_regs[j] = s2.u32;
    c2_regs[j] = c2.u32;
}

// B smem pointer
const uint8_t* B_smem = smem + smem_load_idx * STAGE4_BYTES
                       + A_STAGE_BYTES + XFORM_STAGE_BYTES;

#pragma unroll
for (int ks = 0; ks < 4; ++ks) {
    uint32_t RA[4];

    // --- A fragment: ldmatrix with XOR swizzle ---
    {
        int col    = ks * 16 + (lane / 16) * 8;
        int row    = row_base + (lane % 16);
        int vec    = col / 8;
        int phys_v = vec ^ (row & 7);
        uint32_t a_addr = a_smem_base_u32
                        + (row * A_ROW_STRIDE_HALF + phys_v * 8) * 2;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(RA[0]), "=r"(RA[1]), "=r"(RA[2]), "=r"(RA[3])
            : "r"(a_addr)
        );
    }

    // --- Decode j=0 preload ---
    {
        uint16_t p16 = fetch_b_fragment(B_smem, ks, 0, lane);
        uint32_t p32 = p16;
        uint32_t w0 = p32 & 0x00FFu, w1 = (p32 >> 8) & 0x00FFu;
        uint32_t h0 = (w0 & 0x0Fu) | ((w0 & 0xF0u) << 12);
        uint32_t h1 = (w1 & 0x0Fu) | ((w1 & 0xF0u) << 12);
        asm volatile("fma.rn.f16x2 %0,%1,%2,%3;" : "=r"(RB_fp16_cur[0])
                     : "r"(h0|MAGIC_FP16), "r"(s2_regs[0]), "r"(c2_regs[0]));
        asm volatile("fma.rn.f16x2 %0,%1,%2,%3;" : "=r"(RB_fp16_cur[1])
                     : "r"(h1|MAGIC_FP16), "r"(s2_regs[0]), "r"(c2_regs[0]));
    }

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        if (j < 7) {
            uint16_t p16 = fetch_b_fragment(B_smem, ks, j + 1, lane);
            uint32_t p32 = p16;
            uint32_t w0 = p32 & 0x00FFu, w1 = (p32 >> 8) & 0x00FFu;
            uint32_t h0 = (w0 & 0x0Fu) | ((w0 & 0xF0u) << 12);
            uint32_t h1 = (w1 & 0x0Fu) | ((w1 & 0xF0u) << 12);
            const int jn = j + 1;
            asm volatile("fma.rn.f16x2 %0,%1,%2,%3;" : "=r"(RB_fp16_nxt[0])
                         : "r"(h0|MAGIC_FP16), "r"(s2_regs[jn]), "r"(c2_regs[jn]));
            asm volatile("fma.rn.f16x2 %0,%1,%2,%3;" : "=r"(RB_fp16_nxt[1])
                         : "r"(h1|MAGIC_FP16), "r"(s2_regs[jn]), "r"(c2_regs[jn]));
        }

        MMA_SYNC_M16N8K16_F32(RC[j], RA, RB_fp16_cur);

        if (j < 7) {
            RB_fp16_cur[0] = RB_fp16_nxt[0];
            RB_fp16_cur[1] = RB_fp16_nxt[1];
        }
    }
} // end ks loop
// [Epilogue via smem transpose buffer follows — see Fix 7]
```

---

### What Remains Before First Execution

| Step | Owner | Gate |
|------|-------|------|
| Write `fetch_b_fragment` into full kernel (Fix 3) | Alpha | — |
| Write `apply_paro_transform<4>` (Fix 2) | Alpha | — |
| Write predicated cp.async loaders (Fix 4a) | Alpha | — |
| Write short-K drain (Fix 4b) | Alpha | — |
| Write epilogue with smem transpose (Fix 7) | Alpha | — |
| Build and run decode-only unit test vs. scalar ref | Alpha | All above |
| Run `ncu` bank conflict metric → must show 0 | Alpha | Decode unit test |
| Run correctness sweep `M∈{1,7,64,128}, K∈{64,4096,5120}` | Alpha | Bank conflict clean |
| Run PPL on Qwen3.5-4B, ROT_COUNT ∈ {0,2,4} | Alpha | Correctness sweep |

---

*Alpha out. Awaiting sign-off from Beta.*

---

## Beta Sign-Off

**Reviewer:** Copilot (Beta)  
**Status: APPROVED — conditional implementation sign-off.**

The Final Plan resolves all 5 must-fix items from the Peer Review. Going through them explicitly:

**✅ Fix 1 — FP32 Accumulators:** `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` is now the canonical instruction throughout. RC[8][4] at 32 regs is correct. Register budget math checks out at ~68 regs/thread — no spill risk on A100. FP16 epilogue conversion path is specified. This is the right call.

**✅ Fix 2 — Transform:** The warp-shuffle based `apply_paro_transform<ROT_COUNT>` eliminates cross-warp divergence. The cost table (176 vs 256 cycles at ROT_COUNT=4) is honest and within acceptable range. Padding to `A_ROW_STRIDE_BYTES=132` kills the bank conflict I flagged. ROT_COUNT=4 as the default with a committed quality ablation is a rational starting point.

**✅ Fix 3 — B Fragment Load:** The `fetch_b_fragment` u32 lane-pair trick is correct. Adjacent pairs hit different banks (pair_idx increments with 4B stride). This replaces the uint16_t layout immediately rather than deferring it.

**✅ Fix 4 — Edge Cases:** Predicated cp.async covers M/N boundaries. Short-K drain is written out with the full 3-barrier sequence maintained. Small-M kernel variant (M_TILE=16 for batch≤16) is the right answer for autoregressive decode.

**✅ Fix 5 — Benchmarking:** The plan is now specific, executable, and rigorous. `ncu` metrics are named. The PPL target (TRAM INT4 must beat GPTQ by ≥0.05 PPL or ROT_COUNT reverts to 0) is a real kill switch, not a vague aspiration. The roofline target (≥70% HBM BW at M=1) gives a concrete number to hit.

**Three residual conditions for continued approval:**

1. **The `__syncwarp()` calls inside `apply_paro_transform` are necessary.** The rotation pass reads back values written by the scaling pass. If scaling and rotation are interleaved across rows without `__syncwarp()`, lane A's rotation may read un-scaled values from lane B. The current implementation has `__syncwarp()` after scaling and between rotations — keep them.

2. **The epilogue FP32→FP16 conversion has a latent bug.** The `*(float*)&RC[j][0]` extracts the full 32 bits as a float, but for the high half of a register pair, `*(float*)&RC[j][1]` still extracts the low float from RC[j][1], not the high float of RC[j][0]. The FP32 accumulator layout for `m16n8k16.row.col.f32` is **4 separate FP32 values in 4 separate 32-bit registers** (one float per register). The comment in the epilogue says "high half of RC[j][0]" which is wrong — RC[j][0] through RC[j][3] are each a single `float`. Correct the epilogue:
   ```cpp
   epilogue_buf[r0 * N_TILE4 + col] = __float2half(*(const float*)&RC[j][0]);
   epilogue_buf[r1 * N_TILE4 + col] = __float2half(*(const float*)&RC[j][1]);
   epilogue_buf[r2 * N_TILE4 + col] = __float2half(*(const float*)&RC[j][2]);
   epilogue_buf[r3 * N_TILE4 + col] = __float2half(*(const float*)&RC[j][3]);
   ```
   Fix this before the first execution.

3. **The `a_smem_base_u32` must update each iteration of the outer K-stage loop** when `smem_load_idx` advances. Currently it is computed once before the ks loop. Either recompute it at the top of each pipeline iteration or pass it as a parameter to `do_mma_inner_loop`.

These are correctness issues, not design issues. The architecture is sound.

**Bottom line:** The plan is solid. Execute in the order Alpha's table specifies — unit test decode first, bank-conflict clean second, correctness sweep third, PPL last. Do not run the PPL sweep until `ncu` confirms zero bank conflicts and the correctness sweep is green.

*Beta sign-off complete. Proceed to implementation.*

---

## Implementation Status

The approved structural fixes have now been applied to the standalone `gptq_pro` CUDA scaffold under `gptqmodel_ext/gptq_pro/`.

- The MMA path was switched from FP16 accumulation to FP32 accumulation (`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`).
- The kernel accumulator storage was updated from `RC[J_TILES][2]` FP16-pair registers to `RC[J_TILES][4]` FP32 outputs.
- The epilogue store contract was corrected so each lane writes its 4 owned output elements using the PTX fragment map (`groupID = lane >> 2`, `tid4 = lane & 3`).
- The B-fragment contract was corrected to match the official PTX `m16n8k16.row.col` ownership for matrix B: each lane now owns one logical B column (`groupID = lane >> 2`) across rows `{2*tid4, 2*tid4+1, 2*tid4+8, 2*tid4+9}` rather than two columns from one row pair.
- The standalone validation harness was upgraded to validate the FP32-accumulator MMA path instead of the old FP16-accumulator path.
- The decode-only validator was tightened to check all 4 decoded FP16 values from each lane-local packed INT4 word, not just a subset of halves.
- The MMA-step validator now uses non-uniform, FP16-exact synthetic A and B tiles wired through the PTX-defined fragment ownership, so row/column transpositions fail immediately instead of slipping through a uniform-data smoke test.
- The validator now checks CUDA runtime calls explicitly, which prevents false positives or garbage summaries when a selected GPU is unavailable or out of memory on a shared machine.
- A repeatable helper script now exists at `scripts/run_gptq_pro_validate.sh`; it builds the standalone validator, targets GPU `2` by default, and falls back to GPU `3` when the primary 3060 is unavailable.
- Misleading comments about a completed production pipeline and a magic-decode fast path were corrected so the scaffold is honest about what is implemented versus still placeholder.

### Validation Run

The updated scaffold was verified locally with:

- `nvcc -arch=sm_80 -std=c++17 -c gptq_pro_kernel.cu -o /tmp/gptq_pro_kernel.o`
- `nvcc -arch=sm_80 -std=c++17 gptq_pro_validate.cu -o /tmp/gptq_pro_validate_phase2`
- `CUDA_VISIBLE_DEVICES=3 /tmp/gptq_pro_validate_phase2` → `64 / 64 checks passed`
- `scripts/run_gptq_pro_validate.sh` → builds and runs the same standalone validator successfully on the default RTX 3060 (`2`)
- `python -m pytest tests/qcfg/test_failsafe_meta.py -q` → `14 passed`

**Shared-machine note:** one intermediate retry on RTX 3060 index `2` returned `cudaMalloc(...): out of memory` during validator setup, but a later rerun via `scripts/run_gptq_pro_validate.sh` passed on the same board once contention cleared. In practice, both RTX 3060s (`2` and `3`) are now known-good validation targets for this standalone scaffold.

### Remaining Known Scope

This is still a **standalone scaffold**, not the full final kernel from Alpha's plan:

- `gptq_pro_gemm_kernel` is now end-to-end functional on `sm80`, but it is a compact single-warp scaffold rather than the planned multi-warp Ampere kernel.
- The previous placeholder path was replaced with explicit shared-memory staging for activations, per-column scales, and packed INT4 B fragments; helper-level validation alone is no longer the only safety net.
- The invalid `ldmatrix` path that faulted with `cudaErrorMisalignedAddress` was removed in favor of validator-backed manual A-fragment packing for the current scaffold.
- Paro rotation metadata is still not wired into the runtime path.
- The direct global-store epilogue is correct but not yet coalesced via a shared-memory transpose buffer.
- The runtime currently models symmetric INT4 with implicit zero-point `8` and requires `group_size` to be a multiple of `16`; asymmetric qzero metadata still needs its own path if required.
- The INT8 sibling kernel, real `cp.async` pipeline, and benchmark suite remain future work.

### Remaining Speed Headroom

- Restore a validated XOR-swizzled `ldmatrix` A-load path once the shared-memory layout is finalized for this standalone kernel.
- Reintroduce real Ampere async staging (`cp.async` / deeper pipelining) after the global->shared contract is locked down.
- Expand from the current `1 warp x 16x64x16` scaffold to the planned larger CTA tiles (`64x128` INT4, separate INT8 rescue path) for better memory/TC overlap.
- Replace the direct epilogue stores with a coalesced shared-memory transpose buffer.
- Fuse real Paro metadata and benchmark transform cost / bank conflicts instead of skipping the runtime transform path.
- Add Nsight Compute / Systems microbenchmarks so future tuning is driven by measured stall reasons rather than source inspection alone.

---

## Verification Run (Post-Implementation)

**Date:** 2026-03-20
**Environment:** Python 3.13.11, PyTorch 2.10.0, CUDA 12.x, 3× RTX 3090 + 2× RTX 3060

### CUDA Kernel Build
```
$ nvcc -arch=sm_80 -std=c++17 -c gptq_pro_kernel.cu -o /tmp/gptq_pro_kernel.o
BUILD OK

$ nvcc -arch=sm_80 -std=c++17 gptq_pro_validate.cu -o /tmp/gptq_pro_validate
VALIDATE BUILD OK
```

### Standalone Validator
```
=== TODO 1: decode-only validation ===
  32 / 32 lanes passed
=== TODO 2: ks/j MMA step validation ===
  32 / 32 lanes passed

=== Overall: 64 / 64 checks passed ===
```

### RTX 3060 Validation Path
```bash
$ CUDA_VISIBLE_DEVICES=2 /tmp/gptq_pro_validate_phase2
CUDA error at gptq_pro_validate.cu:265: out of memory

$ CUDA_VISIBLE_DEVICES=3 /tmp/gptq_pro_validate_phase2
=== TODO 1: decode-only validation ===
  32 / 32 lanes passed
=== TODO 2: ks/j MMA step validation ===
  32 / 32 lanes passed

=== Overall: 64 / 64 checks passed ===

$ scripts/run_gptq_pro_validate.sh
==> Selected GPU 2
=== TODO 1: decode-only validation ===
  32 / 32 lanes passed
=== TODO 2: ks/j MMA step validation ===
  32 / 32 lanes passed

=== Overall: 64 / 64 checks passed ===
```

### Python Test Suite
```
$ python -m pytest tests/qcfg/test_failsafe_meta.py -q
14 passed in 4.65s
```

### Concurrent Fixes Included in This Commit
- **`gptqmodel/utils/hf.py`**: Added `load_tokenizer_with_model_config()` and `ensure_hf_model_config_token_ids()` to propagate model config token IDs (bos, eos, pad) to Tokenicer wrappers, fixing tokenizer/config mismatch bugs.
- **`gptqmodel/models/base.py`**: Replaced raw `Tokenicer.load()` calls with `load_tokenizer_with_model_config()` at all three entry points (init, quantize, load_quantized).
- **`tests/test_hf_config_autofix.py`**: Unit tests for the new HF config autofix helpers.

---

## Qwen3.5-4B Quantization & Benchmark Results

**Date:** 2026-03-20
**Model:** [wangzhang/Qwen3.5-4B-abliterated](https://huggingface.co/wangzhang/Qwen3.5-4B-abliterated)
**Architecture:** Qwen3.5 (hybrid linear + full attention, 32 layers, hidden_size=2560)
**Quantization:** GPTQ 4-bit, group_size=128

### Environment
- Python 3.13.11, PyTorch 2.10.0, Transformers 5.3.0
- GPTQModel 5.8.0 (dev), TritonV2 kernel backend
- GPUs: 3× RTX 3090 (24 GB) + 2× RTX 3060 (12 GB), Driver 570.211.01

### Quantization Summary

| Metric | Value |
|--------|-------|
| FP16 model size | 7.83 GB |
| GPTQ 4-bit model size | 2.92 GB |
| Size reduction | 62.71% (4.91 GB saved) |
| Effective BPW | 4.29 bpw |
| Calibration samples | 128 (WikiText-2 train, min 512 chars) |
| Quantization time | 181.4s (1× RTX 3090) |

### Perplexity (WikiText-2, test split)

| Configuration | Perplexity |
|---------------|------------|
| GPTQ 4-bit g128 | **8.6759** |

Sliding-window evaluation: max_length=2048, stride=512, 578 windows, 297,053 tokens total.

### Baseline Generation Speed Benchmark (GPTQModel / Transformers runtime)

Greedy decoding (do_sample=False), 10 prompts averaged per setting.
This first-pass benchmark is preserved for comparison, but it is **not** the final answer to the
corrected request because it used the GPTQModel / Transformers runtime rather than vLLM.

| GPU Config | max_new_tokens | Tokens/sec |
|------------|---------------|------------|
| **1× RTX 3090** | 128 | **24.16** |
| 1× RTX 3090 | 256 | 24.31 |
| 1× RTX 3090 | 512 | 24.53 |
| **2× RTX 3090** | 128 | **17.62** |
| 2× RTX 3090 | 256 | 17.73 |
| 2× RTX 3090 | 512 | 17.67 |

### Baseline Analysis

- **1× RTX 3090** delivers ~24.3 tok/s sustained across all sequence lengths, consistent
  with the model fitting entirely in 24 GB VRAM (2.92 GB quantized).
- **2× RTX 3090** (device_map="auto") is **~27% slower** than 1× due to pipeline-parallel
  cross-GPU communication overhead. Since the model fits in a single GPU's memory,
  splitting across 2 GPUs adds inter-GPU transfer latency without a memory benefit.
  Multi-GPU shines for models that exceed single-GPU capacity.
- Throughput is stable across 128→512 token generation lengths, indicating the kernel
  is compute-bound rather than launch-overhead-bound at these sequence lengths.
- The TritonV2 kernel backend was auto-selected for inference.

### Baseline Notes
- Linear attention layers (conv1d, in_proj_a/b) were intentionally left unquantized
  as they use different compute patterns from standard attention projections.
- The `flash-linear-attention` fast path was unavailable; torch fallback was used for
  the linear attention layers. Installing `fla` + `causal-conv1d` would likely improve
  throughput further.

### Follow-up: Original PPL, GPTQ-Pro, and vLLM

The corrected follow-up run compared against the original BF16 model, quantized a fresh
`QuantizeConfig.gptq_pro()` checkpoint, and benchmarked that checkpoint with vLLM rather than the
Transformers runtime.

#### Original vs Quantized Perplexity (WikiText-2, test split)

All three numbers below use the same sliding-window setup: `max_length=2048`, `stride=512`,
578 windows, and 297,053 tokens total.

| Configuration | Perplexity | Delta vs original |
|---------------|------------|-------------------|
| **Original BF16** | **8.3116** | baseline |
| GPTQ 4-bit g128 | 8.6759 | +0.3643 |
| **GPTQ-Pro 4-bit g128** | **8.6314** | **+0.3198** |

GPTQ-Pro recovered `0.0445` PPL versus the earlier plain GPTQ run under the same evaluation setup.

#### GPTQ-Pro Quantization Summary

| Metric | Value |
|--------|-------|
| Model load time | 4.9s |
| Quantization time | 324.9s |
| Calibration samples | 128 |
| Output format | GPTQ-compatible (`format=gptq`, `checkpoint_format=gptq`) |
| Key quality knobs | `act_group_aware=true`, `mse=2.0`, adaptive damping, `SmoothAuto` failsafe |

Saved GPTQ-Pro metadata confirms GAR, MSE search, adaptive damping, and failsafe smoothing while
still producing a GPTQ-compatible checkpoint that vLLM can consume through Marlin.

#### vLLM GPTQ-Pro Benchmark

Stock vLLM 0.17.0 and the tested nightly build still do not load this `qwen3_5_text` checkpoint
cleanly out of the box in this environment. The benchmark therefore used a temporary runtime patch
that:

- wraps the Hugging Face `qwen3_5_text` config in vLLM's `Qwen3_5Config`
- forces `language_model_only=True`
- skips multimodal / vision initialization
- remaps checkpoint weights from `model.*` to `language_model.model.*`

Once patched, vLLM selected `gptq_marlin` automatically and reported
`Using MarlinLinearKernel for GPTQMarlinLinearMethod`.

| GPU Config | TP Size | max_new_tokens | Tokens/sec | Engine init |
|------------|---------|----------------|------------|-------------|
| **1× RTX 3090** | 1 | 128 | **175.21** | 37.03s |
| 1× RTX 3090 | 1 | 256 | 178.14 | 37.03s |
| **2× RTX 3090** | 2 | 128 | **194.20** | 56.53s |
| 2× RTX 3090 | 2 | 256 | 206.53 | 56.53s |

#### Why the Earlier Speed Was Slow

- The original speed run above was measured with the GPTQModel / Transformers inference path,
  not vLLM, so it never exercised vLLM scheduling or Marlin's end-to-end runtime path.
- Qwen3.5 linear-attention layers were on the torch fallback path because the
  `flash-linear-attention` / `fla` and `causal-conv1d` fast-path dependencies were unavailable.
- The plain GPTQ checkpoint already fit comfortably on one 24 GB RTX 3090, so the earlier
  `device_map="auto"` two-GPU split only added inter-GPU communication overhead.
- After switching to the GPTQ-Pro checkpoint and the vLLM `gptq_marlin` path, throughput improved
  to roughly `7.3×` the earlier 1×-GPU Transformers result (`178.14 / 24.53`) and `11.7×` the
  earlier 2×-GPU Transformers result (`206.53 / 17.67`).

#### Follow-up Notes

- Two-GPU tensor parallelism helped only modestly here: `+10.8%` at 128 tokens and `+15.9%`
  at 256 tokens, which is expected for a 4-bit 4B model that already fits on one card.
- The patched vLLM path still emits noisy shutdown warnings (`destroy_process_group()` /
  `Engine core proc ... died unexpectedly`) after writing the benchmark JSON, so the text-only
  Qwen3.5 integration should still be treated as upstream-incomplete.
- vLLM also logged an unrelated plugin load error (`ModuleNotFoundError: No module named 'reap'`)
  and FLA shape warnings during warmup/inference. These did not prevent successful runs, but they
  are worth cleaning up before treating this path as production-ready.

### Follow-up: 27B replacement for the GGUF-only HauhauCS request

The requested repository
`HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive` turned out to be GGUF-only, with the
model card explicitly saying `GPTQ — coming soon`. Because GPTQModel requires a Transformers /
Safetensors source checkpoint for GPTQ-Pro quantization, the replacement run used the
user-approved Transformers checkpoint `huihui-ai/Huihui-Qwen3.5-27B-abliterated`.

#### 27B original vs GPTQ-Pro perplexity

Shared-machine VRAM pressure from another long-lived process on the second RTX 3090 repeatedly
OOMed the original-model evaluation path, even after reducing context length. The stable fallback
was a fixed regression slice on one clean RTX 3090 plus CPU offload:

- dataset: WikiText-2 raw test
- `max_length=256`
- `stride=256`
- `max_windows=16`
- `4096` scored tokens total

| Configuration | Perplexity | Delta vs original |
|---------------|------------|-------------------|
| **Original BF16** | **11.6266** | baseline |
| **GPTQ-Pro 4-bit g32** | **12.0161** | **+0.3895** |

These absolute numbers should not be compared directly against the earlier 4B full-dataset sweep:
they use much shorter context windows and fewer total tokens because the shared environment could
not sustain the full-length BF16 evaluation for this larger model.

#### 27B GPTQ-Pro quantization summary

| Metric | Value |
|--------|-------|
| Model | `huihui-ai/Huihui-Qwen3.5-27B-abliterated` |
| Load time | `5.3s` |
| Quantization time | `2273.6s` |
| Save time | `21.8s` |
| Calibration samples | `128` |
| Output size | ~`18G` |
| Output shards | `5` safetensors files |
| Key quality / stability knobs | `group_size=32`, `balanced` VRAM strategy, `gc_mode=on_stage_end`, `auto_forward_data_parallel=false`, `wait_for_submodule_finalizers=true`, `ExpertsRoutingBypass(batch_size=2)`, disk offload |

The offload scratch directory peaked in the mid-teens of gigabytes and showed active per-module
staging throughout the run, which confirmed that the MoE-aware serial / offload path was doing the
heavy lifting rather than silently hanging.

#### 27B vLLM smoke result

The installed `vLLM 0.17.0` still does not cleanly deploy this Qwen 3.5 text-family checkpoint in
the tested environment. A one-shot offline `LLM.generate()` smoke test on the new quantized GPTQ-Pro
checkpoint did select `gptq_marlin`, but then failed before generation with:

- `TypeError: Invalid type of HuggingFace config`
- expected `Qwen3_5Config`
- found `Qwen3_5TextConfig`

For this replacement model, the mismatch surfaced through vLLM's multimodal renderer path rather
than the earlier text-only load path, but the root integration problem is the same: upstream vLLM
still does not fully normalize the Hugging Face Qwen 3.5 config family in this environment.
