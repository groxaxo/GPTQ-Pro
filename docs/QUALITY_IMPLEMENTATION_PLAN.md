# GPTQ-Pro quality implementation plan — sm_86

Status: implementation contract, not a completed quantization or quality claim.
Inspection: 7 October 2026 (12700 host clock), base `cdbe16d`.
Original: `/home/op/src/GPTQ-Pro-int8-20261002` (dirty; preserve unchanged).
Branch: `fix/quality-foundation-20261007`.
Worktree: `/home/op/src/GPTQ-Pro-quality-20261007`.
Model tests: **Qwen/Qwen3.5-4B**, local original-precision checkpoint only.

## 0. Safety and scope

No reset, stash, deletion, dependency upgrades, serving changes, power-limit changes,
or interruptions to other jobs. Submit all GPU tests through the existing `gpuq`;
never rewrite its assigned CUDA_VISIBLE_DEVICES. Hide GPUs for CPU tests. No GitHub
Actions. Initial inspection: about 14 GB disk free, 53 GiB RAM available; GPU0
reserved, GPU1/GPU2 free. Recheck admission; these are not future guarantees.
No large model download or checkpoint export in this first patch. A missing official
4B source blocks model tests; do not silently substitute a 27B or abliterated model.
Derive 4B shapes, layer types, skips and MTP presence from its actual metadata.

Verified source bugs before edits:
1. gptq_pro() overwrites explicit fallback because it tests failsafe is None.
2. Explicit gptaq=None is treated as omitted when alpha is provided.
3. FOEM-only settings are dropped by feedback metadata serialization.
4. CUDA-to-CPU fallbacks exist in accumulation, two permutations and inversion.
5. tests/test_hessian.py gates even CPU tests on CUDA device 6; new CPU regression
   tests must live outside that benchmark file.
6. Per-device Hessian partials already exist; reuse before designing a replacement.

## 1. Numerical contract before expensive recipe search

V3 stores INT4 and dequantizes into floating-point MMA, not native W4A4 IMMA.
Build explicitly for sm_86; do not import A100 FP64 Tensor Core/resource assumptions.
Keep Hessian tensors/factorization FP32. Disable TF32 for eligible FP32 GEMMs;
allow_tf32 alone does not certify factorization or every mixed-precision reduction.
Record relevant reduced-reduction settings; restore settings after tests. Do not
mix incompatible legacy/new PyTorch precision APIs. Calibration forwards remain
source BF16 where supported, not a whole-model FP32 upcast.

FP16 has more significand bits but less range than BF16. BF16->FP16 can overflow to
infinity, not merely saturate. Measure range/dtype error before prioritizing BF16 MMA.

```python
# Design pseudocode: use the actual pack/unpack orientation and dtype rules.
r0 = source_dtype_dense(x, w_source)
r1 = source_dtype_dense(x, unpack(stored_quantized_weights))
r2 = emulate_v3_fp16_contract(x, unpack(stored_quantized_weights))
r3 = actual_v3_kernel(x, stored_quantized_weights)
quant_error = metric(r1, r0)
dtype_error = metric(r2, r1)
kernel_error = metric(r3, r2)
# Pairwise comparisons use identical x. Error norms and KL are NOT additive.
```

Also compare candidate FP32 scales, rounded stored scales, quantized codes, packed
and unpacked tensors. Integer packing is exact; fused floating reductions need
predeclared dtype/shape-aware tolerances, not torch.matmul bitwise equality.
Check finite outputs before absolute/relative/RMS/cosine metrics. Cosine alone
can hide bias. Use all real K/N/group shapes and M=1,2,3,4,5,7,8,15,16,17,64,
plus valid tail/fallback shapes. Add zeros, sparse/alternating signs, small scales,
large representable activations and overflow probes. Unsupported cases must reject.
Correctness and speed are separate gates. Keep V4 retired absent new evidence.

## 2. Initial implementation: recipe integrity and strict OOM

Preserve omitted defaults, but honor explicit overrides and nulls:

```python
if "fallback" in kwargs:
    fallback = kwargs.pop("fallback")  # None explicitly disables fallback
elif failsafe is not None:
    fallback = failsafe                # legacy alias
else:
    fallback = Fallback(...)           # existing preset default

if "gptaq" in kwargs:
    gptaq = kwargs.pop("gptaq")         # None disables preset feedback
else:
    gptaq = GPTAQConfig(alpha=gptaq_alpha, device=gptaq_device) \
        if gptaq_alpha is not None else None
```

Serialize GPTAQ and FOEM independently, including nulls clearing stale metadata.
Preserve both supplied configs; do not change solver selection implicitly. Test
JSON and disk round-trips for FOEM-only/GPTAQ-only/both/neither, legacy aliases,
explicit nulls and custom fallback objects.

Add HessianConfig.cuda_oom_policy: "cpu" legacy default; "error" quality mode.
Validate and serialize it. Centralize handling before any CPU transfer; include
module, stage, source device and policy. Record permitted fallback events.
Cover accumulation, activation-order permutation, group-aware permutation and
inversion. Do not swallow non-OOM RuntimeErrors. Intentional CPU runs remain valid:
strict mode rejects emergency CUDA migration, not every CPU tensor.

```python
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.config import HessianConfig, FOEMConfig
qcfg = QuantizeConfig.quality_4bit(
    group_size=64,
    hessian=HessianConfig(staging_dtype="float32", cuda_oom_policy="error"),
    fallback=None, gptaq=None, foem=FOEMConfig(alpha=0.0, beta=0.2),
)
restored = QuantizeConfig.from_quant_config(qcfg.to_dict())
assert restored.foem.beta == 0.2
assert restored.fallback is None
assert restored.hessian.cuda_oom_policy == "error"
```

RTN/SmoothMSE is not universally worse than an unstable solve. Keep legacy behavior;
strict experiments may disable it. Log reason, valid observations and reconstruction
evidence. CPU fallback is an admission/reproducibility failure, not proof of
intrinsically inaccurate CPU arithmetic.

## 3. Frozen provenance, data and architecture manifests

```json
{
  "schema_version": 1,
  "source": {"path": "local source", "revision": "pinned", "sha256": {}},
  "software": {"git_commit": "...", "dirty_diff_sha256": "..."},
  "tokenizer": {"revision": "...", "chat_template_sha256": "..."},
  "calibration": {"token_budget": 262144, "tokens_by_domain": {}, "sha256": "..."},
  "validation": {"sha256": "..."}, "test": {"sha256": "..."},
  "architecture": {"layer_types": [], "module_shapes": {}, "mtp_present": false},
  "recipe": {}, "runtime": {"backend": "...", "dtype": "...", "gpu_uuid": "..."}
}
```

Use calibration/tuning-validation/untouched-final-test splits. Deduplicate by source
conversation/document, not only prompt equality. Freeze tokenizer/chat template,
rendering, seed, order, truncation and padding. Count nonpadding activation tokens
at each module, domain token shares, positions, lengths and recurrent-state resets.
Preserve assistant/tool roles; never wrap an entire conversation as one user string.
No benchmark answers in calibration. Text tests do not certify vision quality.

Compare nested 64k/128k/256k/512k token budgets with enough independent documents.
Keep domain/length composition matched. Long-context ablation is separate: one 32k
sequence is not equivalent to sixteen 2k sequences even with the same token count.

## 4. Controlled uniform search — Qwen3.5-4B first

Historical calib128 artifacts are regression references, not causal baselines.
Fresh baseline, identical source/data/environment. Begin with this small matrix;
do not Cartesian-product every knob:

| Candidate | Group | GPTAQ | FOEM | Data |
|---|---:|---:|---:|---|
| A0 | 64 | off | off | frozen shared token budget |
| A1 | 64 | alpha .25 | off | identical |
| A2 | 32 | off | off | identical |
| A3 | 32 | alpha .25 | off | identical |
| A4 | 64 | off | alpha 0, beta .2 | identical |

Then screen damping .01/.025/.05/.075/.10; GPTAQ alpha 0/.125/.25/.5;
GAR on/off; MSE on/off; activation weighting on/off; optional g128; FOEM beta
separately. Alpha zero is not presumed identical to bypassing its processor.
Record effective damping, retries, dead directions, conditioning, coverage, local
loss, dtype, CPU-fallback events, memory and timings. These are hypotheses, not
preset names proving optimal quality.

Use early/middle/late windows covering attention and GDN, with states captured
from the correct quantized prefix. Independent chunk assembly changes calibration;
do not claim it inevitably loses, or label it true-sequential. Windows reject ideas,
not certify checkpoints. Advance only a few finalists to full sequential solves.

Measure token-weighted NLL/PPL and full-vocabulary KL(p_BF16 || p_candidate), stable
FP32 log_softmax and vocab chunking where necessary. Top-k proxy logprobs are not
exact KL. Use coding, tool/JSON and multilingual slices with fixed templates,
decoding and reason-token budgets. Bootstrap matched documents/conversations, not
correlated tokens. Repeat finalists across calibration seeds. 4B validates tooling;
a later target-model test is necessary before claiming a 27B quality gain.

## 5. Ampere utilization without changing the objective

Profile forward, XTX, clipping and solve separately. Report qualifying GEMM FLOP/s,
bandwidth, launch distribution, copies, host waits and peak allocations; no
whole-layer near-roofline pass rule.

Prefer independent sequential candidate jobs on GPU1/GPU2, subject to RAM/I/O
admission. Pinned-host prefetch is not a dedicated GPU job. Before parallelizing
q/k/v or gate/up, inspect actual stage DAG, shared hooks, feedback inputs and state
ownership; sibling names do not prove independence. Reuse existing per-device
partials and normalize once by total valid observations. Reduction order changes
FP32 rounding. Measure >1 GiB Hessian transfer cost before distributed accumulation.

Vectorize MSE in bounded candidate chunks, not an unbounded candidate x row x column
allocation. Preserve objective, exponent, weighting, zero convention, candidate
order and tie-breaking. Test scale/code parity for zeros/outliers/ties and measure
memory/speed before changing defaults. Tensor parity is required for pure solver
refactors, not just unchanged noisy task scores.

Prefer RAM/pinned-host staging over disk when memory permits; disk offload is not
a quality invariant. Respect other host jobs and GPU queue admission. Do not occupy
all cards merely to show high utilization.

## 6. Optional rotation requires algebraic equivalence first

Do not enable llama/qwen2 rotation on qwen3_5. A whitelist of linears is not proof.
Derive basis changes across residual paths, RMSNorm, RoPE/head structure, MLP
gates, GDN convolution/norm/state and embedding/head/MTP/vision interfaces.
Arbitrary Hadamards cannot commute through SiLU/gating. Internal MLP transforms may
need online work; promise offline fusion only after accounting for every transform.

First test unquantized FP32 local algebra and BF16 full-model logits, prefill,
cached decode and long GDN state trajectories. Finite outputs or improved quantized
KL do not prove equivalence. Reject unexplained functional drift. A winning valid
rotation requires recalibration and sensitivity remeasurement before bit allocation.
Learned rotations, 5-bit, true-desc_act and asymmetric oracles remain bounded
research branches, not unsupported production formats.

## 7. Sensitivity, EoRA and format gates before new kernels

Shortlist using Hessian/output distortion; measure paired downstream KL/NLL recovery
from g32/g16/INT8/BF16 and sparse EoRA r8/r16/r32. avg_loss/weight energy alone is
not global sensitivity. Rank gain per byte/latency and remeasure interactions.
Single-module gains are not additive. Small early EoRA experiments can avoid
unnecessary INT8 kernel work; final adapters follow frozen rotation, bits and dtype.
Small-M adapters may be launch-bound: measure rather than assume bandwidth-bound.

Before a mixed full model, save/reload a tiny graph with INT4 g64, INT4 g32, INT8
and skipped BF16 linears. Verify exact packed data, per-module config, qlinear
classes, dtype, g_idx, bias and regex precedence. Test native and each claimed
external loader. GPTQ_PRO is currently INT4-only; uncommitted INT8 reference is not
a fused kernel and must not enter AUTO. Preserve it in the original dirty checkout.

Uniform GPTQ/GPTQ_V2 does not guarantee universal backend compatibility. Mixed-bit
or mixed-group outputs are local-only until tested elsewhere. Reference INT8 tests
quality, not speed. Integer pack/unpack is bit-exact; floating MMA parity uses
numerical tolerance, not bit-identical torch.matmul.

## 8. Memory-constrained allocation and production kernels

Account from actual stored tensors: qweight, ceil-rounded scales/zeros, g_idx,
unquantized tensors, padding and adapters. Example: 25 billion INT8 bytes are
25 GB / 23.28 GiB before everything else. Approximate INT4 FP16-scale + stored
4-bit-zero metadata change g64->g32 is P*(2+.5)*(1/32-1/64) bytes, plus rounding.

Select a Pareto set under measured peak memory, required context/concurrency,
KV/recurrent state, MTP, graphs/workspaces and throughput. Allocated KV can make
resident memory misleading. TP2 is not one flat 48 GB pool: include replicated
state and per-rank peaks. Record actual interconnect/P2P/NCCL without changing it.
4B MTP tests are explicitly not applicable when source has no MTP; never transplant
an MTP head to manufacture coverage.

Build fused INT8/BF16 only when reference quality justifies it. BF16 work moves
earlier if measured dtype error warrants it. Prove CUDA/stream/graph and numerical
correctness separately from performance. Fit final EoRA against the exact runtime.
Freeze throughput budgets (such as <=8% decode loss) before observing candidates.

## 9. Gate order and deliverables

1. Isolated worktree, plan and protected-dirty snapshot.
2. Reproduce/fix overrides and metadata; strict OOM; CPU regression tests.
3. Source/data/shape manifests and dtype/pack/kernel ladder.
4. Controlled Qwen3.5-4B search and full-model finalists.
5. Math-preserving solver performance work.
6. Optional function-preserving rotation, recalibration.
7. Precision/EoRA oracles and heterogeneous loader gate.
8. Allocation, final correction, justified serving kernels.
9. Untouched final-test and actual deployment-profile validation.

Correct misleading protocol/backend docs now. Implement multi-rule lowering only
when supported dynamic semantics/loader coverage exist. Do not gate reference-only
recipe research on kernel speedup; real serving claims do require kernel correctness.
Report passed, failed, blocked and not-run independently.

## 10. Commands and executable entrypoints

```bash
cd /home/op/src/GPTQ-Pro-quality-20261007
PY=/home/op/quant-jobs/qwen38-gptqpro-single24-hq-20261005/.venv/bin/python
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 "$PY" -m pytest -q \
  tests/qcfg/test_gptq_pro.py tests/test_quality_config_integrity.py \
  tests/test_hessian_oom_policy.py

# Only when implemented and validated; no physical GPU override inside the job.
/home/op/.local/bin/gpuq submit --name gptq-pro-quality-qwen35-4b \
  --cwd "$PWD" --count 1 --candidates 1,2 -- \
  "$PY" scripts/quality_qwen35_4b_probe.py \
  --source "$LOCAL_QWEN35_4B_SOURCE" --output "$REPORT_DIR/probe.json"
```

The initial phase authorizes no full 27B quantization, production checkpoint
replacement or model download. Snippets labelled pseudocode are design examples,
not claims of implemented APIs. Track implemented entrypoints in the work report.

## Primary references

- https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
- https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html
- https://huggingface.co/Qwen/Qwen3.5-4B
- Local source at cdbe16d is authoritative for fork behavior; upstream capability
  claims do not prove support in this checkout.
