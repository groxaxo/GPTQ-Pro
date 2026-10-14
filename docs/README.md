# GPTQ-Pro documentation

Start with [support and qualification status](STATUS.md), then choose the guide
for the exact model and runtime. **An implemented architecture, a saved artifact,
a successful load, and a deployment-qualified model are different milestones.**

This index owns navigation. [STATUS.md](STATUS.md) owns the consolidated status
summary; the linked source and dated evidence establish each claim.

## Start here

| Goal | Canonical document |
| --- | --- |
| Install the package and inspect a basic example | [Repository quick start](../README.md#quick-start) |
| Check runtime limits, model support and known blockers | [Support and qualification status](STATUS.md) |
| Choose the default quantization recipe | [Validated quality default](VALIDATED_QUALITY_DEFAULT.md) |
| Run local checks and understand release gates | [Validation guide](VALIDATION.md) |
| Understand native CUDA execution | [Ampere V3 kernel](KERNEL_V3.md) |

## Model and campaign guides

| Model or workflow | Guide | Boundary |
| --- | --- | --- |
| Qwen3.5 / Qwen3.6 dense, text and MoE routing | [Architecture family](QWEN35_QWEN36.md) | Model-definition support does not certify every artifact or consumer. |
| Qwen3.8 dense 27B | [Qwen3.8 guide](QWEN38.md) | Prefer the full sequential driver for quality; chunk assembly is a separate research path. |
| Qwen3.6 Heretic selective g64/g32/BF16 recipe | [Campaign safety gates](qwen36-campaign-safety.md) | Automatic consumer admission remains blocked. |
| Recover an interrupted Heretic export or inspect native loading | [Native qualification and recovery](qwen36-native-qualification.md) | Native reload evidence is not fused-vLLM, production TP2 or workload qualification. |

The Heretic recovery path preserves the original completed sequential result.
It is **not** the independent layer-chunk assembly described in the Qwen3.8 guide.

## Numerical and kernel references

| Topic | Reference |
| --- | --- |
| Current recipe defaults and bounded quality evidence | [Validated quality default](VALIDATED_QUALITY_DEFAULT.md) |
| Strict checks, OOM handling and effective solver features | [Numerical integrity](QUALITY_NUMERICS.md) |
| Packing, storage round trips and reference error metrics | [Storage and numerical contract](STORAGE_NUMERICAL_CONTRACT.md) |
| Trusted prebuilt binary selection and GPU probes | [Pinned V3 validation](PINNED_V3_VALIDATION.md) |
| V3 dispatch, shapes and arithmetic | [Kernel V3](KERNEL_V3.md) |
| Register-cached V4 experiment and its negative result | [Kernel V4 experiment](KERNEL_V4_EXPERIMENT.md) |

## Dated evidence

Evidence files are snapshots of specific code, inputs and environments, not live
job status. Do not add overlapping test counts or transfer a result between models.

| Evidence | Scope |
| --- | --- |
| [Quality default, 7 October 2026](verification/validated-quality-default-20261007.json) | Controlled Qwen3.5-9B quality experiment and separately scoped MTP smoke; 27B timing remains an estimate. |
| [Storage contract, 7 October 2026](verification/storage-contract-20261007.json) | Earlier CPU/storage stage, not a GPU qualification report. |
| [Pinned V3, 7 October 2026](verification/pinned-v3-20261007.json) | Specific existing binary, synthetic shapes and two real source linears. |
| [Campaign CPU regressions, 14 October 2026](verification/qwen36-campaign-cpu-20261014.json) | Software safety tests, not generated model answers. |
| [Native Heretic recovery, 14 October 2026](verification/qwen36-native-qualification-20261014.json) | Recovered artifact, fixture/native reload results, functional failure and unresolved release gates. |
| [Audit index](audits/README.md) and [14 October audit](audits/2026-10-14/README.md) | Historical diagnosis of the exact audited campaign; later implementation does not rewrite its observations. |

## Historical plans and external-runtime material

These files retain their original paths so existing links remain useful. They are
not the current installation guide or an alternative source of default settings.

| Document | How to read it |
| --- | --- |
| [Assessment and roadmap](ASSESSMENT_AND_ROADMAP.md) | July 2026 implementation snapshot and research roadmap, predating newer defaults and native BF16 boundaries. |
| [Quality implementation plan](QUALITY_IMPLEMENTATION_PLAN.md) | Original 7 October implementation contract. |
| [Numerical-integrity plan](QUALITY_NEXT_STEP_PLAN.md) | Pre-implementation plan; use the numerical guide and evidence for outcomes. |
| [Storage-contract plan](STORAGE_NUMERICAL_CONTRACT_PLAN.md) | Historical diagnostic-tool design. |
| [Pinned-binary plan](PINNED_V3_VALIDATION_PLAN.md) | Historical plan preceding the pinned-binary implementation/evidence. |
| [Quantization protocol](quantization_protocol.md) | Design proposal with limited implemented surface; illustrative interfaces are not general runtime support. |
| [Qwen3.5 vLLM launch](qwen35_vllm_launch.md) | Historical external-vLLM/Marlin procedure for particular dense text artifacts. |
| [Qwen3.5 vLLM comparison](qwen35_vllm_comparison.md) | Version-specific historical measurements, not native GPTQ-Pro kernel speed claims. |
| [Torch fused INT4 transformations](torch_fused_int4_transformations.md) | Upstream/removed CPU/XPU backend background, not an executable path in this fork. |
| [EoRA development notes](eora/README.md) | Auxiliary research material, subject to current runtime and adapter limits. |

## Documentation maintenance

Keep commands and detailed procedures in the owning guide; link to them instead
of maintaining a second recipe in the README. Update STATUS.md when checked
implementation or qualification changes, with the precise evidence and date.
Configuration source owns defaults; evidence owns measured results. An old plan
or a benchmark result for another checkpoint cannot override either.

Preserve dated audit/evidence files. Record new runs separately, retaining failed,
blocked, skipped and unrun classifications. Use portable placeholder paths in new
examples. Check relative links, section anchors, fenced examples and local tests
before pushing documentation changes. Never fix a prose inconsistency by quietly
changing model metadata, qualification pins or runtime behavior.
