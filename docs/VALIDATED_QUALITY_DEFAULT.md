# Validated GPTQ-Pro quality default

As of 7 October 2026, QuantizeConfig.quality_4bit() is the default recommended quality path.

Resolved defaults:

- bits: 4
- group size: 64
- symmetric: true
- desc_act: false
- GAR / act_group_aware: true
- MSE clipping exponent: 2.0
- activation-weighted MSE: true
- damping: 0.05, auto increment 0.01
- strict numerics: true
- Hessian staging: FP32
- CUDA OOM policy: error / fail closed
- fallback: none
- GPTAQ: none
- FOEM: none

The lower-level gptq_pro() constructor resolves the same quality defaults.
legacy_quality_4bit() preserves the previous g128 + 0.5% RTN/SmoothMSE fallback + non-strict CPU-OOM-recovery profile for historical reproducibility.

## Why this is the default

A controlled full-decoder QwenPaw/Qwen3.5-9B experiment quantized all 32 decoder layers / 200 selected linears twice at identical symmetric INT4 g64 precision and packed decoder size. The only recipe differences were GAR, MSE clipping and activation weighting.

| Metric | Plain GPTQ g64 | Validated quality g64 |
|---|---:|---:|
| KL(source || quant), nats/token | 0.04767384 | 0.03317633 |
| NLL, nats/token | 0.86555421 | 0.86474651 |
| Perplexity | 2.37632271 | 2.37440411 |
| Source top-1 agreement | 94.6708% | 95.0893% |

The KL divergence reduction was 30.41%. Its paired-document bootstrap 95% interval for baseline-minus-quality was [0.00588705, 0.02662250], above zero. The NLL interval included zero; do not describe this as a proven task-accuracy or perplexity improvement.

The final MTP-preserving exported candidate also passed a full saved-candidate reload and RTX 3090 draft-MTP smoke. These runtime checks validate preservation, not the quantization-quality delta itself.

## Qwen3.8-27B expected duration on one RTX 3090

The same 9B quality run recorded about 932 seconds of quantization-stage wall time for 32 layers / 200 linears and 16,272 calibration tokens. Qwen3.8-27B has 64 layers / 400 selected linears, hidden size 5120 and FFN 17408.

A shape-aware projection of solver work plus calibration-forward/load/save overhead puts the same ~16k-token g64 quality recipe at approximately:

- pure solver/calibration work: 45-60 minutes
- end-to-end checkpoint build on one RTX 3090: 55-75 minutes
- conservative operational budget with shared-host/offload variance: up to about 90 minutes

This estimate does not apply to the earlier 512-sample/~658k-token GPTAQ pilot; that workload incurred very large activation spilling and is a multi-hour class job. The validated default intentionally does not use GPTAQ.

For a reliable 27B production build, use the full sequential driver rather than independently assembled layer chunks when quality is the priority.
