# GPTQ-Pro ZONOS3-Mini INT4/BF16 overlay — validation, 15 November 2026

All values below were measured on local Ubuntu RTX 3090 machines or calculated from the saved and SHA-checked output artifacts. See the machine-readable [validation summary](validation_summary.json) and [ASR transcripts](asr_extended.json).

## Provenance and method
- Original checkpoint SHA-256: `3b4ac0a6e3002bd575ad9a0761ae7524806a908f553042171dead8f3a2146caa`.
- Original config SHA-256: `28dc488d7142831d739f427e048c699b7ed12852d88da0a9123a137f15aa5a5b`.
- Original ZONOS3 source Git SHA `4c66ceac5a5004dafd852f42674a6c38f657c7cd`.
- 192 generated English/Spanish calibration sequences using 10 different publicly available upstream voice embeddings and optional public reference-code prefixes. Two held-out suites of 24 distinct prompts (16/32 generated token budgets). Calibration data and voice embeddings are not redistributed.
- GPTQ-Pro `QuantizeConfig.quality_4bit()`, symmetric INT4, group size 64, `desc_act=False`, no fake activation replay. Under-calibrated or architecturally sensitive projections were retained BF16.
- Exact packed overlay: 11/11 layer shards, 1005 Q4 projections, 312 preserved BF16 projections, 323,729,736 packed-shard bytes. The source BF16 checkpoint and unchanged ZAC codec are still required.
- Native CUDA packed kernel parity NMSE never exceeded `3.33262505591847e-05` for the quantized projections. Largest fitted per-projection NMSE `0.0027962857857346535`.
- 11/11 local integration tests passed in the final pre-release development session.

## Held-out teacher vs quantized logits

| Set | Logit NMSE | KL divergence | Argmax agreement |
|---|---:|---:|---:|
| Standard 24 | 0.004083 | 0.018779 | 91.04% |
| Longer independent 24 | 0.003794 | 0.024353 | 87.57% |

The two distinct held-out suites passed the designated NMSE 0.10 and KL 0.10 numerical gates. Note that BF16 versus Q4 sampling may differ even with the same seed; token argmax disagreement is expected for some frames.

## Runtime and VRAM

Two RTX 3090 GPUs and separate processes generated ten matched pairs (five EN, five ES). Each test generated 90 codebook rows yielding ~5.6 seconds of audio using the same Zyphra ZAC decoder. The first sample included decoder initialization and is excluded from the warm timings.

| Metric | BF16 source | Experts-only GPTQ-Pro |
| --- | ---: | ---: |
| LM-only allocated CUDA tensor memory | 1.650640 GiB | 0.858202 GiB |
| Full TTS peak CUDA tensor memory | 2.272368 GiB | 1.479429 GiB |
| Average warm utterance latency (n=9) | 0.856241 seconds | 0.997513 seconds |
| Median warm utterance latency | 0.857686 seconds | 0.990513 seconds |

**Outcome:** ~48% less LM weight memory, ~35% lower peak full-TTS CUDA allocation, ~16.5% **slower** warm TTS with the custom PyTorch reference runtime. Figures are not throughput measurements for fused-serving backends.

## Speech intelligibility proxy

Faster-Whisper small (CPU INT8), ten paired 5.6-second clips, original prompt vs approximate transcribed-prefix edit distance. This is **not** a complete standard WER study: original prompts can be longer than the 90-token generation budgets, and Whisper may omit ending words. The short-clip proxy yielded:

| Language | BF16 prefix-match error | INT4 prefix-match error |
| --- | ---: | ---: |
| English (n=5) | 0.0000 | 0.0167 |
| Spanish (n=5) | 0.0143 | 0.0558 |

**Interpretation:** INT4 retains generally intelligible speech in these clips, but the ASR proxy sees some degradation, particularly Spanish. This is not evidence of parity in naturalness, intonation, voice identity or all languages. A blind listening and longer-audio evaluation is still required before production use.

## Gates not yet satisfied

- No subjective MOS/ABX testing or speaker-similarity comparison.
- No SGLang/vLLM/MLX/GGUF deployment integration.
- No broad quality certification across all languages, speakers, text markup and long-form TTS.
- Original BF16 weights are still required on disk.

No GitHub Actions or cloud CI were used for these tests.
