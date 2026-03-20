# PR Quality Review

*Exported from chatGPT on 19/03/2026, 11:01:46 pm*

---

### User:
WHAT DO YOU THINK ABOUT THIS PR ? https://github.com/groxaxo/GPTQ-Pro/pull/4

---

### ChatGPT:
I’m pulling the PR diff and checking what it changes, plus whether it looks correct, maintainable, and likely to help in practice.

---

I’ve got the headline change already: it keeps standard GPTQ export/runtime compatibility, but replaces a single fallback smoother with a best-of offline search. I’m checking the actual code path and tests now to see whether it’s genuinely solid or just sounds good in the PR description.

---

My take: **the idea is good, and the implementation looks directionally strong**. I would not reject it. I’d call it a **quality-oriented PR with low runtime risk**, because it keeps `quant_method=GPTQ` and `format=GPTQ`, and the new behavior is limited to the offline failsafe path rather than changing inference kernels. The core change is that `QuantizeConfig.gptq_pro()` now defaults to `SmoothAuto()` instead of a single `SmoothMSE`, and `SmoothAuto` tries several kernel-safe candidates—unsmoothed baseline, `SmoothMSE`, `SmoothMAD`, `SmoothPercentile`, and `SmoothPercentileAsymmetric`—then keeps the lowest per-row reconstruction MSE. [GitHub+2GitHub+2](https://github.com/groxaxo/GPTQ-Pro/pull/4)

What I like:

- It is **properly scoped**. This is not pretending to be AWQ or AutoRound-in-the-loop; it is a conservative offline improvement that still emits ordinary GPTQ weights/scales/zeros for the same inference path. That is the right way to squeeze more quality without breaking downstream kernel compatibility. [GitHub+1](https://github.com/groxaxo/GPTQ-Pro/pull/4)
- The **“include_none” baseline** is a smart failsafe. Because the unsmoothed candidate can win, the PR explicitly guards against cases where smoothing makes a row worse. [GitHub+1](https://github.com/groxaxo/GPTQ-Pro/pull/4/files)
- The refactor into `_failsafe_quantize_block()` plus `_failsafe_quantize_auto()` is cleaner than jamming more branches into the old path, and the config surface is wired through parsing/serialization/export, so this is not a hacky one-off. [GitHub+1](https://github.com/groxaxo/GPTQ-Pro/pull/4/commits/3b6de40c6f78a6fd5228f514d369a8b71c516ad3)

What I’d watch:

- The selection criterion is **local row-wise reconstruction MSE**. That is a sensible proxy, but it is still only a proxy. A row having lower MSE does **not automatically guarantee** better perplexity or downstream task quality at the model level; that part still needs empirical validation. This is an inference from the implementation, which explicitly chooses candidates by per-row MSE. [GitHub](https://github.com/groxaxo/GPTQ-Pro/pull/4/commits/3b6de40c6f78a6fd5228f514d369a8b71c516ad3)
- It will **increase offline quantization work** on blocks that enter the failsafe path, because instead of running one smoother/quantization pass, it now evaluates multiple candidates and compares them. Since the preset says this is for low-sample fallback blocks, the blast radius is limited, but quantization-time overhead should still be measured. [GitHub+1](https://github.com/groxaxo/GPTQ-Pro/pull/4)
- The follow-up “polish” commit matters. The first version used direct shape checks for row replacement of `scale`/`zero`; the next commit replaced that with `_row_replace_mask()` and constants for fallback steps/maxshrink. That tells me the author already hit at least one tensor-shape corner case, so I’d want CI plus one real model smoke test before merging. [GitHub+1](https://github.com/groxaxo/GPTQ-Pro/pull/4/commits/169f1d534186c512788837025fbd3ca415ab70d7)

Tests: the PR does add/adjust tests so that the auto smoother must match or beat the candidate smoothers within tolerance, which is exactly the right unit-level invariant for this feature. But from what is visible in the PR, I do **not** see model-level benchmark evidence yet. [GitHub](https://github.com/groxaxo/GPTQ-Pro/pull/4/commits/169f1d534186c512788837025fbd3ca415ab70d7)

So my verdict: **good PR, probably worth merging, but only after adding a small benchmark note**. I’d want:

1. perplexity or eval numbers on at least one awkward model,
2. quantization-time overhead versus prior `gptq_pro`,
3. one smoke test proving the exported artifact still loads cleanly in the intended GPTQ runtime stack.

If you want, I can give you a **proper line-by-line code review comment set** you could paste into the PR.

---

### User:
yes please

---

### ChatGPT:
I’m opening the PR and its diff so I can turn this into concrete review comments you can paste directly.

---

I’ve confirmed the config/docs side. Now I’m trying to extract the `gptq.py` diff itself, since that’s where the real correctness risk lives.

---

Paste this as your review.

Overall review text:

This looks solid to me. I like the scope: it keeps `quant_method=GPTQ` / `format=GPTQ`, moves the preset default from a single `SmoothMSE(steps=32, maxshrink=0.9)` to `SmoothAuto()`, and limits the new search to the offline failsafe path rather than changing the runtime kernel path. The refactor into `_failsafe_quantize_block()` plus `_failsafe_quantize_auto()` is also cleaner than growing the old branch inline. [GitHub+3GitHub+3GitHub+3](https://github.com/groxaxo/GPTQ-Pro/pull/4)

Then add these review comments where relevant:

1. On `gptqmodel/quantization/config.py` near `class SmoothAuto(...)`:

Nice addition. Since `SmoothAuto` is now part of the serialized config surface, I’d add lightweight validation here or in `_build_smooth_method_from_dict()` for obviously invalid values: `mse_steps > 0`, `0 < mse_maxshrink <= 1`, `0 < percentile <= 100`, and `0 <= low < high <= 100`. That would make malformed configs fail fast instead of producing confusing quantization behavior later. [GitHub+1](https://github.com/groxaxo/GPTQ-Pro/pull/4/files)

1. On `QuantizeConfig.gptq_pro()` where the default smoother changes:

This is a meaningful preset behavior change, not just an internal cleanup: `gptq_pro()` now defaults to `SmoothAuto()` instead of `SmoothMSE(steps=32, maxshrink=0.9)`. I’d add a short migration note / changelog entry so anyone relying on prior GPTQ-Pro reproducibility knows why outputs changed after upgrading. [GitHub+1](https://github.com/groxaxo/GPTQ-Pro/pull/4/files)

1. On `gptqmodel/quantization/gptq.py` near `_failsafe_quantize_auto()`:

I like the design here. My main ask is empirical validation in the PR description: the selector chooses the best candidate by per-row reconstruction MSE, which is a sensible local proxy, but it would be great to attach one small end-to-end datapoint as well (for example perplexity/eval delta plus quantization-time overhead on one representative model). That would make the case for the new default much stronger. [GitHub+1](https://github.com/groxaxo/GPTQ-Pro/pull/4/commits/3b6de40c6f78a6fd5228f514d369a8b71c516ad3)

1. On `gptqmodel/quantization/gptq.py` near `AUTO_SMOOTH_FALLBACK_STEPS` / `AUTO_SMOOTH_FALLBACK_MAXSHRINK`:

Good cleanup extracting these constants in the polish commit. One small improvement: add a short comment explaining why non-`SmoothMSE` candidates use `32` / `0.8` while the `SmoothAuto` MSE candidate uses its own `mse_steps` / `mse_maxshrink` defaults (`48` / `0.85`). The code is readable, but the rationale is not obvious at a glance. [GitHub+1](https://github.com/groxaxo/GPTQ-Pro/pull/4/commits/169f1d534186c512788837025fbd3ca415ab70d7)

1. On `gptqmodel/quantization/gptq.py` near `_row_replace_mask(...)`:

Good catch fixing the row-wise replacement mask for cases where `scale` / `zero` can be either `[rows, 1]` or `[rows]`. I’d add a targeted regression test for this exact shape-mismatch scenario across at least two failsafe strategies, because this looks like the kind of subtle bug that can easily come back during future refactors. [GitHub](https://github.com/groxaxo/GPTQ-Pro/pull/4/commits/169f1d534186c512788837025fbd3ca415ab70d7)

1. On `tests/test_failsafe.py`:

The invariant test is good: `auto_err` must be no worse than each single candidate within tolerance. I’d add one more config-focused test for round-trip serialization of all `SmoothAuto` fields (`include_none`, `mse_steps`, `mse_maxshrink`, `mad_k`, `percentile`, `low`, `high`) since this PR expands both parsing and `to_dict()` support for the new smoother. [GitHub+1](https://github.com/groxaxo/GPTQ-Pro/pull/4/commits/169f1d534186c512788837025fbd3ca415ab70d7)

My own merge stance: **approve with comments** if you’re comfortable merging without benchmark numbers; **comment but don’t block** if you want the PR merged fast; **request changes** only if this repo requires model-level eval before changing preset defaults. The code and tests shown in the PR look strong enough that I would not reject it on design grounds. [GitHub+2GitHub+2](https://github.com/groxaxo/GPTQ-Pro/pull/4)

---

### User:
just paste it as a reply to copilot please, @copilot and then the answer that i will have to provide with all this suggestions and clear next steps

---

### ChatGPT:
@copilot This PR looks good overall. I like the scope and the way it stays conservative on compatibility: it keeps `quant_method=GPTQ` / `format=GPTQ`, changes the GPTQ-Pro preset from a single `SmoothMSE(...)` fallback to `SmoothAuto()`, and limits the new logic to the offline failsafe path instead of touching runtime kernel behavior.

A few suggestions before merge:

1. Please add lightweight validation for `SmoothAuto` config values, either in the dataclass or in `_build_smooth_method_from_dict()`, so invalid configs fail fast. In particular:

- `mse_steps > 0`
- `0 < mse_maxshrink <= 1`
- `0 < percentile <= 100`
- `0 <= low < high <= 100`

1. Since `QuantizeConfig.gptq_pro()` now changes default behavior from `SmoothMSE(steps=32, maxshrink=0.9)` to `SmoothAuto()`, please add a short changelog / migration note. This is a real preset behavior change, so users depending on reproducibility should know why outputs may differ after upgrading.
2. `_failsafe_quantize_auto()` is a good design, but the selection criterion is still row-wise reconstruction MSE. That is a sensible local proxy, but it would be great to include one small model-level datapoint in the PR description:

- perplexity or eval delta on one representative model
- quantization-time overhead versus the previous GPTQ-Pro behavior

1. The extracted constants for the fallback path are cleaner, but please add a brief code comment explaining why non-`SmoothMSE` candidates use the fallback `32` / `0.8`, while the `SmoothAuto` MSE candidate uses its own configurable defaults (`48` / `0.85`). The code is readable, but the rationale is not obvious.
2. `_row_replace_mask()` looks like the right fix for row-wise replacement when `scale` / `zero` can have different shapes. Please add a targeted regression test for this specific shape-handling case so it does not regress in later refactors.
3. The current failsafe invariant test is good. Please also add a serialization round-trip test for all `SmoothAuto` fields:

- `include_none`
- `mse_steps`
- `mse_maxshrink`
- `mad_k`
- `percentile`
- `low`
- `high`

Next steps I’d suggest:

- add config validation
- add the regression + serialization tests
- add a short migration note
- attach one small benchmark note in the PR description covering quality and quantization-time cost

Once those are in, I’d be comfortable merging this.

---

### User:
what to do now? Copilot AI
commented
3 minutes ago
@copilot This PR looks good overall. I like the scope and the way it stays conservative on compatibility: it keeps quant_method=GPTQ / format=GPTQ, changes the GPTQ-Pro preset from a singl...

Addressed in e6dee44: added SmoothAuto validation, the row-mask shape regression test, explicit invalid-config / serialization coverage, and a migration note for the GPTQ-Pro preset change. I also added the benchmark note in the PR progress update; I could not attach a trustworthy model-level perplexity/time datapoint from this sandbox because torch and transformers are not installed here, so I documented that limitation instead. I kept the percentile contract at 0 < value <= 100 to match your requested validation.

---

### ChatGPT:
Do this now:

**1. Skim the new commit e6dee44**
Check only these things:

- validation was actually added for `SmoothAuto`
- regression test for row-mask shape exists
- serialization + invalid-config tests exist
- migration note is present

**2. Don’t block on the sandbox benchmark limitation**
That part is reasonable. If they couldn’t run `torch` / `transformers` there, that is not a red flag by itself. The important part is they documented the limitation instead of inventing numbers.

**3. If the diff matches what they said, approve and merge**
At this point the review items were addressed. The only thing still missing is a real model-level benchmark, but that is now a follow-up improvement, not a blocker.

Paste this reply:

@copilot Thanks — that addresses my review comments. I’m fine with the validation, regression coverage, serialization tests, and migration note being handled in `e6dee44`.

I also appreciate that you explicitly documented the benchmarking limitation instead of attaching untrustworthy numbers from the sandbox. That’s acceptable for this PR.

I’ll do a quick pass on the updated diff, but assuming the changes match the summary, this is good to merge. A proper model-level perplexity / quantization-time benchmark can be added later as follow-up validation rather than blocking this PR.

If you want the stricter version instead, use this:

@copilot Thanks — the requested code/test/doc changes sound addressed in `e6dee44`, so the remaining gap is only model-level validation. I’m okay with not blocking this PR on sandbox-constrained benchmarks, but I’d like a follow-up issue added for real perplexity + quantization-time measurements on at least one representative model before we treat `SmoothAuto` as fully validated in practice.

My recommendation: **approve unless you personally want benchmark evidence before merge**.

---

### User:
i want evidence now, ive merged, how to test on my personal comptuer? using a qwen 3.5 modoel 4b

---

### ChatGPT:
I’m checking the repo’s current CLI/API plus the exact Qwen 3.5 4B model naming so I can give you a reproducible A/B test instead of hand-wavy advice.

---

I’ve got the model name and the repo’s quantization API. The cleanest proof is an A/B quantization test: same machine, same calibration set, same seed, same prompts, same eval—only changing the repo commit.

---

Yes. The fastest way to get real evidence is an **A/B quantization test on the same machine** with only one variable changed: **old commit vs new commit**. Use **Qwen/Qwen3.5-4B-Base** for the cleanest text-only-style quantization/eval workflow. GPT-QModel main says Qwen 3.5 support was added on **2026-02-28**, supports **Qwen 1/2/3/3.5**, and its documented quantization flow is `GPTQModel.load(..., quant_config) -> model.quantize(...) -> model.save(...)`. Qwen’s model card also says Qwen3.5 needs the latest `transformers` for serving/testing. [GitHub+2Hugging Face+2](https://github.com/ModelCloud/GPTQModel/blob/main/README.md)

Your PR specifically changes the **offline failsafe smoothing path** and is meant to help **low-sample fallback blocks**, while keeping standard GPTQ artifacts/runtime compatibility. So don’t do just one benchmark: do **two** quantizations per commit — one that **stresses** failsafe behavior and one that reflects a more normal setup. [GitHub](https://github.com/groxaxo/GPTQ-Pro/pull/4)

### What to measure

You want 4 outputs for each commit:

1. **Quantization wall-clock time**
2. **Peak GPU memory during quantization**
3. **Eval scores** on the same downstream tasks
4. **Deterministic generation diffs** on the same fixed prompts

For eval, `lm-eval` currently supports Hugging Face backends via `pip install "lm_eval[hf]"`, supports local model paths with `pretrained=/path/to/model`, and lists tasks with `lm-eval ls tasks`. Good quick tasks that exist now include `hellaswag`, `winogrande`, `piqa`, `wikitext`, `arc`, and `gsm8k`. [GitHub+6GitHub+6GitHub+6](https://github.com/EleutherAI/lm-evaluation-harness)

## 1) Make two repo worktrees

Run this from your GPTQ-Pro clone:

```Bashcd ~/GPTQ-Pro# baseline = parent of the first PR code commitBASE_COMMIT=$(git rev-parse 3b6de40^)# patched = last reviewed PR commit with fixesNEW_COMMIT=e6dee44git worktree add ../gptqpro_base "$BASE_COMMIT"git worktree add ../gptqpro_new  "$NEW_COMMIT"
Bashcd ~/GPTQ-Pro# baseline = parent of the first PR code commitBASE_COMMIT=$(git rev-parse 3b6de40^)# patched = last reviewed PR commit with fixesNEW_COMMIT=e6dee44git worktree add ../gptqpro_base "$BASE_COMMIT"git worktree add ../gptqpro_new  "$NEW_COMMIT"
```

That gives you:

- `../gptqpro_base` = before the PR
- `../gptqpro_new` = after the PR fixes

## 2) Create identical envs

Do this once for each worktree, with the same Python / torch stack in both. Since Qwen3.5 wants current Transformers support, install `transformers` from main. [Hugging Face](https://huggingface.co/Qwen/Qwen3.5-4B)

### Base env

```Bashcd ../gptqpro_basepython3 -m venv .venvsource .venv/bin/activatepip install -U pip setuptools wheel ninja# install your normal CUDA-matching torch herepip install "transformers @ git+https://github.com/huggingface/transformers.git@main"pip install datasets accelerate sentencepiece "lm_eval[hf]"pip install -e . --no-build-isolationdeactivate
Bashcd ../gptqpro_basepython3 -m venv .venvsource .venv/bin/activatepip install -U pip setuptools wheel ninja# install your normal CUDA-matching torch herepip install "transformers @ git+https://github.com/huggingface/transformers.git@main"pip install datasets accelerate sentencepiece "lm_eval[hf]"pip install -e . --no-build-isolationdeactivate
```

### New env

```Bashcd ../gptqpro_newpython3 -m venv .venvsource .venv/bin/activatepip install -U pip setuptools wheel ninja# install the same torch build as abovepip install "transformers @ git+https://github.com/huggingface/transformers.git@main"pip install datasets accelerate sentencepiece "lm_eval[hf]"pip install -e . --no-build-isolationdeactivate
Bashcd ../gptqpro_newpython3 -m venv .venvsource .venv/bin/activatepip install -U pip setuptools wheel ninja# install the same torch build as abovepip install "transformers @ git+https://github.com/huggingface/transformers.git@main"pip install datasets accelerate sentencepiece "lm_eval[hf]"pip install -e . --no-build-isolationdeactivate
```

## 3) Save this quantization script as quantize_once.py

Put the same file in both worktrees:

```Pythonimport osimport jsonimport timeimport randomimport argparseimport numpy as npimport torchfrom datasets import load_datasetfrom gptqmodel import GPTQModel, QuantizeConfigdef seed_all(seed: int):    random.seed(seed)    np.random.seed(seed)    torch.manual_seed(seed)    if torch.cuda.is_available():        torch.cuda.manual_seed_all(seed)def main():    ap = argparse.ArgumentParser()    ap.add_argument("--model", required=True)    ap.add_argument("--out", required=True)    ap.add_argument("--nsamples", type=int, default=128)    ap.add_argument("--batch_size", type=int, default=4)    ap.add_argument("--seed", type=int, default=1234)    ap.add_argument("--group_size", type=int, default=128)    ap.add_argument("--bits", type=int, default=4)    args = ap.parse_args()    seed_all(args.seed)    dataset = load_dataset(        "allenai/c4",        data_files="en/c4-train.00001-of-01024.json.gz",        split="train",    ).select(range(args.nsamples))["text"]    qcfg = QuantizeConfig.gptq_pro()    qcfg.bits = args.bits    qcfg.group_size = args.group_size    if torch.cuda.is_available():        torch.cuda.empty_cache()        torch.cuda.reset_peak_memory_stats()    start = time.perf_counter()    model = GPTQModel.load(args.model, qcfg)    model.quantize(dataset, batch_size=args.batch_size)    elapsed = time.perf_counter() - start    os.makedirs(args.out, exist_ok=True)    model.save(args.out)    peak_mem_bytes = None    if torch.cuda.is_available():        peak_mem_bytes = int(torch.cuda.max_memory_allocated())    meta = {        "model": args.model,        "out": args.out,        "nsamples": args.nsamples,        "batch_size": args.batch_size,        "seed": args.seed,        "group_size": args.group_size,        "bits": args.bits,        "elapsed_sec": elapsed,        "peak_cuda_mem_bytes": peak_mem_bytes,    }    with open(os.path.join(args.out, "ab_meta.json"), "w") as f:        json.dump(meta, f, indent=2)    print(json.dumps(meta, indent=2))if __name__ == "__main__":    main()
Pythonimport osimport jsonimport timeimport randomimport argparseimport numpy as npimport torchfrom datasets import load_datasetfrom gptqmodel import GPTQModel, QuantizeConfigdef seed_all(seed: int):    random.seed(seed)    np.random.seed(seed)    torch.manual_seed(seed)    if torch.cuda.is_available():        torch.cuda.manual_seed_all(seed)def main():    ap = argparse.ArgumentParser()    ap.add_argument("--model", required=True)    ap.add_argument("--out", required=True)    ap.add_argument("--nsamples", type=int, default=128)    ap.add_argument("--batch_size", type=int, default=4)    ap.add_argument("--seed", type=int, default=1234)    ap.add_argument("--group_size", type=int, default=128)    ap.add_argument("--bits", type=int, default=4)    args = ap.parse_args()    seed_all(args.seed)    dataset = load_dataset(        "allenai/c4",        data_files="en/c4-train.00001-of-01024.json.gz",        split="train",    ).select(range(args.nsamples))["text"]    qcfg = QuantizeConfig.gptq_pro()    qcfg.bits = args.bits    qcfg.group_size = args.group_size    if torch.cuda.is_available():        torch.cuda.empty_cache()        torch.cuda.reset_peak_memory_stats()    start = time.perf_counter()    model = GPTQModel.load(args.model, qcfg)    model.quantize(dataset, batch_size=args.batch_size)    elapsed = time.perf_counter() - start    os.makedirs(args.out, exist_ok=True)    model.save(args.out)    peak_mem_bytes = None    if torch.cuda.is_available():        peak_mem_bytes = int(torch.cuda.max_memory_allocated())    meta = {        "model": args.model,        "out": args.out,        "nsamples": args.nsamples,        "batch_size": args.batch_size,        "seed": args.seed,        "group_size": args.group_size,        "bits": args.bits,        "elapsed_sec": elapsed,        "peak_cuda_mem_bytes": peak_mem_bytes,    }    with open(os.path.join(args.out, "ab_meta.json"), "w") as f:        json.dump(meta, f, indent=2)    print(json.dumps(meta, indent=2))if __name__ == "__main__":    main()
```

## 4) Run the two key quantization cases

Use the same model, same seed, same GPU, same everything.

### Case A: stress the PR

This is the important one, because your change targets **low-sample fallback blocks**.

#### Baseline

```Bashcd ../gptqpro_basesource .venv/bin/activateCUDA_VISIBLE_DEVICES=0 python quantize_once.py \  --model Qwen/Qwen3.5-4B-Base \  --out ./out/qwen35_4b_base_ns128 \  --nsamples 128 \  --batch_size 4 \  --seed 1234deactivate
Bashcd ../gptqpro_basesource .venv/bin/activateCUDA_VISIBLE_DEVICES=0 python quantize_once.py \  --model Qwen/Qwen3.5-4B-Base \  --out ./out/qwen35_4b_base_ns128 \  --nsamples 128 \  --batch_size 4 \  --seed 1234deactivate
```

#### Patched

```Bashcd ../gptqpro_newsource .venv/bin/activateCUDA_VISIBLE_DEVICES=0 python quantize_once.py \  --model Qwen/Qwen3.5-4B-Base \  --out ./out/qwen35_4b_new_ns128 \  --nsamples 128 \  --batch_size 4 \  --seed 1234deactivate
Bashcd ../gptqpro_newsource .venv/bin/activateCUDA_VISIBLE_DEVICES=0 python quantize_once.py \  --model Qwen/Qwen3.5-4B-Base \  --out ./out/qwen35_4b_new_ns128 \  --nsamples 128 \  --batch_size 4 \  --seed 1234deactivate
```

### Case B: more normal quantization

```Bashcd ../gptqpro_basesource .venv/bin/activateCUDA_VISIBLE_DEVICES=0 python quantize_once.py \  --model Qwen/Qwen3.5-4B-Base \  --out ./out/qwen35_4b_base_ns512 \  --nsamples 512 \  --batch_size 4 \  --seed 1234deactivate
Bashcd ../gptqpro_basesource .venv/bin/activateCUDA_VISIBLE_DEVICES=0 python quantize_once.py \  --model Qwen/Qwen3.5-4B-Base \  --out ./out/qwen35_4b_base_ns512 \  --nsamples 512 \  --batch_size 4 \  --seed 1234deactivate
```

```Bashcd ../gptqpro_newsource .venv/bin/activateCUDA_VISIBLE_DEVICES=0 python quantize_once.py \  --model Qwen/Qwen3.5-4B-Base \  --out ./out/qwen35_4b_new_ns512 \  --nsamples 512 \  --batch_size 4 \  --seed 1234deactivate
Bashcd ../gptqpro_newsource .venv/bin/activateCUDA_VISIBLE_DEVICES=0 python quantize_once.py \  --model Qwen/Qwen3.5-4B-Base \  --out ./out/qwen35_4b_new_ns512 \  --nsamples 512 \  --batch_size 4 \  --seed 1234deactivate
```

## 5) Run lm-eval on both outputs

Start with a fast suite:

```BashTASKS="wikitext,hellaswag,winogrande,piqa"
BashTASKS="wikitext,hellaswag,winogrande,piqa"
```

### Baseline

```Bashcd ../gptqpro_basesource .venv/bin/activatelm_eval --model hf \  --model_args pretrained=$(pwd)/out/qwen35_4b_base_ns128,trust_remote_code=True \  --tasks $TASKS \  --device cuda:0 \  --batch_size auto:4 \  --output_path $(pwd)/eval_ns128deactivate
Bashcd ../gptqpro_basesource .venv/bin/activatelm_eval --model hf \  --model_args pretrained=$(pwd)/out/qwen35_4b_base_ns128,trust_remote_code=True \  --tasks $TASKS \  --device cuda:0 \  --batch_size auto:4 \  --output_path $(pwd)/eval_ns128deactivate
```

### Patched

```Bashcd ../gptqpro_newsource .venv/bin/activatelm_eval --model hf \  --model_args pretrained=$(pwd)/out/qwen35_4b_new_ns128,trust_remote_code=True \  --tasks $TASKS \  --device cuda:0 \  --batch_size auto:4 \  --output_path $(pwd)/eval_ns128deactivate
Bashcd ../gptqpro_newsource .venv/bin/activatelm_eval --model hf \  --model_args pretrained=$(pwd)/out/qwen35_4b_new_ns128,trust_remote_code=True \  --tasks $TASKS \  --device cuda:0 \  --batch_size auto:4 \  --output_path $(pwd)/eval_ns128deactivate
```

Then repeat for the `nsamples=512` models.

If you want a slower but more convincing second pass, add `arc` and `gsm8k`. Those tasks exist in the current harness too. [GitHub+1](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/README.md)

## 6) Compare deterministic generations too

Save this as `gen_compare.py` in both worktrees:

```Pythonimport jsonimport argparseimport torchfrom transformers import AutoTokenizer, AutoModelForCausalLMPROMPTS = [    "Explain why TCP head-of-line blocking happens.",    "Write a Python function to reverse a linked list.",    "Summarize the causes of inflation in simple terms.",    "What is the difference between overfitting and underfitting?",    "Give three practical uses of embeddings in search systems.",    "Why does quantization sometimes hurt coding ability more than chatting?",    "Solve: If a train travels 60 km in 45 minutes, what is its speed in km/h?",    "Write a bash command to find the 20 largest files under /var/log.",    "Explain virtual memory like I'm 12 years old.",    "What are the tradeoffs between GPTQ and AWQ?"]def main():    ap = argparse.ArgumentParser()    ap.add_argument("--model", required=True)    ap.add_argument("--out", required=True)    args = ap.parse_args()    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)    model = AutoModelForCausalLM.from_pretrained(        args.model,        trust_remote_code=True,        device_map="auto",        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,    )    rows = []    for p in PROMPTS:        inputs = tok(p, return_tensors="pt").to(model.device)        with torch.no_grad():            out = model.generate(                **inputs,                do_sample=False,                temperature=0.0,                max_new_tokens=128,                pad_token_id=tok.eos_token_id,            )        text = tok.decode(out[0], skip_special_tokens=True)        rows.append({"prompt": p, "output": text})    with open(args.out, "w") as f:        json.dump(rows, f, indent=2, ensure_ascii=False)if __name__ == "__main__":    main()
Pythonimport jsonimport argparseimport torchfrom transformers import AutoTokenizer, AutoModelForCausalLMPROMPTS = [    "Explain why TCP head-of-line blocking happens.",    "Write a Python function to reverse a linked list.",    "Summarize the causes of inflation in simple terms.",    "What is the difference between overfitting and underfitting?",    "Give three practical uses of embeddings in search systems.",    "Why does quantization sometimes hurt coding ability more than chatting?",    "Solve: If a train travels 60 km in 45 minutes, what is its speed in km/h?",    "Write a bash command to find the 20 largest files under /var/log.",    "Explain virtual memory like I'm 12 years old.",    "What are the tradeoffs between GPTQ and AWQ?"]def main():    ap = argparse.ArgumentParser()    ap.add_argument("--model", required=True)    ap.add_argument("--out", required=True)    args = ap.parse_args()    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)    model = AutoModelForCausalLM.from_pretrained(        args.model,        trust_remote_code=True,        device_map="auto",        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,    )    rows = []    for p in PROMPTS:        inputs = tok(p, return_tensors="pt").to(model.device)        with torch.no_grad():            out = model.generate(                **inputs,                do_sample=False,                temperature=0.0,                max_new_tokens=128,                pad_token_id=tok.eos_token_id,            )        text = tok.decode(out[0], skip_special_tokens=True)        rows.append({"prompt": p, "output": text})    with open(args.out, "w") as f:        json.dump(rows, f, indent=2, ensure_ascii=False)if __name__ == "__main__":    main()
```

Run it on the two quantized models and diff the JSON files.

## 7) How to interpret the result

If your PR is genuinely helping, the **patched** build should usually show:

- **same or slightly slower quantization time** in the stressed run
- **same or lower wikitext perplexity / better language-model score**
- **same or better accuracy** on `hellaswag`, `winogrande`, `piqa`
- **cleaner deterministic outputs** on edge prompts
- the **biggest difference at nsamples=128**, because that is where the new adaptive failsafe search should matter most [GitHub](https://github.com/groxaxo/GPTQ-Pro/pull/4)

What would worry me:

- patched build is materially slower **and**
- `wikitext` gets worse **and**
- downstream tasks are flat or worse

That would mean the extra best-of smoothing search is adding complexity without buying useful quality.

## 8) Minimal verdict rule

Use this decision rule:

- **Keep merged** if patched is better on `wikitext` and at least neutral elsewhere
- **Definitely keep merged** if patched wins mainly at `nsamples=128` and is neutral at `512`
- **Reconsider defaulting it** only if patched is slower and quality is not measurably better

The reason I’d weight `nsamples=128` heavily is that your PR is not a general kernel rewrite; it is a targeted improvement to the **failsafe path for low-sample fallback blocks**. That is exactly where it should prove itself. [GitHub](https://github.com/groxaxo/GPTQ-Pro/pull/4)

If you want, I can turn this into a **single bash harness** that runs the full A/B and prints a compact before/after table.

---

### User:
yes please, provide a full testing suite , using 1x 3090 , gpu 0 for this task please.

---

### ChatGPT:
I’m turning this into a runnable harness instead of loose snippets: one command to quantize old vs new, evaluate both, compare generations, and emit a summary for GPU 0 only.

---

I’ve got the structure: one bootstrap for worktrees/envs, one full runner for quantize+eval+generation, and one summarizer that spits out a markdown verdict.

---

Done.

I packaged a runnable A/B harness here:

- 
- 

It’s built around GPTQModel’s documented `load -> quantize -> save` flow, uses `Qwen/Qwen3.5-4B-Base` as the default 4B test model, and evaluates the quantized outputs with lm-eval’s Hugging Face backend against local model paths. [GitHub+3GitHub+3Hugging Face+3](https://github.com/ModelCloud/GPTQModel)

Run it like this:

```Bashcd /path/where/you/unzipped/gptq_pro_ab_suitechmod +x scripts/*.sh./scripts/bootstrap_worktrees_and_envs.sh \  --repo-root ~/GPTQ-Pro \  --base-commit "3b6de40^" \  --new-commit "e6dee44" \  --torch-install-cmd 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128'./scripts/run_full_suite.sh \  --repo-root ~/GPTQ-Pro \  --base-commit "3b6de40^" \  --new-commit "e6dee44" \  --gpu 0 \  --model-id Qwen/Qwen3.5-4B-Base \  --tasks 'wikitext,hellaswag,winogrande,piqa' \  --output-root ~/gptq_ab_results
Bashcd /path/where/you/unzipped/gptq_pro_ab_suitechmod +x scripts/*.sh./scripts/bootstrap_worktrees_and_envs.sh \  --repo-root ~/GPTQ-Pro \  --base-commit "3b6de40^" \  --new-commit "e6dee44" \  --torch-install-cmd 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128'./scripts/run_full_suite.sh \  --repo-root ~/GPTQ-Pro \  --base-commit "3b6de40^" \  --new-commit "e6dee44" \  --gpu 0 \  --model-id Qwen/Qwen3.5-4B-Base \  --tasks 'wikitext,hellaswag,winogrande,piqa' \  --output-root ~/gptq_ab_results
```

What you’ll get:

- `~/gptq_ab_results/summary.md`
- `~/gptq_ab_results/summary.json`

The suite runs:

- base vs new worktrees
- `nsamples=128` and `nsamples=512`
- quantization timing + peak CUDA memory
- lm-eval task results
- deterministic generation comparisons on fixed prompts

GPTQModel’s README also explicitly recommends using `lm-eval` or `evalplus` rather than plain perplexity as the main post-quant quality check, and lm-eval documents both the HF backend install and `lm-eval ls tasks` for checking task availability. [GitHub+1](https://github.com/ModelCloud/GPTQModel)

One thing to watch: swap the PyTorch install command to match your exact CUDA stack before bootstrapping.

---

### User:
i cant download it, can you pass me the files?

---

### ChatGPT:
I’m unpacking the suite so I can paste every file directly here in a usable form.

---

Yes — recreate it locally with this.

Run this in an empty folder:

```Bashmkdir -p gptq_pro_ab_suite/scriptscd gptq_pro_ab_suitecat > README.md <<'EOF'# GPTQ-Pro A/B test suite for Qwen3.5-4B on GPU 0This suite is designed to verify whether your merged PR materially improves quantization quality in practice.It runs a strict A/B comparison:- **base** = repo state before the PR- **new** = repo state after the PR- same machine- same GPU (**GPU 0 only**)- same model- same calibration data- same seed- same eval tasksIt produces:- quantization timing + peak CUDA memory- `lm-eval` results for each quantized model- deterministic generation outputs for side-by-side inspection- a compact markdown + JSON summary## Files- `scripts/bootstrap_worktrees_and_envs.sh`  - creates base/new worktrees  - creates two virtualenvs  - installs dependencies in both- `scripts/run_full_suite.sh`  - runs quantization, lm-eval, deterministic generation, and final summary- `scripts/quantize_once.py`  - quantizes one model once and records timing/memory metadata- `scripts/gen_compare.py`  - runs fixed deterministic prompts- `scripts/summarize_ab.py`  - aggregates metrics into JSON + Markdown## Requirements- Linux- Python 3.10+- NVIDIA driver + CUDA-compatible PyTorch- 1x RTX 3090 on **GPU 0**- enough disk for two worktrees + downloaded model + four quantized outputs## Recommended modelUse:- `Qwen/Qwen3.5-4B-Base`## 1) Bootstrap the two repos and environmentsReplace the torch install command with the one matching your CUDA stack.```bashchmod +x scripts/*.sh./scripts/bootstrap_worktrees_and_envs.sh \  --repo-root ~/GPTQ-Pro \  --base-commit "3b6de40^" \  --new-commit "e6dee44" \  --torch-install-cmd 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128'
Bashmkdir -p gptq_pro_ab_suite/scriptscd gptq_pro_ab_suitecat > README.md <<'EOF'# GPTQ-Pro A/B test suite for Qwen3.5-4B on GPU 0This suite is designed to verify whether your merged PR materially improves quantization quality in practice.It runs a strict A/B comparison:- **base** = repo state before the PR- **new** = repo state after the PR- same machine- same GPU (**GPU 0 only**)- same model- same calibration data- same seed- same eval tasksIt produces:- quantization timing + peak CUDA memory- `lm-eval` results for each quantized model- deterministic generation outputs for side-by-side inspection- a compact markdown + JSON summary## Files- `scripts/bootstrap_worktrees_and_envs.sh`  - creates base/new worktrees  - creates two virtualenvs  - installs dependencies in both- `scripts/run_full_suite.sh`  - runs quantization, lm-eval, deterministic generation, and final summary- `scripts/quantize_once.py`  - quantizes one model once and records timing/memory metadata- `scripts/gen_compare.py`  - runs fixed deterministic prompts- `scripts/summarize_ab.py`  - aggregates metrics into JSON + Markdown## Requirements- Linux- Python 3.10+- NVIDIA driver + CUDA-compatible PyTorch- 1x RTX 3090 on **GPU 0**- enough disk for two worktrees + downloaded model + four quantized outputs## Recommended modelUse:- `Qwen/Qwen3.5-4B-Base`## 1) Bootstrap the two repos and environmentsReplace the torch install command with the one matching your CUDA stack.```bashchmod +x scripts/*.sh./scripts/bootstrap_worktrees_and_envs.sh \  --repo-root ~/GPTQ-Pro \  --base-commit "3b6de40^" \  --new-commit "e6dee44" \  --torch-install-cmd 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128'
```

This creates:

- `~/GPTQ-Pro_ab_base`
- `~/GPTQ-Pro_ab_new`

## 2) Run the full suite on GPU 0

```Bash./scripts/run_full_suite.sh \  --repo-root ~/GPTQ-Pro \  --base-commit "3b6de40^" \  --new-commit "e6dee44" \  --gpu 0 \  --model-id Qwen/Qwen3.5-4B-Base \  --tasks 'wikitext,hellaswag,winogrande,piqa' \  --output-root ~/gptq_ab_results
Bash./scripts/run_full_suite.sh \  --repo-root ~/GPTQ-Pro \  --base-commit "3b6de40^" \  --new-commit "e6dee44" \  --gpu 0 \  --model-id Qwen/Qwen3.5-4B-Base \  --tasks 'wikitext,hellaswag,winogrande,piqa' \  --output-root ~/gptq_ab_results
```

That runs two calibration regimes:

- `nsamples=128` -> stress test for the PR
- `nsamples=512` -> more normal case

## 3) Read the final verdict

Outputs land under:

- `~/gptq_ab_results/summary.md`
- `~/gptq_ab_results/summary.json`
EOF

cat > scripts/bootstrap_worktrees_and_envs.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

usage() {
cat <<USAGE
Usage:
$0 --repo-root PATH --base-commit REV --new-commit REV --torch-install-cmd 'pip install ...'

Notes:

- Creates sibling worktrees: <repo>_ab_base and <repo>_ab_new
- Creates .venv inside each worktree
- Installs identical dependencies in both envs
USAGE
}

REPO_ROOT=""
BASE_COMMIT=""
NEW_COMMIT=""
TORCH_INSTALL_CMD=""

while [[ $# -gt 0 ]]; do
case "$1" in
--repo-root) REPO_ROOT="$2"; shift 2 ;;
--base-commit) BASE_COMMIT="$2"; shift 2 ;;
--new-commit) NEW_COMMIT="$2"; shift 2 ;;
--torch-install-cmd) TORCH_INSTALL_CMD="$2"; shift 2 ;;
-h|--help) usage; exit 0 ;;
*) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
esac
done

[[ -n "$REPO_ROOT" ]] || { echo "--repo-root is required" >&2; exit 1; }
[[ -n "$BASE_COMMIT" ]] || { echo "--base-commit is required" >&2; exit 1; }
[[ -n "$NEW_COMMIT" ]] || { echo "--new-commit is required" >&2; exit 1; }
[[ -n "$TORCH_INSTALL_CMD" ]] || { echo "--torch-install-cmd is required" >&2; exit 1; }

REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
BASE_WT="${REPO_ROOT}_ab_base"
NEW_WT="${REPO_ROOT}_ab_new"

ensure_worktree() {
local wt="$1"
local rev="$2"
if [[ -d "$wt/.git" || -f "$wt/.git" ]]; then
echo "[info] worktree already exists: $wt"
else
echo "[info] creating worktree $wt at $rev"
git -C "$REPO_ROOT" worktree add "$wt" "$rev"
fi
}

install_env() {
local wt="$1"
echo "[info] setting up env in $wt"
pushd "$wt" >/dev/null
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel ninja packaging
eval "$TORCH_INSTALL_CMD"
python -m pip install "transformers @ git+[https://github.com/huggingface/transformers.git@main]()"
python -m pip install datasets accelerate sentencepiece psutil "lm_eval[hf]"
python -m pip install -e . --no-build-isolation
python - <<'PY'
import sys
try:
import torch
print('[info] torch', torch.**version**, 'cuda=', torch.version.cuda, 'available=', torch.cuda.is_available())
except Exception as e:
print('[warn] torch import failed:', e, file=sys.stderr)
raise
PY
deactivate
popd >/dev/null
}

ensure_worktree "$BASE_WT" "$BASE_COMMIT"
ensure_worktree "$NEW_WT" "$NEW_COMMIT"
install_env "$BASE_WT"
install_env "$NEW_WT"

echo "[done]"
echo "BASE_WT=$BASE_WT"
echo "NEW_WT=$NEW_WT"
EOF

cat > scripts/quantize_once.py <<'EOF'
#!/usr/bin/env python3
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

def seed_all(seed: int) -> None:
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
torch.cuda.manual_seed_all(seed)

def main() -> int:
ap = argparse.ArgumentParser()
ap.add_argument("--model-id", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--nsamples", type=int, default=128)
ap.add_argument("--batch-size", type=int, default=4)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--bits", type=int, default=4)
ap.add_argument("--group-size", type=int, default=128)
ap.add_argument("--dataset", default="allenai/c4")
ap.add_argument("--dataset-file", default="en/c4-train.00001-of-01024.json.gz")
args = ap.parse_args()

```out_dir = Path(args.out_dir)out_dir.mkdir(parents=True, exist_ok=True)seed_all(args.seed)if torch.cuda.is_available():    torch.cuda.empty_cache()    torch.cuda.reset_peak_memory_stats()ds = load_dataset(    args.dataset,    data_files=args.dataset_file,    split="train",).select(range(args.nsamples))calibration_dataset = ds["text"]qcfg = QuantizeConfig.gptq_pro()qcfg.bits = args.bitsqcfg.group_size = args.group_sizestart = time.perf_counter()model = GPTQModel.load(args.model_id, qcfg)model.quantize(calibration_dataset, batch_size=args.batch_size)elapsed = time.perf_counter() - startmodel.save(str(out_dir))peak_cuda_mem_bytes = Noneif torch.cuda.is_available():    peak_cuda_mem_bytes = int(torch.cuda.max_memory_allocated())metadata = {    "model_id": args.model_id,    "out_dir": str(out_dir),    "nsamples": args.nsamples,    "batch_size": args.batch_size,    "seed": args.seed,    "bits": args.bits,    "group_size": args.group_size,    "elapsed_sec": elapsed,    "peak_cuda_mem_bytes": peak_cuda_mem_bytes,    "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,    "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,    "torch_version": torch.__version__,    "torch_cuda": torch.version.cuda,}with open(out_dir / "ab_quant_meta.json", "w", encoding="utf-8") as f:    json.dump(metadata, f, indent=2)print(json.dumps(metadata, indent=2))return 0
out_dir = Path(args.out_dir)out_dir.mkdir(parents=True, exist_ok=True)seed_all(args.seed)if torch.cuda.is_available():    torch.cuda.empty_cache()    torch.cuda.reset_peak_memory_stats()ds = load_dataset(    args.dataset,    data_files=args.dataset_file,    split="train",).select(range(args.nsamples))calibration_dataset = ds["text"]qcfg = QuantizeConfig.gptq_pro()qcfg.bits = args.bitsqcfg.group_size = args.group_sizestart = time.perf_counter()model = GPTQModel.load(args.model_id, qcfg)model.quantize(calibration_dataset, batch_size=args.batch_size)elapsed = time.perf_counter() - startmodel.save(str(out_dir))peak_cuda_mem_bytes = Noneif torch.cuda.is_available():    peak_cuda_mem_bytes = int(torch.cuda.max_memory_allocated())metadata = {    "model_id": args.model_id,    "out_dir": str(out_dir),    "nsamples": args.nsamples,    "batch_size": args.batch_size,    "seed": args.seed,    "bits": args.bits,    "group_size": args.group_size,    "elapsed_sec": elapsed,    "peak_cuda_mem_bytes": peak_cuda_mem_bytes,    "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,    "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,    "torch_version": torch.__version__,    "torch_cuda": torch.version.cuda,}with open(out_dir / "ab_quant_meta.json", "w", encoding="utf-8") as f:    json.dump(metadata, f, indent=2)print(json.dumps(metadata, indent=2))return 0
```

if **name** == "**main**":
raise SystemExit(main())
EOF

cat > scripts/gen_compare.py <<'EOF'
#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
"Explain why TCP head-of-line blocking happens.",
"Write a Python function to reverse a linked list.",
"Summarize the causes of inflation in simple terms.",
"What is the difference between overfitting and underfitting?",
"Give three practical uses of embeddings in search systems.",
"Why can quantization hurt coding ability more than casual chat?",
"Solve: If a train travels 60 km in 45 minutes, what is its speed in km/h?",
"Write a bash command to find the 20 largest files under /var/log.",
"Explain virtual memory like I'm 12 years old.",
"What are the tradeoffs between GPTQ and AWQ?",
]

def main() -> int:
ap = argparse.ArgumentParser()
ap.add_argument("--model-path", required=True)
ap.add_argument("--out-json", required=True)
ap.add_argument("--max-new-tokens", type=int, default=160)
args = ap.parse_args()

```model_path = Path(args.model_path)out_json = Path(args.out_json)out_json.parent.mkdir(parents=True, exist_ok=True)tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)dtype = torch.float16 if torch.cuda.is_available() else torch.float32model = AutoModelForCausalLM.from_pretrained(    str(model_path),    trust_remote_code=True,    device_map="auto",    torch_dtype=dtype,)rows = []for prompt in PROMPTS:    inputs = tok(prompt, return_tensors="pt")    inputs = {k: v.to(model.device) for k, v in inputs.items()}    with torch.no_grad():        out = model.generate(            **inputs,            do_sample=False,            temperature=0.0,            top_p=1.0,            max_new_tokens=args.max_new_tokens,            pad_token_id=tok.eos_token_id,        )    text = tok.decode(out[0], skip_special_tokens=True)    rows.append({"prompt": prompt, "output": text})with open(out_json, "w", encoding="utf-8") as f:    json.dump(rows, f, indent=2, ensure_ascii=False)print(f"wrote {out_json}")return 0
model_path = Path(args.model_path)out_json = Path(args.out_json)out_json.parent.mkdir(parents=True, exist_ok=True)tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)dtype = torch.float16 if torch.cuda.is_available() else torch.float32model = AutoModelForCausalLM.from_pretrained(    str(model_path),    trust_remote_code=True,    device_map="auto",    torch_dtype=dtype,)rows = []for prompt in PROMPTS:    inputs = tok(prompt, return_tensors="pt")    inputs = {k: v.to(model.device) for k, v in inputs.items()}    with torch.no_grad():        out = model.generate(            **inputs,            do_sample=False,            temperature=0.0,            top_p=1.0,            max_new_tokens=args.max_new_tokens,            pad_token_id=tok.eos_token_id,        )    text = tok.decode(out[0], skip_special_tokens=True)    rows.append({"prompt": prompt, "output": text})with open(out_json, "w", encoding="utf-8") as f:    json.dump(rows, f, indent=2, ensure_ascii=False)print(f"wrote {out_json}")return 0
```

if **name** == "**main**":
raise SystemExit(main())
EOF

cat > scripts/summarize_ab.py <<'EOF'
#!/usr/bin/env python3
import argparse
import difflib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HIGHER_IS_BETTER = {"acc", "acc_norm", "exact_match", "mc1", "mc2", "f1", "bleu"}
LOWER_IS_BETTER = {"word_perplexity", "perplexity", "byte_perplexity", "bpb", "bpc"}

def load_json(path: Path):
with open(path, "r", encoding="utf-8") as f:
return json.load(f)

def find_lm_eval_json(eval_dir: Path) -> Optional[Path]:
candidates = sorted(eval_dir.rglob("*.json"))
for p in candidates:
try:
data = load_json(p)
except Exception:
continue
if isinstance(data, dict) and "results" in data:
return p
return None

def extract_task_metrics(eval_json: Dict) -> Dict[str, Dict[str, float]]:
results = eval_json.get("results", {})
out: Dict[str, Dict[str, float]] = {}
for task, task_metrics in results.items():
flat: Dict[str, float] = {}
if not isinstance(task_metrics, dict):
continue
for k, v in task_metrics.items():
if isinstance(v, (int, float)):
flat[k] = float(v)
out[task] = flat
return out

def choose_primary_metric(task: str, metrics: Dict[str, float]) -> Tuple[Optional[str], Optional[float], Optional[str]]:
preferred = [
"acc_norm,none", "acc,none", "exact_match,strict-match", "exact_match,none",
"mc2,none", "mc1,none", "word_perplexity,none", "perplexity,none",
"byte_perplexity,none", "bpb,none", "bpc,none",
"acc_norm", "acc", "exact_match", "mc2", "mc1", "word_perplexity",
"perplexity", "byte_perplexity", "bpb", "bpc",
]
for key in preferred:
if key in metrics:
short = key.split(",", 1)[0]
direction = "higher" if short in HIGHER_IS_BETTER else "lower" if short in LOWER_IS_BETTER else "unknown"
return key, metrics[key], direction
return None, None, None

def compare_generations(base_path: Path, new_path: Path) -> Dict:
base_rows = load_json(base_path)
new_rows = load_json(new_path)
exact = 0
ratios: List[float] = []
changed_examples = []
for b, n in zip(base_rows, new_rows):
br = b.get("output", "")
nr = n.get("output", "")
if br == nr:
exact += 1
ratio = difflib.SequenceMatcher(None, br, nr).ratio()
ratios.append(ratio)
if br != nr and len(changed_examples) < 3:
changed_examples.append({
"prompt": b.get("prompt", ""),
"base_output": br[:900],
"new_output": nr[:900],
"similarity": ratio,
})
return {
"num_prompts": len(base_rows),
"exact_match_count": exact,
"exact_match_rate": exact / max(1, len(base_rows)),
"avg_similarity": sum(ratios) / max(1, len(ratios)),
"changed_examples": changed_examples,
}

def load_quant_meta(path: Path) -> Dict:
return load_json(path)

def build_case_summary(case_dir: Path) -> Dict:
base_quant = load_quant_meta(case_dir / "base" / "ab_quant_meta.json")
new_quant = load_quant_meta(case_dir / "new" / "ab_quant_meta.json")

```base_eval_json_path = find_lm_eval_json(case_dir / "base" / "eval")new_eval_json_path = find_lm_eval_json(case_dir / "new" / "eval")if base_eval_json_path is None or new_eval_json_path is None:    raise FileNotFoundError(f"Could not find lm-eval JSON under {case_dir}")base_eval = load_json(base_eval_json_path)new_eval = load_json(new_eval_json_path)base_metrics = extract_task_metrics(base_eval)new_metrics = extract_task_metrics(new_eval)task_summaries = []all_tasks = sorted(set(base_metrics) | set(new_metrics))for task in all_tasks:    b = base_metrics.get(task, {})    n = new_metrics.get(task, {})    metric_name, bval, direction = choose_primary_metric(task, b)    metric_name2, nval, direction2 = choose_primary_metric(task, n)    chosen = metric_name or metric_name2    direction = direction or direction2 or "unknown"    if chosen is not None:        bval = b.get(chosen, bval)        nval = n.get(chosen, nval)    delta = None    winner = "tie"    if bval is not None and nval is not None:        delta = nval - bval        if direction == "higher":            winner = "new" if nval > bval else "base" if bval > nval else "tie"        elif direction == "lower":            winner = "new" if nval < bval else "base" if bval < nval else "tie"    task_summaries.append({        "task": task,        "metric": chosen,        "direction": direction,        "base": bval,        "new": nval,        "delta_new_minus_base": delta,        "winner": winner,    })gen = compare_generations(case_dir / "base" / "generations.json", case_dir / "new" / "generations.json")quant_delta = {    "elapsed_sec_new_minus_base": new_quant["elapsed_sec"] - base_quant["elapsed_sec"],    "peak_cuda_mem_bytes_new_minus_base": (        (new_quant.get("peak_cuda_mem_bytes") or 0) - (base_quant.get("peak_cuda_mem_bytes") or 0)    ),}return {    "case": case_dir.name,    "base_quant": base_quant,    "new_quant": new_quant,    "quant_delta": quant_delta,    "base_eval_json": str(base_eval_json_path),    "new_eval_json": str(new_eval_json_path),    "tasks": task_summaries,    "generation_compare": gen,}
base_eval_json_path = find_lm_eval_json(case_dir / "base" / "eval")new_eval_json_path = find_lm_eval_json(case_dir / "new" / "eval")if base_eval_json_path is None or new_eval_json_path is None:    raise FileNotFoundError(f"Could not find lm-eval JSON under {case_dir}")base_eval = load_json(base_eval_json_path)new_eval = load_json(new_eval_json_path)base_metrics = extract_task_metrics(base_eval)new_metrics = extract_task_metrics(new_eval)task_summaries = []all_tasks = sorted(set(base_metrics) | set(new_metrics))for task in all_tasks:    b = base_metrics.get(task, {})    n = new_metrics.get(task, {})    metric_name, bval, direction = choose_primary_metric(task, b)    metric_name2, nval, direction2 = choose_primary_metric(task, n)    chosen = metric_name or metric_name2    direction = direction or direction2 or "unknown"    if chosen is not None:        bval = b.get(chosen, bval)        nval = n.get(chosen, nval)    delta = None    winner = "tie"    if bval is not None and nval is not None:        delta = nval - bval        if direction == "higher":            winner = "new" if nval > bval else "base" if bval > nval else "tie"        elif direction == "lower":            winner = "new" if nval < bval else "base" if bval < nval else "tie"    task_summaries.append({        "task": task,        "metric": chosen,        "direction": direction,        "base": bval,        "new": nval,        "delta_new_minus_base": delta,        "winner": winner,    })gen = compare_generations(case_dir / "base" / "generations.json", case_dir / "new" / "generations.json")quant_delta = {    "elapsed_sec_new_minus_base": new_quant["elapsed_sec"] - base_quant["elapsed_sec"],    "peak_cuda_mem_bytes_new_minus_base": (        (new_quant.get("peak_cuda_mem_bytes") or 0) - (base_quant.get("peak_cuda_mem_bytes") or 0)    ),}return {    "case": case_dir.name,    "base_quant": base_quant,    "new_quant": new_quant,    "quant_delta": quant_delta,    "base_eval_json": str(base_eval_json_path),    "new_eval_json": str(new_eval_json_path),    "tasks": task_summaries,    "generation_compare": gen,}
```

def overall_verdict(case_summaries: List[Dict]) -> str:
new_wins = 0
base_wins = 0
for case in case_summaries:
for t in case["tasks"]:
if t["winner"] == "new":
new_wins += 1
elif t["winner"] == "base":
base_wins += 1
if new_wins > base_wins:
return "new looks better overall"
if base_wins > new_wins:
return "base looks better overall"
return "mixed / inconclusive"

def render_md(case_summaries: List[Dict], summary_path: Path) -> None:
lines: List[str] = []
lines.append("# GPTQ-Pro A/B summary")
lines.append("")
lines.append(f"Overall verdict: **{overall_verdict(case_summaries)}**")
lines.append("")
for case in case_summaries:
lines.append(f"## {case['case']}")
lines.append("")
lines.append("### Quantization")
lines.append("")
lines.append(f"- base elapsed: {case['base_quant']['elapsed_sec']:.2f}s")
lines.append(f"- new elapsed: {case['new_quant']['elapsed_sec']:.2f}s")
lines.append(f"- delta (new - base): {case['quant_delta']['elapsed_sec_new_minus_base']:.2f}s")
lines.append(f"- base peak CUDA mem: {case['base_quant'].get('peak_cuda_mem_bytes')}")
lines.append(f"- new peak CUDA mem: {case['new_quant'].get('peak_cuda_mem_bytes')}")
lines.append(f"- delta peak mem (new - base): {case['quant_delta']['peak_cuda_mem_bytes_new_minus_base']}")
lines.append("")
lines.append("### Eval")
lines.append("")
lines.append("| task | metric | direction | base | new | delta (new-base) | winner |")
lines.append("|---|---|---:|---:|---:|---:|---|")
for t in case["tasks"]:
lines.append(
f"| {t['task']} | {t['metric']} | {t['direction']} | {t['base']} | {t['new']} | {t['delta_new_minus_base']} | {t['winner']} |"
)
lines.append("")
gc = case["generation_compare"]
lines.append("### Deterministic generations")
lines.append("")
lines.append(f"- exact matches: {gc['exact_match_count']} / {gc['num_prompts']}")
lines.append(f"- exact match rate: {gc['exact_match_rate']:.3f}")
lines.append(f"- average similarity: {gc['avg_similarity']:.3f}")
if gc["changed_examples"]:
lines.append("")
lines.append("#### Sample changed outputs")
lines.append("")
for ex in gc["changed_examples"]:
lines.append(f"- Prompt: `{ex['prompt']}`")
lines.append(f"  - similarity: {ex['similarity']:.3f}")
lines.append(f"  - base: {ex['base_output'][:220].replace(chr(10), ' ')}")
lines.append(f"  - new: {ex['new_output'][:220].replace(chr(10), ' ')}")
lines.append("")
summary_path.write_text("\n".join(lines), encoding="utf-8")

def main() -> int:
ap = argparse.ArgumentParser()
ap.add_argument("--output-root", required=True)
args = ap.parse_args()

```root = Path(args.output_root)cases = []for name in ("ns128", "ns512"):    case_dir = root / name    if case_dir.exists():        cases.append(build_case_summary(case_dir))if not cases:    raise FileNotFoundError(f"No case directories found under {root}")summary_json = root / "summary.json"summary_md = root / "summary.md"payload = {    "overall_verdict": overall_verdict(cases),    "cases": cases,}summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")render_md(cases, summary_md)print(summary_md)print(summary_json)return 0
root = Path(args.output_root)cases = []for name in ("ns128", "ns512"):    case_dir = root / name    if case_dir.exists():        cases.append(build_case_summary(case_dir))if not cases:    raise FileNotFoundError(f"No case directories found under {root}")summary_json = root / "summary.json"summary_md = root / "summary.md"payload = {    "overall_verdict": overall_verdict(cases),    "cases": cases,}summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")render_md(cases, summary_md)print(summary_md)print(summary_json)return 0
```

if **name** == "**main**":
raise SystemExit(main())
EOF

cat > scripts/run_full_suite.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

usage() {
cat <<USAGE
Usage:
$0 --repo-root PATH --base-commit REV --new-commit REV --gpu 0 [options]

Options:
--model-id MODEL            default: Qwen/Qwen3.5-4B-Base
--tasks CSV                 default: wikitext,hellaswag,winogrande,piqa
--output-root PATH          default: ~/gptq_ab_results
--small-nsamples INT        default: 128
--large-nsamples INT        default: 512
--batch-size INT            default: 4
--seed INT                  default: 1234
--bits INT                  default: 4
--group-size INT            default: 128
USAGE
}

REPO_ROOT=""
BASE_COMMIT=""
NEW_COMMIT=""
GPU="0"
MODEL_ID="Qwen/Qwen3.5-4B-Base"
TASKS="wikitext,hellaswag,winogrande,piqa"
OUTPUT_ROOT="$HOME/gptq_ab_results"
SMALL_NSAMPLES=128
LARGE_NSAMPLES=512
BATCH_SIZE=4
SEED=1234
BITS=4
GROUP_SIZE=128

while [[ $# -gt 0 ]]; do
case "$1" in
--repo-root) REPO_ROOT="$2"; shift 2 ;;
--base-commit) BASE_COMMIT="$2"; shift 2 ;;
--new-commit) NEW_COMMIT="$2"; shift 2 ;;
--gpu) GPU="$2"; shift 2 ;;
--model-id) MODEL_ID="$2"; shift 2 ;;
--tasks) TASKS="$2"; shift 2 ;;
--output-root) OUTPUT_ROOT="$2"; shift 2 ;;
--small-nsamples) SMALL_NSAMPLES="$2"; shift 2 ;;
--large-nsamples) LARGE_NSAMPLES="$2"; shift 2 ;;
--batch-size) BATCH_SIZE="$2"; shift 2 ;;
--seed) SEED="$2"; shift 2 ;;
--bits) BITS="$2"; shift 2 ;;
--group-size) GROUP_SIZE="$2"; shift 2 ;;
-h|--help) usage; exit 0 ;;
*) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
esac
done

[[ -n "$REPO_ROOT" ]] || { echo "--repo-root is required" >&2; exit 1; }
[[ -n "$BASE_COMMIT" ]] || { echo "--base-commit is required" >&2; exit 1; }
[[ -n "$NEW_COMMIT" ]] || { echo "--new-commit is required" >&2; exit 1; }

REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
BASE_WT="${REPO_ROOT}_ab_base"
NEW_WT="${REPO_ROOT}_ab_new"
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT/#~/$HOME}"
mkdir -p "$OUTPUT_ROOT"

if [[ ! -d "$BASE_WT/.venv" || ! -d "$NEW_WT/.venv" ]]; then
echo "[error] missing virtualenvs. Run bootstrap_worktrees_and_envs.sh first." >&2
exit 1
fi

write_meta() {
cat > "$OUTPUT_ROOT/run_meta.json" <<JSON
{
"repo_root": "${REPO_ROOT}",
"base_worktree": "${BASE_WT}",
"new_worktree": "${NEW_WT}",
"base_commit_arg": "${BASE_COMMIT}",
"new_commit_arg": "${NEW_COMMIT}",
"gpu": "${GPU}",
"model_id": "${MODEL_ID}",
"tasks": "${TASKS}",
"small_nsamples": ${SMALL_NSAMPLES},
"large_nsamples": ${LARGE_NSAMPLES},
"batch_size": ${BATCH_SIZE},
"seed": ${SEED},
"bits": ${BITS},
"group_size": ${GROUP_SIZE}
}
JSON
}

run_lm_eval() {
if command -v lm-eval >/dev/null 2>&1; then
lm-eval "$@"
else
python -m lm_eval "$@"
fi
}

run_one_side() {
local side="$1"
local wt="$2"
local case_name="$3"
local nsamples="$4"

local case_root="$OUTPUT_ROOT/$case_name/$side"
local model_out="$case_root/model"
local eval_out="$case_root/eval"
local gen_out="$case_root/generations.json"
mkdir -p "$case_root" "$eval_out"

echo "[info] running $side $case_name in $wt"
pushd "$wt" >/dev/null
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false

python "$SELF_DIR/quantize_once.py" 
--model-id "$MODEL_ID" 
--out-dir "$model_out" 
--nsamples "$nsamples" 
--batch-size "$BATCH_SIZE" 
--seed "$SEED" 
--bits "$BITS" 
--group-size "$GROUP_SIZE" | tee "$case_root/quantize.log"

run_lm_eval 
--model hf 
--model_args "pretrained=$model_out,trust_remote_code=True" 
--tasks "$TASKS" 
--device cuda:0 
--batch_size auto:4 
--output_path "$eval_out" 
--seed "$SEED" 
--trust_remote_code | tee "$case_root/lm_eval.log"

python "$SELF_DIR/gen_compare.py" 
--model-path "$model_out" 
--out-json "$gen_out" | tee "$case_root/generation.log"

deactivate
popd >/dev/null
}

write_meta
run_one_side base "$BASE_WT" ns128 "$SMALL_NSAMPLES"
run_one_side new  "$NEW_WT"  ns128 "$SMALL_NSAMPLES"
run_one_side base "$BASE_WT" ns512 "$LARGE_NSAMPLES"
run_one_side new  "$NEW_WT"  ns512 "$LARGE_NSAMPLES"

python "$SELF_DIR/summarize_ab.py" --output-root "$OUTPUT_ROOT" | tee "$OUTPUT_ROOT/summary_paths.txt"

echo "[done] summary: $OUTPUT_ROOT/summary.md"
EOF

chmod +x scripts/*.sh scripts/*.py

```Then run:```bashcd gptq_pro_ab_suite./scripts/bootstrap_worktrees_and_envs.sh \  --repo-root ~/GPTQ-Pro \  --base-commit "3b6de40^" \  --new-commit "e6dee44" \  --torch-install-cmd 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128'./scripts/run_full_suite.sh \  --repo-root ~/GPTQ-Pro \  --base-commit "3b6de40^" \  --new-commit "e6dee44" \  --gpu 0 \  --model-id Qwen/Qwen3.5-4B-Base \  --tasks 'wikitext,hellaswag,winogrande,piqa' \  --output-root ~/gptq_ab_results
Then run:```bashcd gptq_pro_ab_suite./scripts/bootstrap_worktrees_and_envs.sh \  --repo-root ~/GPTQ-Pro \  --base-commit "3b6de40^" \  --new-commit "e6dee44" \  --torch-install-cmd 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128'./scripts/run_full_suite.sh \  --repo-root ~/GPTQ-Pro \  --base-commit "3b6de40^" \  --new-commit "e6dee44" \  --gpu 0 \  --model-id Qwen/Qwen3.5-4B-Base \  --tasks 'wikitext,hellaswag,winogrande,piqa' \  --output-root ~/gptq_ab_results
```

If you want, next I can give you a **patched v2** that also logs:

- `nvidia-smi` snapshots during quantization
- exact git SHAs from each worktree
- optional `gsm8k` / `arc_challenge`
- a stricter pass/fail verdict for the PR

---

### User:
yes please genius

---

### ChatGPT:
I’m upgrading the harness so it’s actually audit-friendly: GPU telemetry during quantization, pinned git SHAs in the report, optional heavier eval tasks, and a real pass/fail rule instead of a vague summary.

---

Replace these **3 files** with the versions below.

Then run the same bootstrap, and use the new `run_full_suite.sh` flags.

## scripts/quantize_once.py

```Python#!/usr/bin/env python3import argparseimport jsonimport osimport platformimport randomimport subprocessimport sysimport timefrom pathlib import Pathimport numpy as npimport torchfrom datasets import load_datasetfrom gptqmodel import GPTQModel, QuantizeConfigdef seed_all(seed: int) -> None:    random.seed(seed)    np.random.seed(seed)    torch.manual_seed(seed)    if torch.cuda.is_available():        torch.cuda.manual_seed_all(seed)def safe_git_sha() -> str | None:    try:        out = subprocess.check_output(            ["git", "rev-parse", "HEAD"],            stderr=subprocess.DEVNULL,            text=True,        ).strip()        return out    except Exception:        return Nonedef main() -> int:    ap = argparse.ArgumentParser()    ap.add_argument("--model-id", required=True)    ap.add_argument("--out-dir", required=True)    ap.add_argument("--nsamples", type=int, default=128)    ap.add_argument("--batch-size", type=int, default=4)    ap.add_argument("--seed", type=int, default=1234)    ap.add_argument("--bits", type=int, default=4)    ap.add_argument("--group-size", type=int, default=128)    ap.add_argument("--dataset", default="allenai/c4")    ap.add_argument("--dataset-file", default="en/c4-train.00001-of-01024.json.gz")    ap.add_argument("--tag", default=None)    args = ap.parse_args()    out_dir = Path(args.out_dir)    out_dir.mkdir(parents=True, exist_ok=True)    seed_all(args.seed)    if torch.cuda.is_available():        torch.cuda.empty_cache()        torch.cuda.reset_peak_memory_stats()    ds = load_dataset(        args.dataset,        data_files=args.dataset_file,        split="train",    ).select(range(args.nsamples))    calibration_dataset = ds["text"]    qcfg = QuantizeConfig.gptq_pro()    qcfg.bits = args.bits    qcfg.group_size = args.group_size    start = time.perf_counter()    model = GPTQModel.load(args.model_id, qcfg)    model.quantize(calibration_dataset, batch_size=args.batch_size)    elapsed = time.perf_counter() - start    model.save(str(out_dir))    peak_cuda_mem_bytes = None    cuda_name = None    if torch.cuda.is_available():        peak_cuda_mem_bytes = int(torch.cuda.max_memory_allocated())        cuda_name = torch.cuda.get_device_name(0)    metadata = {        "tag": args.tag,        "model_id": args.model_id,        "out_dir": str(out_dir),        "nsamples": args.nsamples,        "batch_size": args.batch_size,        "seed": args.seed,        "bits": args.bits,        "group_size": args.group_size,        "elapsed_sec": elapsed,        "peak_cuda_mem_bytes": peak_cuda_mem_bytes,        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,        "cuda_name": cuda_name,        "torch_version": torch.__version__,        "torch_cuda": torch.version.cuda,        "python_version": sys.version,        "platform": platform.platform(),        "hostname": platform.node(),        "cwd": os.getcwd(),        "git_sha": safe_git_sha(),        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),    }    with open(out_dir / "ab_quant_meta.json", "w", encoding="utf-8") as f:        json.dump(metadata, f, indent=2)    print(json.dumps(metadata, indent=2))    return 0if __name__ == "__main__":    raise SystemExit(main())
Python#!/usr/bin/env python3import argparseimport jsonimport osimport platformimport randomimport subprocessimport sysimport timefrom pathlib import Pathimport numpy as npimport torchfrom datasets import load_datasetfrom gptqmodel import GPTQModel, QuantizeConfigdef seed_all(seed: int) -> None:    random.seed(seed)    np.random.seed(seed)    torch.manual_seed(seed)    if torch.cuda.is_available():        torch.cuda.manual_seed_all(seed)def safe_git_sha() -> str | None:    try:        out = subprocess.check_output(            ["git", "rev-parse", "HEAD"],            stderr=subprocess.DEVNULL,            text=True,        ).strip()        return out    except Exception:        return Nonedef main() -> int:    ap = argparse.ArgumentParser()    ap.add_argument("--model-id", required=True)    ap.add_argument("--out-dir", required=True)    ap.add_argument("--nsamples", type=int, default=128)    ap.add_argument("--batch-size", type=int, default=4)    ap.add_argument("--seed", type=int, default=1234)    ap.add_argument("--bits", type=int, default=4)    ap.add_argument("--group-size", type=int, default=128)    ap.add_argument("--dataset", default="allenai/c4")    ap.add_argument("--dataset-file", default="en/c4-train.00001-of-01024.json.gz")    ap.add_argument("--tag", default=None)    args = ap.parse_args()    out_dir = Path(args.out_dir)    out_dir.mkdir(parents=True, exist_ok=True)    seed_all(args.seed)    if torch.cuda.is_available():        torch.cuda.empty_cache()        torch.cuda.reset_peak_memory_stats()    ds = load_dataset(        args.dataset,        data_files=args.dataset_file,        split="train",    ).select(range(args.nsamples))    calibration_dataset = ds["text"]    qcfg = QuantizeConfig.gptq_pro()    qcfg.bits = args.bits    qcfg.group_size = args.group_size    start = time.perf_counter()    model = GPTQModel.load(args.model_id, qcfg)    model.quantize(calibration_dataset, batch_size=args.batch_size)    elapsed = time.perf_counter() - start    model.save(str(out_dir))    peak_cuda_mem_bytes = None    cuda_name = None    if torch.cuda.is_available():        peak_cuda_mem_bytes = int(torch.cuda.max_memory_allocated())        cuda_name = torch.cuda.get_device_name(0)    metadata = {        "tag": args.tag,        "model_id": args.model_id,        "out_dir": str(out_dir),        "nsamples": args.nsamples,        "batch_size": args.batch_size,        "seed": args.seed,        "bits": args.bits,        "group_size": args.group_size,        "elapsed_sec": elapsed,        "peak_cuda_mem_bytes": peak_cuda_mem_bytes,        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,        "cuda_name": cuda_name,        "torch_version": torch.__version__,        "torch_cuda": torch.version.cuda,        "python_version": sys.version,        "platform": platform.platform(),        "hostname": platform.node(),        "cwd": os.getcwd(),        "git_sha": safe_git_sha(),        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),    }    with open(out_dir / "ab_quant_meta.json", "w", encoding="utf-8") as f:        json.dump(metadata, f, indent=2)    print(json.dumps(metadata, indent=2))    return 0if __name__ == "__main__":    raise SystemExit(main())
```

## scripts/run_full_suite.sh

```Bash#!/usr/bin/env bashset -euo pipefailusage() {  cat <<USAGEUsage:  $0 --repo-root PATH --base-commit REV --new-commit REV --gpu 0 [options]Options:  --model-id MODEL                    default: Qwen/Qwen3.5-4B-Base  --tasks CSV                         default: wikitext,hellaswag,winogrande,piqa  --extra-tasks CSV                   default: ""  --output-root PATH                  default: ~/gptq_ab_results  --small-nsamples INT                default: 128  --large-nsamples INT                default: 512  --batch-size INT                    default: 4  --seed INT                          default: 1234  --bits INT                          default: 4  --group-size INT                    default: 128  --gpu-log-interval SEC              default: 2  --strict-max-slowdown-pct FLOAT     default: 25  --strict-max-mem-increase-pct FLOAT default: 20  --strict-allow-task-regressions INT default: 0  --strict-require-ns128-win          default: trueUSAGE}REPO_ROOT=""BASE_COMMIT=""NEW_COMMIT=""GPU="0"MODEL_ID="Qwen/Qwen3.5-4B-Base"TASKS="wikitext,hellaswag,winogrande,piqa"EXTRA_TASKS=""OUTPUT_ROOT="$HOME/gptq_ab_results"SMALL_NSAMPLES=128LARGE_NSAMPLES=512BATCH_SIZE=4SEED=1234BITS=4GROUP_SIZE=128GPU_LOG_INTERVAL=2STRICT_MAX_SLOWDOWN_PCT=25STRICT_MAX_MEM_INCREASE_PCT=20STRICT_ALLOW_TASK_REGRESSIONS=0STRICT_REQUIRE_NS128_WIN=truewhile [[ $# -gt 0 ]]; do  case "$1" in    --repo-root) REPO_ROOT="$2"; shift 2 ;;    --base-commit) BASE_COMMIT="$2"; shift 2 ;;    --new-commit) NEW_COMMIT="$2"; shift 2 ;;    --gpu) GPU="$2"; shift 2 ;;    --model-id) MODEL_ID="$2"; shift 2 ;;    --tasks) TASKS="$2"; shift 2 ;;    --extra-tasks) EXTRA_TASKS="$2"; shift 2 ;;    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;    --small-nsamples) SMALL_NSAMPLES="$2"; shift 2 ;;    --large-nsamples) LARGE_NSAMPLES="$2"; shift 2 ;;    --batch-size) BATCH_SIZE="$2"; shift 2 ;;    --seed) SEED="$2"; shift 2 ;;    --bits) BITS="$2"; shift 2 ;;    --group-size) GROUP_SIZE="$2"; shift 2 ;;    --gpu-log-interval) GPU_LOG_INTERVAL="$2"; shift 2 ;;    --strict-max-slowdown-pct) STRICT_MAX_SLOWDOWN_PCT="$2"; shift 2 ;;    --strict-max-mem-increase-pct) STRICT_MAX_MEM_INCREASE_PCT="$2"; shift 2 ;;    --strict-allow-task-regressions) STRICT_ALLOW_TASK_REGRESSIONS="$2"; shift 2 ;;    --strict-require-ns128-win) STRICT_REQUIRE_NS128_WIN="$2"; shift 2 ;;    -h|--help) usage; exit 0 ;;    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;  esacdone[[ -n "$REPO_ROOT" ]] || { echo "--repo-root is required" >&2; exit 1; }[[ -n "$BASE_COMMIT" ]] || { echo "--base-commit is required" >&2; exit 1; }[[ -n "$NEW_COMMIT" ]] || { echo "--new-commit is required" >&2; exit 1; }REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"BASE_WT="${REPO_ROOT}_ab_base"NEW_WT="${REPO_ROOT}_ab_new"SELF_DIR="$(cd "$(dirname "$0")" && pwd)"OUTPUT_ROOT="${OUTPUT_ROOT/#\~/$HOME}"mkdir -p "$OUTPUT_ROOT"if [[ ! -d "$BASE_WT/.venv" || ! -d "$NEW_WT/.venv" ]]; then  echo "[error] missing virtualenvs. Run bootstrap_worktrees_and_envs.sh first." >&2  exit 1fiFULL_TASKS="$TASKS"if [[ -n "$EXTRA_TASKS" ]]; then  FULL_TASKS="${FULL_TASKS},${EXTRA_TASKS}"fibase_sha="$(git -C "$BASE_WT" rev-parse HEAD)"new_sha="$(git -C "$NEW_WT" rev-parse HEAD)"write_meta() {  cat > "$OUTPUT_ROOT/run_meta.json" <<JSON{  "repo_root": "${REPO_ROOT}",  "base_worktree": "${BASE_WT}",  "new_worktree": "${NEW_WT}",  "base_commit_arg": "${BASE_COMMIT}",  "new_commit_arg": "${NEW_COMMIT}",  "base_git_sha": "${base_sha}",  "new_git_sha": "${new_sha}",  "gpu": "${GPU}",  "model_id": "${MODEL_ID}",  "tasks": "${TASKS}",  "extra_tasks": "${EXTRA_TASKS}",  "full_tasks": "${FULL_TASKS}",  "small_nsamples": ${SMALL_NSAMPLES},  "large_nsamples": ${LARGE_NSAMPLES},  "batch_size": ${BATCH_SIZE},  "seed": ${SEED},  "bits": ${BITS},  "group_size": ${GROUP_SIZE},  "gpu_log_interval_sec": ${GPU_LOG_INTERVAL},  "strict_max_slowdown_pct": ${STRICT_MAX_SLOWDOWN_PCT},  "strict_max_mem_increase_pct": ${STRICT_MAX_MEM_INCREASE_PCT},  "strict_allow_task_regressions": ${STRICT_ALLOW_TASK_REGRESSIONS},  "strict_require_ns128_win": ${STRICT_REQUIRE_NS128_WIN}}JSON}run_lm_eval() {  if command -v lm-eval >/dev/null 2>&1; then    lm-eval "$@"  else    python -m lm_eval "$@"  fi}start_gpu_logger() {  local logfile="$1"  local pidfile="$2"  if ! command -v nvidia-smi >/dev/null 2>&1; then    echo "[warn] nvidia-smi not found, skipping GPU telemetry" | tee -a "$logfile"    return 0  fi  {    echo "unix_ts,timestamp,index,name,util_gpu,util_mem,memory_used_mb,memory_total_mb,temp_c,power_w"    while true; do      ts="$(date +%s)"      row="$(nvidia-smi \        --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \        --format=csv,noheader,nounits \        -i "$GPU" 2>/dev/null || true)"      if [[ -n "$row" ]]; then        echo "${ts},${row}"      fi      sleep "$GPU_LOG_INTERVAL"    done  } >> "$logfile" &  echo $! > "$pidfile"}stop_gpu_logger() {  local pidfile="$1"  if [[ -f "$pidfile" ]]; then    pid="$(cat "$pidfile" || true)"    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then      kill "$pid" 2>/dev/null || true      wait "$pid" 2>/dev/null || true    fi    rm -f "$pidfile"  fi}run_one_side() {  local side="$1"  local wt="$2"  local case_name="$3"  local nsamples="$4"  local case_root="$OUTPUT_ROOT/$case_name/$side"  local model_out="$case_root/model"  local eval_out="$case_root/eval"  local gen_out="$case_root/generations.json"  local gpu_log="$case_root/gpu_telemetry.csv"  local gpu_pid="$case_root/gpu_telemetry.pid"  mkdir -p "$case_root" "$eval_out"  echo "[info] running $side $case_name in $wt"  pushd "$wt" >/dev/null  source .venv/bin/activate  export CUDA_VISIBLE_DEVICES="$GPU"  export TOKENIZERS_PARALLELISM=false  start_gpu_logger "$gpu_log" "$gpu_pid"  python "$SELF_DIR/quantize_once.py" \    --model-id "$MODEL_ID" \    --out-dir "$model_out" \    --nsamples "$nsamples" \    --batch-size "$BATCH_SIZE" \    --seed "$SEED" \    --bits "$BITS" \    --group-size "$GROUP_SIZE" \    --tag "${side}_${case_name}" | tee "$case_root/quantize.log"  stop_gpu_logger "$gpu_pid"  run_lm_eval \    --model hf \    --model_args "pretrained=$model_out,trust_remote_code=True" \    --tasks "$FULL_TASKS" \    --device cuda:0 \    --batch_size auto:4 \    --output_path "$eval_out" \    --seed "$SEED" \    --trust_remote_code | tee "$case_root/lm_eval.log"  python "$SELF_DIR/gen_compare.py" \    --model-path "$model_out" \    --out-json "$gen_out" | tee "$case_root/generation.log"  deactivate  popd >/dev/null}write_metarun_one_side base "$BASE_WT" ns128 "$SMALL_NSAMPLES"run_one_side new  "$NEW_WT"  ns128 "$SMALL_NSAMPLES"run_one_side base "$BASE_WT" ns512 "$LARGE_NSAMPLES"run_one_side new  "$NEW_WT"  ns512 "$LARGE_NSAMPLES"python "$SELF_DIR/summarize_ab.py" \  --output-root "$OUTPUT_ROOT" \  --strict-max-slowdown-pct "$STRICT_MAX_SLOWDOWN_PCT" \  --strict-max-mem-increase-pct "$STRICT_MAX_MEM_INCREASE_PCT" \  --strict-allow-task-regressions "$STRICT_ALLOW_TASK_REGRESSIONS" \  --strict-require-ns128-win "$STRICT_REQUIRE_NS128_WIN" | tee "$OUTPUT_ROOT/summary_paths.txt"echo "[done] summary: $OUTPUT_ROOT/summary.md"
Bash#!/usr/bin/env bashset -euo pipefailusage() {  cat <<USAGEUsage:  $0 --repo-root PATH --base-commit REV --new-commit REV --gpu 0 [options]Options:  --model-id MODEL                    default: Qwen/Qwen3.5-4B-Base  --tasks CSV                         default: wikitext,hellaswag,winogrande,piqa  --extra-tasks CSV                   default: ""  --output-root PATH                  default: ~/gptq_ab_results  --small-nsamples INT                default: 128  --large-nsamples INT                default: 512  --batch-size INT                    default: 4  --seed INT                          default: 1234  --bits INT                          default: 4  --group-size INT                    default: 128  --gpu-log-interval SEC              default: 2  --strict-max-slowdown-pct FLOAT     default: 25  --strict-max-mem-increase-pct FLOAT default: 20  --strict-allow-task-regressions INT default: 0  --strict-require-ns128-win          default: trueUSAGE}REPO_ROOT=""BASE_COMMIT=""NEW_COMMIT=""GPU="0"MODEL_ID="Qwen/Qwen3.5-4B-Base"TASKS="wikitext,hellaswag,winogrande,piqa"EXTRA_TASKS=""OUTPUT_ROOT="$HOME/gptq_ab_results"SMALL_NSAMPLES=128LARGE_NSAMPLES=512BATCH_SIZE=4SEED=1234BITS=4GROUP_SIZE=128GPU_LOG_INTERVAL=2STRICT_MAX_SLOWDOWN_PCT=25STRICT_MAX_MEM_INCREASE_PCT=20STRICT_ALLOW_TASK_REGRESSIONS=0STRICT_REQUIRE_NS128_WIN=truewhile [[ $# -gt 0 ]]; do  case "$1" in    --repo-root) REPO_ROOT="$2"; shift 2 ;;    --base-commit) BASE_COMMIT="$2"; shift 2 ;;    --new-commit) NEW_COMMIT="$2"; shift 2 ;;    --gpu) GPU="$2"; shift 2 ;;    --model-id) MODEL_ID="$2"; shift 2 ;;    --tasks) TASKS="$2"; shift 2 ;;    --extra-tasks) EXTRA_TASKS="$2"; shift 2 ;;    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;    --small-nsamples) SMALL_NSAMPLES="$2"; shift 2 ;;    --large-nsamples) LARGE_NSAMPLES="$2"; shift 2 ;;    --batch-size) BATCH_SIZE="$2"; shift 2 ;;    --seed) SEED="$2"; shift 2 ;;    --bits) BITS="$2"; shift 2 ;;    --group-size) GROUP_SIZE="$2"; shift 2 ;;    --gpu-log-interval) GPU_LOG_INTERVAL="$2"; shift 2 ;;    --strict-max-slowdown-pct) STRICT_MAX_SLOWDOWN_PCT="$2"; shift 2 ;;    --strict-max-mem-increase-pct) STRICT_MAX_MEM_INCREASE_PCT="$2"; shift 2 ;;    --strict-allow-task-regressions) STRICT_ALLOW_TASK_REGRESSIONS="$2"; shift 2 ;;    --strict-require-ns128-win) STRICT_REQUIRE_NS128_WIN="$2"; shift 2 ;;    -h|--help) usage; exit 0 ;;    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;  esacdone[[ -n "$REPO_ROOT" ]] || { echo "--repo-root is required" >&2; exit 1; }[[ -n "$BASE_COMMIT" ]] || { echo "--base-commit is required" >&2; exit 1; }[[ -n "$NEW_COMMIT" ]] || { echo "--new-commit is required" >&2; exit 1; }REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"BASE_WT="${REPO_ROOT}_ab_base"NEW_WT="${REPO_ROOT}_ab_new"SELF_DIR="$(cd "$(dirname "$0")" && pwd)"OUTPUT_ROOT="${OUTPUT_ROOT/#\~/$HOME}"mkdir -p "$OUTPUT_ROOT"if [[ ! -d "$BASE_WT/.venv" || ! -d "$NEW_WT/.venv" ]]; then  echo "[error] missing virtualenvs. Run bootstrap_worktrees_and_envs.sh first." >&2  exit 1fiFULL_TASKS="$TASKS"if [[ -n "$EXTRA_TASKS" ]]; then  FULL_TASKS="${FULL_TASKS},${EXTRA_TASKS}"fibase_sha="$(git -C "$BASE_WT" rev-parse HEAD)"new_sha="$(git -C "$NEW_WT" rev-parse HEAD)"write_meta() {  cat > "$OUTPUT_ROOT/run_meta.json" <<JSON{  "repo_root": "${REPO_ROOT}",  "base_worktree": "${BASE_WT}",  "new_worktree": "${NEW_WT}",  "base_commit_arg": "${BASE_COMMIT}",  "new_commit_arg": "${NEW_COMMIT}",  "base_git_sha": "${base_sha}",  "new_git_sha": "${new_sha}",  "gpu": "${GPU}",  "model_id": "${MODEL_ID}",  "tasks": "${TASKS}",  "extra_tasks": "${EXTRA_TASKS}",  "full_tasks": "${FULL_TASKS}",  "small_nsamples": ${SMALL_NSAMPLES},  "large_nsamples": ${LARGE_NSAMPLES},  "batch_size": ${BATCH_SIZE},  "seed": ${SEED},  "bits": ${BITS},  "group_size": ${GROUP_SIZE},  "gpu_log_interval_sec": ${GPU_LOG_INTERVAL},  "strict_max_slowdown_pct": ${STRICT_MAX_SLOWDOWN_PCT},  "strict_max_mem_increase_pct": ${STRICT_MAX_MEM_INCREASE_PCT},  "strict_allow_task_regressions": ${STRICT_ALLOW_TASK_REGRESSIONS},  "strict_require_ns128_win": ${STRICT_REQUIRE_NS128_WIN}}JSON}run_lm_eval() {  if command -v lm-eval >/dev/null 2>&1; then    lm-eval "$@"  else    python -m lm_eval "$@"  fi}start_gpu_logger() {  local logfile="$1"  local pidfile="$2"  if ! command -v nvidia-smi >/dev/null 2>&1; then    echo "[warn] nvidia-smi not found, skipping GPU telemetry" | tee -a "$logfile"    return 0  fi  {    echo "unix_ts,timestamp,index,name,util_gpu,util_mem,memory_used_mb,memory_total_mb,temp_c,power_w"    while true; do      ts="$(date +%s)"      row="$(nvidia-smi \        --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \        --format=csv,noheader,nounits \        -i "$GPU" 2>/dev/null || true)"      if [[ -n "$row" ]]; then        echo "${ts},${row}"      fi      sleep "$GPU_LOG_INTERVAL"    done  } >> "$logfile" &  echo $! > "$pidfile"}stop_gpu_logger() {  local pidfile="$1"  if [[ -f "$pidfile" ]]; then    pid="$(cat "$pidfile" || true)"    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then      kill "$pid" 2>/dev/null || true      wait "$pid" 2>/dev/null || true    fi    rm -f "$pidfile"  fi}run_one_side() {  local side="$1"  local wt="$2"  local case_name="$3"  local nsamples="$4"  local case_root="$OUTPUT_ROOT/$case_name/$side"  local model_out="$case_root/model"  local eval_out="$case_root/eval"  local gen_out="$case_root/generations.json"  local gpu_log="$case_root/gpu_telemetry.csv"  local gpu_pid="$case_root/gpu_telemetry.pid"  mkdir -p "$case_root" "$eval_out"  echo "[info] running $side $case_name in $wt"  pushd "$wt" >/dev/null  source .venv/bin/activate  export CUDA_VISIBLE_DEVICES="$GPU"  export TOKENIZERS_PARALLELISM=false  start_gpu_logger "$gpu_log" "$gpu_pid"  python "$SELF_DIR/quantize_once.py" \    --model-id "$MODEL_ID" \    --out-dir "$model_out" \    --nsamples "$nsamples" \    --batch-size "$BATCH_SIZE" \    --seed "$SEED" \    --bits "$BITS" \    --group-size "$GROUP_SIZE" \    --tag "${side}_${case_name}" | tee "$case_root/quantize.log"  stop_gpu_logger "$gpu_pid"  run_lm_eval \    --model hf \    --model_args "pretrained=$model_out,trust_remote_code=True" \    --tasks "$FULL_TASKS" \    --device cuda:0 \    --batch_size auto:4 \    --output_path "$eval_out" \    --seed "$SEED" \    --trust_remote_code | tee "$case_root/lm_eval.log"  python "$SELF_DIR/gen_compare.py" \    --model-path "$model_out" \    --out-json "$gen_out" | tee "$case_root/generation.log"  deactivate  popd >/dev/null}write_metarun_one_side base "$BASE_WT" ns128 "$SMALL_NSAMPLES"run_one_side new  "$NEW_WT"  ns128 "$SMALL_NSAMPLES"run_one_side base "$BASE_WT" ns512 "$LARGE_NSAMPLES"run_one_side new  "$NEW_WT"  ns512 "$LARGE_NSAMPLES"python "$SELF_DIR/summarize_ab.py" \  --output-root "$OUTPUT_ROOT" \  --strict-max-slowdown-pct "$STRICT_MAX_SLOWDOWN_PCT" \  --strict-max-mem-increase-pct "$STRICT_MAX_MEM_INCREASE_PCT" \  --strict-allow-task-regressions "$STRICT_ALLOW_TASK_REGRESSIONS" \  --strict-require-ns128-win "$STRICT_REQUIRE_NS128_WIN" | tee "$OUTPUT_ROOT/summary_paths.txt"echo "[done] summary: $OUTPUT_ROOT/summary.md"
```

## scripts/summarize_ab.py

```Python#!/usr/bin/env python3import argparseimport difflibimport jsonfrom pathlib import Pathfrom typing import Dict, List, Optional, TupleHIGHER_IS_BETTER = {"acc", "acc_norm", "exact_match", "mc1", "mc2", "f1", "bleu"}LOWER_IS_BETTER = {"word_perplexity", "perplexity", "byte_perplexity", "bpb", "bpc"}def load_json(path: Path):    with open(path, "r", encoding="utf-8") as f:        return json.load(f)def find_lm_eval_json(eval_dir: Path) -> Optional[Path]:    candidates = sorted(eval_dir.rglob("*.json"))    for p in candidates:        try:            data = load_json(p)        except Exception:            continue        if isinstance(data, dict) and "results" in data:            return p    return Nonedef extract_task_metrics(eval_json: Dict) -> Dict[str, Dict[str, float]]:    results = eval_json.get("results", {})    out: Dict[str, Dict[str, float]] = {}    for task, task_metrics in results.items():        if not isinstance(task_metrics, dict):            continue        flat: Dict[str, float] = {}        for k, v in task_metrics.items():            if isinstance(v, (int, float)):                flat[k] = float(v)        out[task] = flat    return outdef choose_primary_metric(task: str, metrics: Dict[str, float]) -> Tuple[Optional[str], Optional[float], Optional[str]]:    preferred = [        "acc_norm,none", "acc,none", "exact_match,strict-match", "exact_match,none",        "mc2,none", "mc1,none", "word_perplexity,none", "perplexity,none",        "byte_perplexity,none", "bpb,none", "bpc,none",        "acc_norm", "acc", "exact_match", "mc2", "mc1", "word_perplexity",        "perplexity", "byte_perplexity", "bpb", "bpc",    ]    for key in preferred:        if key in metrics:            short = key.split(",", 1)[0]            direction = "higher" if short in HIGHER_IS_BETTER else "lower" if short in LOWER_IS_BETTER else "unknown"            return key, metrics[key], direction    return None, None, Nonedef compare_generations(base_path: Path, new_path: Path) -> Dict:    base_rows = load_json(base_path)    new_rows = load_json(new_path)    exact = 0    ratios: List[float] = []    changed_examples = []    for b, n in zip(base_rows, new_rows):        br = b.get("output", "")        nr = n.get("output", "")        if br == nr:            exact += 1        ratio = difflib.SequenceMatcher(None, br, nr).ratio()        ratios.append(ratio)        if br != nr and len(changed_examples) < 3:            changed_examples.append({                "prompt": b.get("prompt", ""),                "base_output": br[:900],                "new_output": nr[:900],                "similarity": ratio,            })    return {        "num_prompts": len(base_rows),        "exact_match_count": exact,        "exact_match_rate": exact / max(1, len(base_rows)),        "avg_similarity": sum(ratios) / max(1, len(ratios)),        "changed_examples": changed_examples,    }def load_quant_meta(path: Path) -> Dict:    return load_json(path)def pct_change(new: Optional[float], base: Optional[float]) -> Optional[float]:    if new is None or base is None:        return None    if base == 0:        return None    return ((new - base) / base) * 100.0def build_case_summary(case_dir: Path) -> Dict:    base_quant = load_quant_meta(case_dir / "base" / "model" / "ab_quant_meta.json")    new_quant = load_quant_meta(case_dir / "new" / "model" / "ab_quant_meta.json")    base_eval_json_path = find_lm_eval_json(case_dir / "base" / "eval")    new_eval_json_path = find_lm_eval_json(case_dir / "new" / "eval")    if base_eval_json_path is None or new_eval_json_path is None:        raise FileNotFoundError(f"Could not find lm-eval JSON under {case_dir}")    base_eval = load_json(base_eval_json_path)    new_eval = load_json(new_eval_json_path)    base_metrics = extract_task_metrics(base_eval)    new_metrics = extract_task_metrics(new_eval)    task_summaries = []    all_tasks = sorted(set(base_metrics) | set(new_metrics))    new_wins = 0    base_wins = 0    ties = 0    for task in all_tasks:        b = base_metrics.get(task, {})        n = new_metrics.get(task, {})        metric_name, bval, direction = choose_primary_metric(task, b)        metric_name2, nval, direction2 = choose_primary_metric(task, n)        chosen = metric_name or metric_name2        direction = direction or direction2 or "unknown"        if chosen is not None:            bval = b.get(chosen, bval)            nval = n.get(chosen, nval)        delta = None        winner = "tie"        if bval is not None and nval is not None:            delta = nval - bval            if direction == "higher":                winner = "new" if nval > bval else "base" if bval > nval else "tie"            elif direction == "lower":                winner = "new" if nval < bval else "base" if bval < nval else "tie"        if winner == "new":            new_wins += 1        elif winner == "base":            base_wins += 1        else:            ties += 1        task_summaries.append({            "task": task,            "metric": chosen,            "direction": direction,            "base": bval,            "new": nval,            "delta_new_minus_base": delta,            "winner": winner,        })    gen = compare_generations(case_dir / "base" / "generations.json", case_dir / "new" / "generations.json")    quant_delta = {        "elapsed_sec_new_minus_base": new_quant["elapsed_sec"] - base_quant["elapsed_sec"],        "elapsed_pct_new_minus_base": pct_change(new_quant["elapsed_sec"], base_quant["elapsed_sec"]),        "peak_cuda_mem_bytes_new_minus_base": (            (new_quant.get("peak_cuda_mem_bytes") or 0) - (base_quant.get("peak_cuda_mem_bytes") or 0)        ),        "peak_cuda_mem_pct_new_minus_base": pct_change(            (new_quant.get("peak_cuda_mem_bytes") or 0),            (base_quant.get("peak_cuda_mem_bytes") or 0),        ),    }    return {        "case": case_dir.name,        "base_quant": base_quant,        "new_quant": new_quant,        "quant_delta": quant_delta,        "base_eval_json": str(base_eval_json_path),        "new_eval_json": str(new_eval_json_path),        "tasks": task_summaries,        "task_win_counts": {            "new": new_wins,            "base": base_wins,            "tie": ties,        },        "generation_compare": gen,        "gpu_logs": {            "base": str(case_dir / "base" / "gpu_telemetry.csv"),            "new": str(case_dir / "new" / "gpu_telemetry.csv"),        },    }def parse_bool(s: str) -> bool:    return str(s).strip().lower() in {"1", "true", "yes", "y", "on"}def strict_case_pass(    case: Dict,    max_slowdown_pct: float,    max_mem_increase_pct: float,    allow_task_regressions: int,    require_win: bool,) -> Tuple[bool, List[str]]:    reasons = []    slowdown_pct = case["quant_delta"].get("elapsed_pct_new_minus_base")    if slowdown_pct is not None and slowdown_pct > max_slowdown_pct:        reasons.append(f"slowdown {slowdown_pct:.2f}% > allowed {max_slowdown_pct:.2f}%")    mem_increase_pct = case["quant_delta"].get("peak_cuda_mem_pct_new_minus_base")    if mem_increase_pct is not None and mem_increase_pct > max_mem_increase_pct:        reasons.append(f"peak mem increase {mem_increase_pct:.2f}% > allowed {max_mem_increase_pct:.2f}%")    regressions = case["task_win_counts"]["base"]    if regressions > allow_task_regressions:        reasons.append(f"{regressions} task regressions > allowed {allow_task_regressions}")    if require_win and case["task_win_counts"]["new"] < 1:        reasons.append("no task wins for new build")    return (len(reasons) == 0, reasons)def strict_overall_verdict(    cases: List[Dict],    max_slowdown_pct: float,    max_mem_increase_pct: float,    allow_task_regressions: int,    require_ns128_win: bool,) -> Dict:    per_case = {}    overall_pass = True    for case in cases:        require_win = require_ns128_win if case["case"] == "ns128" else False        ok, reasons = strict_case_pass(            case,            max_slowdown_pct=max_slowdown_pct,            max_mem_increase_pct=max_mem_increase_pct,            allow_task_regressions=allow_task_regressions,            require_win=require_win,        )        per_case[case["case"]] = {            "pass": ok,            "reasons": reasons,        }        if not ok:            overall_pass = False    return {        "pass": overall_pass,        "per_case": per_case,    }def overall_soft_verdict(case_summaries: List[Dict]) -> str:    new_wins = 0    base_wins = 0    for case in case_summaries:        new_wins += case["task_win_counts"]["new"]        base_wins += case["task_win_counts"]["base"]    if new_wins > base_wins:        return "new looks better overall"    if base_wins > new_wins:        return "base looks better overall"    return "mixed / inconclusive"def render_md(case_summaries: List[Dict], summary_path: Path, strict_result: Dict, run_meta: Dict) -> None:    lines: List[str] = []    lines.append("# GPTQ-Pro A/B summary")    lines.append("")    lines.append(f"Soft verdict: **{overall_soft_verdict(case_summaries)}**")    lines.append(f"Strict verdict: **{'PASS' if strict_result['pass'] else 'FAIL'}**")    lines.append("")    lines.append("## Run metadata")    lines.append("")    lines.append(f"- base git sha: `{run_meta.get('base_git_sha')}`")    lines.append(f"- new git sha: `{run_meta.get('new_git_sha')}`")    lines.append(f"- model: `{run_meta.get('model_id')}`")    lines.append(f"- tasks: `{run_meta.get('full_tasks')}`")    lines.append(f"- GPU: `{run_meta.get('gpu')}`")    lines.append("")    lines.append("## Strict criteria")    lines.append("")    lines.append(f"- max slowdown pct: {run_meta.get('strict_max_slowdown_pct')}")    lines.append(f"- max peak mem increase pct: {run_meta.get('strict_max_mem_increase_pct')}")    lines.append(f"- allowed task regressions per case: {run_meta.get('strict_allow_task_regressions')}")    lines.append(f"- require ns128 to have at least one win: {run_meta.get('strict_require_ns128_win')}")    lines.append("")    for case in case_summaries:        strict_case = strict_result["per_case"][case["case"]]        lines.append(f"## {case['case']}")        lines.append("")        lines.append(f"- strict result: **{'PASS' if strict_case['pass'] else 'FAIL'}**")        if strict_case["reasons"]:            for r in strict_case["reasons"]:                lines.append(f"- reason: {r}")        lines.append("")        lines.append("### Quantization")        lines.append("")        lines.append(f"- base elapsed: {case['base_quant']['elapsed_sec']:.2f}s")        lines.append(f"- new elapsed: {case['new_quant']['elapsed_sec']:.2f}s")        lines.append(f"- delta sec (new - base): {case['quant_delta']['elapsed_sec_new_minus_base']:.2f}s")        lines.append(f"- delta pct (new - base): {case['quant_delta']['elapsed_pct_new_minus_base']}")        lines.append(f"- base peak CUDA mem: {case['base_quant'].get('peak_cuda_mem_bytes')}")        lines.append(f"- new peak CUDA mem: {case['new_quant'].get('peak_cuda_mem_bytes')}")        lines.append(f"- delta peak mem bytes: {case['quant_delta']['peak_cuda_mem_bytes_new_minus_base']}")        lines.append(f"- delta peak mem pct: {case['quant_delta']['peak_cuda_mem_pct_new_minus_base']}")        lines.append(f"- base GPU log: `{case['gpu_logs']['base']}`")        lines.append(f"- new GPU log: `{case['gpu_logs']['new']}`")        lines.append("")        lines.append("### Eval")        lines.append("")        lines.append(f"- task wins -> new: {case['task_win_counts']['new']}, base: {case['task_win_counts']['base']}, tie: {case['task_win_counts']['tie']}")        lines.append("")        lines.append("| task | metric | direction | base | new | delta (new-base) | winner |")        lines.append("|---|---|---:|---:|---:|---:|---|")        for t in case["tasks"]:            lines.append(                f"| {t['task']} | {t['metric']} | {t['direction']} | {t['base']} | {t['new']} | {t['delta_new_minus_base']} | {t['winner']} |"            )        lines.append("")        gc = case["generation_compare"]        lines.append("### Deterministic generations")        lines.append("")        lines.append(f"- exact matches: {gc['exact_match_count']} / {gc['num_prompts']}")        lines.append(f"- exact match rate: {gc['exact_match_rate']:.3f}")        lines.append(f"- average similarity: {gc['avg_similarity']:.3f}")        if gc["changed_examples"]:            lines.append("")            lines.append("#### Sample changed outputs")            lines.append("")            for ex in gc["changed_examples"]:                lines.append(f"- Prompt: `{ex['prompt']}`")                lines.append(f"  - similarity: {ex['similarity']:.3f}")                lines.append(f"  - base: {ex['base_output'][:220].replace(chr(10), ' ')}")                lines.append(f"  - new: {ex['new_output'][:220].replace(chr(10), ' ')}")        lines.append("")    summary_path.write_text("\n".join(lines), encoding="utf-8")def main() -> int:    ap = argparse.ArgumentParser()    ap.add_argument("--output-root", required=True)    ap.add_argument("--strict-max-slowdown-pct", type=float, default=25.0)    ap.add_argument("--strict-max-mem-increase-pct", type=float, default=20.0)    ap.add_argument("--strict-allow-task-regressions", type=int, default=0)    ap.add_argument("--strict-require-ns128-win", default="true")    args = ap.parse_args()    root = Path(args.output_root)    run_meta_path = root / "run_meta.json"    run_meta = load_json(run_meta_path) if run_meta_path.exists() else {}    cases = []    for name in ("ns128", "ns512"):        case_dir = root / name        if case_dir.exists():            cases.append(build_case_summary(case_dir))    if not cases:        raise FileNotFoundError(f"No case directories found under {root}")    strict_result = strict_overall_verdict(        cases=cases,        max_slowdown_pct=args.strict_max_slowdown_pct,        max_mem_increase_pct=args.strict_max_mem_increase_pct,        allow_task_regressions=args.strict_allow_task_regressions,        require_ns128_win=parse_bool(args.strict_require_ns128_win),    )    payload = {        "soft_verdict": overall_soft_verdict(cases),        "strict_verdict": strict_result,        "cases": cases,        "run_meta": run_meta,    }    summary_json = root / "summary.json"    summary_md = root / "summary.md"    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")    render_md(cases, summary_md, strict_result, run_meta)    print(summary_md)    print(summary_json)    return 0if __name__ == "__main__":    raise SystemExit(main())
Python#!/usr/bin/env python3import argparseimport difflibimport jsonfrom pathlib import Pathfrom typing import Dict, List, Optional, TupleHIGHER_IS_BETTER = {"acc", "acc_norm", "exact_match", "mc1", "mc2", "f1", "bleu"}LOWER_IS_BETTER = {"word_perplexity", "perplexity", "byte_perplexity", "bpb", "bpc"}def load_json(path: Path):    with open(path, "r", encoding="utf-8") as f:        return json.load(f)def find_lm_eval_json(eval_dir: Path) -> Optional[Path]:    candidates = sorted(eval_dir.rglob("*.json"))    for p in candidates:        try:            data = load_json(p)        except Exception:            continue        if isinstance(data, dict) and "results" in data:            return p    return Nonedef extract_task_metrics(eval_json: Dict) -> Dict[str, Dict[str, float]]:    results = eval_json.get("results", {})    out: Dict[str, Dict[str, float]] = {}    for task, task_metrics in results.items():        if not isinstance(task_metrics, dict):            continue        flat: Dict[str, float] = {}        for k, v in task_metrics.items():            if isinstance(v, (int, float)):                flat[k] = float(v)        out[task] = flat    return outdef choose_primary_metric(task: str, metrics: Dict[str, float]) -> Tuple[Optional[str], Optional[float], Optional[str]]:    preferred = [        "acc_norm,none", "acc,none", "exact_match,strict-match", "exact_match,none",        "mc2,none", "mc1,none", "word_perplexity,none", "perplexity,none",        "byte_perplexity,none", "bpb,none", "bpc,none",        "acc_norm", "acc", "exact_match", "mc2", "mc1", "word_perplexity",        "perplexity", "byte_perplexity", "bpb", "bpc",    ]    for key in preferred:        if key in metrics:            short = key.split(",", 1)[0]            direction = "higher" if short in HIGHER_IS_BETTER else "lower" if short in LOWER_IS_BETTER else "unknown"            return key, metrics[key], direction    return None, None, Nonedef compare_generations(base_path: Path, new_path: Path) -> Dict:    base_rows = load_json(base_path)    new_rows = load_json(new_path)    exact = 0    ratios: List[float] = []    changed_examples = []    for b, n in zip(base_rows, new_rows):        br = b.get("output", "")        nr = n.get("output", "")        if br == nr:            exact += 1        ratio = difflib.SequenceMatcher(None, br, nr).ratio()        ratios.append(ratio)        if br != nr and len(changed_examples) < 3:            changed_examples.append({                "prompt": b.get("prompt", ""),                "base_output": br[:900],                "new_output": nr[:900],                "similarity": ratio,            })    return {        "num_prompts": len(base_rows),        "exact_match_count": exact,        "exact_match_rate": exact / max(1, len(base_rows)),        "avg_similarity": sum(ratios) / max(1, len(ratios)),        "changed_examples": changed_examples,    }def load_quant_meta(path: Path) -> Dict:    return load_json(path)def pct_change(new: Optional[float], base: Optional[float]) -> Optional[float]:    if new is None or base is None:        return None    if base == 0:        return None    return ((new - base) / base) * 100.0def build_case_summary(case_dir: Path) -> Dict:    base_quant = load_quant_meta(case_dir / "base" / "model" / "ab_quant_meta.json")    new_quant = load_quant_meta(case_dir / "new" / "model" / "ab_quant_meta.json")    base_eval_json_path = find_lm_eval_json(case_dir / "base" / "eval")    new_eval_json_path = find_lm_eval_json(case_dir / "new" / "eval")    if base_eval_json_path is None or new_eval_json_path is None:        raise FileNotFoundError(f"Could not find lm-eval JSON under {case_dir}")    base_eval = load_json(base_eval_json_path)    new_eval = load_json(new_eval_json_path)    base_metrics = extract_task_metrics(base_eval)    new_metrics = extract_task_metrics(new_eval)    task_summaries = []    all_tasks = sorted(set(base_metrics) | set(new_metrics))    new_wins = 0    base_wins = 0    ties = 0    for task in all_tasks:        b = base_metrics.get(task, {})        n = new_metrics.get(task, {})        metric_name, bval, direction = choose_primary_metric(task, b)        metric_name2, nval, direction2 = choose_primary_metric(task, n)        chosen = metric_name or metric_name2        direction = direction or direction2 or "unknown"        if chosen is not None:            bval = b.get(chosen, bval)            nval = n.get(chosen, nval)        delta = None        winner = "tie"        if bval is not None and nval is not None:            delta = nval - bval            if direction == "higher":                winner = "new" if nval > bval else "base" if bval > nval else "tie"            elif direction == "lower":                winner = "new" if nval < bval else "base" if bval < nval else "tie"        if winner == "new":            new_wins += 1        elif winner == "base":            base_wins += 1        else:            ties += 1        task_summaries.append({            "task": task,            "metric": chosen,            "direction": direction,            "base": bval,            "new": nval,            "delta_new_minus_base": delta,            "winner": winner,        })    gen = compare_generations(case_dir / "base" / "generations.json", case_dir / "new" / "generations.json")    quant_delta = {        "elapsed_sec_new_minus_base": new_quant["elapsed_sec"] - base_quant["elapsed_sec"],        "elapsed_pct_new_minus_base": pct_change(new_quant["elapsed_sec"], base_quant["elapsed_sec"]),        "peak_cuda_mem_bytes_new_minus_base": (            (new_quant.get("peak_cuda_mem_bytes") or 0) - (base_quant.get("peak_cuda_mem_bytes") or 0)        ),        "peak_cuda_mem_pct_new_minus_base": pct_change(            (new_quant.get("peak_cuda_mem_bytes") or 0),            (base_quant.get("peak_cuda_mem_bytes") or 0),        ),    }    return {        "case": case_dir.name,        "base_quant": base_quant,        "new_quant": new_quant,        "quant_delta": quant_delta,        "base_eval_json": str(base_eval_json_path),        "new_eval_json": str(new_eval_json_path),        "tasks": task_summaries,        "task_win_counts": {            "new": new_wins,            "base": base_wins,            "tie": ties,        },        "generation_compare": gen,        "gpu_logs": {            "base": str(case_dir / "base" / "gpu_telemetry.csv"),            "new": str(case_dir / "new" / "gpu_telemetry.csv"),        },    }def parse_bool(s: str) -> bool:    return str(s).strip().lower() in {"1", "true", "yes", "y", "on"}def strict_case_pass(    case: Dict,    max_slowdown_pct: float,    max_mem_increase_pct: float,    allow_task_regressions: int,    require_win: bool,) -> Tuple[bool, List[str]]:    reasons = []    slowdown_pct = case["quant_delta"].get("elapsed_pct_new_minus_base")    if slowdown_pct is not None and slowdown_pct > max_slowdown_pct:        reasons.append(f"slowdown {slowdown_pct:.2f}% > allowed {max_slowdown_pct:.2f}%")    mem_increase_pct = case["quant_delta"].get("peak_cuda_mem_pct_new_minus_base")    if mem_increase_pct is not None and mem_increase_pct > max_mem_increase_pct:        reasons.append(f"peak mem increase {mem_increase_pct:.2f}% > allowed {max_mem_increase_pct:.2f}%")    regressions = case["task_win_counts"]["base"]    if regressions > allow_task_regressions:        reasons.append(f"{regressions} task regressions > allowed {allow_task_regressions}")    if require_win and case["task_win_counts"]["new"] < 1:        reasons.append("no task wins for new build")    return (len(reasons) == 0, reasons)def strict_overall_verdict(    cases: List[Dict],    max_slowdown_pct: float,    max_mem_increase_pct: float,    allow_task_regressions: int,    require_ns128_win: bool,) -> Dict:    per_case = {}    overall_pass = True    for case in cases:        require_win = require_ns128_win if case["case"] == "ns128" else False        ok, reasons = strict_case_pass(            case,            max_slowdown_pct=max_slowdown_pct,            max_mem_increase_pct=max_mem_increase_pct,            allow_task_regressions=allow_task_regressions,            require_win=require_win,        )        per_case[case["case"]] = {            "pass": ok,            "reasons": reasons,        }        if not ok:            overall_pass = False    return {        "pass": overall_pass,        "per_case": per_case,    }def overall_soft_verdict(case_summaries: List[Dict]) -> str:    new_wins = 0    base_wins = 0    for case in case_summaries:        new_wins += case["task_win_counts"]["new"]        base_wins += case["task_win_counts"]["base"]    if new_wins > base_wins:        return "new looks better overall"    if base_wins > new_wins:        return "base looks better overall"    return "mixed / inconclusive"def render_md(case_summaries: List[Dict], summary_path: Path, strict_result: Dict, run_meta: Dict) -> None:    lines: List[str] = []    lines.append("# GPTQ-Pro A/B summary")    lines.append("")    lines.append(f"Soft verdict: **{overall_soft_verdict(case_summaries)}**")    lines.append(f"Strict verdict: **{'PASS' if strict_result['pass'] else 'FAIL'}**")    lines.append("")    lines.append("## Run metadata")    lines.append("")    lines.append(f"- base git sha: `{run_meta.get('base_git_sha')}`")    lines.append(f"- new git sha: `{run_meta.get('new_git_sha')}`")    lines.append(f"- model: `{run_meta.get('model_id')}`")    lines.append(f"- tasks: `{run_meta.get('full_tasks')}`")    lines.append(f"- GPU: `{run_meta.get('gpu')}`")    lines.append("")    lines.append("## Strict criteria")    lines.append("")    lines.append(f"- max slowdown pct: {run_meta.get('strict_max_slowdown_pct')}")    lines.append(f"- max peak mem increase pct: {run_meta.get('strict_max_mem_increase_pct')}")    lines.append(f"- allowed task regressions per case: {run_meta.get('strict_allow_task_regressions')}")    lines.append(f"- require ns128 to have at least one win: {run_meta.get('strict_require_ns128_win')}")    lines.append("")    for case in case_summaries:        strict_case = strict_result["per_case"][case["case"]]        lines.append(f"## {case['case']}")        lines.append("")        lines.append(f"- strict result: **{'PASS' if strict_case['pass'] else 'FAIL'}**")        if strict_case["reasons"]:            for r in strict_case["reasons"]:                lines.append(f"- reason: {r}")        lines.append("")        lines.append("### Quantization")        lines.append("")        lines.append(f"- base elapsed: {case['base_quant']['elapsed_sec']:.2f}s")        lines.append(f"- new elapsed: {case['new_quant']['elapsed_sec']:.2f}s")        lines.append(f"- delta sec (new - base): {case['quant_delta']['elapsed_sec_new_minus_base']:.2f}s")        lines.append(f"- delta pct (new - base): {case['quant_delta']['elapsed_pct_new_minus_base']}")        lines.append(f"- base peak CUDA mem: {case['base_quant'].get('peak_cuda_mem_bytes')}")        lines.append(f"- new peak CUDA mem: {case['new_quant'].get('peak_cuda_mem_bytes')}")        lines.append(f"- delta peak mem bytes: {case['quant_delta']['peak_cuda_mem_bytes_new_minus_base']}")        lines.append(f"- delta peak mem pct: {case['quant_delta']['peak_cuda_mem_pct_new_minus_base']}")        lines.append(f"- base GPU log: `{case['gpu_logs']['base']}`")        lines.append(f"- new GPU log: `{case['gpu_logs']['new']}`")        lines.append("")        lines.append("### Eval")        lines.append("")        lines.append(f"- task wins -> new: {case['task_win_counts']['new']}, base: {case['task_win_counts']['base']}, tie: {case['task_win_counts']['tie']}")        lines.append("")        lines.append("| task | metric | direction | base | new | delta (new-base) | winner |")        lines.append("|---|---|---:|---:|---:|---:|---|")        for t in case["tasks"]:            lines.append(                f"| {t['task']} | {t['metric']} | {t['direction']} | {t['base']} | {t['new']} | {t['delta_new_minus_base']} | {t['winner']} |"            )        lines.append("")        gc = case["generation_compare"]        lines.append("### Deterministic generations")        lines.append("")        lines.append(f"- exact matches: {gc['exact_match_count']} / {gc['num_prompts']}")        lines.append(f"- exact match rate: {gc['exact_match_rate']:.3f}")        lines.append(f"- average similarity: {gc['avg_similarity']:.3f}")        if gc["changed_examples"]:            lines.append("")            lines.append("#### Sample changed outputs")            lines.append("")            for ex in gc["changed_examples"]:                lines.append(f"- Prompt: `{ex['prompt']}`")                lines.append(f"  - similarity: {ex['similarity']:.3f}")                lines.append(f"  - base: {ex['base_output'][:220].replace(chr(10), ' ')}")                lines.append(f"  - new: {ex['new_output'][:220].replace(chr(10), ' ')}")        lines.append("")    summary_path.write_text("\n".join(lines), encoding="utf-8")def main() -> int:    ap = argparse.ArgumentParser()    ap.add_argument("--output-root", required=True)    ap.add_argument("--strict-max-slowdown-pct", type=float, default=25.0)    ap.add_argument("--strict-max-mem-increase-pct", type=float, default=20.0)    ap.add_argument("--strict-allow-task-regressions", type=int, default=0)    ap.add_argument("--strict-require-ns128-win", default="true")    args = ap.parse_args()    root = Path(args.output_root)    run_meta_path = root / "run_meta.json"    run_meta = load_json(run_meta_path) if run_meta_path.exists() else {}    cases = []    for name in ("ns128", "ns512"):        case_dir = root / name        if case_dir.exists():            cases.append(build_case_summary(case_dir))    if not cases:        raise FileNotFoundError(f"No case directories found under {root}")    strict_result = strict_overall_verdict(        cases=cases,        max_slowdown_pct=args.strict_max_slowdown_pct,        max_mem_increase_pct=args.strict_max_mem_increase_pct,        allow_task_regressions=args.strict_allow_task_regressions,        require_ns128_win=parse_bool(args.strict_require_ns128_win),    )    payload = {        "soft_verdict": overall_soft_verdict(cases),        "strict_verdict": strict_result,        "cases": cases,        "run_meta": run_meta,    }    summary_json = root / "summary.json"    summary_md = root / "summary.md"    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")    render_md(cases, summary_md, strict_result, run_meta)    print(summary_md)    print(summary_json)    return 0if __name__ == "__main__":    raise SystemExit(main())
```

## Run it

Your bootstrap command stays the same:

```Bashcd gptq_pro_ab_suite./scripts/bootstrap_worktrees_and_envs.sh \  --repo-root ~/GPTQ-Pro \  --base-commit "3b6de40^" \  --new-commit "e6dee44" \  --torch-install-cmd 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128'
Bashcd gptq_pro_ab_suite./scripts/bootstrap_worktrees_and_envs.sh \  --repo-root ~/GPTQ-Pro \  --base-commit "3b6de40^" \  --new-commit "e6dee44" \  --torch-install-cmd 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128'
```

Then run the stricter suite like this:

```Bash./scripts/run_full_suite.sh \  --repo-root ~/GPTQ-Pro \  --base-commit "3b6de40^" \  --new-commit "e6dee44" \  --gpu 0 \  --model-id Qwen/Qwen3.5-4B-Base \  --tasks 'wikitext,hellaswag,winogrande,piqa' \  --extra-tasks 'arc_challenge,gsm8k' \  --output-root ~/gptq_ab_results \  --gpu-log-interval 2 \  --strict-max-slowdown-pct 25 \  --strict-max-mem-increase-pct 20 \  --strict-allow-task-regressions 0 \  --strict-require-ns128-win true
Bash./scripts/run_full_suite.sh \  --repo-root ~/GPTQ-Pro \  --base-commit "3b6de40^" \  --new-commit "e6dee44" \  --gpu 0 \  --model-id Qwen/Qwen3.5-4B-Base \  --tasks 'wikitext,hellaswag,winogrande,piqa' \  --extra-tasks 'arc_challenge,gsm8k' \  --output-root ~/gptq_ab_results \  --gpu-log-interval 2 \  --strict-max-slowdown-pct 25 \  --strict-max-mem-increase-pct 20 \  --strict-allow-task-regressions 0 \  --strict-require-ns128-win true
```

## What this adds

You asked for 4 upgrades. This version gives you all 4:

- `nvidia-smi` snapshots during **quantization** into:

`~/gptq_ab_results/ns128/base/gpu_telemetry.csv`

`~/gptq_ab_results/ns128/new/gpu_telemetry.csv`

and same for `ns512`
- `~/gptq_ab_results/ns128/base/gpu_telemetry.csv`
- `~/gptq_ab_results/ns128/new/gpu_telemetry.csv`
- and same for `ns512`
- exact git SHAs from each worktree in:

`~/gptq_ab_results/run_meta.json`

`~/gptq_ab_results/summary.md`
- `~/gptq_ab_results/run_meta.json`
- `~/gptq_ab_results/summary.md`
- optional heavier tasks:

`--extra-tasks 'arc_challenge,gsm8k'`
- `--extra-tasks 'arc_challenge,gsm8k'`
- strict pass/fail verdict:

fail if too slow

fail if peak memory grows too much

fail if task regressions exceed threshold

optionally require at least one win in the **ns128** stress case
- fail if too slow
- fail if peak memory grows too much
- fail if task regressions exceed threshold
- optionally require at least one win in the **ns128** stress case

## What I’d use as the real decision rule

For your PR, I’d judge it like this:

- **PASS** if `ns128` gets at least one meaningful win and no regressions, while slowdown stays under 25%
- **PASS** if `ns512` is neutral
- **FAIL** if `ns128` is flat or worse and quantization gets slower for no gain

Because this PR is supposed to help the **low-sample fallback path**, `ns128` matters much more than `ns512`.

Send me your `summary.md` or `summary.json` after the run and I’ll interpret whether the merged default was actually justified.

---

### User:
how can we add an external judge llm to classify the level of quant that should be applied to each lay4er?

---

### ChatGPT:
I’m checking what the current GPTQ/quantization toolchain already supports so the design fits real hooks instead of fighting the repo.

---

Yes — and the clean way is **not** “LLM reads raw tensors and vibes out a bitwidth.”
Use the external judge as a **policy model over measured layer features**, then map its output into `QuantizeConfig.dynamic`, because GPTQModel already supports per-module overrides and even skipping specific modules from quantization entirely. [GitHub+1](https://github.com/ModelCloud/GPTQModel)

### The right architecture

Use a **two-stage controller**:

1. **Probe pass**
Run the model on a calibration set and compute per-layer features such as:

- output reconstruction error after temporary W4/W8 quantization
- activation sensitivity
- weight kurtosis / outlier score
- Hessian-ish proxy or input covariance proxy
- layer type (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj`, `gate_proj`)
- FLOPs / parameter count / memory cost of upgrading that layer
2. **Judge pass**
Send only those features to the external judge LLM and force it to choose from a **small allowed action set**, for example:

- `skip`
- `w8_g128`
- `w4_g128`
- `w4_g64`
- `w4_g128_no_desc_act`
- `w4_g128_mse_on`
3. **Compile pass**
Convert the judge output into GPTQModel `dynamic` regex overrides.
4. **Quantize + eval**
Quantize once with that per-layer plan, then score with `lm-eval`.

This is a good fit because GPTQModel already documents dynamic per-module mixed quantization, with overrides for `bits`, `group_size`, `sym`, `desc_act`, `mse`, and `pack_dtype`, plus negative matching to skip modules. [GitHub](https://github.com/ModelCloud/GPTQModel)

### Why this makes sense

Layer sensitivity is not uniform. Recent work specifically argues that **activation sensitivity** and **weight kurtosis** are useful signals for finding layers that need more precision, and reported up to **9% lower perplexity** with only about a **2% memory increase** when extra budget is targeted to sensitive layers. AWQ also rests on the idea that only a small fraction of weights/channels are especially salient, while GPTQ uses second-order information from calibration data to reduce quantization error. [arXiv+2ar5iv+2](https://arxiv.org/html/2503.06518v1)

### What I would actually build

#### Option A — pragmatic and good

Use the judge LLM only for **tier classification**.

Input to judge:

```JSON{  "layer": "model.layers.18.mlp.down_proj",  "type": "down_proj",  "params_millions": 37.7,  "w4_recon_mse": 0.0182,  "w8_recon_mse": 0.0021,  "sensitivity_score": 0.91,  "kurtosis": 14.7,  "activation_outlier_rate": 0.032,  "delta_ppl_if_w4_est": 0.19,  "memory_cost_w8_mb": 72,  "latency_cost_w8_ms": 0.4}
JSON{  "layer": "model.layers.18.mlp.down_proj",  "type": "down_proj",  "params_millions": 37.7,  "w4_recon_mse": 0.0182,  "w8_recon_mse": 0.0021,  "sensitivity_score": 0.91,  "kurtosis": 14.7,  "activation_outlier_rate": 0.032,  "delta_ppl_if_w4_est": 0.19,  "memory_cost_w8_mb": 72,  "latency_cost_w8_ms": 0.4}
```

Constrained output:

```JSON{  "action": "w8_g128",  "reason_code": "high_sensitivity_high_kurtosis"}
JSON{  "action": "w8_g128",  "reason_code": "high_sensitivity_high_kurtosis"}
```

Then compile:

```Pythondynamic = {    r".*\.18\..*down_proj.*": {"bits": 8, "group_size": 128},}
Pythondynamic = {    r".*\.18\..*down_proj.*": {"bits": 8, "group_size": 128},}
```

#### Option B — stronger and less gimmicky

Use the LLM to **propose**, but let a **scorer** accept/reject.

Pipeline:

- judge LLM emits plan
- deterministic scorer estimates total memory budget and predicted risk
- if over budget, auto-downgrade least sensitive upgrades
- quantize + eval
- feed results back into a small dataset for later fine-tuning of the judge

That gives you something closer to a real controller instead of a fancy prompt wrapper.

### Best action space for Qwen 3.5 4B

For a first version, do **not** let the judge choose arbitrary numeric values. Keep it discrete:

- `skip`
- `w8_g128`
- `w4_g64`
- `w4_g128`
- `w4_g128_mse`
- `w4_g128_no_desc_act`

That matches GPTQ-style knobs that are actually exposed today through dynamic overrides. [GitHub](https://github.com/ModelCloud/GPTQModel)

### Which layers usually deserve special handling

As a starting heuristic before the judge learns anything:

- `lm_head`: often keep higher precision or skip quantizing if quality matters a lot
- `o_proj`, `down_proj`, `gate_proj`: commonly more fragile than some other linears
- early and late transformer blocks: often more sensitive than middle blocks
- layers with high activation sensitivity / high kurtosis: promote to W8 or tighter group size
- layers with low sensitivity and low outlier score: leave at W4/G128

That heuristic is exactly the sort of structure your judge should learn from the probe features, rather than inventing from names alone. The layer-sensitive paper’s core point is that sensitivity varies materially by layer and can be exploited for mixed allocation. [arXiv](https://arxiv.org/html/2503.06518v1)

## Concrete implementation plan

### 1) Add a probe script

For each linear module in Qwen 3.5 4B:

- capture FP activations on a small calibration set
- run temporary quantization candidates:

W4/G128

W4/G64

W8/G128
- W4/G128
- W4/G64
- W8/G128
- compute:

output MSE

cosine similarity

max absolute error

activation outlier rate

weight kurtosis
- output MSE
- cosine similarity
- max absolute error
- activation outlier rate
- weight kurtosis
- save one JSON row per layer

Example row:

```JSON{  "name": "model.layers.7.self_attn.o_proj",  "type": "o_proj",  "block": 7,  "params": 16777216,  "fp_norm": 92.1,  "w4g128_mse": 0.0127,  "w4g64_mse": 0.0084,  "w8g128_mse": 0.0012,  "cos_w4g128": 0.984,  "cos_w4g64": 0.989,  "cos_w8g128": 0.998,  "kurtosis": 11.4,  "act_outlier_rate": 0.027,  "memory_mb_w4g128": 8.2,  "memory_mb_w8g128": 16.4}
JSON{  "name": "model.layers.7.self_attn.o_proj",  "type": "o_proj",  "block": 7,  "params": 16777216,  "fp_norm": 92.1,  "w4g128_mse": 0.0127,  "w4g64_mse": 0.0084,  "w8g128_mse": 0.0012,  "cos_w4g128": 0.984,  "cos_w4g64": 0.989,  "cos_w8g128": 0.998,  "kurtosis": 11.4,  "act_outlier_rate": 0.027,  "memory_mb_w4g128": 8.2,  "memory_mb_w8g128": 16.4}
```

### 2) Ask the judge for a plan

Prompt pattern:

```You are a quantization policy judge.Goal:Choose the lowest-precision setting for each layer that preserves model quality under a global memory budget.Allowed actions:- skip- w8_g128- w4_g64- w4_g128- w4_g128_mse- w4_g128_no_desc_actRules:- Prefer w4_g128 unless the evidence suggests fragility.- Use w8_g128 for layers with high sensitivity and large estimated quality loss at w4.- Use skip only for extreme outliers.- Keep total upgraded memory under the supplied budget.- Return only JSON.Input:{... layer table ...}
You are a quantization policy judge.Goal:Choose the lowest-precision setting for each layer that preserves model quality under a global memory budget.Allowed actions:- skip- w8_g128- w4_g64- w4_g128- w4_g128_mse- w4_g128_no_desc_actRules:- Prefer w4_g128 unless the evidence suggests fragility.- Use w8_g128 for layers with high sensitivity and large estimated quality loss at w4.- Use skip only for extreme outliers.- Keep total upgraded memory under the supplied budget.- Return only JSON.Input:{... layer table ...}
```

Return:

```JSON{  "budget_mb_extra": 220,  "assignments": [    {"name": "model.layers.0.self_attn.o_proj", "action": "w8_g128"},    {"name": "model.layers.1.mlp.down_proj", "action": "w4_g64"},    {"name": "model.layers.2.self_attn.q_proj", "action": "w4_g128"}  ]}
JSON{  "budget_mb_extra": 220,  "assignments": [    {"name": "model.layers.0.self_attn.o_proj", "action": "w8_g128"},    {"name": "model.layers.1.mlp.down_proj", "action": "w4_g64"},    {"name": "model.layers.2.self_attn.q_proj", "action": "w4_g128"}  ]}
```

### 3) Compile to GPTQModel dynamic

```PythonACTION_MAP = {    "w8_g128": {"bits": 8, "group_size": 128},    "w4_g64": {"bits": 4, "group_size": 64},    "w4_g128": {"bits": 4, "group_size": 128},    "w4_g128_mse": {"bits": 4, "group_size": 128, "mse": True},    "w4_g128_no_desc_act": {"bits": 4, "group_size": 128, "desc_act": False},}def compile_dynamic(assignments):    dynamic = {}    for row in assignments:        name = row["name"]        action = row["action"]        pattern = rf"{name.replace('.', r'\.')}"        if action == "skip":            dynamic[rf"-:{pattern}"] = {}        else:            dynamic[rf"+:{pattern}"] = ACTION_MAP[action]    return dynamic
PythonACTION_MAP = {    "w8_g128": {"bits": 8, "group_size": 128},    "w4_g64": {"bits": 4, "group_size": 64},    "w4_g128": {"bits": 4, "group_size": 128},    "w4_g128_mse": {"bits": 4, "group_size": 128, "mse": True},    "w4_g128_no_desc_act": {"bits": 4, "group_size": 128, "desc_act": False},}def compile_dynamic(assignments):    dynamic = {}    for row in assignments:        name = row["name"]        action = row["action"]        pattern = rf"{name.replace('.', r'\.')}"        if action == "skip":            dynamic[rf"-:{pattern}"] = {}        else:            dynamic[rf"+:{pattern}"] = ACTION_MAP[action]    return dynamic
```

And then:

```Pythonfrom gptqmodel import QuantizeConfigqcfg = QuantizeConfig.gptq_pro()qcfg.dynamic = compile_dynamic(plan["assignments"])
Pythonfrom gptqmodel import QuantizeConfigqcfg = QuantizeConfig.gptq_pro()qcfg.dynamic = compile_dynamic(plan["assignments"])
```

GPTQModel explicitly supports positive matches for overrides and negative matches for skipping modules. [GitHub](https://github.com/ModelCloud/GPTQModel)

## What I would *not* do

I would **not** let the LLM directly see:

- raw tensors
- full histograms for hundreds of layers in one prompt
- unconstrained actions like “choose any bits from 2–16”
- free-form natural language reasoning that then gets parsed loosely

That turns the system into prompt soup.

Use the LLM as:

- a **budgeted classifier**
- with **small discrete actions**
- over **measured numeric features**
- followed by **deterministic compilation**

### Better than an LLM in v1

Honestly, the strongest v1 is:

- collect probe data
- train a tiny XGBoost / LightGBM / logistic model to predict:

“safe at W4/G128”

“needs tighter grouping”

“needs W8”

“skip”
- “safe at W4/G128”
- “needs tighter grouping”
- “needs W8”
- “skip”
- optionally use the LLM only to explain decisions

That is usually more stable than a judge LLM. The LLM becomes useful when you want to combine many signals plus a **global budget constraint**, or when you want an agent loop that proposes and revises mixed-precision recipes. Mixed-precision/non-uniform quantization is already an active direction in current toolchains like AutoRound and LLM Compressor. [GitHub+1](https://github.com/intel/auto-round)

## Best practical roadmap for your setup

For **Qwen 3.5 4B on 1x3090**, I’d do this:

1. Build probe dataset on 128–256 calibration samples.
2. Start with only these module types:

- `o_proj`
- `down_proj`
- `gate_proj`
- `lm_head`
3. Judge action space:

- `w4_g128`
- `w4_g64`
- `w8_g128`
- `skip`
4. Compile to `QuantizeConfig.dynamic`
5. Run your existing A/B suite
6. Keep the judge only if it beats simple heuristics

That gets you a real mixed-precision controller fast, using hooks GPTQModel already exposes. [GitHub+1](https://github.com/ModelCloud/GPTQModel)

I can write you the **full probe + judge + dynamic-compiler scaffold** next.

---

### User:
provide full please for me

---

### ChatGPT:
Yes. Paste this into an empty folder. It creates a full scaffold that does:

- probe per-layer sensitivity from real activations
- ask an external OpenAI-compatible judge LLM for discrete actions
- enforce a hard extra-memory budget deterministically
- compile the result into `QuantizeConfig.dynamic`

That matches GPTQModel’s documented per-module mixed-quantization and `-:` skip controls. [GitHub](https://github.com/ModelCloud/GPTQModel)

```Bashmkdir -p quant_judge_scaffold && cd quant_judge_scaffoldcat > README.md <<'EOF'# Quant Judge Scaffold for GPTQModelThis project adds an external "judge" LLM on top of GPTQModel mixed quantization.Pipeline:1. Probe layer/module sensitivity on a calibration set using the FP model.2. Compute proxy quantization errors for a small discrete action set.3. Ask an external judge LLM to choose one action per module.4. Enforce a deterministic extra-memory budget.5. Compile the plan into `QuantizeConfig.dynamic`.6. Quantize with GPTQModel using the resulting mixed plan.Supported actions in this scaffold:- w4_g128   (base)- w4_g64- w8_g128- skip      (do not quantize that module)Default targeted modules:- o_proj- down_proj- gate_proj- lm_headYou can extend to q_proj/k_proj/v_proj/up_proj later, but start conservative.## InstallInstall torch yourself to match your CUDA stack, then:```bashpip install -r requirements.txt
Bashmkdir -p quant_judge_scaffold && cd quant_judge_scaffoldcat > README.md <<'EOF'# Quant Judge Scaffold for GPTQModelThis project adds an external "judge" LLM on top of GPTQModel mixed quantization.Pipeline:1. Probe layer/module sensitivity on a calibration set using the FP model.2. Compute proxy quantization errors for a small discrete action set.3. Ask an external judge LLM to choose one action per module.4. Enforce a deterministic extra-memory budget.5. Compile the plan into `QuantizeConfig.dynamic`.6. Quantize with GPTQModel using the resulting mixed plan.Supported actions in this scaffold:- w4_g128   (base)- w4_g64- w8_g128- skip      (do not quantize that module)Default targeted modules:- o_proj- down_proj- gate_proj- lm_headYou can extend to q_proj/k_proj/v_proj/up_proj later, but start conservative.## InstallInstall torch yourself to match your CUDA stack, then:```bashpip install -r requirements.txt
```

## Probe only

```Bashpython -m quant_judge.pipeline probe \  --model-id Qwen/Qwen3.5-4B-Base \  --out-dir ./runs/qwen35_4b_probe \  --dataset-name wikitext \  --dataset-config wikitext-2-raw-v1 \  --split train \  --text-field text \  --num-texts 64 \  --max-length 256 \  --max-rows-per-module 128 \  --include-modules o_proj,down_proj,gate_proj,lm_head
Bashpython -m quant_judge.pipeline probe \  --model-id Qwen/Qwen3.5-4B-Base \  --out-dir ./runs/qwen35_4b_probe \  --dataset-name wikitext \  --dataset-config wikitext-2-raw-v1 \  --split train \  --text-field text \  --num-texts 64 \  --max-length 256 \  --max-rows-per-module 128 \  --include-modules o_proj,down_proj,gate_proj,lm_head
```

## Plan with heuristic fallback only

```Bashpython -m quant_judge.pipeline plan \  --features-jsonl ./runs/qwen35_4b_probe/probe_features.jsonl \  --plan-out ./runs/qwen35_4b_probe/plan.json \  --dynamic-out ./runs/qwen35_4b_probe/dynamic.json \  --budget-mb-extra 220 \  --heuristic-only
Bashpython -m quant_judge.pipeline plan \  --features-jsonl ./runs/qwen35_4b_probe/probe_features.jsonl \  --plan-out ./runs/qwen35_4b_probe/plan.json \  --dynamic-out ./runs/qwen35_4b_probe/dynamic.json \  --budget-mb-extra 220 \  --heuristic-only
```

## Plan with external judge

Use any OpenAI-compatible endpoint, local or remote.

```Bashexport JUDGE_API_KEY=your_key_herepython -m quant_judge.pipeline plan \  --features-jsonl ./runs/qwen35_4b_probe/probe_features.jsonl \  --plan-out ./runs/qwen35_4b_probe/plan.json \  --dynamic-out ./runs/qwen35_4b_probe/dynamic.json \  --budget-mb-extra 220 \  --judge-base-url http://127.0.0.1:8000/v1 \  --judge-model Qwen/Qwen2.5-14B-Instruct \  --judge-api-key-env JUDGE_API_KEY \  --chunk-size 24
Bashexport JUDGE_API_KEY=your_key_herepython -m quant_judge.pipeline plan \  --features-jsonl ./runs/qwen35_4b_probe/probe_features.jsonl \  --plan-out ./runs/qwen35_4b_probe/plan.json \  --dynamic-out ./runs/qwen35_4b_probe/dynamic.json \  --budget-mb-extra 220 \  --judge-base-url http://127.0.0.1:8000/v1 \  --judge-model Qwen/Qwen2.5-14B-Instruct \  --judge-api-key-env JUDGE_API_KEY \  --chunk-size 24
```

## Quantize with the compiled plan

```Bashpython -m quant_judge.pipeline quantize \  --model-id Qwen/Qwen3.5-4B-Base \  --plan-json ./runs/qwen35_4b_probe/plan.json \  --out-dir ./runs/qwen35_4b_quant \  --dataset-name wikitext \  --dataset-config wikitext-2-raw-v1 \  --split train \  --text-field text \  --num-texts 128 \  --batch-size 4 \  --base-bits 4 \  --base-group-size 128
Bashpython -m quant_judge.pipeline quantize \  --model-id Qwen/Qwen3.5-4B-Base \  --plan-json ./runs/qwen35_4b_probe/plan.json \  --out-dir ./runs/qwen35_4b_quant \  --dataset-name wikitext \  --dataset-config wikitext-2-raw-v1 \  --split train \  --text-field text \  --num-texts 128 \  --batch-size 4 \  --base-bits 4 \  --base-group-size 128
```

## One-shot run

```Bashexport JUDGE_API_KEY=your_key_herepython -m quant_judge.pipeline all \  --model-id Qwen/Qwen3.5-4B-Base \  --work-dir ./runs/qwen35_4b_full \  --dataset-name wikitext \  --dataset-config wikitext-2-raw-v1 \  --split train \  --text-field text \  --probe-num-texts 64 \  --quant-num-texts 128 \  --max-length 256 \  --max-rows-per-module 128 \  --include-modules o_proj,down_proj,gate_proj,lm_head \  --budget-mb-extra 220 \  --judge-base-url http://127.0.0.1:8000/v1 \  --judge-model Qwen/Qwen2.5-14B-Instruct \  --judge-api-key-env JUDGE_API_KEY
Bashexport JUDGE_API_KEY=your_key_herepython -m quant_judge.pipeline all \  --model-id Qwen/Qwen3.5-4B-Base \  --work-dir ./runs/qwen35_4b_full \  --dataset-name wikitext \  --dataset-config wikitext-2-raw-v1 \  --split train \  --text-field text \  --probe-num-texts 64 \  --quant-num-texts 128 \  --max-length 256 \  --max-rows-per-module 128 \  --include-modules o_proj,down_proj,gate_proj,lm_head \  --budget-mb-extra 220 \  --judge-base-url http://127.0.0.1:8000/v1 \  --judge-model Qwen/Qwen2.5-14B-Instruct \  --judge-api-key-env JUDGE_API_KEY
```

## Notes

- The probe uses a proxy group-wise weight quantizer to estimate per-module sensitivity. It is not a full internal GPTQ re-run for every module/action.
- That is intentional: it stays fast enough to be practical and gives the judge stable numeric features.
- The final quantization still happens through GPTQModel.
- Start with the default modules. Add q_proj/k_proj/v_proj/up_proj only after the loop is working.
EOF

cat > requirements.txt <<'EOF'
transformers
datasets
accelerate
requests
numpy
tqdm
psutil
gptqmodel
EOF

mkdir -p quant_judge

cat > quant_judge/**init**.py <<'EOF'
**all** = []
EOF

cat > quant_judge/actions.py <<'EOF'
from **future** import annotations

from dataclasses import dataclass
from typing import Dict, Any

@dataclass(frozen=True)
class ActionSpec:
name: str
bits: int | None
group_size: int | None
skip: bool
notes: str = ""

ACTION_SPECS: dict[str, ActionSpec] = {
"w4_g128": ActionSpec("w4_g128", bits=4, group_size=128, skip=False, notes="base action"),
"w4_g64": ActionSpec("w4_g64", bits=4, group_size=64, skip=False, notes="tighter grouping"),
"w8_g128": ActionSpec("w8_g128", bits=8, group_size=128, skip=False, notes="higher precision"),
"skip": ActionSpec("skip", bits=None, group_size=None, skip=True, notes="leave module unquantized"),
}

BASE_ACTION = "w4_g128"

def bytes_per_param_for_action(action: str, fp_bytes: int = 2) -> float:
if action == "skip":
return float(fp_bytes)
spec = ACTION_SPECS[action]
assert spec.bits is not None
return spec.bits / 8.0

def action_to_dynamic_override(action: str) -> Dict[str, Any]:
if action == "skip":
return {}
spec = ACTION_SPECS[action]
return {
"bits": spec.bits,
"group_size": spec.group_size,
}

def allowed_actions_csv() -> str:
return ",".join(ACTION_SPECS.keys())
EOF

cat > quant_judge/probe.py <<'EOF'
from **future** import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .actions import ACTION_SPECS, BASE_ACTION, bytes_per_param_for_action

@dataclass
class ProbeConfig:
model_id: str
out_dir: str
dataset_name: str
dataset_config: str | None
split: str
text_field: str
num_texts: int
max_length: int
max_rows_per_module: int
include_modules: list[str]
trust_remote_code: bool = True

def _dtype_auto() -> torch.dtype:
if not torch.cuda.is_available():
return torch.float32
major, _minor = torch.cuda.get_device_capability(0)
if major >= 8:
return torch.bfloat16
return torch.float16

def load_texts(
dataset_name: str,
dataset_config: str | None,
split: str,
text_field: str,
num_texts: int,
) -> list[str]:
ds = load_dataset(dataset_name, dataset_config, split=split)
texts: list[str] = []
for row in ds:
text = row.get(text_field, "")
if isinstance(text, str) and text.strip():
texts.append(text.strip())
if len(texts) >= num_texts:
break
if not texts:
raise ValueError("No non-empty texts found in dataset.")
return texts

def load_model_and_tokenizer(model_id: str, trust_remote_code: bool = True):
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
if tokenizer.pad_token_id is None:
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
model_id,
trust_remote_code=trust_remote_code,
torch_dtype=_dtype_auto(),
device_map="auto",
low_cpu_mem_usage=True,
)
model.eval()
return model, tokenizer

def module_name_selected(name: str, include_modules: list[str]) -> bool:
return any(key in name for key in include_modules)

def iter_target_linears(model, include_modules: list[str]):
for name, module in model.named_modules():
if isinstance(module, torch.nn.Linear) and module_name_selected(name, include_modules):
yield name, module

def _flatten_rows(x: torch.Tensor) -> torch.Tensor:
if x.dim() == 2:
return x
if x.dim() >= 3:
return x.reshape(-1, x.shape[-1])
return x.unsqueeze(0)

def _take_rows(x: torch.Tensor, max_rows: int) -> torch.Tensor:
x = _flatten_rows(x)
if x.shape[0] <= max_rows:
return x
idx = torch.randperm(x.shape[0], device=x.device)[:max_rows]
return x.index_select(0, idx)

def _cpu_half(x: torch.Tensor) -> torch.Tensor:
return x.detach().to("cpu", dtype=torch.float16, copy=True)

def _weight_kurtosis(weight: torch.Tensor) -> float:
w = weight.detach().float().flatten()
mu = w.mean()
var = ((w - mu) ** 2).mean().clamp_min(1e-12)
kurt = (((w - mu) ** 4).mean() / (var ** 2)).item()
return float(kurt)

def _act_outlier_rate(x: torch.Tensor, sigma: float = 6.0) -> float:
xf = x.float()
std = xf.std().clamp_min(1e-6)
rate = (xf.abs() > (sigma * std)).float().mean().item()
return float(rate)

def _groupwise_symmetric_quantize(weight: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
"""
Quantizes along the input dimension per output row.
weight shape: [out_features, in_features]
"""
assert weight.dim() == 2
w = weight.detach().float()
out_features, in_features = w.shape
if group_size <= 0:
raise ValueError("group_size must be > 0")

```pad = (group_size - (in_features % group_size)) % group_sizeif pad:    w = F.pad(w, (0, pad), value=0.0)maxq = (2 ** (bits - 1)) - 1groups = w.view(out_features, -1, group_size)absmax = groups.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)scale = absmax / maxqq = torch.round(groups / scale).clamp(-maxq, maxq)dq = q * scaledq = dq.view(out_features, -1)if pad:    dq = dq[:, :in_features]return dq.to(weight.device, dtype=weight.dtype)
pad = (group_size - (in_features % group_size)) % group_sizeif pad:    w = F.pad(w, (0, pad), value=0.0)maxq = (2 ** (bits - 1)) - 1groups = w.view(out_features, -1, group_size)absmax = groups.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)scale = absmax / maxqq = torch.round(groups / scale).clamp(-maxq, maxq)dq = q * scaledq = dq.view(out_features, -1)if pad:    dq = dq[:, :in_features]return dq.to(weight.device, dtype=weight.dtype)
```

def _probe_linear_module(
module: torch.nn.Linear,
x_rows: torch.Tensor,
y_rows: torch.Tensor,
module_name: str,
) -> dict:
device = module.weight.device
x = x_rows.to(device=device, dtype=module.weight.dtype)
y = y_rows.to(device=device, dtype=module.weight.dtype)

```W = module.weight.detach()bias = None if module.bias is None else module.bias.detach()fp_params = int(W.numel())fp_mb = (fp_params * 2.0) / (1024 ** 2)candidates = {}for action_name in ("w4_g128", "w4_g64", "w8_g128"):    spec = ACTION_SPECS[action_name]    Wq = _groupwise_symmetric_quantize(W, bits=spec.bits, group_size=spec.group_size)    y_hat = F.linear(x, Wq, bias)    mse = F.mse_loss(y_hat.float(), y.float()).item()    max_abs = (y_hat.float() - y.float()).abs().max().item()    y_norm = y.float().norm(dim=-1).clamp_min(1e-8)    yh_norm = y_hat.float().norm(dim=-1).clamp_min(1e-8)    cos = F.cosine_similarity(y_hat.float(), y.float(), dim=-1).mean().item()    q_mb = (fp_params * bytes_per_param_for_action(action_name)) / (1024 ** 2)    candidates[action_name] = {        "mse": float(mse),        "cosine": float(cos),        "max_abs": float(max_abs),        "estimated_model_mb_for_module": float(q_mb),        "estimated_extra_mb_vs_base": float(            q_mb - (fp_params * bytes_per_param_for_action(BASE_ACTION)) / (1024 ** 2)        ),    }skip_mb = fp_mbcandidates["skip"] = {    "mse": 0.0,    "cosine": 1.0,    "max_abs": 0.0,    "estimated_model_mb_for_module": float(skip_mb),    "estimated_extra_mb_vs_base": float(        skip_mb - (fp_params * bytes_per_param_for_action(BASE_ACTION)) / (1024 ** 2)    ),}w4 = candidates["w4_g128"]["mse"]w4g64 = candidates["w4_g64"]["mse"]w8 = candidates["w8_g128"]["mse"]gain_w8_vs_w4 = max(0.0, w4 - w8)gain_w4g64_vs_w4 = max(0.0, w4 - w4g64)record = {    "name": module_name,    "module_type": module_name.split(".")[-1],    "out_features": int(W.shape[0]),    "in_features": int(W.shape[1]),    "params": int(fp_params),    "captured_rows": int(x_rows.shape[0]),    "weight_kurtosis": _weight_kurtosis(W),    "act_outlier_rate": _act_outlier_rate(x_rows),    "estimated_fp16_mb_for_module": float(fp_mb),    "proxy_gain_w8_vs_w4": float(gain_w8_vs_w4),    "proxy_gain_w4g64_vs_w4": float(gain_w4g64_vs_w4),    "candidates": candidates,}return record
W = module.weight.detach()bias = None if module.bias is None else module.bias.detach()fp_params = int(W.numel())fp_mb = (fp_params * 2.0) / (1024 ** 2)candidates = {}for action_name in ("w4_g128", "w4_g64", "w8_g128"):    spec = ACTION_SPECS[action_name]    Wq = _groupwise_symmetric_quantize(W, bits=spec.bits, group_size=spec.group_size)    y_hat = F.linear(x, Wq, bias)    mse = F.mse_loss(y_hat.float(), y.float()).item()    max_abs = (y_hat.float() - y.float()).abs().max().item()    y_norm = y.float().norm(dim=-1).clamp_min(1e-8)    yh_norm = y_hat.float().norm(dim=-1).clamp_min(1e-8)    cos = F.cosine_similarity(y_hat.float(), y.float(), dim=-1).mean().item()    q_mb = (fp_params * bytes_per_param_for_action(action_name)) / (1024 ** 2)    candidates[action_name] = {        "mse": float(mse),        "cosine": float(cos),        "max_abs": float(max_abs),        "estimated_model_mb_for_module": float(q_mb),        "estimated_extra_mb_vs_base": float(            q_mb - (fp_params * bytes_per_param_for_action(BASE_ACTION)) / (1024 ** 2)        ),    }skip_mb = fp_mbcandidates["skip"] = {    "mse": 0.0,    "cosine": 1.0,    "max_abs": 0.0,    "estimated_model_mb_for_module": float(skip_mb),    "estimated_extra_mb_vs_base": float(        skip_mb - (fp_params * bytes_per_param_for_action(BASE_ACTION)) / (1024 ** 2)    ),}w4 = candidates["w4_g128"]["mse"]w4g64 = candidates["w4_g64"]["mse"]w8 = candidates["w8_g128"]["mse"]gain_w8_vs_w4 = max(0.0, w4 - w8)gain_w4g64_vs_w4 = max(0.0, w4 - w4g64)record = {    "name": module_name,    "module_type": module_name.split(".")[-1],    "out_features": int(W.shape[0]),    "in_features": int(W.shape[1]),    "params": int(fp_params),    "captured_rows": int(x_rows.shape[0]),    "weight_kurtosis": _weight_kurtosis(W),    "act_outlier_rate": _act_outlier_rate(x_rows),    "estimated_fp16_mb_for_module": float(fp_mb),    "proxy_gain_w8_vs_w4": float(gain_w8_vs_w4),    "proxy_gain_w4g64_vs_w4": float(gain_w4g64_vs_w4),    "candidates": candidates,}return record
```

def run_probe(cfg: ProbeConfig) -> dict:
out_dir = Path(cfg.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

```model, tokenizer = load_model_and_tokenizer(cfg.model_id, trust_remote_code=cfg.trust_remote_code)texts = load_texts(    dataset_name=cfg.dataset_name,    dataset_config=cfg.dataset_config,    split=cfg.split,    text_field=cfg.text_field,    num_texts=cfg.num_texts,)targets = list(iter_target_linears(model, cfg.include_modules))if not targets:    raise ValueError("No target linear modules found. Check --include-modules.")captured_inputs: Dict[str, list[torch.Tensor]] = {name: [] for name, _ in targets}captured_outputs: Dict[str, list[torch.Tensor]] = {name: [] for name, _ in targets}remaining_rows: Dict[str, int] = {name: cfg.max_rows_per_module for name, _ in targets}handles = []def make_hook(name: str):    def hook(_module, inp, out):        rem = remaining_rows[name]        if rem <= 0:            return        x = inp[0]        y = out        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):            return        x_rows = _take_rows(x, rem)        y_rows = _take_rows(y, rem)        take = min(x_rows.shape[0], y_rows.shape[0], rem)        if take <= 0:            return        captured_inputs[name].append(_cpu_half(x_rows[:take]))        captured_outputs[name].append(_cpu_half(y_rows[:take]))        remaining_rows[name] -= take    return hookfor name, module in targets:    handles.append(module.register_forward_hook(make_hook(name)))with torch.no_grad():    for text in tqdm(texts, desc="collect_activations"):        batch = tokenizer(            text,            return_tensors="pt",            truncation=True,            max_length=cfg.max_length,            padding=False,        )        batch = {k: v.to(model.device) for k, v in batch.items()}        _ = model(**batch)        if all(v <= 0 for v in remaining_rows.values()):            breakfor h in handles:    h.remove()probe_rows: list[dict] = []for name, module in tqdm(targets, desc="probe_modules"):    if not captured_inputs[name] or not captured_outputs[name]:        continue    x_rows = torch.cat(captured_inputs[name], dim=0).float()    y_rows = torch.cat(captured_outputs[name], dim=0).float()    row = _probe_linear_module(module, x_rows, y_rows, name)    probe_rows.append(row)features_jsonl = out_dir / "probe_features.jsonl"compact_json = out_dir / "probe_features_compact.json"metadata_json = out_dir / "probe_metadata.json"with open(features_jsonl, "w", encoding="utf-8") as f:    for row in probe_rows:        f.write(json.dumps(row, ensure_ascii=False) + "\n")compact = {    "model_id": cfg.model_id,    "num_modules": len(probe_rows),    "include_modules": cfg.include_modules,    "top_w4_mse_modules": sorted(        [{"name": r["name"], "w4_g128_mse": r["candidates"]["w4_g128"]["mse"]} for r in probe_rows],        key=lambda x: x["w4_g128_mse"],        reverse=True,    )[:20],}compact_json.write_text(json.dumps(compact, indent=2), encoding="utf-8")metadata = {    "model_id": cfg.model_id,    "dataset_name": cfg.dataset_name,    "dataset_config": cfg.dataset_config,    "split": cfg.split,    "text_field": cfg.text_field,    "num_texts": cfg.num_texts,    "max_length": cfg.max_length,    "max_rows_per_module": cfg.max_rows_per_module,    "include_modules": cfg.include_modules,    "num_probe_rows": len(probe_rows),}metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")return {    "probe_rows": len(probe_rows),    "features_jsonl": str(features_jsonl),    "compact_json": str(compact_json),    "metadata_json": str(metadata_json),}
model, tokenizer = load_model_and_tokenizer(cfg.model_id, trust_remote_code=cfg.trust_remote_code)texts = load_texts(    dataset_name=cfg.dataset_name,    dataset_config=cfg.dataset_config,    split=cfg.split,    text_field=cfg.text_field,    num_texts=cfg.num_texts,)targets = list(iter_target_linears(model, cfg.include_modules))if not targets:    raise ValueError("No target linear modules found. Check --include-modules.")captured_inputs: Dict[str, list[torch.Tensor]] = {name: [] for name, _ in targets}captured_outputs: Dict[str, list[torch.Tensor]] = {name: [] for name, _ in targets}remaining_rows: Dict[str, int] = {name: cfg.max_rows_per_module for name, _ in targets}handles = []def make_hook(name: str):    def hook(_module, inp, out):        rem = remaining_rows[name]        if rem <= 0:            return        x = inp[0]        y = out        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):            return        x_rows = _take_rows(x, rem)        y_rows = _take_rows(y, rem)        take = min(x_rows.shape[0], y_rows.shape[0], rem)        if take <= 0:            return        captured_inputs[name].append(_cpu_half(x_rows[:take]))        captured_outputs[name].append(_cpu_half(y_rows[:take]))        remaining_rows[name] -= take    return hookfor name, module in targets:    handles.append(module.register_forward_hook(make_hook(name)))with torch.no_grad():    for text in tqdm(texts, desc="collect_activations"):        batch = tokenizer(            text,            return_tensors="pt",            truncation=True,            max_length=cfg.max_length,            padding=False,        )        batch = {k: v.to(model.device) for k, v in batch.items()}        _ = model(**batch)        if all(v <= 0 for v in remaining_rows.values()):            breakfor h in handles:    h.remove()probe_rows: list[dict] = []for name, module in tqdm(targets, desc="probe_modules"):    if not captured_inputs[name] or not captured_outputs[name]:        continue    x_rows = torch.cat(captured_inputs[name], dim=0).float()    y_rows = torch.cat(captured_outputs[name], dim=0).float()    row = _probe_linear_module(module, x_rows, y_rows, name)    probe_rows.append(row)features_jsonl = out_dir / "probe_features.jsonl"compact_json = out_dir / "probe_features_compact.json"metadata_json = out_dir / "probe_metadata.json"with open(features_jsonl, "w", encoding="utf-8") as f:    for row in probe_rows:        f.write(json.dumps(row, ensure_ascii=False) + "\n")compact = {    "model_id": cfg.model_id,    "num_modules": len(probe_rows),    "include_modules": cfg.include_modules,    "top_w4_mse_modules": sorted(        [{"name": r["name"], "w4_g128_mse": r["candidates"]["w4_g128"]["mse"]} for r in probe_rows],        key=lambda x: x["w4_g128_mse"],        reverse=True,    )[:20],}compact_json.write_text(json.dumps(compact, indent=2), encoding="utf-8")metadata = {    "model_id": cfg.model_id,    "dataset_name": cfg.dataset_name,    "dataset_config": cfg.dataset_config,    "split": cfg.split,    "text_field": cfg.text_field,    "num_texts": cfg.num_texts,    "max_length": cfg.max_length,    "max_rows_per_module": cfg.max_rows_per_module,    "include_modules": cfg.include_modules,    "num_probe_rows": len(probe_rows),}metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")return {    "probe_rows": len(probe_rows),    "features_jsonl": str(features_jsonl),    "compact_json": str(compact_json),    "metadata_json": str(metadata_json),}
```

EOF

cat > quant_judge/judge_client.py <<'EOF'
from **future** import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import requests

from .actions import ACTION_SPECS, BASE_ACTION, allowed_actions_csv

@dataclass
class JudgeConfig:
features_jsonl: str
plan_out: str
dynamic_out: str
budget_mb_extra: float
heuristic_only: bool = False
judge_base_url: str | None = None
judge_model: str | None = None
judge_api_key_env: str = "JUDGE_API_KEY"
chunk_size: int = 24

def load_features_jsonl(path: str) -> list[dict]:
rows = []
with open(path, "r", encoding="utf-8") as f:
for line in f:
line = line.strip()
if line:
rows.append(json.loads(line))
return rows

def _module_bonus(name: str) -> float:
bonus = 0.0
if "lm_head" in name:
bonus += 0.25
if "down_proj" in name:
bonus += 0.15
if "o_proj" in name:
bonus += 0.12
if "gate_proj" in name:
bonus += 0.10
return bonus

def heuristic_decide(row: dict) -> dict:
name = row["name"]
w4 = row["candidates"]["w4_g128"]["mse"]
w4g64 = row["candidates"]["w4_g64"]["mse"]
w8 = row["candidates"]["w8_g128"]["mse"]
kurt = row["weight_kurtosis"]
outlier = row["act_outlier_rate"]

```w8_relief = (w4 - w8) / max(w4, 1e-9)g64_relief = (w4 - w4g64) / max(w4, 1e-9)score = 0.0score += min(1.0, w8_relief / 0.60) * 0.40score += min(1.0, g64_relief / 0.35) * 0.20score += min(1.0, kurt / 20.0) * 0.15score += min(1.0, outlier / 0.05) * 0.15score += _module_bonus(name)score = max(0.0, min(1.0, score))reason = "default_low_sensitivity"action = "w4_g128"if "lm_head" in name and (w8_relief > 0.55 or kurt > 25.0):    action = "skip"    reason = "lm_head_extreme_sensitivity"elif score >= 0.75 or w8_relief > 0.50:    action = "w8_g128"    reason = "high_sensitivity_prefers_w8"elif score >= 0.42 or g64_relief > 0.18:    action = "w4_g64"    reason = "moderate_sensitivity_prefers_tighter_groups"return {    "name": name,    "action": action,    "priority": float(round(score, 4)),    "reason_code": reason,    "source": "heuristic",}
w8_relief = (w4 - w8) / max(w4, 1e-9)g64_relief = (w4 - w4g64) / max(w4, 1e-9)score = 0.0score += min(1.0, w8_relief / 0.60) * 0.40score += min(1.0, g64_relief / 0.35) * 0.20score += min(1.0, kurt / 20.0) * 0.15score += min(1.0, outlier / 0.05) * 0.15score += _module_bonus(name)score = max(0.0, min(1.0, score))reason = "default_low_sensitivity"action = "w4_g128"if "lm_head" in name and (w8_relief > 0.55 or kurt > 25.0):    action = "skip"    reason = "lm_head_extreme_sensitivity"elif score >= 0.75 or w8_relief > 0.50:    action = "w8_g128"    reason = "high_sensitivity_prefers_w8"elif score >= 0.42 or g64_relief > 0.18:    action = "w4_g64"    reason = "moderate_sensitivity_prefers_tighter_groups"return {    "name": name,    "action": action,    "priority": float(round(score, 4)),    "reason_code": reason,    "source": "heuristic",}
```

def _strip_code_fence(text: str) -> str:
t = text.strip()
if t.startswith("`"):         t = t.split("\n", 1)[1]         if t.endswith("`"):
t = t.rsplit("\n", 1)[0]
return t.strip()

def _chunk_rows(rows: list[dict], chunk_size: int) -> list[list[dict]]:
return [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]

def _reduced_row(row: dict) -> dict:
return {
"name": row["name"],
"module_type": row["module_type"],
"params": row["params"],
"captured_rows": row["captured_rows"],
"weight_kurtosis": row["weight_kurtosis"],
"act_outlier_rate": row["act_outlier_rate"],
"proxy_gain_w8_vs_w4": row["proxy_gain_w8_vs_w4"],
"proxy_gain_w4g64_vs_w4": row["proxy_gain_w4g64_vs_w4"],
"w4_g128_mse": row["candidates"]["w4_g128"]["mse"],
"w4_g64_mse": row["candidates"]["w4_g64"]["mse"],
"w8_g128_mse": row["candidates"]["w8_g128"]["mse"],
"extra_mb_w4_g64": row["candidates"]["w4_g64"]["estimated_extra_mb_vs_base"],
"extra_mb_w8_g128": row["candidates"]["w8_g128"]["estimated_extra_mb_vs_base"],
"extra_mb_skip": row["candidates"]["skip"]["estimated_extra_mb_vs_base"],
}

def call_openai_compatible(
base_url: str,
model: str,
api_key: str,
payload_rows: list[dict],
budget_mb_extra: float,
) -> list[dict]:
system = (
"You are a quantization policy judge. "
"Choose exactly one action per module from: "
f"{allowed_actions_csv()}. "
"Return JSON only. No markdown. "
"Prefer w4_g128 unless the evidence says the module is fragile. "
"Use w4_g64 for moderate sensitivity. "
"Use w8_g128 for high sensitivity. "
"Use skip only for extreme outliers, usually lm_head or very fragile modules. "
"Also emit a priority score between 0 and 1 where 1 means strongest need to preserve precision."
)
user = {
"budget_mb_extra_global": budget_mb_extra,
"rules": {
"one_assignment_per_input_row": True,
"allowed_actions": list(ACTION_SPECS.keys()),
"json_schema": {
"assignments": [
{
"name": "exact module name from input",
"action": "one of allowed actions",
"priority": "float from 0 to 1",
"reason_code": "short snake_case code"
}
]
},
},
"rows": payload_rows,
}

```url = base_url.rstrip("/") + "/chat/completions"headers = {"Content-Type": "application/json"}if api_key:    headers["Authorization"] = f"Bearer {api_key}"body = {    "model": model,    "temperature": 0.0,    "response_format": {"type": "json_object"},    "messages": [        {"role": "system", "content": system},        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},    ],}resp = requests.post(url, headers=headers, json=body, timeout=180)resp.raise_for_status()data = resp.json()content = data["choices"][0]["message"]["content"]parsed = json.loads(_strip_code_fence(content))assignments = parsed.get("assignments", [])if not isinstance(assignments, list):    raise ValueError("Judge response missing 'assignments' list.")return assignments
url = base_url.rstrip("/") + "/chat/completions"headers = {"Content-Type": "application/json"}if api_key:    headers["Authorization"] = f"Bearer {api_key}"body = {    "model": model,    "temperature": 0.0,    "response_format": {"type": "json_object"},    "messages": [        {"role": "system", "content": system},        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},    ],}resp = requests.post(url, headers=headers, json=body, timeout=180)resp.raise_for_status()data = resp.json()content = data["choices"][0]["message"]["content"]parsed = json.loads(_strip_code_fence(content))assignments = parsed.get("assignments", [])if not isinstance(assignments, list):    raise ValueError("Judge response missing 'assignments' list.")return assignments
```

def merge_assignments(rows: list[dict], llm_assignments: list[dict]) -> list[dict]:
by_name = {a["name"]: a for a in llm_assignments if "name" in a}
merged = []
for row in rows:
a = by_name.get(row["name"])
if a is None:
merged.append(heuristic_decide(row))
continue
action = a.get("action", BASE_ACTION)
if action not in ACTION_SPECS:
action = BASE_ACTION
priority = float(a.get("priority", 0.5))
priority = max(0.0, min(1.0, priority))
merged.append({
"name": row["name"],
"action": action,
"priority": priority,
"reason_code": str(a.get("reason_code", "judge_no_reason")),
"source": "judge",
})
return merged

def enforce_budget(rows: list[dict], assignments: list[dict], budget_mb_extra: float) -> dict:
row_by_name = {r["name"]: r for r in rows}
normalized = []

```for a in assignments:    row = row_by_name[a["name"]]    action = a["action"]    extra = row["candidates"][action]["estimated_extra_mb_vs_base"]    normalized.append({        **a,        "estimated_extra_mb_vs_base": extra,    })keep = []extras = []for a in normalized:    if a["action"] == BASE_ACTION:        keep.append(a)    else:        extras.append(a)extras.sort(key=lambda x: (x["priority"], x["estimated_extra_mb_vs_base"]), reverse=True)used = 0.0final_assignments = list(keep)downgraded = []for a in extras:    extra = max(0.0, float(a["estimated_extra_mb_vs_base"]))    if used + extra <= budget_mb_extra:        final_assignments.append(a)        used += extra    else:        downgraded.append({            **a,            "downgraded_to": BASE_ACTION,            "downgrade_reason": "budget_clamp",        })        final_assignments.append({            "name": a["name"],            "action": BASE_ACTION,            "priority": a["priority"],            "reason_code": "budget_clamp_to_base",            "source": a["source"],            "estimated_extra_mb_vs_base": 0.0,        })final_assignments.sort(key=lambda x: x["name"])return {    "budget_mb_extra": budget_mb_extra,    "used_mb_extra": round(used, 4),    "assignments": final_assignments,    "downgraded": downgraded,}
for a in assignments:    row = row_by_name[a["name"]]    action = a["action"]    extra = row["candidates"][action]["estimated_extra_mb_vs_base"]    normalized.append({        **a,        "estimated_extra_mb_vs_base": extra,    })keep = []extras = []for a in normalized:    if a["action"] == BASE_ACTION:        keep.append(a)    else:        extras.append(a)extras.sort(key=lambda x: (x["priority"], x["estimated_extra_mb_vs_base"]), reverse=True)used = 0.0final_assignments = list(keep)downgraded = []for a in extras:    extra = max(0.0, float(a["estimated_extra_mb_vs_base"]))    if used + extra <= budget_mb_extra:        final_assignments.append(a)        used += extra    else:        downgraded.append({            **a,            "downgraded_to": BASE_ACTION,            "downgrade_reason": "budget_clamp",        })        final_assignments.append({            "name": a["name"],            "action": BASE_ACTION,            "priority": a["priority"],            "reason_code": "budget_clamp_to_base",            "source": a["source"],            "estimated_extra_mb_vs_base": 0.0,        })final_assignments.sort(key=lambda x: x["name"])return {    "budget_mb_extra": budget_mb_extra,    "used_mb_extra": round(used, 4),    "assignments": final_assignments,    "downgraded": downgraded,}
```

def make_plan(cfg: JudgeConfig) -> dict:
rows = load_features_jsonl(cfg.features_jsonl)

```if cfg.heuristic_only:    initial_assignments = [heuristic_decide(r) for r in rows]else:    if not cfg.judge_base_url or not cfg.judge_model:        raise ValueError("Judge mode requires --judge-base-url and --judge-model.")    api_key = os.environ.get(cfg.judge_api_key_env, "")    chunks = _chunk_rows(rows, cfg.chunk_size)    all_judge_assignments = []    for chunk in chunks:        reduced = [_reduced_row(r) for r in chunk]        try:            chunk_assignments = call_openai_compatible(                base_url=cfg.judge_base_url,                model=cfg.judge_model,                api_key=api_key,                payload_rows=reduced,                budget_mb_extra=cfg.budget_mb_extra,            )            merged = merge_assignments(chunk, chunk_assignments)            all_judge_assignments.extend(merged)        except Exception:            all_judge_assignments.extend([heuristic_decide(r) for r in chunk])    initial_assignments = all_judge_assignmentsplan = enforce_budget(rows, initial_assignments, cfg.budget_mb_extra)return plan
if cfg.heuristic_only:    initial_assignments = [heuristic_decide(r) for r in rows]else:    if not cfg.judge_base_url or not cfg.judge_model:        raise ValueError("Judge mode requires --judge-base-url and --judge-model.")    api_key = os.environ.get(cfg.judge_api_key_env, "")    chunks = _chunk_rows(rows, cfg.chunk_size)    all_judge_assignments = []    for chunk in chunks:        reduced = [_reduced_row(r) for r in chunk]        try:            chunk_assignments = call_openai_compatible(                base_url=cfg.judge_base_url,                model=cfg.judge_model,                api_key=api_key,                payload_rows=reduced,                budget_mb_extra=cfg.budget_mb_extra,            )            merged = merge_assignments(chunk, chunk_assignments)            all_judge_assignments.extend(merged)        except Exception:            all_judge_assignments.extend([heuristic_decide(r) for r in chunk])    initial_assignments = all_judge_assignmentsplan = enforce_budget(rows, initial_assignments, cfg.budget_mb_extra)return plan
```

EOF

cat > quant_judge/compiler.py <<'EOF'
from **future** import annotations

import json
import re
from pathlib import Path

from .actions import action_to_dynamic_override

def compile_dynamic_from_plan(plan: dict) -> dict:
dynamic = {}
for item in plan["assignments"]:
name = item["name"]
action = item["action"]
escaped = re.escape(name)
if action == "skip":
dynamic[rf"-:^{escaped}$"] = {}
else:
dynamic[rf"+:^{escaped}$"] = action_to_dynamic_override(action)
return dynamic

def save_plan_and_dynamic(plan: dict, plan_out: str, dynamic_out: str) -> None:
plan_path = Path(plan_out)
dyn_path = Path(dynamic_out)
plan_path.parent.mkdir(parents=True, exist_ok=True)
dyn_path.parent.mkdir(parents=True, exist_ok=True)

```dynamic = compile_dynamic_from_plan(plan)plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")dyn_path.write_text(json.dumps(dynamic, indent=2), encoding="utf-8")
dynamic = compile_dynamic_from_plan(plan)plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")dyn_path.write_text(json.dumps(dynamic, indent=2), encoding="utf-8")
```

EOF

cat > quant_judge/pipeline.py <<'EOF'
from **future** import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

from .compiler import compile_dynamic_from_plan, save_plan_and_dynamic
from .judge_client import JudgeConfig, make_plan
from .probe import ProbeConfig, run_probe

def load_texts_for_quant(dataset_name: str, dataset_config: str | None, split: str, text_field: str, num_texts: int) -> list[str]:
ds = load_dataset(dataset_name, dataset_config, split=split)
texts = []
for row in ds:
text = row.get(text_field, "")
if isinstance(text, str) and text.strip():
texts.append(text.strip())
if len(texts) >= num_texts:
break
if not texts:
raise ValueError("No texts found for quantization dataset.")
return texts

def do_probe(args):
cfg = ProbeConfig(
model_id=args.model_id,
out_dir=args.out_dir,
dataset_name=args.dataset_name,
dataset_config=args.dataset_config,
split=args.split,
text_field=args.text_field,
num_texts=args.num_texts,
max_length=args.max_length,
max_rows_per_module=args.max_rows_per_module,
include_modules=[x.strip() for x in args.include_modules.split(",") if x.strip()],
)
result = run_probe(cfg)
print(json.dumps(result, indent=2))

def do_plan(args):
cfg = JudgeConfig(
features_jsonl=args.features_jsonl,
plan_out=args.plan_out,
dynamic_out=args.dynamic_out,
budget_mb_extra=args.budget_mb_extra,
heuristic_only=args.heuristic_only,
judge_base_url=args.judge_base_url,
judge_model=args.judge_model,
judge_api_key_env=args.judge_api_key_env,
chunk_size=args.chunk_size,
)
plan = make_plan(cfg)
save_plan_and_dynamic(plan, cfg.plan_out, cfg.dynamic_out)
print(json.dumps({
"plan_out": cfg.plan_out,
"dynamic_out": cfg.dynamic_out,
"used_mb_extra": plan["used_mb_extra"],
"downgraded_count": len(plan["downgraded"]),
"assignments": len(plan["assignments"]),
}, indent=2))

def do_quantize(args):
plan = json.loads(Path(args.plan_json).read_text(encoding="utf-8"))
dynamic = compile_dynamic_from_plan(plan)

```calibration_dataset = load_texts_for_quant(    dataset_name=args.dataset_name,    dataset_config=args.dataset_config,    split=args.split,    text_field=args.text_field,    num_texts=args.num_texts,)qcfg = QuantizeConfig.gptq_pro()qcfg.bits = args.base_bitsqcfg.group_size = args.base_group_sizeqcfg.dynamic = dynamicmodel = GPTQModel.load(args.model_id, qcfg)model.quantize(calibration_dataset, batch_size=args.batch_size)Path(args.out_dir).mkdir(parents=True, exist_ok=True)model.save(args.out_dir)meta = {    "model_id": args.model_id,    "out_dir": args.out_dir,    "base_bits": args.base_bits,    "base_group_size": args.base_group_size,    "batch_size": args.batch_size,    "num_texts": args.num_texts,    "dynamic_rules": len(dynamic),    "plan_json": args.plan_json,}Path(args.out_dir, "quant_judge_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")print(json.dumps(meta, indent=2))
calibration_dataset = load_texts_for_quant(    dataset_name=args.dataset_name,    dataset_config=args.dataset_config,    split=args.split,    text_field=args.text_field,    num_texts=args.num_texts,)qcfg = QuantizeConfig.gptq_pro()qcfg.bits = args.base_bitsqcfg.group_size = args.base_group_sizeqcfg.dynamic = dynamicmodel = GPTQModel.load(args.model_id, qcfg)model.quantize(calibration_dataset, batch_size=args.batch_size)Path(args.out_dir).mkdir(parents=True, exist_ok=True)model.save(args.out_dir)meta = {    "model_id": args.model_id,    "out_dir": args.out_dir,    "base_bits": args.base_bits,    "base_group_size": args.base_group_size,    "batch_size": args.batch_size,    "num_texts": args.num_texts,    "dynamic_rules": len(dynamic),    "plan_json": args.plan_json,}Path(args.out_dir, "quant_judge_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")print(json.dumps(meta, indent=2))
```

def do_all(args):
work_dir = Path(args.work_dir)
probe_dir = work_dir / "probe"
probe_dir.mkdir(parents=True, exist_ok=True)

```probe_args = argparse.Namespace(    model_id=args.model_id,    out_dir=str(probe_dir),    dataset_name=args.dataset_name,    dataset_config=args.dataset_config,    split=args.split,    text_field=args.text_field,    num_texts=args.probe_num_texts,    max_length=args.max_length,    max_rows_per_module=args.max_rows_per_module,    include_modules=args.include_modules,)do_probe(probe_args)features_jsonl = str(probe_dir / "probe_features.jsonl")plan_out = str(work_dir / "plan.json")dynamic_out = str(work_dir / "dynamic.json")plan_args = argparse.Namespace(    features_jsonl=features_jsonl,    plan_out=plan_out,    dynamic_out=dynamic_out,    budget_mb_extra=args.budget_mb_extra,    heuristic_only=args.heuristic_only,    judge_base_url=args.judge_base_url,    judge_model=args.judge_model,    judge_api_key_env=args.judge_api_key_env,    chunk_size=args.chunk_size,)do_plan(plan_args)quant_args = argparse.Namespace(    model_id=args.model_id,    plan_json=plan_out,    out_dir=str(work_dir / "quantized"),    dataset_name=args.dataset_name,    dataset_config=args.dataset_config,    split=args.split,    text_field=args.text_field,    num_texts=args.quant_num_texts,    batch_size=args.batch_size,    base_bits=args.base_bits,    base_group_size=args.base_group_size,)do_quantize(quant_args)
probe_args = argparse.Namespace(    model_id=args.model_id,    out_dir=str(probe_dir),    dataset_name=args.dataset_name,    dataset_config=args.dataset_config,    split=args.split,    text_field=args.text_field,    num_texts=args.probe_num_texts,    max_length=args.max_length,    max_rows_per_module=args.max_rows_per_module,    include_modules=args.include_modules,)do_probe(probe_args)features_jsonl = str(probe_dir / "probe_features.jsonl")plan_out = str(work_dir / "plan.json")dynamic_out = str(work_dir / "dynamic.json")plan_args = argparse.Namespace(    features_jsonl=features_jsonl,    plan_out=plan_out,    dynamic_out=dynamic_out,    budget_mb_extra=args.budget_mb_extra,    heuristic_only=args.heuristic_only,    judge_base_url=args.judge_base_url,    judge_model=args.judge_model,    judge_api_key_env=args.judge_api_key_env,    chunk_size=args.chunk_size,)do_plan(plan_args)quant_args = argparse.Namespace(    model_id=args.model_id,    plan_json=plan_out,    out_dir=str(work_dir / "quantized"),    dataset_name=args.dataset_name,    dataset_config=args.dataset_config,    split=args.split,    text_field=args.text_field,    num_texts=args.quant_num_texts,    batch_size=args.batch_size,    base_bits=args.base_bits,    base_group_size=args.base_group_size,)do_quantize(quant_args)
```

def build_parser():
p = argparse.ArgumentParser(prog="quant_judge")
sp = p.add_subparsers(dest="cmd", required=True)

```p_probe = sp.add_parser("probe")p_probe.add_argument("--model-id", required=True)p_probe.add_argument("--out-dir", required=True)p_probe.add_argument("--dataset-name", default="wikitext")p_probe.add_argument("--dataset-config", default="wikitext-2-raw-v1")p_probe.add_argument("--split", default="train")p_probe.add_argument("--text-field", default="text")p_probe.add_argument("--num-texts", type=int, default=64)p_probe.add_argument("--max-length", type=int, default=256)p_probe.add_argument("--max-rows-per-module", type=int, default=128)p_probe.add_argument("--include-modules", default="o_proj,down_proj,gate_proj,lm_head")p_probe.set_defaults(func=do_probe)p_plan = sp.add_parser("plan")p_plan.add_argument("--features-jsonl", required=True)p_plan.add_argument("--plan-out", required=True)p_plan.add_argument("--dynamic-out", required=True)p_plan.add_argument("--budget-mb-extra", type=float, default=220.0)p_plan.add_argument("--heuristic-only", action="store_true")p_plan.add_argument("--judge-base-url", default=None)p_plan.add_argument("--judge-model", default=None)p_plan.add_argument("--judge-api-key-env", default="JUDGE_API_KEY")p_plan.add_argument("--chunk-size", type=int, default=24)p_plan.set_defaults(func=do_plan)p_quant = sp.add_parser("quantize")p_quant.add_argument("--model-id", required=True)p_quant.add_argument("--plan-json", required=True)p_quant.add_argument("--out-dir", required=True)p_quant.add_argument("--dataset-name", default="wikitext")p_quant.add_argument("--dataset-config", default="wikitext-2-raw-v1")p_quant.add_argument("--split", default="train")p_quant.add_argument("--text-field", default="text")p_quant.add_argument("--num-texts", type=int, default=128)p_quant.add_argument("--batch-size", type=int, default=4)p_quant.add_argument("--base-bits", type=int, default=4)p_quant.add_argument("--base-group-size", type=int, default=128)p_quant.set_defaults(func=do_quantize)p_all = sp.add_parser("all")p_all.add_argument("--model-id", required=True)p_all.add_argument("--work-dir", required=True)p_all.add_argument("--dataset-name", default="wikitext")p_all.add_argument("--dataset-config", default="wikitext-2-raw-v1")p_all.add_argument("--split", default="train")p_all.add_argument("--text-field", default="text")p_all.add_argument("--probe-num-texts", type=int, default=64)p_all.add_argument("--quant-num-texts", type=int, default=128)p_all.add_argument("--max-length", type=int, default=256)p_all.add_argument("--max-rows-per-module", type=int, default=128)p_all.add_argument("--include-modules", default="o_proj,down_proj,gate_proj,lm_head")p_all.add_argument("--budget-mb-extra", type=float, default=220.0)p_all.add_argument("--heuristic-only", action="store_true")p_all.add_argument("--judge-base-url", default=None)p_all.add_argument("--judge-model", default=None)p_all.add_argument("--judge-api-key-env", default="JUDGE_API_KEY")p_all.add_argument("--chunk-size", type=int, default=24)p_all.add_argument("--batch-size", type=int, default=4)p_all.add_argument("--base-bits", type=int, default=4)p_all.add_argument("--base-group-size", type=int, default=128)p_all.set_defaults(func=do_all)return p
p_probe = sp.add_parser("probe")p_probe.add_argument("--model-id", required=True)p_probe.add_argument("--out-dir", required=True)p_probe.add_argument("--dataset-name", default="wikitext")p_probe.add_argument("--dataset-config", default="wikitext-2-raw-v1")p_probe.add_argument("--split", default="train")p_probe.add_argument("--text-field", default="text")p_probe.add_argument("--num-texts", type=int, default=64)p_probe.add_argument("--max-length", type=int, default=256)p_probe.add_argument("--max-rows-per-module", type=int, default=128)p_probe.add_argument("--include-modules", default="o_proj,down_proj,gate_proj,lm_head")p_probe.set_defaults(func=do_probe)p_plan = sp.add_parser("plan")p_plan.add_argument("--features-jsonl", required=True)p_plan.add_argument("--plan-out", required=True)p_plan.add_argument("--dynamic-out", required=True)p_plan.add_argument("--budget-mb-extra", type=float, default=220.0)p_plan.add_argument("--heuristic-only", action="store_true")p_plan.add_argument("--judge-base-url", default=None)p_plan.add_argument("--judge-model", default=None)p_plan.add_argument("--judge-api-key-env", default="JUDGE_API_KEY")p_plan.add_argument("--chunk-size", type=int, default=24)p_plan.set_defaults(func=do_plan)p_quant = sp.add_parser("quantize")p_quant.add_argument("--model-id", required=True)p_quant.add_argument("--plan-json", required=True)p_quant.add_argument("--out-dir", required=True)p_quant.add_argument("--dataset-name", default="wikitext")p_quant.add_argument("--dataset-config", default="wikitext-2-raw-v1")p_quant.add_argument("--split", default="train")p_quant.add_argument("--text-field", default="text")p_quant.add_argument("--num-texts", type=int, default=128)p_quant.add_argument("--batch-size", type=int, default=4)p_quant.add_argument("--base-bits", type=int, default=4)p_quant.add_argument("--base-group-size", type=int, default=128)p_quant.set_defaults(func=do_quantize)p_all = sp.add_parser("all")p_all.add_argument("--model-id", required=True)p_all.add_argument("--work-dir", required=True)p_all.add_argument("--dataset-name", default="wikitext")p_all.add_argument("--dataset-config", default="wikitext-2-raw-v1")p_all.add_argument("--split", default="train")p_all.add_argument("--text-field", default="text")p_all.add_argument("--probe-num-texts", type=int, default=64)p_all.add_argument("--quant-num-texts", type=int, default=128)p_all.add_argument("--max-length", type=int, default=256)p_all.add_argument("--max-rows-per-module", type=int, default=128)p_all.add_argument("--include-modules", default="o_proj,down_proj,gate_proj,lm_head")p_all.add_argument("--budget-mb-extra", type=float, default=220.0)p_all.add_argument("--heuristic-only", action="store_true")p_all.add_argument("--judge-base-url", default=None)p_all.add_argument("--judge-model", default=None)p_all.add_argument("--judge-api-key-env", default="JUDGE_API_KEY")p_all.add_argument("--chunk-size", type=int, default=24)p_all.add_argument("--batch-size", type=int, default=4)p_all.add_argument("--base-bits", type=int, default=4)p_all.add_argument("--base-group-size", type=int, default=128)p_all.set_defaults(func=do_all)return p
```

def main():
parser = build_parser()
args = parser.parse_args()
args.func(args)

if **name** == "**main**":
main()
EOF

```Then install and run it:```bashpython -m venv .venvsource .venv/bin/activate# install your CUDA-matching torch first, for example:# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128pip install -r requirements.txt
Then install and run it:```bashpython -m venv .venvsource .venv/bin/activate# install your CUDA-matching torch first, for example:# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128pip install -r requirements.txt
```

Heuristic-only dry run:

```Bashpython -m quant_judge.pipeline all \  --model-id Qwen/Qwen3.5-4B-Base \  --work-dir ./runs/qwen35_4b_full \  --dataset-name wikitext \  --dataset-config wikitext-2-raw-v1 \  --split train \  --text-field text \  --probe-num-texts 64 \  --quant-num-texts 128 \  --max-length 256 \  --max-rows-per-module 128 \  --include-modules o_proj,down_proj,gate_proj,lm_head \  --budget-mb-extra 220 \  --heuristic-only
Bashpython -m quant_judge.pipeline all \  --model-id Qwen/Qwen3.5-4B-Base \  --work-dir ./runs/qwen35_4b_full \  --dataset-name wikitext \  --dataset-config wikitext-2-raw-v1 \  --split train \  --text-field text \  --probe-num-texts 64 \  --quant-num-texts 128 \  --max-length 256 \  --max-rows-per-module 128 \  --include-modules o_proj,down_proj,gate_proj,lm_head \  --budget-mb-extra 220 \  --heuristic-only
```

External judge run:

```Bashexport JUDGE_API_KEY=your_key_herepython -m quant_judge.pipeline all \  --model-id Qwen/Qwen3.5-4B-Base \  --work-dir ./runs/qwen35_4b_full \  --dataset-name wikitext \  --dataset-config wikitext-2-raw-v1 \  --split train \  --text-field text \  --probe-num-texts 64 \  --quant-num-texts 128 \  --max-length 256 \  --max-rows-per-module 128 \  --include-modules o_proj,down_proj,gate_proj,lm_head \  --budget-mb-extra 220 \  --judge-base-url http://127.0.0.1:8000/v1 \  --judge-model Qwen/Qwen2.5-14B-Instruct \  --judge-api-key-env JUDGE_API_KEY
Bashexport JUDGE_API_KEY=your_key_herepython -m quant_judge.pipeline all \  --model-id Qwen/Qwen3.5-4B-Base \  --work-dir ./runs/qwen35_4b_full \  --dataset-name wikitext \  --dataset-config wikitext-2-raw-v1 \  --split train \  --text-field text \  --probe-num-texts 64 \  --quant-num-texts 128 \  --max-length 256 \  --max-rows-per-module 128 \  --include-modules o_proj,down_proj,gate_proj,lm_head \  --budget-mb-extra 220 \  --judge-base-url http://127.0.0.1:8000/v1 \  --judge-model Qwen/Qwen2.5-14B-Instruct \  --judge-api-key-env JUDGE_API_KEY
```

What you should inspect first:

- `./runs/qwen35_4b_full/probe/probe_features.jsonl`
- `./runs/qwen35_4b_full/plan.json`
- `./runs/qwen35_4b_full/dynamic.json`

The important part is that `dynamic.json` is what gets applied to GPTQModel’s mixed per-module quantization rules, including exact-match `+:` overrides and `-:` skips. [GitHub](https://github.com/ModelCloud/GPTQModel)

Best first upgrade after this works:

- add `q_proj,v_proj,up_proj`
- keep `k_proj` conservative
- run your A/B harness on:

plain base `gptq_pro()`

heuristic plan

judge plan
- plain base `gptq_pro()`
- heuristic plan
- judge plan

Paste `plan.json` and I’ll tell you whether the judge is being smart or just expensive.

---

### User:
This architecture is a significant leap over standard "brute-force" quantization. You are essentially proposing a **Software-Defined Quantization (SDQ)** controller.

By moving from a global policy (e.g., "everything is 4-bit") to a **telemetry-driven per-layer policy**, you solve the "Pareto Frontier" problem of quantization: getting the best possible accuracy for a specific memory budget.

Here is the concrete implementation scaffold to build exactly what you described for **Qwen 3.5 4B**.

---

### Phase 1: The Probe (Feature Extraction)
You need a script that hooks into `GPTQModel` or `AutoGPTQ` to run "shadow quantization" trials. 

```
python
import torch
from gptqmodel import GPTQModel, QuantizeConfig
from copy import deepcopy

def probe_layer_sensitivity(model, calibration_data):
    features = []
    # Identify all linear layers that aren't the embedding
    layers = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    
    for layer_name in layers:
        # 1. Calculate Weight Kurtosis (Outlier Indicator)
        weights = model.get_submodule(layer_name).weight.data.float()
        kurtosis = ((weights - weights.mean())**4).mean() / (weights.var()**2)
        
        # 2. Run mini-quantization trials (MSE check)
        # We compare original FP16 output vs quantized W4 output on 1 batch
        input_feat = []
        def hook(m, i, o): input_feat.append(i[0])
        handle = model.get_submodule(layer_name).register_forward_hook(hook)
        
        # Run one calibration batch
        model(calibration_data[0]) 
        handle.remove()
        
        x = input_feat[0]
        original_output = model.get_submodule(layer_name)(x)
        
        # Simple MSE Proxy: Quantize weights manually to 4-bit and measure error
        # (This is a 'fast' proxy for actual GPTQ error)
        w_min, w_max = weights.min(), weights.max()
        # ... (Insert standard symmetric quant math here) ...
        q_weights = torch.quantize_per_tensor(weights, scale=..., zero_point=0, dtype=torch.qint8)
        
        mse_error = torch.nn.functional.mse_norm(original_output, reconstructed_output)

        features.append({
            "layer": layer_name,
            "kurtosis": kurtosis.item(),
            "mse_proxy": mse_error.item(),
            "params": weights.numel()
        })
    return features
```

### Phase 2: The Judge (The Policy Agent)
The key here is **constrained output**. You don't want the LLM to write code; you want it to act as a **Router**.

**System Prompt for the Judge:**
> "You are a Quantization Strategy Engine. You will receive a JSON list of model layers and their sensitivity metrics. 
> 
> **Your Constraints:**
> 1. Total 'Extra Memory' budget: 150MB.
> 2. `w8_g128` costs 2x memory of `w4_g128`.
> 3. `skip` costs 4x memory of `w4_g128`.
> 
> **Decision Logic:**
> - If `kurtosis` > 15 OR `mse_proxy` > 0.05: Prioritize `w8_g128`.
> - If `layer` is `lm_head` or the first/last 2 blocks: Prioritize `skip` or `w8`.
> - Otherwise: Default to `w4_g128`.
> 
> Output ONLY a JSON map of `{ "layer_regex": "action" }`."

### Phase 3: The Compiler (Mapping to GPTQModel)
`GPTQModel`’s `dynamic` parameter is the "clean way" you mentioned. It allows for regex-based overrides.

```
python
from gptqmodel import GPTQModel, QuantizeConfig

# This is what your Judge produced
judge_decision = {
    r".*lm_head": "skip",
    r".*layers\.0\..*": "w8_g128",
    r".*down_proj": "w4_g64",
}

def build_dynamic_config(decisions):
    dynamic_map = {}
    for regex, action in decisions.items():
        if action == "skip":
            dynamic_map[f"-:{regex}"] = {} # Negative match to skip
        elif action == "w8_g128":
            dynamic_map[f"+:{regex}"] = {"bits": 8, "group_size": 128}
        elif action == "w4_g64":
            dynamic_map[f"+:{regex}"] = {"bits": 4, "group_size": 64}
    return dynamic_map

quant_config = QuantizeConfig(
    bits=4, 
    group_size=128, 
    dynamic=build_dynamic_config(judge_decision)
)

# Apply and Quantize
model = GPTQModel.load(model_id_or_path)
model.quantize(calibration_dataset, quant_config)
model.save_quantized("qwen-3.5-4b-judge-enhanced")
```

---

### Why this beats current GPTQ implementations:

1.  **The "Vibe Check" is Measurable:** By feeding `kurtosis` and `MSE` to the judge, the LLM isn't guessing which layers are important; it is performing a multi-objective optimization (Quality vs. VRAM) that is difficult to express in a simple `if/else` script.
2.  **Solves the "Down_Proj" Fragility:** In Qwen and Llama models, the `down_proj` in the MLP block and the `o_proj` in Attention are notoriously sensitive. A standard GPTQ run destroys them. Your Judge will see the high MSE in the probe and automatically up-tier them to `w4_g64` or `w8`.
3.  **Future-Proofing for Qwen 3.5:** Since Qwen 3.5 4B is a small model, every bit counts. Using 8-bit for just the most "painful" 5% of layers will likely keep perplexity within 0.1 of the FP16 baseline while still fitting on a 24GB consumer card easily.

### The "Pro" Upgrade: The Feedback Loop
If you want to go beyond "v1", implement a **Verify Pass**:
1. Judge proposes config.
2. Quantize.
3. Run `lm-eval` (HellaSwag or ARC-Challenge).
4. If score drops > 1%, feed the scores back to the Judge: *"The previous plan failed on reasoning tasks. Tier up the 'gate_proj' layers in the middle blocks."* 

This turns GPTQ from a **compression tool** into an **autonomous optimization agent**.

---

### ChatGPT:
This is **directionally very strong**, but I’d patch a few things before you treat it as production-ready.

The biggest correction is API shape: in GPTQModel’s current documented flow, you build `QuantizeConfig`, pass it into `GPTQModel.load(model_id, quant_config)`, then call `model.quantize(calibration_dataset, batch_size=...)`, then `model.save(...)`. The per-module override mechanism you want is real: `QuantizeConfig.dynamic` supports regex matches, positive `+:` overrides, and negative `-:` matches to skip modules; the override surface includes `bits`, `group_size`, `sym`, `desc_act`, `mse`, and `pack_dtype`. GPTQModel also explicitly recommends validating post-quant quality with `lm-eval` or `evalplus` rather than relying on perplexity alone. [GitHub+1](https://github.com/ModelCloud/GPTQModel)

The second correction is your probe math. `torch.quantize_per_tensor(... dtype=torch.qint8)` is not a real 4-bit GPTQ proxy, and `torch.nn.functional.mse_norm` is not a valid PyTorch function. For a fast probe, use a **custom group-wise fake quantizer** on the weight matrix, then compare `F.linear(x, Wq, bias)` against the FP output with `torch.nn.functional.mse_loss`. That gives you a stable screening signal without pretending you ran full GPTQ per layer. [GitHub](https://github.com/ModelCloud/GPTQModel)

Third: `model(calibration_data[0])` is too loose. If your calibration set is text, you need tokenization first, and for `Qwen/Qwen3.5-4B-Base` specifically, the current model card describes it as a **causal language model with a vision encoder**, with 32 layers and tied LM output. So the safest initial target set is exactly what you suggested on the LM side only—`o_proj`, `down_proj`, `gate_proj`, and maybe `lm_head`—instead of trying to blanket every `torch.nn.Linear` in the full multimodal graph on day one. [Hugging Face](https://huggingface.co/Qwen/Qwen3.5-4B-Base)

Fourth: your “judge” should not emit regexes directly as its primary object. Have it emit **exact module names + discrete actions** like `w4_g128`, `w4_g64`, `w8_g128`, `skip`, and then compile those exact names into anchored regexes yourself. That removes one whole class of bad outputs and keeps the LLM in the role of a constrained classifier, not a config author. GPTQModel’s dynamic rules are regex-based, so compiling exact names into `+:^...$` or `-:^...$` is the clean deterministic boundary. [GitHub](https://github.com/ModelCloud/GPTQModel)

Fifth: I would tighten the control loop conceptually. The LLM is **not** the optimizer; the deterministic budget clamp is. The judge proposes upgrades, then a compiler enforces the extra-memory budget by downgrading the lowest-priority promotions back to base `w4_g128`. That makes the system auditable and reproducible. Without that clamp, the prompt can drift into “seems important” behavior.

So the production shape I’d use is:

- **Probe:** collect per-module features from real activations.
- **Judge:** classify each module into a small action set.
- **Compiler:** enforce budget and emit `dynamic`.
- **Quantize:** run GPTQModel with base `bits=4, group_size=128` plus overrides.
- **Verify:** score with `lm-eval`, not just proxy MSE. [GitHub+1](https://github.com/ModelCloud/GPTQModel)

The feature set I’d keep for v1:

- `w4_g128_mse`
- `w4_g64_mse`
- `w8_g128_mse`
- `weight_kurtosis`
- `act_outlier_rate`
- `params`
- `estimated_extra_mb_vs_base`
- `module_type`
- `block_index`

And the hard rules I’d start with:

- default `w4_g128`
- promote to `w4_g64` if `w4_g64_mse` materially beats `w4_g128_mse`
- promote to `w8_g128` if `w8_g128_mse` gives a large gain and the module is high-priority
- reserve `skip` mostly for `lm_head` or extreme outliers

One more practical note: GPTQModel’s README explicitly notes `lm_head` quantization support, so testing `lm_head` as a special case is reasonable now rather than assuming it must always stay FP16. [GitHub](https://github.com/ModelCloud/GPTQModel)

So my verdict on that writeup is:

- **architecture idea:** good
- **API direction:** mostly correct
- **probe example code:** needs fixes
- **judge concept:** good if constrained
- **compiler to dynamic:** exactly the right path
- **feedback loop with lm-eval:** absolutely worth doing

The next concrete step is to replace the toy probe snippet with a real **group-wise fake-quant probe** and make the judge output exact names instead of regexes. If you want, I’ll write the **v2 scaffold** with those fixes baked in and add a verifier pass that compares:

1. plain `gptq_pro()`
2. heuristic mixed plan
3. judge mixed plan

---

### User:
Your corrections are spot on. Moving from a fuzzy "LLM config author" to a **constrained classifier with a deterministic budget compiler** is exactly how you take this from a weekend experiment to a production-grade tool. 

Fixing the fake-quant proxy to be group-wise, fixing the PyTorch math, anchoring the regexes, and properly tokenizing the calibration data are all essential. 

Here is the **v2 Scaffold**. This represents a fully mature Software-Defined Quantization (SDQ) pipeline for GPTQModel.

---

### 1. The Real Group-Wise Fake-Quant Probe
We replace the toy quantizer with a fast group-wise symmetric fake quantizer. We tokenize real text to get valid activations, and use `F.linear` and `F.mse_loss`.

```
python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from gptqmodel import GPTQModel

def fake_quant_groupwise(weight, bits, group_size):
    """Fast symmetric group-wise fake quantization proxy."""
    org_shape = weight.shape
    # Pad if necessary for group_size alignment
    pad_len = (group_size - (weight.numel() % group_size)) % group_size
    w_flat = weight.flatten()
    if pad_len > 0:
        w_flat = torch.cat([w_flat, torch.zeros(pad_len, device=weight.device)])
    
    w_grouped = w_flat.view(-1, group_size)
    q_max = (1 << (bits - 1)) - 1
    
    # Calculate scales per group
    max_val = w_grouped.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-5)
    scales = max_val / q_max
    
    # Quantize and Dequantize
    w_q = torch.round(w_grouped / scales).clamp(-q_max, q_max)
    w_dq = w_q * scales
    
    w_dq_flat = w_dq.flatten()[:weight.numel()]
    return w_dq_flat.view(org_shape)

def probe_model(model_id, calibration_texts):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = GPTQModel.load(model_id, device_map="auto") # Load in FP16 for probing
    
    # Tokenize input
    inputs = tokenizer(calibration_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    features =[]
    target_types = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head")
    
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or not any(t in name for t in target_types):
            continue

        weight = module.weight.data
        
        # 1. Kurtosis
        w_mean, w_var = weight.mean(), weight.var()
        kurtosis = ((weight - w_mean)**4).mean() / (w_var**2)
        
        # 2. Get Activation
        act = []
        def hook(m, i, o): act.append(i[0].detach())
        handle = module.register_forward_hook(hook)
        with torch.no_grad():
            model(**inputs)
        handle.remove()
        x = act[0]
        
        # 3. Fast MSE Proxies
        out_fp = F.linear(x, weight)
        
        w_w4g128 = fake_quant_groupwise(weight, bits=4, group_size=128)
        w4g128_mse = F.mse_loss(F.linear(x, w_w4g128), out_fp).item()
        
        w_w4g64 = fake_quant_groupwise(weight, bits=4, group_size=64)
        w4g64_mse = F.mse_loss(F.linear(x, w_w4g64), out_fp).item()
        
        w_w8g128 = fake_quant_groupwise(weight, bits=8, group_size=128)
        w8g128_mse = F.mse_loss(F.linear(x, w_w8g128), out_fp).item()

        params = weight.numel()
        
        features.append({
            "name": name,
            "params": params,
            "kurtosis": kurtosis.item(),
            "w4g128_mse": w4g128_mse,
            "w4g64_mse": w4g64_mse,
            "w8g128_mse": w8g128_mse,
            # Est. extra memory vs base W4
            "w4g64_extra_mb": (params * 4 / 8) / 1024**2 * 0.1,  # roughly scale up for smaller group overhead
            "w8g128_extra_mb": (params * 4 / 8) / 1024**2,       # roughly 4 extra bits per param
        })
    
    return features
```

### 2. The Deterministic Compiler (The Budget Clamp)
The LLM Judge receives the JSON from the probe and returns a list of proposed exact module names and actions (`w4_g64`, `w8_g128`, `skip`). The Compiler enforces the global VRAM budget by ranking the requested upgrades by their delta MSE (the quality saved by upgrading), and drops the rest back to the `w4_g128` baseline.

```
python
import re

def compile_dynamic_config(judge_proposals, probe_features, extra_budget_mb=150.0):
    """
    judge_proposals: dict of { "exact.module.name": "action" }
    probe_features: dict of features from probe keyed by module name
    """
    # 1. Match proposals to costs and benefits
    upgrades =[]
    for name, action in judge_proposals.items():
        if action == "w4_g128": 
            continue # Base, no extra cost
            
        feat = next((f for f in probe_features if f["name"] == name), None)
        if not feat: continue
        
        # Calculate how much error we save by doing this upgrade vs base
        if action == "w8_g128":
            benefit = feat["w4g128_mse"] - feat["w8g128_mse"]
            cost_mb = feat["w8g128_extra_mb"]
        elif action == "w4_g64":
            benefit = feat["w4g128_mse"] - feat["w4g64_mse"]
            cost_mb = feat["w4g64_extra_mb"]
        elif action == "skip":
            benefit = feat["w4g128_mse"] # Absolute max benefit
            cost_mb = (feat["params"] * 12 / 8) / 1024**2 # FP16 vs W4 difference
        
        upgrades.append({
            "name": name, "action": action, 
            "cost": cost_mb, "benefit": benefit, 
            "roi": benefit / (cost_mb + 1e-9)
        })

    # 2. Sort by Return on Investment (ROI) and Clamp Budget
    upgrades.sort(key=lambda x: x["roi"], reverse=True)
    
    approved_dynamic = {}
    spent_mb = 0.0
    
    for up in upgrades:
        if spent_mb + up["cost"] <= extra_budget_mb:
            spent_mb += up["cost"]
            # Convert exact name to anchored regex for GPTQModel
            anchored_regex = f"^{re.escape(up['name'])}$"
            
            if up["action"] == "skip":
                approved_dynamic[f"-:{anchored_regex}"] = {}
            elif up["action"] == "w8_g128":
                approved_dynamic[f"+:{anchored_regex}"] = {"bits": 8, "group_size": 128}
            elif up["action"] == "w4_g64":
                approved_dynamic[f"+:{anchored_regex}"] = {"bits": 4, "group_size": 64}

    print(f"Compiler Approved: {len(approved_dynamic)} upgrades. Budget Spent: {spent_mb:.1f}/{extra_budget_mb} MB")
    return approved_dynamic
```

### 3. Execution & Verifier Pass
We run three models: Base `gptq_pro`, a simple heuristic plan, and the compiled Judge plan. Then we evaluate via `lm-eval`.

```
python
from gptqmodel import QuantizeConfig, GPTQModel
import lm_eval

def run_quantization_and_eval(model_id, calib_data, dynamic_overrides, run_name):
    # Base is W4 G128. We inject the compiled overrides.
    quant_config = QuantizeConfig(
        bits=4, 
        group_size=128, 
        sym=True, 
        desc_act=True,
        dynamic=dynamic_overrides 
    )
    
    # 1. Load and Quantize
    print(f"--- Running: {run_name} ---")
    model = GPTQModel.load(model_id, quant_config=quant_config)
    model.quantize(calib_data, batch_size=4)
    
    save_path = f"./qwen-3.5-4b-{run_name}"
    model.save(save_path)
    
    # 2. Verify with lm-eval (e.g., ARC-Challenge or HellaSwag)
    # Note: lm-eval handles the HF model wrapping
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={save_path}",
        tasks=["arc_challenge", "hellaswag"],
        batch_size="auto",
        device="cuda:0"
    )
    
    print(f"Results for {run_name}:")
    print(f"ARC-Challenge: {results['results']['arc_challenge']['acc_norm,none']:.4f}")
    print(f"HellaSwag: {results['results']['hellaswag']['acc_norm,none']:.4f}")
    return results

# Example Execution Flow:
# 1. features = probe_model("Qwen/Qwen3.5-4B-Base", text_dataset)
# 2. judge_proposals = call_llm_judge(features) 
# 3. final_dynamic = compile_dynamic_config(judge_proposals, features, extra_budget_mb=150)
# 4. run_quantization_and_eval("Qwen/Qwen3.5-4B-Base", text_dataset, final_dynamic, "judge_sdq")
```

### Why this is the ultimate "Correct" version:
1. **The LLM is no longer an implicit risk.** By ranking via mathematically grounded `ROI` (Return on Investment: `Delta MSE / Delta MB`), the LLM becomes a high-level strategic proposer. If it hallucinates or prioritizes poorly, the Compiler's budget clamp and `ROI` sorter neutralizes the damage.
2. **GPTQModel Native Constraints:** The exact regex anchoring (`^model\.layers\.0\.mlp\.down_proj$`) ensures `GPTQModel`'s `dynamic` routing matches exactly one layer without cascading side-effects.
3. **Hardware Truth:** `torch.quantize_per_tensor` was mathematically wrong for GPTQ. The custom `fake_quant_groupwise` replicates the actual local chunking (like `group_size=128`) that dictates how much precision a linear layer retains, meaning the `MSE` the probe spits out is a highly accurate proxy for the final GPTQ loss.

---

### ChatGPT:
This v2 is **very close**. The architecture is right: probe → judge → deterministic compiler → GPTQModel `dynamic` → `lm-eval`. That matches GPTQModel’s current API and feature surface: `GPTQModel.load(model_id, quant_config)`, `model.quantize(...)`, `model.save(...)`, plus `QuantizeConfig.dynamic` for per-module overrides and `-:` negative matches to skip modules. `lm-eval` still supports the HF backend with local paths and `simple_evaluate()`. Qwen3.5-4B-Base is currently described as a 32-layer causal LM with a vision encoder and tied LM output, so being selective about which linears you target is still the right move. [Hugging Face+5GitHub+5GitHub+5](https://github.com/ModelCloud/GPTQModel/blob/main/README.md)

The main thing I would still fix is the fake quantizer. In your snippet, flattening the whole weight tensor into one long vector is **not** a faithful proxy for GPTQ-style group quantization. For linear layers, grouping should happen **per output row along the input dimension**, not across rows, otherwise scales leak across unrelated channels and your MSE ranking gets distorted. Also, your FP reference path should include `bias`, and for probing I would load the model with regular `transformers.AutoModelForCausalLM`, not `GPTQModel.load(..., device_map="auto")`; GPTQModel’s documented load path is tied to a quant config workflow, while your probe is just a forward-feature extractor. [GitHub+1](https://github.com/ModelCloud/GPTQModel/blob/main/README.md)

Use this fake quantizer instead:

```Pythonimport torchimport torch.nn.functional as Fdef fake_quant_groupwise_linear_weight(weight: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:    """    Symmetric per-row, per-group fake quantization for Linear weights.    weight shape: [out_features, in_features]    Groups are taken along the input dimension, per output row.    """    assert weight.dim() == 2    w = weight.detach().float()    out_features, in_features = w.shape    pad = (group_size - (in_features % group_size)) % group_size    if pad:        w = F.pad(w, (0, pad), value=0.0)    maxq = (2 ** (bits - 1)) - 1    w = w.view(out_features, -1, group_size)  # [out, n_groups, group]    absmax = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)    scale = absmax / maxq    q = torch.round(w / scale).clamp(-maxq, maxq)    dq = q * scale    dq = dq.view(out_features, -1)    if pad:        dq = dq[:, :in_features]    return dq.to(weight.device, dtype=weight.dtype)
Pythonimport torchimport torch.nn.functional as Fdef fake_quant_groupwise_linear_weight(weight: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:    """    Symmetric per-row, per-group fake quantization for Linear weights.    weight shape: [out_features, in_features]    Groups are taken along the input dimension, per output row.    """    assert weight.dim() == 2    w = weight.detach().float()    out_features, in_features = w.shape    pad = (group_size - (in_features % group_size)) % group_size    if pad:        w = F.pad(w, (0, pad), value=0.0)    maxq = (2 ** (bits - 1)) - 1    w = w.view(out_features, -1, group_size)  # [out, n_groups, group]    absmax = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)    scale = absmax / maxq    q = torch.round(w / scale).clamp(-maxq, maxq)    dq = q * scale    dq = dq.view(out_features, -1)    if pad:        dq = dq[:, :in_features]    return dq.to(weight.device, dtype=weight.dtype)
```

And the probe path should look more like this:

```Pythonfrom transformers import AutoTokenizer, AutoModelForCausalLMdef probe_module(module, x):    weight = module.weight.detach()    bias = None if module.bias is None else module.bias.detach()    with torch.no_grad():        out_fp = F.linear(x, weight, bias)        w4g128 = fake_quant_groupwise_linear_weight(weight, bits=4, group_size=128)        out_w4g128 = F.linear(x, w4g128, bias)        w4g128_mse = F.mse_loss(out_w4g128.float(), out_fp.float()).item()        w4g64 = fake_quant_groupwise_linear_weight(weight, bits=4, group_size=64)        out_w4g64 = F.linear(x, w4g64, bias)        w4g64_mse = F.mse_loss(out_w4g64.float(), out_fp.float()).item()        w8g128 = fake_quant_groupwise_linear_weight(weight, bits=8, group_size=128)        out_w8g128 = F.linear(x, w8g128, bias)        w8g128_mse = F.mse_loss(out_w8g128.float(), out_fp.float()).item()    return w4g128_mse, w4g64_mse, w8g128_mse
Pythonfrom transformers import AutoTokenizer, AutoModelForCausalLMdef probe_module(module, x):    weight = module.weight.detach()    bias = None if module.bias is None else module.bias.detach()    with torch.no_grad():        out_fp = F.linear(x, weight, bias)        w4g128 = fake_quant_groupwise_linear_weight(weight, bits=4, group_size=128)        out_w4g128 = F.linear(x, w4g128, bias)        w4g128_mse = F.mse_loss(out_w4g128.float(), out_fp.float()).item()        w4g64 = fake_quant_groupwise_linear_weight(weight, bits=4, group_size=64)        out_w4g64 = F.linear(x, w4g64, bias)        w4g64_mse = F.mse_loss(out_w4g64.float(), out_fp.float()).item()        w8g128 = fake_quant_groupwise_linear_weight(weight, bits=8, group_size=128)        out_w8g128 = F.linear(x, w8g128, bias)        w8g128_mse = F.mse_loss(out_w8g128.float(), out_fp.float()).item()    return w4g128_mse, w4g64_mse, w8g128_mse
```

Your compiler idea is good. I would add two hard guards: ignore upgrades whose `benefit <= 0`, and require a minimum ROI threshold so you do not spend budget on noise. Also keep the judge output as **exact names + actions**, then compile those exact names into anchored regexes yourself, which is exactly what GPTQModel’s dynamic routing expects. [GitHub](https://github.com/ModelCloud/GPTQModel/blob/main/README.md)

This compiler version is safer:

```Pythonimport redef compile_dynamic_config(judge_proposals, probe_features, extra_budget_mb=150.0, min_roi=1e-6):    feat_by_name = {f["name"]: f for f in probe_features}    upgrades = []    for name, action in judge_proposals.items():        if action == "w4_g128":            continue        feat = feat_by_name.get(name)        if feat is None:            continue        if action == "w8_g128":            benefit = feat["w4g128_mse"] - feat["w8g128_mse"]            cost_mb = feat["w8g128_extra_mb"]        elif action == "w4_g64":            benefit = feat["w4g128_mse"] - feat["w4g64_mse"]            cost_mb = feat["w4g64_extra_mb"]        elif action == "skip":            benefit = feat["w4g128_mse"]            cost_mb = (feat["params"] * (2.0 - 0.5)) / (1024 ** 2)  # fp16 bytes - w4 bytes        else:            continue        if benefit <= 0 or cost_mb <= 0:            continue        roi = benefit / cost_mb        if roi < min_roi:            continue        upgrades.append({            "name": name,            "action": action,            "benefit": benefit,            "cost_mb": cost_mb,            "roi": roi,        })    upgrades.sort(key=lambda x: x["roi"], reverse=True)    dynamic = {}    spent_mb = 0.0    for up in upgrades:        if spent_mb + up["cost_mb"] > extra_budget_mb:            continue        spent_mb += up["cost_mb"]        anchored = f"^{re.escape(up['name'])}$"        if up["action"] == "skip":            dynamic[f"-:{anchored}"] = {}        elif up["action"] == "w8_g128":            dynamic[f"+:{anchored}"] = {"bits": 8, "group_size": 128}        elif up["action"] == "w4_g64":            dynamic[f"+:{anchored}"] = {"bits": 4, "group_size": 64}    return dynamic, spent_mb
Pythonimport redef compile_dynamic_config(judge_proposals, probe_features, extra_budget_mb=150.0, min_roi=1e-6):    feat_by_name = {f["name"]: f for f in probe_features}    upgrades = []    for name, action in judge_proposals.items():        if action == "w4_g128":            continue        feat = feat_by_name.get(name)        if feat is None:            continue        if action == "w8_g128":            benefit = feat["w4g128_mse"] - feat["w8g128_mse"]            cost_mb = feat["w8g128_extra_mb"]        elif action == "w4_g64":            benefit = feat["w4g128_mse"] - feat["w4g64_mse"]            cost_mb = feat["w4g64_extra_mb"]        elif action == "skip":            benefit = feat["w4g128_mse"]            cost_mb = (feat["params"] * (2.0 - 0.5)) / (1024 ** 2)  # fp16 bytes - w4 bytes        else:            continue        if benefit <= 0 or cost_mb <= 0:            continue        roi = benefit / cost_mb        if roi < min_roi:            continue        upgrades.append({            "name": name,            "action": action,            "benefit": benefit,            "cost_mb": cost_mb,            "roi": roi,        })    upgrades.sort(key=lambda x: x["roi"], reverse=True)    dynamic = {}    spent_mb = 0.0    for up in upgrades:        if spent_mb + up["cost_mb"] > extra_budget_mb:            continue        spent_mb += up["cost_mb"]        anchored = f"^{re.escape(up['name'])}$"        if up["action"] == "skip":            dynamic[f"-:{anchored}"] = {}        elif up["action"] == "w8_g128":            dynamic[f"+:{anchored}"] = {"bits": 8, "group_size": 128}        elif up["action"] == "w4_g64":            dynamic[f"+:{anchored}"] = {"bits": 4, "group_size": 64}    return dynamic, spent_mb
```

Your verifier pass is the right idea too. `lm-eval.simple_evaluate()` is a valid current entry point, and the HF backend supports local paths via `pretrained=/path/to/model`. I would still save all three runs separately—baseline `gptq_pro()`, heuristic mixed plan, judge mixed plan—and compare them on the same task set. [GitHub+1](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/python-api.md?utm_source=chatgpt.com)

For the quantization run itself, I’d tighten it to this shape:

```Pythonfrom gptqmodel import GPTQModel, QuantizeConfigdef run_quantization(model_id, calib_data, dynamic_overrides, save_path):    qcfg = QuantizeConfig(bits=4, group_size=128, sym=True, desc_act=True, dynamic=dynamic_overrides)    model = GPTQModel.load(model_id, qcfg)    model.quantize(calib_data, batch_size=4)    model.save(save_path)
Pythonfrom gptqmodel import GPTQModel, QuantizeConfigdef run_quantization(model_id, calib_data, dynamic_overrides, save_path):    qcfg = QuantizeConfig(bits=4, group_size=128, sym=True, desc_act=True, dynamic=dynamic_overrides)    model = GPTQModel.load(model_id, qcfg)    model.quantize(calib_data, batch_size=4)    model.save(save_path)
```

My verdict: **the control architecture is now correct**, and the only real remaining gap is making the probe mathematically faithful enough that the judge is ranking real sensitivity instead of artifacts. Once you swap in per-row group quantization and keep exact-name compilation, this becomes a serious SDQ controller rather than a prompt-driven experiment.

---

### User:
You absolutely nailed the mathematical flaw in the v2 fake quantizer. Flattening the entire weight tensor destroys the structural reality of GPTQ, which operates on **local chunks along the input dimension (in_features)** per output channel. If the probe doesn't respect that grouping geometry, the MSE proxy is practically noise, and the LLM judge is optimizing based on hallucinations.

Loading the FP model for probing via standard `transformers` instead of `GPTQModel.load` is also the cleanest boundary: let Transformers handle the pure forward pass for feature extraction, and strictly reserve GPTQModel for the actual quantization execution loop.

Here is the **v3 Production-Ready Architecture**. This integrates your exact per-row grouping math, the strict ROI-driven budget compiler, and a full A/B/C testing suite comparing Baseline, Heuristic, and Judge-driven quantization.

### The Complete SDQ Pipeline for Qwen3.5-4B

#### 1. The Mathematically Faithful Probe
This uses pure PyTorch and Transformers to extract genuine sensitivity metrics.

```
python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

def fake_quant_groupwise_linear_weight(weight: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    """Symmetric per-row, per-group fake quantization for Linear weights."""
    assert weight.dim() == 2
    w = weight.detach().float()
    out_features, in_features = w.shape

    pad = (group_size - (in_features % group_size)) % group_size
    if pad:
        w = F.pad(w, (0, pad), value=0.0)

    maxq = (2 ** (bits - 1)) - 1
    w = w.view(out_features, -1, group_size)  # [out, n_groups, group]
    absmax = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = absmax / maxq

    q = torch.round(w / scale).clamp(-maxq, maxq)
    dq = q * scale
    dq = dq.view(out_features, -1)

    if pad:
        dq = dq[:, :in_features]

    return dq.to(weight.device, dtype=weight.dtype)

def extract_features(model_id, calib_texts):
    print("Loading FP16 model for probing...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
    
    inputs = tokenizer(calib_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    target_types = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head")
    features =[]
    
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or not any(t in name for t in target_types):
            continue

        # 1. Capture Activation
        act =[]
        def hook(m, i, o): act.append(i[0].detach())
        handle = module.register_forward_hook(hook)
        
        with torch.no_grad():
            model(**inputs)
        handle.remove()
        
        x = act[0]
        weight = module.weight.detach()
        bias = None if module.bias is None else module.bias.detach()
        
        # 2. Kurtosis
        w_float = weight.float()
        kurtosis = ((w_float - w_float.mean())**4).mean() / (w_float.var()**2)

        # 3. Simulate Quantization Error
        with torch.no_grad():
            out_fp = F.linear(x, weight, bias)

            w4g128 = fake_quant_groupwise_linear_weight(weight, bits=4, group_size=128)
            out_w4g128 = F.linear(x, w4g128, bias)
            w4g128_mse = F.mse_loss(out_w4g128.float(), out_fp.float()).item()

            w4g64 = fake_quant_groupwise_linear_weight(weight, bits=4, group_size=64)
            out_w4g64 = F.linear(x, w4g64, bias)
            w4g64_mse = F.mse_loss(out_w4g64.float(), out_fp.float()).item()

            w8g128 = fake_quant_groupwise_linear_weight(weight, bits=8, group_size=128)
            out_w8g128 = F.linear(x, w8g128, bias)
            w8g128_mse = F.mse_loss(out_w8g128.float(), out_fp.float()).item()

        params = weight.numel()
        
        features.append({
            "name": name,
            "params": params,
            "kurtosis": kurtosis.item(),
            "w4g128_mse": w4g128_mse,
            "w4g64_mse": w4g64_mse,
            "w8g128_mse": w8g128_mse,
            "w4g64_extra_mb": (params * 0.5) / (1024**2) * 0.1, # Est. group overhead
            "w8g128_extra_mb": (params * 0.5) / (1024**2),      # 4 extra bits per param
        })
        
    del model # Free VRAM for GPTQ step
    torch.cuda.empty_cache()
    return features
```

#### 2. The Strict ROI Compiler
This enforces your mathematical guards (`benefit > 0`, `min_roi`) and locks the output into `GPTQModel`’s exact regex format.

```
python
import re

def compile_dynamic_config(judge_proposals, probe_features, extra_budget_mb=150.0, min_roi=1e-6):
    feat_by_name = {f["name"]: f for f in probe_features}
    upgrades =[]

    for name, action in judge_proposals.items():
        if action == "w4_g128": continue

        feat = feat_by_name.get(name)
        if feat is None: continue

        if action == "w8_g128":
            benefit = feat["w4g128_mse"] - feat["w8g128_mse"]
            cost_mb = feat["w8g128_extra_mb"]
        elif action == "w4_g64":
            benefit = feat["w4g128_mse"] - feat["w4g64_mse"]
            cost_mb = feat["w4g64_extra_mb"]
        elif action == "skip":
            benefit = feat["w4g128_mse"]
            cost_mb = (feat["params"] * 1.5) / (1024 ** 2)  # fp16 bytes (2.0) - w4 bytes (0.5)
        else: continue

        # Strict Guards
        if benefit <= 0 or cost_mb <= 0: continue
        
        roi = benefit / cost_mb
        if roi < min_roi: continue

        upgrades.append({
            "name": name, "action": action, 
            "cost_mb": cost_mb, "roi": roi
        })

    # Sort strictly by Return on Investment
    upgrades.sort(key=lambda x: x["roi"], reverse=True)

    dynamic = {}
    spent_mb = 0.0

    for up in upgrades:
        if spent_mb + up["cost_mb"] > extra_budget_mb:
            continue

        spent_mb += up["cost_mb"]
        # Exact module match via anchored regex
        anchored = f"^{re.escape(up['name'])}$"

        if up["action"] == "skip":
            dynamic[f"-:{anchored}"] = {}
        elif up["action"] == "w8_g128":
            dynamic[f"+:{anchored}"] = {"bits": 8, "group_size": 128}
        elif up["action"] == "w4_g64":
            dynamic[f"+:{anchored}"] = {"bits": 4, "group_size": 64}

    print(f"Compiler Allocated {spent_mb:.1f}MB / {extra_budget_mb}MB across {len(dynamic)} layers.")
    return dynamic
```

#### 3. Execution & The Verifier Suite
We map everything back to `GPTQModel` and `lm-eval`.

```
python
from gptqmodel import GPTQModel, QuantizeConfig
import lm_eval

def run_quantization(model_id, calib_data, dynamic_overrides, save_path):
    print(f"\n--- Quantizing into {save_path} ---")
    qcfg = QuantizeConfig(
        bits=4, 
        group_size=128, 
        sym=True, 
        desc_act=True, 
        dynamic=dynamic_overrides
    )
    
    # Standard GPTQModel documented flow
    model = GPTQModel.load(model_id, quant_config=qcfg)
    model.quantize(calib_data, batch_size=4)
    model.save(save_path)
    
    # Cleanup to avoid VRAM OOMs during eval
    del model
    torch.cuda.empty_cache()

def evaluate_model(model_path):
    print(f"\n--- Evaluating {model_path} ---")
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path}",
        tasks=["arc_challenge", "hellaswag"],
        batch_size="auto",
        device="cuda:0"
    )
    arc_acc = results['results']['arc_challenge']['acc_norm,none']
    hs_acc = results['results']['hellaswag']['acc_norm,none']
    return arc_acc, hs_acc

# ==========================================
# Master Pipeline Execution
# ==========================================
model_id = "Qwen/Qwen3.5-4B-Base"
calib_dataset =["Calibration text sample 1...", "Calibration text sample 2..."]

# 1. Feature Extraction
features = extract_features(model_id, calib_dataset)

# 2. Get LLM Judge Proposals (Mocked here)
# -> You pass `features` to your LLM and parse the JSON response.
judge_proposals = {
    "model.layers.31.mlp.down_proj": "w8_g128",
    "lm_head": "skip",
    "model.layers.0.self_attn.o_proj": "w4_g64"
}

# 3. Compile Configs
judge_dynamic = compile_dynamic_config(judge_proposals, features, extra_budget_mb=150)

# Build a baseline heuristic for A/B testing
heuristic_dynamic = {
    r"-:^lm_head$": {},
    r"+:.*down_proj$": {"bits": 4, "group_size": 64}
}

# 4. Run Suite
configs = {
    "baseline_w4g128": None,
    "heuristic_mixed": heuristic_dynamic,
    "judge_sdq": judge_dynamic
}

scores = {}
for name, dynamic_cfg in configs.items():
    save_path = f"./models/{model_id.split('/')[-1]}-{name}"
    run_quantization(model_id, calib_dataset, dynamic_cfg, save_path)
    arc, hs = evaluate_model(save_path)
    scores[name] = {"ARC": arc, "HellaSwag": hs}

print("\n=== Final SDQ Leaderboard ===")
for name, score in scores.items():
    print(f"{name:15} | ARC: {score['ARC']:.4f} | HS: {score['HellaSwag']:.4f}")
```

### Why this architecture is fundamentally correct:
1. **Mathematical Isolation:** Fake quantization correctly models the physical chunk boundaries of the eventual kernel, validating true sensitivity.
2. **Economic Guardrails:** The LLM's role is restricted to a policy proposal. The deterministic compiler converts that into an exact ROI curve and enforces the VRAM budget, ignoring "hallucinated urgency" if the math says the MSE delta isn't worth it.
3. **API Alignment:** It correctly uses `transformers` for pure FP16 probing, and leaves the actual tensor modification to `GPTQModel` using the official `QuantizeConfig.dynamic` surface.

This is arguably the most sophisticated way to run local GPTQ compression currently available. provide all project in one answer with json format

---

### ChatGPT:
```JSON{  "project_name": "sdq_quant_judge_qwen35_gptqmodel",  "files": [    {      "path": "README.md",      "content": "# SDQ Quant Judge for GPTQModel\n\nA software-defined quantization scaffold for Qwen/Qwen3.5-4B-Base.\n\nCommands:\n- probe: extract per-module fake-quant sensitivity features from the FP model\n- plan: build a mixed-precision plan with either heuristic logic or an external OpenAI-compatible judge\n- quantize: quantize with GPTQModel using the compiled dynamic overrides\n- eval: evaluate a saved model with lm-eval\n- suite: run baseline vs heuristic vs judge and save a leaderboard\n\nExample:\n\n```bash\npython -m venv .venv\nsource .venv/bin/activate\npip install -r requirements.txt\n\npython -m quant_judge.pipeline probe \\\n  --model-id Qwen/Qwen3.5-4B-Base \\\n  --out-dir ./runs/probe \\\n  --dataset-name wikitext \\\n  --dataset-config wikitext-2-raw-v1 \\\n  --split train \\\n  --text-field text \\\n  --num-texts 64 \\\n  --max-length 256 \\\n  --max-rows-per-module 128 \\\n  --include-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head\n\npython -m quant_judge.pipeline plan \\\n  --features-jsonl ./runs/probe/probe_features.jsonl \\\n  --plan-out ./runs/plan.json \\\n  --dynamic-out ./runs/dynamic.json \\\n  --budget-mb-extra 150 \\\n  --heuristic-only\n\npython -m quant_judge.pipeline quantize \\\n  --model-id Qwen/Qwen3.5-4B-Base \\\n  --plan-json ./runs/plan.json \\\n  --out-dir ./runs/quantized \\\n  --dataset-name wikitext \\\n  --dataset-config wikitext-2-raw-v1 \\\n  --split train \\\n  --text-field text \\\n  --num-texts 128 \\\n  --batch-size 4\n\npython -m quant_judge.pipeline eval \\\n  --model-path ./runs/quantized \\\n  --tasks arc_challenge,hellaswag \\\n  --output-json ./runs/eval.json\n```\n"    },    {      "path": "requirements.txt",      "content": "transformers\ndatasets\naccelerate\nrequests\nnumpy\ntqdm\npsutil\ngptqmodel\nlm-eval\n"    },    {      "path": "quant_judge/__init__.py",      "content": "\"\"\"SDQ quantization scaffold.\"\"\"\n"    },    {      "path": "quant_judge/actions.py",      "content": "from __future__ import annotations\n\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass ActionSpec:\n    name: str\n    bits: int | None\n    group_size: int | None\n    skip: bool\n\n\nACTION_SPECS: dict[str, ActionSpec] = {\n    \"w4_g128\": ActionSpec(\"w4_g128\", bits=4, group_size=128, skip=False),\n    \"w4_g64\": ActionSpec(\"w4_g64\", bits=4, group_size=64, skip=False),\n    \"w8_g128\": ActionSpec(\"w8_g128\", bits=8, group_size=128, skip=False),\n    \"skip\": ActionSpec(\"skip\", bits=None, group_size=None, skip=True),\n}\n\nBASE_ACTION = \"w4_g128\"\n\n\ndef allowed_actions() -> list[str]:\n    return list(ACTION_SPECS.keys())\n\n\ndef bytes_per_param(action: str, fp_bytes: float = 2.0) -> float:\n    if action == \"skip\":\n        return fp_bytes\n    spec = ACTION_SPECS[action]\n    assert spec.bits is not None\n    return spec.bits / 8.0\n\n\ndef extra_mb_vs_base(params: int, action: str) -> float:\n    base = params * bytes_per_param(BASE_ACTION)\n    now = params * bytes_per_param(action)\n    return (now - base) / (1024 ** 2)\n\n\ndef action_to_override(action: str) -> dict:\n    if action == \"skip\":\n        return {}\n    spec = ACTION_SPECS[action]\n    return {\"bits\": spec.bits, \"group_size\": spec.group_size}\n"    },    {      "path": "quant_judge/probe.py",      "content": "from __future__ import annotations\n\nimport json\nfrom dataclasses import dataclass\nfrom pathlib import Path\n\nimport torch\nimport torch.nn.functional as F\nfrom datasets import load_dataset\nfrom tqdm import tqdm\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\n\nfrom .actions import extra_mb_vs_base\n\n\n@dataclass\nclass ProbeConfig:\n    model_id: str\n    out_dir: str\n    dataset_name: str\n    dataset_config: str | None\n    split: str\n    text_field: str\n    num_texts: int\n    max_length: int\n    max_rows_per_module: int\n    include_modules: list[str]\n    trust_remote_code: bool = True\n\n\ndef pick_dtype() -> torch.dtype:\n    if not torch.cuda.is_available():\n        return torch.float32\n    major, _minor = torch.cuda.get_device_capability(0)\n    if major >= 8:\n        return torch.bfloat16\n    return torch.float16\n\n\ndef fake_quant_groupwise_linear_weight(weight: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:\n    assert weight.dim() == 2\n    w = weight.detach().float()\n    out_features, in_features = w.shape\n    pad = (group_size - (in_features % group_size)) % group_size\n    if pad:\n        w = F.pad(w, (0, pad), value=0.0)\n    maxq = (2 ** (bits - 1)) - 1\n    w = w.view(out_features, -1, group_size)\n    absmax = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)\n    scale = absmax / maxq\n    q = torch.round(w / scale).clamp(-maxq, maxq)\n    dq = q * scale\n    dq = dq.view(out_features, -1)\n    if pad:\n        dq = dq[:, :in_features]\n    return dq.to(weight.device, dtype=weight.dtype)\n\n\ndef load_texts(dataset_name: str, dataset_config: str | None, split: str, text_field: str, num_texts: int) -> list[str]:\n    ds = load_dataset(dataset_name, dataset_config, split=split)\n    texts: list[str] = []\n    for row in ds:\n        value = row.get(text_field, \"\")\n        if isinstance(value, str) and value.strip():\n            texts.append(value.strip())\n        if len(texts) >= num_texts:\n            break\n    if not texts:\n        raise ValueError(\"No usable texts found.\")\n    return texts\n\n\ndef module_selected(name: str, include_modules: list[str]) -> bool:\n    return any(token in name for token in include_modules)\n\n\ndef iter_target_linears(model, include_modules: list[str]):\n    for name, module in model.named_modules():\n        if isinstance(module, torch.nn.Linear) and module_selected(name, include_modules):\n            yield name, module\n\n\ndef flatten_rows(x: torch.Tensor) -> torch.Tensor:\n    if x.dim() == 2:\n        return x\n    if x.dim() >= 3:\n        return x.reshape(-1, x.shape[-1])\n    return x.unsqueeze(0)\n\n\ndef sample_rows(x: torch.Tensor, max_rows: int) -> torch.Tensor:\n    rows = flatten_rows(x)\n    if rows.shape[0] <= max_rows:\n        return rows\n    idx = torch.randperm(rows.shape[0], device=rows.device)[:max_rows]\n    return rows.index_select(0, idx)\n\n\ndef weight_kurtosis(weight: torch.Tensor) -> float:\n    w = weight.detach().float().flatten()\n    mu = w.mean()\n    var = ((w - mu) ** 2).mean().clamp_min(1e-12)\n    return float((((w - mu) ** 4).mean() / (var ** 2)).item())\n\n\ndef activation_outlier_rate(x: torch.Tensor, sigma: float = 6.0) -> float:\n    xf = x.float()\n    std = xf.std().clamp_min(1e-6)\n    return float((xf.abs() > sigma * std).float().mean().item())\n\n\ndef block_index_from_name(name: str) -> int | None:\n    parts = name.split(\".\")\n    for i, p in enumerate(parts):\n        if p == \"layers\" and i + 1 < len(parts):\n            try:\n                return int(parts[i + 1])\n            except Exception:\n                return None\n    return None\n\n\ndef reduced_metrics_for_action(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, bits: int, group_size: int) -> dict:\n    wq = fake_quant_groupwise_linear_weight(weight, bits=bits, group_size=group_size)\n    out_fp = F.linear(x, weight, bias)\n    out_q = F.linear(x, wq, bias)\n    mse = F.mse_loss(out_q.float(), out_fp.float()).item()\n    cos = F.cosine_similarity(out_q.float(), out_fp.float(), dim=-1).mean().item()\n    max_abs = (out_q.float() - out_fp.float()).abs().max().item()\n    return {\"mse\": float(mse), \"cosine\": float(cos), \"max_abs\": float(max_abs)}\n\n\ndef run_probe(cfg: ProbeConfig) -> dict:\n    out_dir = Path(cfg.out_dir)\n    out_dir.mkdir(parents=True, exist_ok=True)\n\n    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code)\n    if tokenizer.pad_token_id is None:\n        tokenizer.pad_token = tokenizer.eos_token\n\n    model = AutoModelForCausalLM.from_pretrained(\n        cfg.model_id,\n        trust_remote_code=cfg.trust_remote_code,\n        torch_dtype=pick_dtype(),\n        device_map=\"auto\",\n        low_cpu_mem_usage=True,\n    )\n    model.eval()\n\n    texts = load_texts(cfg.dataset_name, cfg.dataset_config, cfg.split, cfg.text_field, cfg.num_texts)\n    targets = list(iter_target_linears(model, cfg.include_modules))\n    if not targets:\n        raise ValueError(\"No target modules found.\")\n\n    captured_inputs = {name: [] for name, _ in targets}\n    remaining_rows = {name: cfg.max_rows_per_module for name, _ in targets}\n    handles = []\n\n    def make_hook(name: str):\n        def hook(_module, inp, _out):\n            rem = remaining_rows[name]\n            if rem <= 0 or not inp or not isinstance(inp[0], torch.Tensor):\n                return\n            x = sample_rows(inp[0], rem)\n            take = min(x.shape[0], rem)\n            if take <= 0:\n                return\n            captured_inputs[name].append(x[:take].detach().cpu().to(torch.float16))\n            remaining_rows[name] -= take\n        return hook\n\n    for name, module in targets:\n        handles.append(module.register_forward_hook(make_hook(name)))\n\n    with torch.no_grad():\n        for text in tqdm(texts, desc=\"collect_activations\"):\n            batch = tokenizer(text, return_tensors=\"pt\", truncation=True, max_length=cfg.max_length, padding=False)\n            batch = {k: v.to(model.device) for k, v in batch.items()}\n            _ = model(**batch)\n            if all(v <= 0 for v in remaining_rows.values()):\n                break\n\n    for h in handles:\n        h.remove()\n\n    rows = []\n    for name, module in tqdm(targets, desc=\"probe_modules\"):\n        if not captured_inputs[name]:\n            continue\n        x = torch.cat(captured_inputs[name], dim=0).to(module.weight.device, dtype=module.weight.dtype)\n        weight = module.weight.detach()\n        bias = None if module.bias is None else module.bias.detach()\n\n        m_w4g128 = reduced_metrics_for_action(x, weight, bias, bits=4, group_size=128)\n        m_w4g64 = reduced_metrics_for_action(x, weight, bias, bits=4, group_size=64)\n        m_w8g128 = reduced_metrics_for_action(x, weight, bias, bits=8, group_size=128)\n\n        params = int(weight.numel())\n        row = {\n            \"name\": name,\n            \"module_type\": name.split(\".\")[-1],\n            \"block_index\": block_index_from_name(name),\n            \"params\": params,\n            \"captured_rows\": int(x.shape[0]),\n            \"weight_kurtosis\": weight_kurtosis(weight),\n            \"act_outlier_rate\": activation_outlier_rate(x),\n            \"candidates\": {\n                \"w4_g128\": {**m_w4g128, \"estimated_extra_mb_vs_base\": 0.0},\n                \"w4_g64\": {**m_w4g64, \"estimated_extra_mb_vs_base\": max(0.0, extra_mb_vs_base(params, \"w4_g64\")) + (params * 0.05 / (1024 ** 2))},\n                \"w8_g128\": {**m_w8g128, \"estimated_extra_mb_vs_base\": max(0.0, extra_mb_vs_base(params, \"w8_g128\"))},\n                \"skip\": {\"mse\": 0.0, \"cosine\": 1.0, \"max_abs\": 0.0, \"estimated_extra_mb_vs_base\": max(0.0, extra_mb_vs_base(params, \"skip\"))},\n            },\n            \"proxy_gain_w4g64_vs_w4\": float(max(0.0, m_w4g128[\"mse\"] - m_w4g64[\"mse\"])),\n            \"proxy_gain_w8_vs_w4\": float(max(0.0, m_w4g128[\"mse\"] - m_w8g128[\"mse\"])),\n        }\n        rows.append(row)\n\n    features_jsonl = out_dir / \"probe_features.jsonl\"\n    metadata_json = out_dir / \"probe_metadata.json\"\n    summary_json = out_dir / \"probe_summary.json\"\n\n    with open(features_jsonl, \"w\", encoding=\"utf-8\") as f:\n        for row in rows:\n            f.write(json.dumps(row, ensure_ascii=False) + \"\\n\")\n\n    metadata = {\n        \"model_id\": cfg.model_id,\n        \"dataset_name\": cfg.dataset_name,\n        \"dataset_config\": cfg.dataset_config,\n        \"split\": cfg.split,\n        \"text_field\": cfg.text_field,\n        \"num_texts\": cfg.num_texts,\n        \"max_length\": cfg.max_length,\n        \"max_rows_per_module\": cfg.max_rows_per_module,\n        \"include_modules\": cfg.include_modules,\n        \"num_rows\": len(rows),\n    }\n    metadata_json.write_text(json.dumps(metadata, indent=2), encoding=\"utf-8\")\n    summary_json.write_text(json.dumps({\"num_rows\": len(rows)}, indent=2), encoding=\"utf-8\")\n\n    del model\n    if torch.cuda.is_available():\n        torch.cuda.empty_cache()\n\n    return {\"features_jsonl\": str(features_jsonl), \"metadata_json\": str(metadata_json), \"summary_json\": str(summary_json), \"num_rows\": len(rows)}\n"    },    {      "path": "quant_judge/judge.py",      "content": "from __future__ import annotations\n\nimport json\nimport os\nfrom dataclasses import dataclass\n\nimport requests\n\nfrom .actions import ACTION_SPECS, BASE_ACTION, allowed_actions\n\n\n@dataclass\nclass JudgeConfig:\n    features_jsonl: str\n    budget_mb_extra: float\n    heuristic_only: bool = False\n    judge_base_url: str | None = None\n    judge_model: str | None = None\n    judge_api_key_env: str = \"JUDGE_API_KEY\"\n    chunk_size: int = 24\n\n\ndef load_rows(path: str) -> list[dict]:\n    rows = []\n    with open(path, \"r\", encoding=\"utf-8\") as f:\n        for line in f:\n            line = line.strip()\n            if line:\n                rows.append(json.loads(line))\n    return rows\n\n\ndef heuristic_decision(row: dict) -> dict:\n    name = row[\"name\"]\n    w4 = row[\"candidates\"][\"w4_g128\"][\"mse\"]\n    w4g64 = row[\"candidates\"][\"w4_g64\"][\"mse\"]\n    w8 = row[\"candidates\"][\"w8_g128\"][\"mse\"]\n    kurt = row[\"weight_kurtosis\"]\n    outlier = row[\"act_outlier_rate\"]\n    block = row.get(\"block_index\")\n\n    w8_relief = (w4 - w8) / max(w4, 1e-9)\n    g64_relief = (w4 - w4g64) / max(w4, 1e-9)\n\n    score = 0.0\n    score += min(1.0, w8_relief / 0.60) * 0.40\n    score += min(1.0, g64_relief / 0.35) * 0.20\n    score += min(1.0, kurt / 20.0) * 0.15\n    score += min(1.0, outlier / 0.05) * 0.15\n    if \"lm_head\" in name:\n        score += 0.25\n    if row[\"module_type\"] in (\"down_proj\", \"o_proj\", \"gate_proj\"):\n        score += 0.10\n    if block is not None and (block <= 1 or block >= 30):\n        score += 0.10\n    score = max(0.0, min(1.0, score))\n\n    action = BASE_ACTION\n    reason = \"default\"\n    if \"lm_head\" in name and (w8_relief > 0.55 or kurt > 25.0):\n        action = \"skip\"\n        reason = \"lm_head_extreme\"\n    elif score >= 0.75 or w8_relief > 0.50:\n        action = \"w8_g128\"\n        reason = \"high_sensitivity\"\n    elif score >= 0.42 or g64_relief > 0.18:\n        action = \"w4_g64\"\n        reason = \"moderate_sensitivity\"\n\n    return {\"name\": name, \"action\": action, \"priority\": round(score, 6), \"reason_code\": reason, \"source\": \"heuristic\"}\n\n\ndef reduced_row(row: dict) -> dict:\n    return {\n        \"name\": row[\"name\"],\n        \"module_type\": row[\"module_type\"],\n        \"block_index\": row.get(\"block_index\"),\n        \"params\": row[\"params\"],\n        \"weight_kurtosis\": row[\"weight_kurtosis\"],\n        \"act_outlier_rate\": row[\"act_outlier_rate\"],\n        \"w4_g128_mse\": row[\"candidates\"][\"w4_g128\"][\"mse\"],\n        \"w4_g64_mse\": row[\"candidates\"][\"w4_g64\"][\"mse\"],\n        \"w8_g128_mse\": row[\"candidates\"][\"w8_g128\"][\"mse\"],\n        \"extra_mb_w4_g64\": row[\"candidates\"][\"w4_g64\"][\"estimated_extra_mb_vs_base\"],\n        \"extra_mb_w8_g128\": row[\"candidates\"][\"w8_g128\"][\"estimated_extra_mb_vs_base\"],\n        \"extra_mb_skip\": row[\"candidates\"][\"skip\"][\"estimated_extra_mb_vs_base\"],\n    }\n\n\ndef strip_code_fence(text: str) -> str:\n    t = text.strip()\n    if t.startswith(\"```\"):\n        t = t.split(\"\\n\", 1)[1]\n        if t.endswith(\"```\"):\n            t = t.rsplit(\"\\n\", 1)[0]\n    return t.strip()\n\n\ndef chunk(rows: list[dict], size: int) -> list[list[dict]]:\n    return [rows[i:i + size] for i in range(0, len(rows), size)]\n\n\ndef call_openai_compatible(base_url: str, model: str, api_key: str, rows: list[dict], budget_mb_extra: float) -> list[dict]:\n    system = (\n        \"You are a quantization policy judge. \"\n        \"Choose exactly one action per module from: \" + \", \".join(allowed_actions()) + \". \"\n        \"Prefer w4_g128 unless the evidence shows fragility. \"\n        \"Use w4_g64 for moderate sensitivity. Use w8_g128 for high sensitivity. \"\n        \"Use skip only for extreme outliers, usually lm_head. \"\n        \"Return JSON only with schema {\\\\\\\"assignments\\\\\\\":[{\\\\\\\"name\\\\\\\":...,\\\\\\\"action\\\\\\\":...,\\\\\\\"priority\\\\\\\":0..1,\\\\\\\"reason_code\\\\\\\":...}]}\"\n    )\n    user = {\"budget_mb_extra_global\": budget_mb_extra, \"allowed_actions\": allowed_actions(), \"rows\": rows}\n    url = base_url.rstrip(\"/\") + \"/chat/completions\"\n    headers = {\"Content-Type\": \"application/json\"}\n    if api_key:\n        headers[\"Authorization\"] = f\"Bearer {api_key}\"\n    body = {\n        \"model\": model,\n        \"temperature\": 0.0,\n        \"response_format\": {\"type\": \"json_object\"},\n        \"messages\": [\n            {\"role\": \"system\", \"content\": system},\n            {\"role\": \"user\", \"content\": json.dumps(user, ensure_ascii=False)},\n        ],\n    }\n    resp = requests.post(url, headers=headers, json=body, timeout=180)\n    resp.raise_for_status()\n    content = resp.json()[\"choices\"][0][\"message\"][\"content\"]\n    parsed = json.loads(strip_code_fence(content))\n    assignments = parsed.get(\"assignments\", [])\n    if not isinstance(assignments, list):\n        raise ValueError(\"Judge response missing assignments.\")\n    return assignments\n\n\ndef merge_assignments(rows: list[dict], proposed: list[dict]) -> list[dict]:\n    by_name = {a[\"name\"]: a for a in proposed if isinstance(a, dict) and \"name\" in a}\n    merged = []\n    for row in rows:\n        item = by_name.get(row[\"name\"])\n        if item is None:\n            merged.append(heuristic_decision(row))\n            continue\n        action = item.get(\"action\", BASE_ACTION)\n        if action not in ACTION_SPECS:\n            action = BASE_ACTION\n        try:\n            priority = float(item.get(\"priority\", 0.5))\n        except Exception:\n            priority = 0.5\n        priority = max(0.0, min(1.0, priority))\n        merged.append({\"name\": row[\"name\"], \"action\": action, \"priority\": priority, \"reason_code\": str(item.get(\"reason_code\", \"judge\")), \"source\": \"judge\"})\n    return merged\n\n\ndef make_judge_proposals(cfg: JudgeConfig) -> dict:\n    rows = load_rows(cfg.features_jsonl)\n    if cfg.heuristic_only:\n        return {\"assignments\": [heuristic_decision(r) for r in rows]}\n    if not cfg.judge_base_url or not cfg.judge_model:\n        raise ValueError(\"Judge mode requires judge_base_url and judge_model.\")\n    api_key = os.environ.get(cfg.judge_api_key_env, \"\")\n    assignments = []\n    for part in chunk(rows, cfg.chunk_size):\n        reduced = [reduced_row(r) for r in part]\n        try:\n            proposed = call_openai_compatible(cfg.judge_base_url, cfg.judge_model, api_key, reduced, cfg.budget_mb_extra)\n            assignments.extend(merge_assignments(part, proposed))\n        except Exception:\n            assignments.extend([heuristic_decision(r) for r in part])\n    return {\"assignments\": assignments}\n"    },    {      "path": "quant_judge/compiler.py",      "content": "from __future__ import annotations\n\nimport json\nimport re\nfrom pathlib import Path\n\nfrom .actions import BASE_ACTION, action_to_override\n\n\ndef compile_plan(assignments: list[dict], probe_rows: list[dict], extra_budget_mb: float, min_roi: float = 1e-6) -> dict:\n    probe_by_name = {row[\"name\"]: row for row in probe_rows}\n    upgrades = []\n    for item in assignments:\n        name = item[\"name\"]\n        action = item[\"action\"]\n        priority = float(item.get(\"priority\", 0.5))\n        if action == BASE_ACTION:\n            continue\n        row = probe_by_name.get(name)\n        if row is None:\n            continue\n        if action == \"w8_g128\":\n            benefit = row[\"candidates\"][\"w4_g128\"][\"mse\"] - row[\"candidates\"][\"w8_g128\"][\"mse\"]\n            cost_mb = row[\"candidates\"][\"w8_g128\"][\"estimated_extra_mb_vs_base\"]\n        elif action == \"w4_g64\":\n            benefit = row[\"candidates\"][\"w4_g128\"][\"mse\"] - row[\"candidates\"][\"w4_g64\"][\"mse\"]\n            cost_mb = row[\"candidates\"][\"w4_g64\"][\"estimated_extra_mb_vs_base\"]\n        elif action == \"skip\":\n            benefit = row[\"candidates\"][\"w4_g128\"][\"mse\"]\n            cost_mb = row[\"candidates\"][\"skip\"][\"estimated_extra_mb_vs_base\"]\n        else:\n            continue\n        if benefit <= 0 or cost_mb <= 0:\n            continue\n        roi = benefit / cost_mb\n        if roi < min_roi:\n            continue\n        upgrades.append({\n            \"name\": name,\n            \"action\": action,\n            \"benefit\": float(benefit),\n            \"cost_mb\": float(cost_mb),\n            \"roi\": float(roi),\n            \"priority\": priority,\n            \"reason_code\": item.get(\"reason_code\", \"\"),\n            \"source\": item.get(\"source\", \"\"),\n        })\n    upgrades.sort(key=lambda x: (x[\"roi\"], x[\"priority\"]), reverse=True)\n\n    approved = []\n    spent_mb = 0.0\n    for up in upgrades:\n        if spent_mb + up[\"cost_mb\"] > extra_budget_mb:\n            continue\n        spent_mb += up[\"cost_mb\"]\n        approved.append(up)\n\n    approved_by_name = {x[\"name\"]: x for x in approved}\n    final_assignments = []\n    for row in probe_rows:\n        ap = approved_by_name.get(row[\"name\"])\n        if ap is None:\n            final_assignments.append({\"name\": row[\"name\"], \"action\": BASE_ACTION, \"priority\": 0.0, \"reason_code\": \"base_default\", \"source\": \"compiler\", \"estimated_extra_mb_vs_base\": 0.0})\n        else:\n            final_assignments.append({\"name\": ap[\"name\"], \"action\": ap[\"action\"], \"priority\": ap[\"priority\"], \"reason_code\": ap[\"reason_code\"], \"source\": ap[\"source\"], \"estimated_extra_mb_vs_base\": ap[\"cost_mb\"], \"roi\": ap[\"roi\"], \"benefit\": ap[\"benefit\"]})\n    final_assignments.sort(key=lambda x: x[\"name\"])\n    return {\"budget_mb_extra\": float(extra_budget_mb), \"used_mb_extra\": round(float(spent_mb), 6), \"upgrades_considered\": len(upgrades), \"upgrades_approved\": len(approved), \"assignments\": final_assignments}\n\n\ndef plan_to_dynamic(plan: dict) -> dict:\n    dynamic = {}\n    for item in plan[\"assignments\"]:\n        name = item[\"name\"]\n        action = item[\"action\"]\n        if action == BASE_ACTION:\n            continue\n        anchored = f\"^{re.escape(name)}$\"\n        if action == \"skip\":\n            dynamic[f\"-:{anchored}\"] = {}\n        else:\n            dynamic[f\"+:{anchored}\"] = action_to_override(action)\n    return dynamic\n\n\ndef save_plan_and_dynamic(plan: dict, dynamic: dict, plan_out: str, dynamic_out: str) -> None:\n    Path(plan_out).parent.mkdir(parents=True, exist_ok=True)\n    Path(dynamic_out).parent.mkdir(parents=True, exist_ok=True)\n    Path(plan_out).write_text(json.dumps(plan, indent=2), encoding=\"utf-8\")\n    Path(dynamic_out).write_text(json.dumps(dynamic, indent=2), encoding=\"utf-8\")\n"    },    {      "path": "quant_judge/runner.py",      "content": "from __future__ import annotations\n\nimport json\nimport subprocess\nimport sys\nfrom pathlib import Path\n\nimport torch\nfrom datasets import load_dataset\nfrom gptqmodel import GPTQModel, QuantizeConfig\n\n\ndef load_texts(dataset_name: str, dataset_config: str | None, split: str, text_field: str, num_texts: int) -> list[str]:\n    ds = load_dataset(dataset_name, dataset_config, split=split)\n    texts = []\n    for row in ds:\n        text = row.get(text_field, \"\")\n        if isinstance(text, str) and text.strip():\n            texts.append(text.strip())\n        if len(texts) >= num_texts:\n            break\n    if not texts:\n        raise ValueError(\"No texts found for quantization.\")\n    return texts\n\n\ndef quantize_with_dynamic(model_id: str, calibration_texts: list[str], out_dir: str, dynamic: dict | None, bits: int = 4, group_size: int = 128, batch_size: int = 4, sym: bool = True, desc_act: bool = True) -> dict:\n    qcfg = QuantizeConfig(bits=bits, group_size=group_size, sym=sym, desc_act=desc_act, dynamic=dynamic)\n    model = GPTQModel.load(model_id, qcfg)\n    model.quantize(calibration_texts, batch_size=batch_size)\n    out = Path(out_dir)\n    out.mkdir(parents=True, exist_ok=True)\n    model.save(str(out))\n    meta = {\"model_id\": model_id, \"out_dir\": str(out), \"bits\": bits, \"group_size\": group_size, \"batch_size\": batch_size, \"dynamic_rules\": 0 if dynamic is None else len(dynamic)}\n    (out / \"quant_meta.json\").write_text(json.dumps(meta, indent=2), encoding=\"utf-8\")\n    del model\n    if torch.cuda.is_available():\n        torch.cuda.empty_cache()\n    return meta\n\n\ndef heuristic_dynamic_from_flags(flags: list[str]) -> dict:\n    dynamic = {}\n    for flag in flags:\n        if flag == \"skip_lm_head\":\n            dynamic[r\"-:^lm_head$\"] = {}\n        elif flag == \"down_proj_g64\":\n            dynamic[r\"+:.*down_proj$\"] = {\"bits\": 4, \"group_size\": 64}\n        elif flag == \"o_proj_g64\":\n            dynamic[r\"+:.*o_proj$\"] = {\"bits\": 4, \"group_size\": 64}\n        elif flag == \"edge_blocks_w8\":\n            dynamic[r\"+:^model\\\\.layers\\\\.(0|1|30|31)\\\\..*$\"] = {\"bits\": 8, \"group_size\": 128}\n    return dynamic\n\n\ndef run_lm_eval(model_path: str, tasks: list[str], output_json: str, device: str = \"cuda:0\", batch_size: str = \"auto\") -> dict:\n    output_path = Path(output_json)\n    output_path.parent.mkdir(parents=True, exist_ok=True)\n    cmd = [\n        sys.executable, \"-m\", \"lm_eval\",\n        \"--model\", \"hf\",\n        \"--model_args\", f\"pretrained={model_path},trust_remote_code=True\",\n        \"--tasks\", \",\".join(tasks),\n        \"--device\", device,\n        \"--batch_size\", batch_size,\n        \"--output_path\", str(output_path.parent),\n    ]\n    subprocess.run(cmd, check=True)\n    for path in sorted(output_path.parent.rglob(\"*.json\")):\n        try:\n            data = json.loads(path.read_text(encoding=\"utf-8\"))\n        except Exception:\n            continue\n        if isinstance(data, dict) and \"results\" in data:\n            output_path.write_text(json.dumps(data, indent=2), encoding=\"utf-8\")\n            return data\n    raise FileNotFoundError(\"Could not find lm_eval results json.\")\n\n\ndef extract_primary_scores(eval_json: dict) -> dict:\n    out = {}\n    for task, metrics in eval_json.get(\"results\", {}).items():\n        if \"acc_norm,none\" in metrics:\n            out[task] = metrics[\"acc_norm,none\"]\n        elif \"acc,none\" in metrics:\n            out[task] = metrics[\"acc,none\"]\n        elif \"word_perplexity,none\" in metrics:\n            out[task] = metrics[\"word_perplexity,none\"]\n    return out\n"    },    {      "path": "quant_judge/pipeline.py",      "content": "from __future__ import annotations\n\nimport argparse\nimport json\nfrom pathlib import Path\n\nfrom .compiler import compile_plan, plan_to_dynamic, save_plan_and_dynamic\nfrom .judge import JudgeConfig, load_rows, make_judge_proposals\nfrom .probe import ProbeConfig, run_probe\nfrom .runner import extract_primary_scores, heuristic_dynamic_from_flags, load_texts, quantize_with_dynamic, run_lm_eval\n\n\ndef cmd_probe(args):\n    cfg = ProbeConfig(\n        model_id=args.model_id,\n        out_dir=args.out_dir,\n        dataset_name=args.dataset_name,\n        dataset_config=args.dataset_config,\n        split=args.split,\n        text_field=args.text_field,\n        num_texts=args.num_texts,\n        max_length=args.max_length,\n        max_rows_per_module=args.max_rows_per_module,\n        include_modules=[x.strip() for x in args.include_modules.split(\",\") if x.strip()],\n    )\n    print(json.dumps(run_probe(cfg), indent=2))\n\n\ndef cmd_plan(args):\n    judge_cfg = JudgeConfig(\n        features_jsonl=args.features_jsonl,\n        budget_mb_extra=args.budget_mb_extra,\n        heuristic_only=args.heuristic_only,\n        judge_base_url=args.judge_base_url,\n        judge_model=args.judge_model,\n        judge_api_key_env=args.judge_api_key_env,\n        chunk_size=args.chunk_size,\n    )\n    proposals = make_judge_proposals(judge_cfg)\n    probe_rows = load_rows(args.features_jsonl)\n    plan = compile_plan(proposals[\"assignments\"], probe_rows, args.budget_mb_extra, args.min_roi)\n    dynamic = plan_to_dynamic(plan)\n    save_plan_and_dynamic(plan, dynamic, args.plan_out, args.dynamic_out)\n    print(json.dumps({\"plan_out\": args.plan_out, \"dynamic_out\": args.dynamic_out, \"used_mb_extra\": plan[\"used_mb_extra\"], \"upgrades_approved\": plan[\"upgrades_approved\"], \"dynamic_rules\": len(dynamic)}, indent=2))\n\n\ndef cmd_quantize(args):\n    calibration_texts = load_texts(args.dataset_name, args.dataset_config, args.split, args.text_field, args.num_texts)\n    plan = json.loads(Path(args.plan_json).read_text(encoding=\"utf-8\"))\n    dynamic = plan_to_dynamic(plan)\n    print(json.dumps(quantize_with_dynamic(args.model_id, calibration_texts, args.out_dir, dynamic, args.base_bits, args.base_group_size, args.batch_size), indent=2))\n\n\ndef cmd_eval(args):\n    tasks = [x.strip() for x in args.tasks.split(\",\") if x.strip()]\n    result = run_lm_eval(args.model_path, tasks, args.output_json, args.device, args.batch_size)\n    print(json.dumps({\"output_json\": args.output_json, \"scores\": extract_primary_scores(result)}, indent=2))\n\n\ndef cmd_suite(args):\n    work_dir = Path(args.work_dir)\n    work_dir.mkdir(parents=True, exist_ok=True)\n    probe_dir = work_dir / \"probe\"\n\n    probe_result = run_probe(ProbeConfig(\n        model_id=args.model_id,\n        out_dir=str(probe_dir),\n        dataset_name=args.dataset_name,\n        dataset_config=args.dataset_config,\n        split=args.split,\n        text_field=args.text_field,\n        num_texts=args.probe_num_texts,\n        max_length=args.max_length,\n        max_rows_per_module=args.max_rows_per_module,\n        include_modules=[x.strip() for x in args.include_modules.split(\",\") if x.strip()],\n    ))\n    features_jsonl = probe_result[\"features_jsonl\"]\n    probe_rows = load_rows(features_jsonl)\n\n    judge_cfg = JudgeConfig(\n        features_jsonl=features_jsonl,\n        budget_mb_extra=args.budget_mb_extra,\n        heuristic_only=(not args.judge_base_url or not args.judge_model),\n        judge_base_url=args.judge_base_url,\n        judge_model=args.judge_model,\n        judge_api_key_env=args.judge_api_key_env,\n        chunk_size=args.chunk_size,\n    )\n    judge_proposals = make_judge_proposals(judge_cfg)\n    judge_plan = compile_plan(judge_proposals[\"assignments\"], probe_rows, args.budget_mb_extra, args.min_roi)\n    judge_dynamic = plan_to_dynamic(judge_plan)\n    save_plan_and_dynamic(judge_plan, judge_dynamic, str(work_dir / \"judge_plan.json\"), str(work_dir / \"judge_dynamic.json\"))\n\n    heuristic_flags = [x.strip() for x in args.heuristic_include.split(\",\") if x.strip()]\n    heuristic_dynamic = heuristic_dynamic_from_flags(heuristic_flags)\n    (work_dir / \"heuristic_dynamic.json\").write_text(json.dumps(heuristic_dynamic, indent=2), encoding=\"utf-8\")\n\n    calibration_texts = load_texts(args.dataset_name, args.dataset_config, args.split, args.text_field, args.quant_num_texts)\n    tasks = [x.strip() for x in args.tasks.split(\",\") if x.strip()]\n\n    configs = {\n        \"baseline_w4g128\": None,\n        \"heuristic_mixed\": heuristic_dynamic,\n        \"judge_sdq\": judge_dynamic,\n    }\n\n    leaderboard = {}\n    for name, dynamic in configs.items():\n        model_dir = work_dir / \"models\" / name\n        eval_json = work_dir / \"eval\" / f\"{name}.json\"\n        quantize_with_dynamic(args.model_id, calibration_texts, str(model_dir), dynamic, args.base_bits, args.base_group_size, args.batch_size)\n        result = run_lm_eval(str(model_dir), tasks, str(eval_json), args.device, args.eval_batch_size)\n        leaderboard[name] = extract_primary_scores(result)\n\n    payload = {\n        \"probe\": probe_result,\n        \"leaderboard\": leaderboard,\n        \"judge_used\": bool(args.judge_base_url and args.judge_model),\n        \"judge_plan_path\": str(work_dir / \"judge_plan.json\"),\n        \"judge_dynamic_path\": str(work_dir / \"judge_dynamic.json\"),\n        \"heuristic_dynamic_path\": str(work_dir / \"heuristic_dynamic.json\"),\n    }\n    (work_dir / \"leaderboard.json\").write_text(json.dumps(payload, indent=2), encoding=\"utf-8\")\n    print(json.dumps(payload, indent=2))\n\n\ndef build_parser():\n    p = argparse.ArgumentParser(prog=\"quant_judge\")\n    sub = p.add_subparsers(dest=\"cmd\", required=True)\n\n    probe = sub.add_parser(\"probe\")\n    probe.add_argument(\"--model-id\", required=True)\n    probe.add_argument(\"--out-dir\", required=True)\n    probe.add_argument(\"--dataset-name\", default=\"wikitext\")\n    probe.add_argument(\"--dataset-config\", default=\"wikitext-2-raw-v1\")\n    probe.add_argument(\"--split\", default=\"train\")\n    probe.add_argument(\"--text-field\", default=\"text\")\n    probe.add_argument(\"--num-texts\", type=int, default=64)\n    probe.add_argument(\"--max-length\", type=int, default=256)\n    probe.add_argument(\"--max-rows-per-module\", type=int, default=128)\n    probe.add_argument(\"--include-modules\", default=\"q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head\")\n    probe.set_defaults(func=cmd_probe)\n\n    plan = sub.add_parser(\"plan\")\n    plan.add_argument(\"--features-jsonl\", required=True)\n    plan.add_argument(\"--plan-out\", required=True)\n    plan.add_argument(\"--dynamic-out\", required=True)\n    plan.add_argument(\"--budget-mb-extra\", type=float, default=150.0)\n    plan.add_argument(\"--min-roi\", type=float, default=1e-6)\n    plan.add_argument(\"--heuristic-only\", action=\"store_true\")\n    plan.add_argument(\"--judge-base-url\", default=None)\n    plan.add_argument(\"--judge-model\", default=None)\n    plan.add_argument(\"--judge-api-key-env\", default=\"JUDGE_API_KEY\")\n    plan.add_argument(\"--chunk-size\", type=int, default=24)\n    plan.set_defaults(func=cmd_plan)\n\n    quant = sub.add_parser(\"quantize\")\n    quant.add_argument(\"--model-id\", required=True)\n    quant.add_argument(\"--plan-json\", required=True)\n    quant.add_argument(\"--out-dir\", required=True)\n    quant.add_argument(\"--dataset-name\", default=\"wikitext\")\n    quant.add_argument(\"--dataset-config\", default=\"wikitext-2-raw-v1\")\n    quant.add_argument(\"--split\", default=\"train\")\n    quant.add_argument(\"--text-field\", default=\"text\")\n    quant.add_argument(\"--num-texts\", type=int, default=128)\n    quant.add_argument(\"--batch-size\", type=int, default=4)\n    quant.add_argument(\"--base-bits\", type=int, default=4)\n    quant.add_argument(\"--base-group-size\", type=int, default=128)\n    quant.set_defaults(func=cmd_quantize)\n\n    ev = sub.add_parser(\"eval\")\n    ev.add_argument(\"--model-path\", required=True)\n    ev.add_argument(\"--tasks\", default=\"arc_challenge,hellaswag\")\n    ev.add_argument(\"--output-json\", required=True)\n    ev.add_argument(\"--device\", default=\"cuda:0\")\n    ev.add_argument(\"--batch-size\", default=\"auto\")\n    ev.set_defaults(func=cmd_eval)\n\n    suite = sub.add_parser(\"suite\")\n    suite.add_argument(\"--model-id\", required=True)\n    suite.add_argument(\"--work-dir\", required=True)\n    suite.add_argument(\"--dataset-name\", default=\"wikitext\")\n    suite.add_argument(\"--dataset-config\", default=\"wikitext-2-raw-v1\")\n    suite.add_argument(\"--split\", default=\"train\")\n    suite.add_argument(\"--text-field\", default=\"text\")\n    suite.add_argument(\"--probe-num-texts\", type=int, default=64)\n    suite.add_argument(\"--quant-num-texts\", type=int, default=128)\n    suite.add_argument(\"--max-length\", type=int, default=256)\n    suite.add_argument(\"--max-rows-per-module\", type=int, default=128)\n    suite.add_argument(\"--include-modules\", default=\"q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head\")\n    suite.add_argument(\"--budget-mb-extra\", type=float, default=150.0)\n    suite.add_argument(\"--min-roi\", type=float, default=1e-6)\n    suite.add_argument(\"--heuristic-include\", default=\"skip_lm_head,down_proj_g64\")\n    suite.add_argument(\"--judge-base-url\", default=None)\n    suite.add_argument(\"--judge-model\", default=None)\n    suite.add_argument(\"--judge-api-key-env\", default=\"JUDGE_API_KEY\")\n    suite.add_argument(\"--chunk-size\", type=int, default=24)\n    suite.add_argument(\"--tasks\", default=\"arc_challenge,hellaswag\")\n    suite.add_argument(\"--device\", default=\"cuda:0\")\n    suite.add_argument(\"--batch-size\", type=int, default=4)\n    suite.add_argument(\"--eval-batch-size\", default=\"auto\")\n    suite.add_argument(\"--base-bits\", type=int, default=4)\n    suite.add_argument(\"--base-group-size\", type=int, default=128)\n    suite.set_defaults(func=cmd_suite)\n\n    return p\n\n\ndef main():\n    parser = build_parser()\n    args = parser.parse_args()\n    args.func(args)\n\n\nif __name__ == \"__main__\":\n    main()\n"    }  ]}
JSON{  "project_name": "sdq_quant_judge_qwen35_gptqmodel",  "files": [    {      "path": "README.md",      "content": "# SDQ Quant Judge for GPTQModel\n\nA software-defined quantization scaffold for Qwen/Qwen3.5-4B-Base.\n\nCommands:\n- probe: extract per-module fake-quant sensitivity features from the FP model\n- plan: build a mixed-precision plan with either heuristic logic or an external OpenAI-compatible judge\n- quantize: quantize with GPTQModel using the compiled dynamic overrides\n- eval: evaluate a saved model with lm-eval\n- suite: run baseline vs heuristic vs judge and save a leaderboard\n\nExample:\n\n```bash\npython -m venv .venv\nsource .venv/bin/activate\npip install -r requirements.txt\n\npython -m quant_judge.pipeline probe \\\n  --model-id Qwen/Qwen3.5-4B-Base \\\n  --out-dir ./runs/probe \\\n  --dataset-name wikitext \\\n  --dataset-config wikitext-2-raw-v1 \\\n  --split train \\\n  --text-field text \\\n  --num-texts 64 \\\n  --max-length 256 \\\n  --max-rows-per-module 128 \\\n  --include-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head\n\npython -m quant_judge.pipeline plan \\\n  --features-jsonl ./runs/probe/probe_features.jsonl \\\n  --plan-out ./runs/plan.json \\\n  --dynamic-out ./runs/dynamic.json \\\n  --budget-mb-extra 150 \\\n  --heuristic-only\n\npython -m quant_judge.pipeline quantize \\\n  --model-id Qwen/Qwen3.5-4B-Base \\\n  --plan-json ./runs/plan.json \\\n  --out-dir ./runs/quantized \\\n  --dataset-name wikitext \\\n  --dataset-config wikitext-2-raw-v1 \\\n  --split train \\\n  --text-field text \\\n  --num-texts 128 \\\n  --batch-size 4\n\npython -m quant_judge.pipeline eval \\\n  --model-path ./runs/quantized \\\n  --tasks arc_challenge,hellaswag \\\n  --output-json ./runs/eval.json\n```\n"    },    {      "path": "requirements.txt",      "content": "transformers\ndatasets\naccelerate\nrequests\nnumpy\ntqdm\npsutil\ngptqmodel\nlm-eval\n"    },    {      "path": "quant_judge/__init__.py",      "content": "\"\"\"SDQ quantization scaffold.\"\"\"\n"    },    {      "path": "quant_judge/actions.py",      "content": "from __future__ import annotations\n\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass ActionSpec:\n    name: str\n    bits: int | None\n    group_size: int | None\n    skip: bool\n\n\nACTION_SPECS: dict[str, ActionSpec] = {\n    \"w4_g128\": ActionSpec(\"w4_g128\", bits=4, group_size=128, skip=False),\n    \"w4_g64\": ActionSpec(\"w4_g64\", bits=4, group_size=64, skip=False),\n    \"w8_g128\": ActionSpec(\"w8_g128\", bits=8, group_size=128, skip=False),\n    \"skip\": ActionSpec(\"skip\", bits=None, group_size=None, skip=True),\n}\n\nBASE_ACTION = \"w4_g128\"\n\n\ndef allowed_actions() -> list[str]:\n    return list(ACTION_SPECS.keys())\n\n\ndef bytes_per_param(action: str, fp_bytes: float = 2.0) -> float:\n    if action == \"skip\":\n        return fp_bytes\n    spec = ACTION_SPECS[action]\n    assert spec.bits is not None\n    return spec.bits / 8.0\n\n\ndef extra_mb_vs_base(params: int, action: str) -> float:\n    base = params * bytes_per_param(BASE_ACTION)\n    now = params * bytes_per_param(action)\n    return (now - base) / (1024 ** 2)\n\n\ndef action_to_override(action: str) -> dict:\n    if action == \"skip\":\n        return {}\n    spec = ACTION_SPECS[action]\n    return {\"bits\": spec.bits, \"group_size\": spec.group_size}\n"    },    {      "path": "quant_judge/probe.py",      "content": "from __future__ import annotations\n\nimport json\nfrom dataclasses import dataclass\nfrom pathlib import Path\n\nimport torch\nimport torch.nn.functional as F\nfrom datasets import load_dataset\nfrom tqdm import tqdm\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\n\nfrom .actions import extra_mb_vs_base\n\n\n@dataclass\nclass ProbeConfig:\n    model_id: str\n    out_dir: str\n    dataset_name: str\n    dataset_config: str | None\n    split: str\n    text_field: str\n    num_texts: int\n    max_length: int\n    max_rows_per_module: int\n    include_modules: list[str]\n    trust_remote_code: bool = True\n\n\ndef pick_dtype() -> torch.dtype:\n    if not torch.cuda.is_available():\n        return torch.float32\n    major, _minor = torch.cuda.get_device_capability(0)\n    if major >= 8:\n        return torch.bfloat16\n    return torch.float16\n\n\ndef fake_quant_groupwise_linear_weight(weight: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:\n    assert weight.dim() == 2\n    w = weight.detach().float()\n    out_features, in_features = w.shape\n    pad = (group_size - (in_features % group_size)) % group_size\n    if pad:\n        w = F.pad(w, (0, pad), value=0.0)\n    maxq = (2 ** (bits - 1)) - 1\n    w = w.view(out_features, -1, group_size)\n    absmax = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)\n    scale = absmax / maxq\n    q = torch.round(w / scale).clamp(-maxq, maxq)\n    dq = q * scale\n    dq = dq.view(out_features, -1)\n    if pad:\n        dq = dq[:, :in_features]\n    return dq.to(weight.device, dtype=weight.dtype)\n\n\ndef load_texts(dataset_name: str, dataset_config: str | None, split: str, text_field: str, num_texts: int) -> list[str]:\n    ds = load_dataset(dataset_name, dataset_config, split=split)\n    texts: list[str] = []\n    for row in ds:\n        value = row.get(text_field, \"\")\n        if isinstance(value, str) and value.strip():\n            texts.append(value.strip())\n        if len(texts) >= num_texts:\n            break\n    if not texts:\n        raise ValueError(\"No usable texts found.\")\n    return texts\n\n\ndef module_selected(name: str, include_modules: list[str]) -> bool:\n    return any(token in name for token in include_modules)\n\n\ndef iter_target_linears(model, include_modules: list[str]):\n    for name, module in model.named_modules():\n        if isinstance(module, torch.nn.Linear) and module_selected(name, include_modules):\n            yield name, module\n\n\ndef flatten_rows(x: torch.Tensor) -> torch.Tensor:\n    if x.dim() == 2:\n        return x\n    if x.dim() >= 3:\n        return x.reshape(-1, x.shape[-1])\n    return x.unsqueeze(0)\n\n\ndef sample_rows(x: torch.Tensor, max_rows: int) -> torch.Tensor:\n    rows = flatten_rows(x)\n    if rows.shape[0] <= max_rows:\n        return rows\n    idx = torch.randperm(rows.shape[0], device=rows.device)[:max_rows]\n    return rows.index_select(0, idx)\n\n\ndef weight_kurtosis(weight: torch.Tensor) -> float:\n    w = weight.detach().float().flatten()\n    mu = w.mean()\n    var = ((w - mu) ** 2).mean().clamp_min(1e-12)\n    return float((((w - mu) ** 4).mean() / (var ** 2)).item())\n\n\ndef activation_outlier_rate(x: torch.Tensor, sigma: float = 6.0) -> float:\n    xf = x.float()\n    std = xf.std().clamp_min(1e-6)\n    return float((xf.abs() > sigma * std).float().mean().item())\n\n\ndef block_index_from_name(name: str) -> int | None:\n    parts = name.split(\".\")\n    for i, p in enumerate(parts):\n        if p == \"layers\" and i + 1 < len(parts):\n            try:\n                return int(parts[i + 1])\n            except Exception:\n                return None\n    return None\n\n\ndef reduced_metrics_for_action(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, bits: int, group_size: int) -> dict:\n    wq = fake_quant_groupwise_linear_weight(weight, bits=bits, group_size=group_size)\n    out_fp = F.linear(x, weight, bias)\n    out_q = F.linear(x, wq, bias)\n    mse = F.mse_loss(out_q.float(), out_fp.float()).item()\n    cos = F.cosine_similarity(out_q.float(), out_fp.float(), dim=-1).mean().item()\n    max_abs = (out_q.float() - out_fp.float()).abs().max().item()\n    return {\"mse\": float(mse), \"cosine\": float(cos), \"max_abs\": float(max_abs)}\n\n\ndef run_probe(cfg: ProbeConfig) -> dict:\n    out_dir = Path(cfg.out_dir)\n    out_dir.mkdir(parents=True, exist_ok=True)\n\n    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code)\n    if tokenizer.pad_token_id is None:\n        tokenizer.pad_token = tokenizer.eos_token\n\n    model = AutoModelForCausalLM.from_pretrained(\n        cfg.model_id,\n        trust_remote_code=cfg.trust_remote_code,\n        torch_dtype=pick_dtype(),\n        device_map=\"auto\",\n        low_cpu_mem_usage=True,\n    )\n    model.eval()\n\n    texts = load_texts(cfg.dataset_name, cfg.dataset_config, cfg.split, cfg.text_field, cfg.num_texts)\n    targets = list(iter_target_linears(model, cfg.include_modules))\n    if not targets:\n        raise ValueError(\"No target modules found.\")\n\n    captured_inputs = {name: [] for name, _ in targets}\n    remaining_rows = {name: cfg.max_rows_per_module for name, _ in targets}\n    handles = []\n\n    def make_hook(name: str):\n        def hook(_module, inp, _out):\n            rem = remaining_rows[name]\n            if rem <= 0 or not inp or not isinstance(inp[0], torch.Tensor):\n                return\n            x = sample_rows(inp[0], rem)\n            take = min(x.shape[0], rem)\n            if take <= 0:\n                return\n            captured_inputs[name].append(x[:take].detach().cpu().to(torch.float16))\n            remaining_rows[name] -= take\n        return hook\n\n    for name, module in targets:\n        handles.append(module.register_forward_hook(make_hook(name)))\n\n    with torch.no_grad():\n        for text in tqdm(texts, desc=\"collect_activations\"):\n            batch = tokenizer(text, return_tensors=\"pt\", truncation=True, max_length=cfg.max_length, padding=False)\n            batch = {k: v.to(model.device) for k, v in batch.items()}\n            _ = model(**batch)\n            if all(v <= 0 for v in remaining_rows.values()):\n                break\n\n    for h in handles:\n        h.remove()\n\n    rows = []\n    for name, module in tqdm(targets, desc=\"probe_modules\"):\n        if not captured_inputs[name]:\n            continue\n        x = torch.cat(captured_inputs[name], dim=0).to(module.weight.device, dtype=module.weight.dtype)\n        weight = module.weight.detach()\n        bias = None if module.bias is None else module.bias.detach()\n\n        m_w4g128 = reduced_metrics_for_action(x, weight, bias, bits=4, group_size=128)\n        m_w4g64 = reduced_metrics_for_action(x, weight, bias, bits=4, group_size=64)\n        m_w8g128 = reduced_metrics_for_action(x, weight, bias, bits=8, group_size=128)\n\n        params = int(weight.numel())\n        row = {\n            \"name\": name,\n            \"module_type\": name.split(\".\")[-1],\n            \"block_index\": block_index_from_name(name),\n            \"params\": params,\n            \"captured_rows\": int(x.shape[0]),\n            \"weight_kurtosis\": weight_kurtosis(weight),\n            \"act_outlier_rate\": activation_outlier_rate(x),\n            \"candidates\": {\n                \"w4_g128\": {**m_w4g128, \"estimated_extra_mb_vs_base\": 0.0},\n                \"w4_g64\": {**m_w4g64, \"estimated_extra_mb_vs_base\": max(0.0, extra_mb_vs_base(params, \"w4_g64\")) + (params * 0.05 / (1024 ** 2))},\n                \"w8_g128\": {**m_w8g128, \"estimated_extra_mb_vs_base\": max(0.0, extra_mb_vs_base(params, \"w8_g128\"))},\n                \"skip\": {\"mse\": 0.0, \"cosine\": 1.0, \"max_abs\": 0.0, \"estimated_extra_mb_vs_base\": max(0.0, extra_mb_vs_base(params, \"skip\"))},\n            },\n            \"proxy_gain_w4g64_vs_w4\": float(max(0.0, m_w4g128[\"mse\"] - m_w4g64[\"mse\"])),\n            \"proxy_gain_w8_vs_w4\": float(max(0.0, m_w4g128[\"mse\"] - m_w8g128[\"mse\"])),\n        }\n        rows.append(row)\n\n    features_jsonl = out_dir / \"probe_features.jsonl\"\n    metadata_json = out_dir / \"probe_metadata.json\"\n    summary_json = out_dir / \"probe_summary.json\"\n\n    with open(features_jsonl, \"w\", encoding=\"utf-8\") as f:\n        for row in rows:\n            f.write(json.dumps(row, ensure_ascii=False) + \"\\n\")\n\n    metadata = {\n        \"model_id\": cfg.model_id,\n        \"dataset_name\": cfg.dataset_name,\n        \"dataset_config\": cfg.dataset_config,\n        \"split\": cfg.split,\n        \"text_field\": cfg.text_field,\n        \"num_texts\": cfg.num_texts,\n        \"max_length\": cfg.max_length,\n        \"max_rows_per_module\": cfg.max_rows_per_module,\n        \"include_modules\": cfg.include_modules,\n        \"num_rows\": len(rows),\n    }\n    metadata_json.write_text(json.dumps(metadata, indent=2), encoding=\"utf-8\")\n    summary_json.write_text(json.dumps({\"num_rows\": len(rows)}, indent=2), encoding=\"utf-8\")\n\n    del model\n    if torch.cuda.is_available():\n        torch.cuda.empty_cache()\n\n    return {\"features_jsonl\": str(features_jsonl), \"metadata_json\": str(metadata_json), \"summary_json\": str(summary_json), \"num_rows\": len(rows)}\n"    },    {      "path": "quant_judge/judge.py",      "content": "from __future__ import annotations\n\nimport json\nimport os\nfrom dataclasses import dataclass\n\nimport requests\n\nfrom .actions import ACTION_SPECS, BASE_ACTION, allowed_actions\n\n\n@dataclass\nclass JudgeConfig:\n    features_jsonl: str\n    budget_mb_extra: float\n    heuristic_only: bool = False\n    judge_base_url: str | None = None\n    judge_model: str | None = None\n    judge_api_key_env: str = \"JUDGE_API_KEY\"\n    chunk_size: int = 24\n\n\ndef load_rows(path: str) -> list[dict]:\n    rows = []\n    with open(path, \"r\", encoding=\"utf-8\") as f:\n        for line in f:\n            line = line.strip()\n            if line:\n                rows.append(json.loads(line))\n    return rows\n\n\ndef heuristic_decision(row: dict) -> dict:\n    name = row[\"name\"]\n    w4 = row[\"candidates\"][\"w4_g128\"][\"mse\"]\n    w4g64 = row[\"candidates\"][\"w4_g64\"][\"mse\"]\n    w8 = row[\"candidates\"][\"w8_g128\"][\"mse\"]\n    kurt = row[\"weight_kurtosis\"]\n    outlier = row[\"act_outlier_rate\"]\n    block = row.get(\"block_index\")\n\n    w8_relief = (w4 - w8) / max(w4, 1e-9)\n    g64_relief = (w4 - w4g64) / max(w4, 1e-9)\n\n    score = 0.0\n    score += min(1.0, w8_relief / 0.60) * 0.40\n    score += min(1.0, g64_relief / 0.35) * 0.20\n    score += min(1.0, kurt / 20.0) * 0.15\n    score += min(1.0, outlier / 0.05) * 0.15\n    if \"lm_head\" in name:\n        score += 0.25\n    if row[\"module_type\"] in (\"down_proj\", \"o_proj\", \"gate_proj\"):\n        score += 0.10\n    if block is not None and (block <= 1 or block >= 30):\n        score += 0.10\n    score = max(0.0, min(1.0, score))\n\n    action = BASE_ACTION\n    reason = \"default\"\n    if \"lm_head\" in name and (w8_relief > 0.55 or kurt > 25.0):\n        action = \"skip\"\n        reason = \"lm_head_extreme\"\n    elif score >= 0.75 or w8_relief > 0.50:\n        action = \"w8_g128\"\n        reason = \"high_sensitivity\"\n    elif score >= 0.42 or g64_relief > 0.18:\n        action = \"w4_g64\"\n        reason = \"moderate_sensitivity\"\n\n    return {\"name\": name, \"action\": action, \"priority\": round(score, 6), \"reason_code\": reason, \"source\": \"heuristic\"}\n\n\ndef reduced_row(row: dict) -> dict:\n    return {\n        \"name\": row[\"name\"],\n        \"module_type\": row[\"module_type\"],\n        \"block_index\": row.get(\"block_index\"),\n        \"params\": row[\"params\"],\n        \"weight_kurtosis\": row[\"weight_kurtosis\"],\n        \"act_outlier_rate\": row[\"act_outlier_rate\"],\n        \"w4_g128_mse\": row[\"candidates\"][\"w4_g128\"][\"mse\"],\n        \"w4_g64_mse\": row[\"candidates\"][\"w4_g64\"][\"mse\"],\n        \"w8_g128_mse\": row[\"candidates\"][\"w8_g128\"][\"mse\"],\n        \"extra_mb_w4_g64\": row[\"candidates\"][\"w4_g64\"][\"estimated_extra_mb_vs_base\"],\n        \"extra_mb_w8_g128\": row[\"candidates\"][\"w8_g128\"][\"estimated_extra_mb_vs_base\"],\n        \"extra_mb_skip\": row[\"candidates\"][\"skip\"][\"estimated_extra_mb_vs_base\"],\n    }\n\n\ndef strip_code_fence(text: str) -> str:\n    t = text.strip()\n    if t.startswith(\"```\"):\n        t = t.split(\"\\n\", 1)[1]\n        if t.endswith(\"```\"):\n            t = t.rsplit(\"\\n\", 1)[0]\n    return t.strip()\n\n\ndef chunk(rows: list[dict], size: int) -> list[list[dict]]:\n    return [rows[i:i + size] for i in range(0, len(rows), size)]\n\n\ndef call_openai_compatible(base_url: str, model: str, api_key: str, rows: list[dict], budget_mb_extra: float) -> list[dict]:\n    system = (\n        \"You are a quantization policy judge. \"\n        \"Choose exactly one action per module from: \" + \", \".join(allowed_actions()) + \". \"\n        \"Prefer w4_g128 unless the evidence shows fragility. \"\n        \"Use w4_g64 for moderate sensitivity. Use w8_g128 for high sensitivity. \"\n        \"Use skip only for extreme outliers, usually lm_head. \"\n        \"Return JSON only with schema {\\\\\\\"assignments\\\\\\\":[{\\\\\\\"name\\\\\\\":...,\\\\\\\"action\\\\\\\":...,\\\\\\\"priority\\\\\\\":0..1,\\\\\\\"reason_code\\\\\\\":...}]}\"\n    )\n    user = {\"budget_mb_extra_global\": budget_mb_extra, \"allowed_actions\": allowed_actions(), \"rows\": rows}\n    url = base_url.rstrip(\"/\") + \"/chat/completions\"\n    headers = {\"Content-Type\": \"application/json\"}\n    if api_key:\n        headers[\"Authorization\"] = f\"Bearer {api_key}\"\n    body = {\n        \"model\": model,\n        \"temperature\": 0.0,\n        \"response_format\": {\"type\": \"json_object\"},\n        \"messages\": [\n            {\"role\": \"system\", \"content\": system},\n            {\"role\": \"user\", \"content\": json.dumps(user, ensure_ascii=False)},\n        ],\n    }\n    resp = requests.post(url, headers=headers, json=body, timeout=180)\n    resp.raise_for_status()\n    content = resp.json()[\"choices\"][0][\"message\"][\"content\"]\n    parsed = json.loads(strip_code_fence(content))\n    assignments = parsed.get(\"assignments\", [])\n    if not isinstance(assignments, list):\n        raise ValueError(\"Judge response missing assignments.\")\n    return assignments\n\n\ndef merge_assignments(rows: list[dict], proposed: list[dict]) -> list[dict]:\n    by_name = {a[\"name\"]: a for a in proposed if isinstance(a, dict) and \"name\" in a}\n    merged = []\n    for row in rows:\n        item = by_name.get(row[\"name\"])\n        if item is None:\n            merged.append(heuristic_decision(row))\n            continue\n        action = item.get(\"action\", BASE_ACTION)\n        if action not in ACTION_SPECS:\n            action = BASE_ACTION\n        try:\n            priority = float(item.get(\"priority\", 0.5))\n        except Exception:\n            priority = 0.5\n        priority = max(0.0, min(1.0, priority))\n        merged.append({\"name\": row[\"name\"], \"action\": action, \"priority\": priority, \"reason_code\": str(item.get(\"reason_code\", \"judge\")), \"source\": \"judge\"})\n    return merged\n\n\ndef make_judge_proposals(cfg: JudgeConfig) -> dict:\n    rows = load_rows(cfg.features_jsonl)\n    if cfg.heuristic_only:\n        return {\"assignments\": [heuristic_decision(r) for r in rows]}\n    if not cfg.judge_base_url or not cfg.judge_model:\n        raise ValueError(\"Judge mode requires judge_base_url and judge_model.\")\n    api_key = os.environ.get(cfg.judge_api_key_env, \"\")\n    assignments = []\n    for part in chunk(rows, cfg.chunk_size):\n        reduced = [reduced_row(r) for r in part]\n        try:\n            proposed = call_openai_compatible(cfg.judge_base_url, cfg.judge_model, api_key, reduced, cfg.budget_mb_extra)\n            assignments.extend(merge_assignments(part, proposed))\n        except Exception:\n            assignments.extend([heuristic_decision(r) for r in part])\n    return {\"assignments\": assignments}\n"    },    {      "path": "quant_judge/compiler.py",      "content": "from __future__ import annotations\n\nimport json\nimport re\nfrom pathlib import Path\n\nfrom .actions import BASE_ACTION, action_to_override\n\n\ndef compile_plan(assignments: list[dict], probe_rows: list[dict], extra_budget_mb: float, min_roi: float = 1e-6) -> dict:\n    probe_by_name = {row[\"name\"]: row for row in probe_rows}\n    upgrades = []\n    for item in assignments:\n        name = item[\"name\"]\n        action = item[\"action\"]\n        priority = float(item.get(\"priority\", 0.5))\n        if action == BASE_ACTION:\n            continue\n        row = probe_by_name.get(name)\n        if row is None:\n            continue\n        if action == \"w8_g128\":\n            benefit = row[\"candidates\"][\"w4_g128\"][\"mse\"] - row[\"candidates\"][\"w8_g128\"][\"mse\"]\n            cost_mb = row[\"candidates\"][\"w8_g128\"][\"estimated_extra_mb_vs_base\"]\n        elif action == \"w4_g64\":\n            benefit = row[\"candidates\"][\"w4_g128\"][\"mse\"] - row[\"candidates\"][\"w4_g64\"][\"mse\"]\n            cost_mb = row[\"candidates\"][\"w4_g64\"][\"estimated_extra_mb_vs_base\"]\n        elif action == \"skip\":\n            benefit = row[\"candidates\"][\"w4_g128\"][\"mse\"]\n            cost_mb = row[\"candidates\"][\"skip\"][\"estimated_extra_mb_vs_base\"]\n        else:\n            continue\n        if benefit <= 0 or cost_mb <= 0:\n            continue\n        roi = benefit / cost_mb\n        if roi < min_roi:\n            continue\n        upgrades.append({\n            \"name\": name,\n            \"action\": action,\n            \"benefit\": float(benefit),\n            \"cost_mb\": float(cost_mb),\n            \"roi\": float(roi),\n            \"priority\": priority,\n            \"reason_code\": item.get(\"reason_code\", \"\"),\n            \"source\": item.get(\"source\", \"\"),\n        })\n    upgrades.sort(key=lambda x: (x[\"roi\"], x[\"priority\"]), reverse=True)\n\n    approved = []\n    spent_mb = 0.0\n    for up in upgrades:\n        if spent_mb + up[\"cost_mb\"] > extra_budget_mb:\n            continue\n        spent_mb += up[\"cost_mb\"]\n        approved.append(up)\n\n    approved_by_name = {x[\"name\"]: x for x in approved}\n    final_assignments = []\n    for row in probe_rows:\n        ap = approved_by_name.get(row[\"name\"])\n        if ap is None:\n            final_assignments.append({\"name\": row[\"name\"], \"action\": BASE_ACTION, \"priority\": 0.0, \"reason_code\": \"base_default\", \"source\": \"compiler\", \"estimated_extra_mb_vs_base\": 0.0})\n        else:\n            final_assignments.append({\"name\": ap[\"name\"], \"action\": ap[\"action\"], \"priority\": ap[\"priority\"], \"reason_code\": ap[\"reason_code\"], \"source\": ap[\"source\"], \"estimated_extra_mb_vs_base\": ap[\"cost_mb\"], \"roi\": ap[\"roi\"], \"benefit\": ap[\"benefit\"]})\n    final_assignments.sort(key=lambda x: x[\"name\"])\n    return {\"budget_mb_extra\": float(extra_budget_mb), \"used_mb_extra\": round(float(spent_mb), 6), \"upgrades_considered\": len(upgrades), \"upgrades_approved\": len(approved), \"assignments\": final_assignments}\n\n\ndef plan_to_dynamic(plan: dict) -> dict:\n    dynamic = {}\n    for item in plan[\"assignments\"]:\n        name = item[\"name\"]\n        action = item[\"action\"]\n        if action == BASE_ACTION:\n            continue\n        anchored = f\"^{re.escape(name)}$\"\n        if action == \"skip\":\n            dynamic[f\"-:{anchored}\"] = {}\n        else:\n            dynamic[f\"+:{anchored}\"] = action_to_override(action)\n    return dynamic\n\n\ndef save_plan_and_dynamic(plan: dict, dynamic: dict, plan_out: str, dynamic_out: str) -> None:\n    Path(plan_out).parent.mkdir(parents=True, exist_ok=True)\n    Path(dynamic_out).parent.mkdir(parents=True, exist_ok=True)\n    Path(plan_out).write_text(json.dumps(plan, indent=2), encoding=\"utf-8\")\n    Path(dynamic_out).write_text(json.dumps(dynamic, indent=2), encoding=\"utf-8\")\n"    },    {      "path": "quant_judge/runner.py",      "content": "from __future__ import annotations\n\nimport json\nimport subprocess\nimport sys\nfrom pathlib import Path\n\nimport torch\nfrom datasets import load_dataset\nfrom gptqmodel import GPTQModel, QuantizeConfig\n\n\ndef load_texts(dataset_name: str, dataset_config: str | None, split: str, text_field: str, num_texts: int) -> list[str]:\n    ds = load_dataset(dataset_name, dataset_config, split=split)\n    texts = []\n    for row in ds:\n        text = row.get(text_field, \"\")\n        if isinstance(text, str) and text.strip():\n            texts.append(text.strip())\n        if len(texts) >= num_texts:\n            break\n    if not texts:\n        raise ValueError(\"No texts found for quantization.\")\n    return texts\n\n\ndef quantize_with_dynamic(model_id: str, calibration_texts: list[str], out_dir: str, dynamic: dict | None, bits: int = 4, group_size: int = 128, batch_size: int = 4, sym: bool = True, desc_act: bool = True) -> dict:\n    qcfg = QuantizeConfig(bits=bits, group_size=group_size, sym=sym, desc_act=desc_act, dynamic=dynamic)\n    model = GPTQModel.load(model_id, qcfg)\n    model.quantize(calibration_texts, batch_size=batch_size)\n    out = Path(out_dir)\n    out.mkdir(parents=True, exist_ok=True)\n    model.save(str(out))\n    meta = {\"model_id\": model_id, \"out_dir\": str(out), \"bits\": bits, \"group_size\": group_size, \"batch_size\": batch_size, \"dynamic_rules\": 0 if dynamic is None else len(dynamic)}\n    (out / \"quant_meta.json\").write_text(json.dumps(meta, indent=2), encoding=\"utf-8\")\n    del model\n    if torch.cuda.is_available():\n        torch.cuda.empty_cache()\n    return meta\n\n\ndef heuristic_dynamic_from_flags(flags: list[str]) -> dict:\n    dynamic = {}\n    for flag in flags:\n        if flag == \"skip_lm_head\":\n            dynamic[r\"-:^lm_head$\"] = {}\n        elif flag == \"down_proj_g64\":\n            dynamic[r\"+:.*down_proj$\"] = {\"bits\": 4, \"group_size\": 64}\n        elif flag == \"o_proj_g64\":\n            dynamic[r\"+:.*o_proj$\"] = {\"bits\": 4, \"group_size\": 64}\n        elif flag == \"edge_blocks_w8\":\n            dynamic[r\"+:^model\\\\.layers\\\\.(0|1|30|31)\\\\..*$\"] = {\"bits\": 8, \"group_size\": 128}\n    return dynamic\n\n\ndef run_lm_eval(model_path: str, tasks: list[str], output_json: str, device: str = \"cuda:0\", batch_size: str = \"auto\") -> dict:\n    output_path = Path(output_json)\n    output_path.parent.mkdir(parents=True, exist_ok=True)\n    cmd = [\n        sys.executable, \"-m\", \"lm_eval\",\n        \"--model\", \"hf\",\n        \"--model_args\", f\"pretrained={model_path},trust_remote_code=True\",\n        \"--tasks\", \",\".join(tasks),\n        \"--device\", device,\n        \"--batch_size\", batch_size,\n        \"--output_path\", str(output_path.parent),\n    ]\n    subprocess.run(cmd, check=True)\n    for path in sorted(output_path.parent.rglob(\"*.json\")):\n        try:\n            data = json.loads(path.read_text(encoding=\"utf-8\"))\n        except Exception:\n            continue\n        if isinstance(data, dict) and \"results\" in data:\n            output_path.write_text(json.dumps(data, indent=2), encoding=\"utf-8\")\n            return data\n    raise FileNotFoundError(\"Could not find lm_eval results json.\")\n\n\ndef extract_primary_scores(eval_json: dict) -> dict:\n    out = {}\n    for task, metrics in eval_json.get(\"results\", {}).items():\n        if \"acc_norm,none\" in metrics:\n            out[task] = metrics[\"acc_norm,none\"]\n        elif \"acc,none\" in metrics:\n            out[task] = metrics[\"acc,none\"]\n        elif \"word_perplexity,none\" in metrics:\n            out[task] = metrics[\"word_perplexity,none\"]\n    return out\n"    },    {      "path": "quant_judge/pipeline.py",      "content": "from __future__ import annotations\n\nimport argparse\nimport json\nfrom pathlib import Path\n\nfrom .compiler import compile_plan, plan_to_dynamic, save_plan_and_dynamic\nfrom .judge import JudgeConfig, load_rows, make_judge_proposals\nfrom .probe import ProbeConfig, run_probe\nfrom .runner import extract_primary_scores, heuristic_dynamic_from_flags, load_texts, quantize_with_dynamic, run_lm_eval\n\n\ndef cmd_probe(args):\n    cfg = ProbeConfig(\n        model_id=args.model_id,\n        out_dir=args.out_dir,\n        dataset_name=args.dataset_name,\n        dataset_config=args.dataset_config,\n        split=args.split,\n        text_field=args.text_field,\n        num_texts=args.num_texts,\n        max_length=args.max_length,\n        max_rows_per_module=args.max_rows_per_module,\n        include_modules=[x.strip() for x in args.include_modules.split(\",\") if x.strip()],\n    )\n    print(json.dumps(run_probe(cfg), indent=2))\n\n\ndef cmd_plan(args):\n    judge_cfg = JudgeConfig(\n        features_jsonl=args.features_jsonl,\n        budget_mb_extra=args.budget_mb_extra,\n        heuristic_only=args.heuristic_only,\n        judge_base_url=args.judge_base_url,\n        judge_model=args.judge_model,\n        judge_api_key_env=args.judge_api_key_env,\n        chunk_size=args.chunk_size,\n    )\n    proposals = make_judge_proposals(judge_cfg)\n    probe_rows = load_rows(args.features_jsonl)\n    plan = compile_plan(proposals[\"assignments\"], probe_rows, args.budget_mb_extra, args.min_roi)\n    dynamic = plan_to_dynamic(plan)\n    save_plan_and_dynamic(plan, dynamic, args.plan_out, args.dynamic_out)\n    print(json.dumps({\"plan_out\": args.plan_out, \"dynamic_out\": args.dynamic_out, \"used_mb_extra\": plan[\"used_mb_extra\"], \"upgrades_approved\": plan[\"upgrades_approved\"], \"dynamic_rules\": len(dynamic)}, indent=2))\n\n\ndef cmd_quantize(args):\n    calibration_texts = load_texts(args.dataset_name, args.dataset_config, args.split, args.text_field, args.num_texts)\n    plan = json.loads(Path(args.plan_json).read_text(encoding=\"utf-8\"))\n    dynamic = plan_to_dynamic(plan)\n    print(json.dumps(quantize_with_dynamic(args.model_id, calibration_texts, args.out_dir, dynamic, args.base_bits, args.base_group_size, args.batch_size), indent=2))\n\n\ndef cmd_eval(args):\n    tasks = [x.strip() for x in args.tasks.split(\",\") if x.strip()]\n    result = run_lm_eval(args.model_path, tasks, args.output_json, args.device, args.batch_size)\n    print(json.dumps({\"output_json\": args.output_json, \"scores\": extract_primary_scores(result)}, indent=2))\n\n\ndef cmd_suite(args):\n    work_dir = Path(args.work_dir)\n    work_dir.mkdir(parents=True, exist_ok=True)\n    probe_dir = work_dir / \"probe\"\n\n    probe_result = run_probe(ProbeConfig(\n        model_id=args.model_id,\n        out_dir=str(probe_dir),\n        dataset_name=args.dataset_name,\n        dataset_config=args.dataset_config,\n        split=args.split,\n        text_field=args.text_field,\n        num_texts=args.probe_num_texts,\n        max_length=args.max_length,\n        max_rows_per_module=args.max_rows_per_module,\n        include_modules=[x.strip() for x in args.include_modules.split(\",\") if x.strip()],\n    ))\n    features_jsonl = probe_result[\"features_jsonl\"]\n    probe_rows = load_rows(features_jsonl)\n\n    judge_cfg = JudgeConfig(\n        features_jsonl=features_jsonl,\n        budget_mb_extra=args.budget_mb_extra,\n        heuristic_only=(not args.judge_base_url or not args.judge_model),\n        judge_base_url=args.judge_base_url,\n        judge_model=args.judge_model,\n        judge_api_key_env=args.judge_api_key_env,\n        chunk_size=args.chunk_size,\n    )\n    judge_proposals = make_judge_proposals(judge_cfg)\n    judge_plan = compile_plan(judge_proposals[\"assignments\"], probe_rows, args.budget_mb_extra, args.min_roi)\n    judge_dynamic = plan_to_dynamic(judge_plan)\n    save_plan_and_dynamic(judge_plan, judge_dynamic, str(work_dir / \"judge_plan.json\"), str(work_dir / \"judge_dynamic.json\"))\n\n    heuristic_flags = [x.strip() for x in args.heuristic_include.split(\",\") if x.strip()]\n    heuristic_dynamic = heuristic_dynamic_from_flags(heuristic_flags)\n    (work_dir / \"heuristic_dynamic.json\").write_text(json.dumps(heuristic_dynamic, indent=2), encoding=\"utf-8\")\n\n    calibration_texts = load_texts(args.dataset_name, args.dataset_config, args.split, args.text_field, args.quant_num_texts)\n    tasks = [x.strip() for x in args.tasks.split(\",\") if x.strip()]\n\n    configs = {\n        \"baseline_w4g128\": None,\n        \"heuristic_mixed\": heuristic_dynamic,\n        \"judge_sdq\": judge_dynamic,\n    }\n\n    leaderboard = {}\n    for name, dynamic in configs.items():\n        model_dir = work_dir / \"models\" / name\n        eval_json = work_dir / \"eval\" / f\"{name}.json\"\n        quantize_with_dynamic(args.model_id, calibration_texts, str(model_dir), dynamic, args.base_bits, args.base_group_size, args.batch_size)\n        result = run_lm_eval(str(model_dir), tasks, str(eval_json), args.device, args.eval_batch_size)\n        leaderboard[name] = extract_primary_scores(result)\n\n    payload = {\n        \"probe\": probe_result,\n        \"leaderboard\": leaderboard,\n        \"judge_used\": bool(args.judge_base_url and args.judge_model),\n        \"judge_plan_path\": str(work_dir / \"judge_plan.json\"),\n        \"judge_dynamic_path\": str(work_dir / \"judge_dynamic.json\"),\n        \"heuristic_dynamic_path\": str(work_dir / \"heuristic_dynamic.json\"),\n    }\n    (work_dir / \"leaderboard.json\").write_text(json.dumps(payload, indent=2), encoding=\"utf-8\")\n    print(json.dumps(payload, indent=2))\n\n\ndef build_parser():\n    p = argparse.ArgumentParser(prog=\"quant_judge\")\n    sub = p.add_subparsers(dest=\"cmd\", required=True)\n\n    probe = sub.add_parser(\"probe\")\n    probe.add_argument(\"--model-id\", required=True)\n    probe.add_argument(\"--out-dir\", required=True)\n    probe.add_argument(\"--dataset-name\", default=\"wikitext\")\n    probe.add_argument(\"--dataset-config\", default=\"wikitext-2-raw-v1\")\n    probe.add_argument(\"--split\", default=\"train\")\n    probe.add_argument(\"--text-field\", default=\"text\")\n    probe.add_argument(\"--num-texts\", type=int, default=64)\n    probe.add_argument(\"--max-length\", type=int, default=256)\n    probe.add_argument(\"--max-rows-per-module\", type=int, default=128)\n    probe.add_argument(\"--include-modules\", default=\"q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head\")\n    probe.set_defaults(func=cmd_probe)\n\n    plan = sub.add_parser(\"plan\")\n    plan.add_argument(\"--features-jsonl\", required=True)\n    plan.add_argument(\"--plan-out\", required=True)\n    plan.add_argument(\"--dynamic-out\", required=True)\n    plan.add_argument(\"--budget-mb-extra\", type=float, default=150.0)\n    plan.add_argument(\"--min-roi\", type=float, default=1e-6)\n    plan.add_argument(\"--heuristic-only\", action=\"store_true\")\n    plan.add_argument(\"--judge-base-url\", default=None)\n    plan.add_argument(\"--judge-model\", default=None)\n    plan.add_argument(\"--judge-api-key-env\", default=\"JUDGE_API_KEY\")\n    plan.add_argument(\"--chunk-size\", type=int, default=24)\n    plan.set_defaults(func=cmd_plan)\n\n    quant = sub.add_parser(\"quantize\")\n    quant.add_argument(\"--model-id\", required=True)\n    quant.add_argument(\"--plan-json\", required=True)\n    quant.add_argument(\"--out-dir\", required=True)\n    quant.add_argument(\"--dataset-name\", default=\"wikitext\")\n    quant.add_argument(\"--dataset-config\", default=\"wikitext-2-raw-v1\")\n    quant.add_argument(\"--split\", default=\"train\")\n    quant.add_argument(\"--text-field\", default=\"text\")\n    quant.add_argument(\"--num-texts\", type=int, default=128)\n    quant.add_argument(\"--batch-size\", type=int, default=4)\n    quant.add_argument(\"--base-bits\", type=int, default=4)\n    quant.add_argument(\"--base-group-size\", type=int, default=128)\n    quant.set_defaults(func=cmd_quantize)\n\n    ev = sub.add_parser(\"eval\")\n    ev.add_argument(\"--model-path\", required=True)\n    ev.add_argument(\"--tasks\", default=\"arc_challenge,hellaswag\")\n    ev.add_argument(\"--output-json\", required=True)\n    ev.add_argument(\"--device\", default=\"cuda:0\")\n    ev.add_argument(\"--batch-size\", default=\"auto\")\n    ev.set_defaults(func=cmd_eval)\n\n    suite = sub.add_parser(\"suite\")\n    suite.add_argument(\"--model-id\", required=True)\n    suite.add_argument(\"--work-dir\", required=True)\n    suite.add_argument(\"--dataset-name\", default=\"wikitext\")\n    suite.add_argument(\"--dataset-config\", default=\"wikitext-2-raw-v1\")\n    suite.add_argument(\"--split\", default=\"train\")\n    suite.add_argument(\"--text-field\", default=\"text\")\n    suite.add_argument(\"--probe-num-texts\", type=int, default=64)\n    suite.add_argument(\"--quant-num-texts\", type=int, default=128)\n    suite.add_argument(\"--max-length\", type=int, default=256)\n    suite.add_argument(\"--max-rows-per-module\", type=int, default=128)\n    suite.add_argument(\"--include-modules\", default=\"q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head\")\n    suite.add_argument(\"--budget-mb-extra\", type=float, default=150.0)\n    suite.add_argument(\"--min-roi\", type=float, default=1e-6)\n    suite.add_argument(\"--heuristic-include\", default=\"skip_lm_head,down_proj_g64\")\n    suite.add_argument(\"--judge-base-url\", default=None)\n    suite.add_argument(\"--judge-model\", default=None)\n    suite.add_argument(\"--judge-api-key-env\", default=\"JUDGE_API_KEY\")\n    suite.add_argument(\"--chunk-size\", type=int, default=24)\n    suite.add_argument(\"--tasks\", default=\"arc_challenge,hellaswag\")\n    suite.add_argument(\"--device\", default=\"cuda:0\")\n    suite.add_argument(\"--batch-size\", type=int, default=4)\n    suite.add_argument(\"--eval-batch-size\", default=\"auto\")\n    suite.add_argument(\"--base-bits\", type=int, default=4)\n    suite.add_argument(\"--base-group-size\", type=int, default=128)\n    suite.set_defaults(func=cmd_suite)\n\n    return p\n\n\ndef main():\n    parser = build_parser()\n    args = parser.parse_args()\n    args.func(args)\n\n\nif __name__ == \"__main__\":\n    main()\n"    }  ]}
```

---

### User:
Here is a detailed critique of the `sdq_quant_judge_qwen35_gptqmodel` project. Overall, the scaffold is well-structured and uses a clever approach (fake-quant probing combined with an LLM judge/heuristic compiler to guide GPTQModel). 

However, there are several critical logical disconnects, mathematical approximations, and workflow inconsistencies that will prevent it from working as intended out-of-the-box.

### 1. Critical Bugs & Logical Flaws

**A. The Budget Compiler Silently Defeats the "Skip" Rule**
In `judge.py`, you have a specific heuristic to skip `lm_head` if it exhibits high outliers. However, `lm_head` is huge (e.g., in Qwen3.5-4B it is `151936 * 3072 ≈ 466M` parameters). 
* Upgrading `lm_head` from W4 (0.5 bytes/param) to Skip (2 bytes/param) costs `~700 MB`.
* Your default `budget_mb_extra` in the CLI is `150.0`.
* In `compiler.py`, the logic dictates: `if spent_mb + up["cost_mb"] > extra_budget_mb: continue`.
**Result:** The compiler will *always* reject skipping the `lm_head` under default settings. The user will be confused as to why the `lm_head_extreme` rule never actually skips the layer.
* **Fix:** Either evaluate `lm_head` skip *outside* the ROI budget calculation (treat skipping the embedding/head as a structural baseline rather than a budget upgrade), or increase the default budget significantly (e.g., `1000 MB`).

**B. `cmd_suite` Inconsistently Evaluates Baselines**
The `suite` command is meant to compare baselines, heuristics, and the judge. However:
1. It assigns `heuristic_mixed` to `heuristic_dynamic_from_flags`, which uses **hardcoded regexes** (e.g., `model.layers.(0|1|30|31)` to W8) and completely ignores the fake-quant metrics in `probe.py`.
2. Meanwhile, if no OpenAI API key is provided, `judge_proposals` falls back to `heuristic_only=True`. This means the `judge_sdq` run will actually execute your data-driven `heuristic_decision(row)` from `judge.py`!
**Result:** The benchmark leaderboard is heavily skewed. It compares a regex baseline to a data-driven heuristic disguised under the label "judge_sdq". The logic for generating the heuristic run in the suite should leverage the output of `make_judge_proposals(heuristic_only=True)` rather than `heuristic_dynamic_from_flags`.

### 2. Mathematical & Algorithmic Issues

**A. Inaccurate Parameter Size Calculations**
In `actions.py`, `bytes_per_param()` calculates size strictly as `bits / 8.0`. It completely ignores quantization metadata (scales and zero-points). 
* A W4G128 config vs a W4G64 config have the *exact same* bits per parameter (4 bits), but W4G64 has double the number of scales and zeros. 
* To patch this, you added a hack in `probe.py`: `+ (params * 0.05 / (1024 ** 2))`.
* **Fix:** Calculate group size overhead explicitly. Assuming 16-bit scales and 4-bit zeros (per group):
  
```
python
  def bytes_per_param(action: str, fp_bytes: float = 2.0) -> float:
      if action == "skip": return fp_bytes
      spec = ACTION_SPECS[action]
      base = spec.bits / 8.0
      metadata_per_group = 2.0 + 0.5  # 16-bit scale + 4-bit zero = 2.5 bytes
      group_overhead = metadata_per_group / spec.group_size
      return base + group_overhead
```

**B. Fake Quantization (RTN) vs. Real Quantization (GPTQ)**
In `probe.py`, `fake_quant_groupwise_linear_weight` implements a standard Round-To-Nearest (RTN) symmetric quantization using `w.abs().amax()`.
* **Issue:** GPTQ uses second-order information (the inverse Hessian of activations) to iteratively update weights, which often mitigates errors that RTN catches. By using RTN to calculate MSE, your "probe" might penalize modules that RTN struggles with, but that GPTQ would actually handle perfectly fine.
* **Fix:** While implementing a full Hessian-based probe is likely too slow, integrating an Activation-Aware Weight Quantization (AWQ) style scaling probe (multiplying weights by activation scales before RTN) would be a much more accurate proxy for GPTQ degradation.

### 3. Data Processing & Performance

**A. Unbatched Forward Passes**
In `probe.py`:

```
python
for text in tqdm(texts, desc="collect_activations"):
    batch = tokenizer(...)
    _ = model(**batch)
```

You are executing forward passes on 64 texts of length 256 *one by one*. For smaller models this is tolerable, but for larger models, it is very inefficient. 
* **Fix:** Batch the tokenized inputs and run them through the model. 

**B. Memory/Precision Loss in Activation Capture**
In `make_hook` (in `probe.py`), you cast captured activations to `torch.float16` and move them to the CPU.

```
python
captured_inputs[name].append(x[:take].detach().cpu().to(torch.float16))
```

If the source model uses `bfloat16`, casting to `float16` can cause outliers to overflow/clip (FP16 maxes out at ~65504), which directly undermines your `activation_outlier_rate` metric. 
* **Fix:** Keep the activations in their native dtype when moving them to the CPU (or explicitly cast to `torch.float32` if memory allows, since you are limiting the rows to 128 anyway).

### 4. Safety and Robustness

**A. LLM Parsing Fragility**
In `judge.py`, `call_openai_compatible` relies on basic string manipulation (`strip_code_fence`) and `json.loads` to extract the output. LLMs often prefix outputs with `Here is the JSON you requested...` which will break the parser if the code fence isn't strictly the first thing.
* **Praise:** The fallback mechanism in `merge_assignments` (which applies the heuristic for any modules the LLM failed to return or parse) is excellent defensively.
* **Fix:** Instead of raw parsing, use OpenAI's newer `response_format: {"type": "json_schema", ...}` to enforce the JSON structure rigorously, or use a library like `instructor` / `pydantic` to validate the payload.

**B. Division by Zero Risk in ROI**
In `compiler.py`, the ROI is `benefit / cost_mb`. 
While `probe.py` forces `max(0.0, ...)` on extra MB, if the `cost_mb` evaluates to exactly `0.0` (which is highly likely due to the size calculation bug mentioned in 2A if W4G64 vs W4G128 round the same way), `benefit / cost_mb` will throw a `ZeroDivisionError`. 
* **Fix:** Clamp the denominator: `roi = benefit / max(cost_mb, 1e-6)`.

---

### ChatGPT:
This critique is good. The biggest blockers are real, and I’d change the project before trusting any leaderboard it produces.

What I’d keep as-is is the overall architecture. What I’d change is the control semantics, the cost model, the suite labeling, and parts of the probe/runtime.

## The most important fixes

### 1) lm_head skip should not compete inside a 150 MB budget

You’re right: with a 4B-class model, `lm_head` skip is structurally expensive enough that the default compiler will almost always reject it.

The clean fix is to split decisions into two classes:

- **structural overrides**: things like `skip lm_head`
- **budgeted upgrades**: `w4_g64`, `w8_g128`, other skip candidates

That means the compiler should apply structural rules first, then spend the remaining budget on ROI-ranked upgrades.

Use this logic in `compiler.py`:

```Pythondef is_structural_skip(name: str, action: str) -> bool:    return action == "skip" and name == "lm_head"def compile_plan(assignments, probe_rows, extra_budget_mb, min_roi=1e-6):    probe_by_name = {row["name"]: row for row in probe_rows}    structural = []    upgrades = []    for item in assignments:        name = item["name"]        action = item["action"]        row = probe_by_name.get(name)        if row is None:            continue        if is_structural_skip(name, action):            structural.append(item)            continue        if action == "w4_g128":            continue        if action == "w8_g128":            benefit = row["candidates"]["w4_g128"]["mse"] - row["candidates"]["w8_g128"]["mse"]            cost_mb = row["candidates"]["w8_g128"]["estimated_extra_mb_vs_base"]        elif action == "w4_g64":            benefit = row["candidates"]["w4_g128"]["mse"] - row["candidates"]["w4_g64"]["mse"]            cost_mb = row["candidates"]["w4_g64"]["estimated_extra_mb_vs_base"]        elif action == "skip":            benefit = row["candidates"]["w4_g128"]["mse"]            cost_mb = row["candidates"]["skip"]["estimated_extra_mb_vs_base"]        else:            continue        if benefit <= 0 or cost_mb <= 0:            continue        roi = benefit / max(cost_mb, 1e-6)        if roi < min_roi:            continue        upgrades.append({            "name": name,            "action": action,            "benefit": float(benefit),            "cost_mb": float(cost_mb),            "roi": float(roi),            "priority": float(item.get("priority", 0.5)),            "reason_code": item.get("reason_code", ""),            "source": item.get("source", ""),        })    upgrades.sort(key=lambda x: (x["roi"], x["priority"]), reverse=True)    approved = []    spent_mb = 0.0    for up in upgrades:        if spent_mb + up["cost_mb"] > extra_budget_mb:            continue        approved.append(up)        spent_mb += up["cost_mb"]    final_assignments = []    structural_by_name = {x["name"]: x for x in structural}    approved_by_name = {x["name"]: x for x in approved}    for row in probe_rows:        name = row["name"]        if name in structural_by_name:            item = structural_by_name[name]            final_assignments.append({                "name": name,                "action": "skip",                "priority": float(item.get("priority", 1.0)),                "reason_code": item.get("reason_code", "structural_skip"),                "source": item.get("source", "compiler"),                "estimated_extra_mb_vs_base": row["candidates"]["skip"]["estimated_extra_mb_vs_base"],                "structural": True,            })        elif name in approved_by_name:            up = approved_by_name[name]            final_assignments.append({                "name": name,                "action": up["action"],                "priority": up["priority"],                "reason_code": up["reason_code"],                "source": up["source"],                "estimated_extra_mb_vs_base": up["cost_mb"],                "benefit": up["benefit"],                "roi": up["roi"],                "structural": False,            })        else:            final_assignments.append({                "name": name,                "action": "w4_g128",                "priority": 0.0,                "reason_code": "base_default",                "source": "compiler",                "estimated_extra_mb_vs_base": 0.0,                "structural": False,            })    return {        "budget_mb_extra": float(extra_budget_mb),        "used_mb_extra": round(float(spent_mb), 6),        "structural_skips": len(structural),        "upgrades_approved": len(approved),        "assignments": sorted(final_assignments, key=lambda x: x["name"]),    }
Pythondef is_structural_skip(name: str, action: str) -> bool:    return action == "skip" and name == "lm_head"def compile_plan(assignments, probe_rows, extra_budget_mb, min_roi=1e-6):    probe_by_name = {row["name"]: row for row in probe_rows}    structural = []    upgrades = []    for item in assignments:        name = item["name"]        action = item["action"]        row = probe_by_name.get(name)        if row is None:            continue        if is_structural_skip(name, action):            structural.append(item)            continue        if action == "w4_g128":            continue        if action == "w8_g128":            benefit = row["candidates"]["w4_g128"]["mse"] - row["candidates"]["w8_g128"]["mse"]            cost_mb = row["candidates"]["w8_g128"]["estimated_extra_mb_vs_base"]        elif action == "w4_g64":            benefit = row["candidates"]["w4_g128"]["mse"] - row["candidates"]["w4_g64"]["mse"]            cost_mb = row["candidates"]["w4_g64"]["estimated_extra_mb_vs_base"]        elif action == "skip":            benefit = row["candidates"]["w4_g128"]["mse"]            cost_mb = row["candidates"]["skip"]["estimated_extra_mb_vs_base"]        else:            continue        if benefit <= 0 or cost_mb <= 0:            continue        roi = benefit / max(cost_mb, 1e-6)        if roi < min_roi:            continue        upgrades.append({            "name": name,            "action": action,            "benefit": float(benefit),            "cost_mb": float(cost_mb),            "roi": float(roi),            "priority": float(item.get("priority", 0.5)),            "reason_code": item.get("reason_code", ""),            "source": item.get("source", ""),        })    upgrades.sort(key=lambda x: (x["roi"], x["priority"]), reverse=True)    approved = []    spent_mb = 0.0    for up in upgrades:        if spent_mb + up["cost_mb"] > extra_budget_mb:            continue        approved.append(up)        spent_mb += up["cost_mb"]    final_assignments = []    structural_by_name = {x["name"]: x for x in structural}    approved_by_name = {x["name"]: x for x in approved}    for row in probe_rows:        name = row["name"]        if name in structural_by_name:            item = structural_by_name[name]            final_assignments.append({                "name": name,                "action": "skip",                "priority": float(item.get("priority", 1.0)),                "reason_code": item.get("reason_code", "structural_skip"),                "source": item.get("source", "compiler"),                "estimated_extra_mb_vs_base": row["candidates"]["skip"]["estimated_extra_mb_vs_base"],                "structural": True,            })        elif name in approved_by_name:            up = approved_by_name[name]            final_assignments.append({                "name": name,                "action": up["action"],                "priority": up["priority"],                "reason_code": up["reason_code"],                "source": up["source"],                "estimated_extra_mb_vs_base": up["cost_mb"],                "benefit": up["benefit"],                "roi": up["roi"],                "structural": False,            })        else:            final_assignments.append({                "name": name,                "action": "w4_g128",                "priority": 0.0,                "reason_code": "base_default",                "source": "compiler",                "estimated_extra_mb_vs_base": 0.0,                "structural": False,            })    return {        "budget_mb_extra": float(extra_budget_mb),        "used_mb_extra": round(float(spent_mb), 6),        "structural_skips": len(structural),        "upgrades_approved": len(approved),        "assignments": sorted(final_assignments, key=lambda x: x["name"]),    }
```

That makes `lm_head` skip explicit instead of silently impossible.

### 2) The suite currently compares mismatched baselines

You’re also right here. The current suite compares:

- a regex rule baseline
- a data-driven heuristic under the “judge” label when no external judge is configured

That is misleading.

The fix is:

- keep the regex one, but call it `rule_based_baseline`
- add a separate `data_driven_heuristic`
- only run `judge_sdq` if an external judge endpoint is actually configured

So `cmd_suite` should become:

```Pythonrule_based_dynamic = heuristic_dynamic_from_flags(heuristic_flags)heuristic_proposals = make_judge_proposals(    JudgeConfig(        features_jsonl=features_jsonl,        budget_mb_extra=args.budget_mb_extra,        heuristic_only=True,        chunk_size=args.chunk_size,    ))heuristic_plan = compile_plan(    heuristic_proposals["assignments"],    probe_rows,    args.budget_mb_extra,    args.min_roi,)data_driven_heuristic_dynamic = plan_to_dynamic(heuristic_plan)configs = {    "baseline_w4g128": None,    "rule_based_baseline": rule_based_dynamic,    "data_driven_heuristic": data_driven_heuristic_dynamic,}if args.judge_base_url and args.judge_model:    judge_proposals = make_judge_proposals(        JudgeConfig(            features_jsonl=features_jsonl,            budget_mb_extra=args.budget_mb_extra,            heuristic_only=False,            judge_base_url=args.judge_base_url,            judge_model=args.judge_model,            judge_api_key_env=args.judge_api_key_env,            chunk_size=args.chunk_size,        )    )    judge_plan = compile_plan(judge_proposals["assignments"], probe_rows, args.budget_mb_extra, args.min_roi)    configs["judge_sdq"] = plan_to_dynamic(judge_plan)
Pythonrule_based_dynamic = heuristic_dynamic_from_flags(heuristic_flags)heuristic_proposals = make_judge_proposals(    JudgeConfig(        features_jsonl=features_jsonl,        budget_mb_extra=args.budget_mb_extra,        heuristic_only=True,        chunk_size=args.chunk_size,    ))heuristic_plan = compile_plan(    heuristic_proposals["assignments"],    probe_rows,    args.budget_mb_extra,    args.min_roi,)data_driven_heuristic_dynamic = plan_to_dynamic(heuristic_plan)configs = {    "baseline_w4g128": None,    "rule_based_baseline": rule_based_dynamic,    "data_driven_heuristic": data_driven_heuristic_dynamic,}if args.judge_base_url and args.judge_model:    judge_proposals = make_judge_proposals(        JudgeConfig(            features_jsonl=features_jsonl,            budget_mb_extra=args.budget_mb_extra,            heuristic_only=False,            judge_base_url=args.judge_base_url,            judge_model=args.judge_model,            judge_api_key_env=args.judge_api_key_env,            chunk_size=args.chunk_size,        )    )    judge_plan = compile_plan(judge_proposals["assignments"], probe_rows, args.budget_mb_extra, args.min_roi)    configs["judge_sdq"] = plan_to_dynamic(judge_plan)
```

That makes the leaderboard honest.

### 3) The parameter-size model is too loose

I agree with your criticism, with one nuance: the exact metadata overhead depends on the real pack format and whether the backend stores zeros for symmetric quant.

So I would not hardcode one “true” formula unless you have confirmed GPTQModel’s current serialization format for your chosen config. But I would definitely replace the current hack with a configurable overhead model.

A safer version:

```Pythondef bytes_per_param(action: str, fp_bytes: float = 2.0, scale_bytes: float = 2.0, zero_bytes: float = 0.0) -> float:    if action == "skip":        return fp_bytes    spec = ACTION_SPECS[action]    assert spec.bits is not None and spec.group_size is not None    base = spec.bits / 8.0    metadata_per_group = scale_bytes + zero_bytes    overhead = metadata_per_group / spec.group_size    return base + overhead
Pythondef bytes_per_param(action: str, fp_bytes: float = 2.0, scale_bytes: float = 2.0, zero_bytes: float = 0.0) -> float:    if action == "skip":        return fp_bytes    spec = ACTION_SPECS[action]    assert spec.bits is not None and spec.group_size is not None    base = spec.bits / 8.0    metadata_per_group = scale_bytes + zero_bytes    overhead = metadata_per_group / spec.group_size    return base + overhead
```

Then wire it like this:

- `sym=True` → `zero_bytes=0.0` as a lower-overhead estimate
- asymmetric / packed-zero path → `zero_bytes=0.5` or whatever the actual backend uses

That gives you a real cost model instead of `+ 0.05 MB`.

### 4) RTN fake-quant is still only a proxy

Yes. This is not GPTQ. It is a ranking proxy.

I would not try to sell it as “highly accurate GPTQ loss.” I’d describe it as:

per-layer RTN-style sensitivity estimate with group geometry matched to the target quantizer

That said, your suggestion is right: an activation-aware probe would be better than plain RTN.

The lightest useful upgrade is:

- compute per-input-channel activation magnitude from the captured rows
- scale weights by that channel salience before fake quant
- unscale afterward

Something like:

```Pythondef activation_aware_fake_quant(weight, x_rows, bits, group_size):    # x_rows shape: [N, in_features]    salience = x_rows.float().abs().mean(dim=0).clamp_min(1e-6)    salience = salience / salience.mean()    w_scaled = weight.float() * salience.unsqueeze(0)    wq_scaled = fake_quant_groupwise_linear_weight(w_scaled, bits=bits, group_size=group_size)    wq = wq_scaled / salience.unsqueeze(0)    return wq.to(weight.device, dtype=weight.dtype)
Pythondef activation_aware_fake_quant(weight, x_rows, bits, group_size):    # x_rows shape: [N, in_features]    salience = x_rows.float().abs().mean(dim=0).clamp_min(1e-6)    salience = salience / salience.mean()    w_scaled = weight.float() * salience.unsqueeze(0)    wq_scaled = fake_quant_groupwise_linear_weight(w_scaled, bits=bits, group_size=group_size)    wq = wq_scaled / salience.unsqueeze(0)    return wq.to(weight.device, dtype=weight.dtype)
```

That is still not GPTQ, but it is a materially better proxy for fragile channels.

### 5) Probe collection should be batched

Agreed.

Right now the probe is doing one forward per text, which is wasteful. The simplest fix is to batch texts before tokenization.

Example:

```Pythondef batched(iterable, batch_size):    for i in range(0, len(iterable), batch_size):        yield iterable[i:i + batch_size]with torch.no_grad():    for batch_texts in tqdm(list(batched(texts, 8)), desc="collect_activations"):        batch = tokenizer(            batch_texts,            return_tensors="pt",            truncation=True,            max_length=cfg.max_length,            padding=True,        )        batch = {k: v.to(model.device) for k, v in batch.items()}        _ = model(**batch)        if all(v <= 0 for v in remaining_rows.values()):            break
Pythondef batched(iterable, batch_size):    for i in range(0, len(iterable), batch_size):        yield iterable[i:i + batch_size]with torch.no_grad():    for batch_texts in tqdm(list(batched(texts, 8)), desc="collect_activations"):        batch = tokenizer(            batch_texts,            return_tensors="pt",            truncation=True,            max_length=cfg.max_length,            padding=True,        )        batch = {k: v.to(model.device) for k, v in batch.items()}        _ = model(**batch)        if all(v <= 0 for v in remaining_rows.values()):            break
```

That speeds the probe up substantially.

### 6) Preserve activation dtype when capturing

Yes. This is a good catch.

Casting BF16 activations to FP16 on CPU can distort outlier statistics. The clean change is either:

- preserve original dtype, or
- normalize everything to FP32 on CPU

Since you are row-limiting aggressively, FP32 is safer:

```Pythoncaptured_inputs[name].append(x[:take].detach().cpu().to(torch.float32))
Pythoncaptured_inputs[name].append(x[:take].detach().cpu().to(torch.float32))
```

Then when feeding the module probe back into the layer, cast to the module dtype on device.

### 7) LLM parsing needs harder validation

Also correct.

One caveat: many OpenAI-compatible endpoints do **not** fully support `json_schema`, even if official OpenAI endpoints do. So I’d implement a two-layer strategy:

- try `json_schema` if the backend supports it
- otherwise fall back to `json_object`
- validate with Pydantic after parsing

At minimum, define a strict validator:

```Pythonfrom pydantic import BaseModel, Fieldfrom typing import Literalclass Assignment(BaseModel):    name: str    action: Literal["w4_g128", "w4_g64", "w8_g128", "skip"]    priority: float = Field(ge=0.0, le=1.0)    reason_code: strclass AssignmentPayload(BaseModel):    assignments: list[Assignment]
Pythonfrom pydantic import BaseModel, Fieldfrom typing import Literalclass Assignment(BaseModel):    name: str    action: Literal["w4_g128", "w4_g64", "w8_g128", "skip"]    priority: float = Field(ge=0.0, le=1.0)    reason_code: strclass AssignmentPayload(BaseModel):    assignments: list[Assignment]
```

Then parse and validate after `json.loads`.

### 8) roi = benefit / cost_mb should clamp denominator

Yes. That one should be fixed immediately.

Use:

```Pythonroi = benefit / max(cost_mb, 1e-6)
Pythonroi = benefit / max(cost_mb, 1e-6)
```

Even after the size model is fixed, that clamp is still the right defensive behavior.

## My verdict on the critique

These are the real severity levels I’d assign:

- **must-fix before trusting results**

suite inconsistency

silent `lm_head` skip defeat

denominator clamp

activation dtype handling
- suite inconsistency
- silent `lm_head` skip defeat
- denominator clamp
- activation dtype handling
- **should-fix before calling the budget model credible**

metadata-aware bytes-per-param

batch the probe
- metadata-aware bytes-per-param
- batch the probe
- **nice next upgrade**

activation-aware fake quant

stronger structured LLM validation
- activation-aware fake quant
- stronger structured LLM validation

So the project is **not broken conceptually**, but it is still **too inconsistent to produce a trustworthy leaderboard without these fixes**.

The cleanest next step is to ship a **v4 JSON project** with:

- structural skip support
- honest suite labels
- metadata-aware cost model
- batched probing
- FP32 activation capture
- safer judge validation

I can produce that as a single JSON artifact next.

---

### User:
yes

---

### ChatGPT:
Here’s the v4 JSON project artifact with the fixes baked in:

It includes:

- structural `lm_head` skip handling outside the ROI budget
- honest suite labels
- metadata-aware bytes-per-param
- batched probing
- FP32 activation capture
- Pydantic-validated judge output
- clamped ROI denominator
- optional activation-aware fake quant proxy

If you want, next I can also turn this JSON artifact into a ready-to-extract bash script that recreates the full folder locally in one command.

---

### User:
i cant download it, provide it in plain text please

---

### ChatGPT:
Run this in an empty folder. It will recreate the full v4 project locally.

```Bashmkdir -p sdq_quant_judge_v4/quant_judgecd sdq_quant_judge_v4cat > README.md <<'EOF'# SDQ Quant Judge v4 for GPTQModelA software-defined quantization scaffold for `Qwen/Qwen3.5-4B-Base` using:- a real FP probe via `transformers`- per-row, per-group fake quantization- optional activation-aware proxy quantization- an external OpenAI-compatible judge- a deterministic compiler with structural skips + ROI budget clamp- GPTQModel `dynamic` mixed quantization- an honest benchmark suite:  - `baseline_w4g128`  - `rule_based_baseline`  - `data_driven_heuristic`  - `judge_sdq` (only when a judge endpoint is actually configured)## InstallCreate a virtualenv and install your CUDA-matching PyTorch manually first.Example:```bashpython -m venv .venvsource .venv/bin/activate# Example only — change to your CUDA build:# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128pip install -r requirements.txt
Bashmkdir -p sdq_quant_judge_v4/quant_judgecd sdq_quant_judge_v4cat > README.md <<'EOF'# SDQ Quant Judge v4 for GPTQModelA software-defined quantization scaffold for `Qwen/Qwen3.5-4B-Base` using:- a real FP probe via `transformers`- per-row, per-group fake quantization- optional activation-aware proxy quantization- an external OpenAI-compatible judge- a deterministic compiler with structural skips + ROI budget clamp- GPTQModel `dynamic` mixed quantization- an honest benchmark suite:  - `baseline_w4g128`  - `rule_based_baseline`  - `data_driven_heuristic`  - `judge_sdq` (only when a judge endpoint is actually configured)## InstallCreate a virtualenv and install your CUDA-matching PyTorch manually first.Example:```bashpython -m venv .venvsource .venv/bin/activate# Example only — change to your CUDA build:# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128pip install -r requirements.txt
```

## Probe

```Bashpython -m quant_judge.pipeline probe \  --model-id Qwen/Qwen3.5-4B-Base \  --out-dir ./runs/probe \  --dataset-name wikitext \  --dataset-config wikitext-2-raw-v1 \  --split train \  --text-field text \  --num-texts 64 \  --max-length 256 \  --max-rows-per-module 128 \  --probe-batch-size 8 \  --include-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head
Bashpython -m quant_judge.pipeline probe \  --model-id Qwen/Qwen3.5-4B-Base \  --out-dir ./runs/probe \  --dataset-name wikitext \  --dataset-config wikitext-2-raw-v1 \  --split train \  --text-field text \  --num-texts 64 \  --max-length 256 \  --max-rows-per-module 128 \  --probe-batch-size 8 \  --include-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head
```

## Plan with data-driven heuristic only

```Bashpython -m quant_judge.pipeline plan \  --features-jsonl ./runs/probe/probe_features.jsonl \  --plan-out ./runs/plan.json \  --dynamic-out ./runs/dynamic.json \  --budget-mb-extra 150 \  --heuristic-only
Bashpython -m quant_judge.pipeline plan \  --features-jsonl ./runs/probe/probe_features.jsonl \  --plan-out ./runs/plan.json \  --dynamic-out ./runs/dynamic.json \  --budget-mb-extra 150 \  --heuristic-only
```

## Plan with external judge

```Bashexport JUDGE_API_KEY=your_key_herepython -m quant_judge.pipeline plan \  --features-jsonl ./runs/probe/probe_features.jsonl \  --plan-out ./runs/plan.json \  --dynamic-out ./runs/dynamic.json \  --budget-mb-extra 150 \  --judge-base-url http://127.0.0.1:8000/v1 \  --judge-model Qwen/Qwen2.5-14B-Instruct
Bashexport JUDGE_API_KEY=your_key_herepython -m quant_judge.pipeline plan \  --features-jsonl ./runs/probe/probe_features.jsonl \  --plan-out ./runs/plan.json \  --dynamic-out ./runs/dynamic.json \  --budget-mb-extra 150 \  --judge-base-url http://127.0.0.1:8000/v1 \  --judge-model Qwen/Qwen2.5-14B-Instruct
```

## Quantize from plan

```Bashpython -m quant_judge.pipeline quantize \  --model-id Qwen/Qwen3.5-4B-Base \  --plan-json ./runs/plan.json \  --out-dir ./runs/quantized \  --dataset-name wikitext \  --dataset-config wikitext-2-raw-v1 \  --split train \  --text-field text \  --num-texts 128 \  --batch-size 4
Bashpython -m quant_judge.pipeline quantize \  --model-id Qwen/Qwen3.5-4B-Base \  --plan-json ./runs/plan.json \  --out-dir ./runs/quantized \  --dataset-name wikitext \  --dataset-config wikitext-2-raw-v1 \  --split train \  --text-field text \  --num-texts 128 \  --batch-size 4
```

## Evaluate a saved model

```Bashpython -m quant_judge.pipeline eval \  --model-path ./runs/quantized \  --tasks arc_challenge,hellaswag \  --output-json ./runs/eval.json
Bashpython -m quant_judge.pipeline eval \  --model-path ./runs/quantized \  --tasks arc_challenge,hellaswag \  --output-json ./runs/eval.json
```

## Full suite

This compares:

- `baseline_w4g128`
- `rule_based_baseline`
- `data_driven_heuristic`
- `judge_sdq` only if a judge endpoint is configured

Without a judge:

```Bashpython -m quant_judge.pipeline suite \  --model-id Qwen/Qwen3.5-4B-Base \  --work-dir ./runs/suite \  --dataset-name wikitext \  --dataset-config wikitext-2-raw-v1 \  --split train \  --text-field text \  --probe-num-texts 64 \  --quant-num-texts 128 \  --tasks arc_challenge,hellaswag
Bashpython -m quant_judge.pipeline suite \  --model-id Qwen/Qwen3.5-4B-Base \  --work-dir ./runs/suite \  --dataset-name wikitext \  --dataset-config wikitext-2-raw-v1 \  --split train \  --text-field text \  --probe-num-texts 64 \  --quant-num-texts 128 \  --tasks arc_challenge,hellaswag
```

With a judge:

```Bashexport JUDGE_API_KEY=your_key_herepython -m quant_judge.pipeline suite \  --model-id Qwen/Qwen3.5-4B-Base \  --work-dir ./runs/suite \  --dataset-name wikitext \  --dataset-config wikitext-2-raw-v1 \  --split train \  --text-field text \  --probe-num-texts 64 \  --quant-num-texts 128 \  --tasks arc_challenge,hellaswag \  --judge-base-url http://127.0.0.1:8000/v1 \  --judge-model Qwen/Qwen2.5-14B-Instruct
Bashexport JUDGE_API_KEY=your_key_herepython -m quant_judge.pipeline suite \  --model-id Qwen/Qwen3.5-4B-Base \  --work-dir ./runs/suite \  --dataset-name wikitext \  --dataset-config wikitext-2-raw-v1 \  --split train \  --text-field text \  --probe-num-texts 64 \  --quant-num-texts 128 \  --tasks arc_challenge,hellaswag \  --judge-base-url http://127.0.0.1:8000/v1 \  --judge-model Qwen/Qwen2.5-14B-Instruct
```

## Notes

- The fake quant probe is still a proxy, not full GPTQ.
- The optional activation-aware path is closer to reality than plain RTN.
- `lm_head` skip is treated as a structural override outside the ROI budget so it is not silently defeated by a tiny default budget.
EOF

cat > requirements.txt <<'EOF'
transformers
datasets
accelerate
requests
numpy
tqdm
psutil
pydantic>=2
gptqmodel
lm-eval
EOF

cat > quant_judge/**init**.py <<'EOF'
"""SDQ Quant Judge v4."""
EOF

cat > quant_judge/actions.py <<'EOF'
from **future** import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class ActionSpec:
name: str
bits: int | None
group_size: int | None
skip: bool

ACTION_SPECS: dict[str, ActionSpec] = {
"w4_g128": ActionSpec("w4_g128", bits=4, group_size=128, skip=False),
"w4_g64": ActionSpec("w4_g64", bits=4, group_size=64, skip=False),
"w8_g128": ActionSpec("w8_g128", bits=8, group_size=128, skip=False),
"skip": ActionSpec("skip", bits=None, group_size=None, skip=True),
}

BASE_ACTION = "w4_g128"

def allowed_actions() -> list[str]:
return list(ACTION_SPECS.keys())

def bytes_per_param(
action: str,
*,
fp_bytes: float = 2.0,
sym: bool = True,
scale_bytes: float = 2.0,
zero_bytes_asym: float = 0.5,
) -> float:
"""
Approximate bytes/param including metadata overhead.

```For quantized actions:  base bytes = bits / 8  overhead per param = (scale_bytes + zero_bytes) / group_sizeFor skip:  bytes/param = fp_bytesThis is still an approximation, but far more honest than bits/8 alone."""if action == "skip":    return fp_bytesspec = ACTION_SPECS[action]assert spec.bits is not Noneassert spec.group_size is not Nonebase = spec.bits / 8.0zero_bytes = 0.0 if sym else zero_bytes_asymoverhead = (scale_bytes + zero_bytes) / spec.group_sizereturn base + overhead
For quantized actions:  base bytes = bits / 8  overhead per param = (scale_bytes + zero_bytes) / group_sizeFor skip:  bytes/param = fp_bytesThis is still an approximation, but far more honest than bits/8 alone."""if action == "skip":    return fp_bytesspec = ACTION_SPECS[action]assert spec.bits is not Noneassert spec.group_size is not Nonebase = spec.bits / 8.0zero_bytes = 0.0 if sym else zero_bytes_asymoverhead = (scale_bytes + zero_bytes) / spec.group_sizereturn base + overhead
```

def extra_mb_vs_base(
params: int,
action: str,
*,
sym: bool = True,
fp_bytes: float = 2.0,
scale_bytes: float = 2.0,
zero_bytes_asym: float = 0.5,
) -> float:
base = params * bytes_per_param(
BASE_ACTION,
fp_bytes=fp_bytes,
sym=sym,
scale_bytes=scale_bytes,
zero_bytes_asym=zero_bytes_asym,
)
now = params * bytes_per_param(
action,
fp_bytes=fp_bytes,
sym=sym,
scale_bytes=scale_bytes,
zero_bytes_asym=zero_bytes_asym,
)
return (now - base) / (1024 ** 2)

def action_to_override(action: str) -> dict:
if action == "skip":
return {}
spec = ACTION_SPECS[action]
return {"bits": spec.bits, "group_size": spec.group_size}
EOF

cat > quant_judge/probe.py <<'EOF'
from **future** import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .actions import extra_mb_vs_base

@dataclass
class ProbeConfig:
model_id: str
out_dir: str
dataset_name: str
dataset_config: str | None
split: str
text_field: str
num_texts: int
max_length: int
max_rows_per_module: int
include_modules: list[str]
probe_batch_size: int = 8
trust_remote_code: bool = True
use_activation_aware_proxy: bool = False
sym: bool = True
scale_bytes: float = 2.0
zero_bytes_asym: float = 0.5

def pick_dtype() -> torch.dtype:
if not torch.cuda.is_available():
return torch.float32
major, _minor = torch.cuda.get_device_capability(0)
if major >= 8:
return torch.bfloat16
return torch.float16

def fake_quant_groupwise_linear_weight(weight: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
"""
Symmetric per-row, per-group fake quantization for Linear weights.
weight shape: [out_features, in_features]
"""
assert weight.dim() == 2
w = weight.detach().float()
out_features, in_features = w.shape

```pad = (group_size - (in_features % group_size)) % group_sizeif pad:    w = F.pad(w, (0, pad), value=0.0)maxq = (2 ** (bits - 1)) - 1w = w.view(out_features, -1, group_size)absmax = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)scale = absmax / maxqq = torch.round(w / scale).clamp(-maxq, maxq)dq = q * scaledq = dq.view(out_features, -1)if pad:    dq = dq[:, :in_features]return dq.to(weight.device, dtype=weight.dtype)
pad = (group_size - (in_features % group_size)) % group_sizeif pad:    w = F.pad(w, (0, pad), value=0.0)maxq = (2 ** (bits - 1)) - 1w = w.view(out_features, -1, group_size)absmax = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)scale = absmax / maxqq = torch.round(w / scale).clamp(-maxq, maxq)dq = q * scaledq = dq.view(out_features, -1)if pad:    dq = dq[:, :in_features]return dq.to(weight.device, dtype=weight.dtype)
```

def activation_aware_fake_quant(weight: torch.Tensor, x_rows: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
"""
AWQ-ish lightweight proxy:
- compute per-input-channel salience from captured activations
- scale weights by salience
- fake quantize
- unscale
"""
salience = x_rows.float().abs().mean(dim=0).clamp_min(1e-6)
salience = salience / salience.mean()
w_scaled = weight.float() * salience.unsqueeze(0)
wq_scaled = fake_quant_groupwise_linear_weight(w_scaled, bits=bits, group_size=group_size)
wq = wq_scaled / salience.unsqueeze(0)
return wq.to(weight.device, dtype=weight.dtype)

def load_texts(dataset_name: str, dataset_config: str | None, split: str, text_field: str, num_texts: int) -> list[str]:
ds = load_dataset(dataset_name, dataset_config, split=split)
texts: list[str] = []
for row in ds:
value = row.get(text_field, "")
if isinstance(value, str) and value.strip():
texts.append(value.strip())
if len(texts) >= num_texts:
break
if not texts:
raise ValueError("No usable texts found.")
return texts

def batched(items: list[str], batch_size: int):
for i in range(0, len(items), batch_size):
yield items[i:i + batch_size]

def module_selected(name: str, include_modules: list[str]) -> bool:
return any(token in name for token in include_modules)

def iter_target_linears(model, include_modules: list[str]):
for name, module in model.named_modules():
if isinstance(module, torch.nn.Linear) and module_selected(name, include_modules):
yield name, module

def flatten_rows(x: torch.Tensor) -> torch.Tensor:
if x.dim() == 2:
return x
if x.dim() >= 3:
return x.reshape(-1, x.shape[-1])
return x.unsqueeze(0)

def sample_rows(x: torch.Tensor, max_rows: int) -> torch.Tensor:
rows = flatten_rows(x)
if rows.shape[0] <= max_rows:
return rows
idx = torch.randperm(rows.shape[0], device=rows.device)[:max_rows]
return rows.index_select(0, idx)

def weight_kurtosis(weight: torch.Tensor) -> float:
w = weight.detach().float().flatten()
mu = w.mean()
var = ((w - mu) ** 2).mean().clamp_min(1e-12)
return float((((w - mu) ** 4).mean() / (var ** 2)).item())

def activation_outlier_rate(x: torch.Tensor, sigma: float = 6.0) -> float:
xf = x.float()
std = xf.std().clamp_min(1e-6)
return float((xf.abs() > sigma * std).float().mean().item())

def block_index_from_name(name: str) -> int | None:
parts = name.split(".")
for i, p in enumerate(parts):
if p == "layers" and i + 1 < len(parts):
try:
return int(parts[i + 1])
except Exception:
return None
return None

def reduced_metrics_for_action(
x: torch.Tensor,
weight: torch.Tensor,
bias: torch.Tensor | None,
*,
bits: int,
group_size: int,
use_activation_aware_proxy: bool,
) -> dict:
if use_activation_aware_proxy:
wq = activation_aware_fake_quant(weight, x, bits=bits, group_size=group_size)
else:
wq = fake_quant_groupwise_linear_weight(weight, bits=bits, group_size=group_size)

```out_fp = F.linear(x, weight, bias)out_q = F.linear(x, wq, bias)mse = F.mse_loss(out_q.float(), out_fp.float()).item()cos = F.cosine_similarity(out_q.float(), out_fp.float(), dim=-1).mean().item()max_abs = (out_q.float() - out_fp.float()).abs().max().item()return {"mse": float(mse), "cosine": float(cos), "max_abs": float(max_abs)}
out_fp = F.linear(x, weight, bias)out_q = F.linear(x, wq, bias)mse = F.mse_loss(out_q.float(), out_fp.float()).item()cos = F.cosine_similarity(out_q.float(), out_fp.float(), dim=-1).mean().item()max_abs = (out_q.float() - out_fp.float()).abs().max().item()return {"mse": float(mse), "cosine": float(cos), "max_abs": float(max_abs)}
```

def run_probe(cfg: ProbeConfig) -> dict:
out_dir = Path(cfg.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

```tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code)if tokenizer.pad_token_id is None:    tokenizer.pad_token = tokenizer.eos_tokenmodel = AutoModelForCausalLM.from_pretrained(    cfg.model_id,    trust_remote_code=cfg.trust_remote_code,    torch_dtype=pick_dtype(),    device_map="auto",    low_cpu_mem_usage=True,)model.eval()texts = load_texts(cfg.dataset_name, cfg.dataset_config, cfg.split, cfg.text_field, cfg.num_texts)targets = list(iter_target_linears(model, cfg.include_modules))if not targets:    raise ValueError("No target modules found.")captured_inputs = {name: [] for name, _ in targets}remaining_rows = {name: cfg.max_rows_per_module for name, _ in targets}handles = []def make_hook(name: str):    def hook(_module, inp, _out):        rem = remaining_rows[name]        if rem <= 0 or not inp or not isinstance(inp[0], torch.Tensor):            return        x = sample_rows(inp[0], rem)        take = min(x.shape[0], rem)        if take <= 0:            return        # keep FP32 on CPU to avoid BF16 -> FP16 clipping issues        captured_inputs[name].append(x[:take].detach().cpu().to(torch.float32))        remaining_rows[name] -= take    return hookfor name, module in targets:    handles.append(module.register_forward_hook(make_hook(name)))with torch.no_grad():    for batch_texts in tqdm(list(batched(texts, cfg.probe_batch_size)), desc="collect_activations"):        batch = tokenizer(            batch_texts,            return_tensors="pt",            truncation=True,            max_length=cfg.max_length,            padding=True,        )        batch = {k: v.to(model.device) for k, v in batch.items()}        _ = model(**batch)        if all(v <= 0 for v in remaining_rows.values()):            breakfor h in handles:    h.remove()rows = []for name, module in tqdm(targets, desc="probe_modules"):    if not captured_inputs[name]:        continue    x = torch.cat(captured_inputs[name], dim=0).to(module.weight.device, dtype=module.weight.dtype)    weight = module.weight.detach()    bias = None if module.bias is None else module.bias.detach()    m_w4g128 = reduced_metrics_for_action(        x, weight, bias,        bits=4, group_size=128,        use_activation_aware_proxy=cfg.use_activation_aware_proxy    )    m_w4g64 = reduced_metrics_for_action(        x, weight, bias,        bits=4, group_size=64,        use_activation_aware_proxy=cfg.use_activation_aware_proxy    )    m_w8g128 = reduced_metrics_for_action(        x, weight, bias,        bits=8, group_size=128,        use_activation_aware_proxy=cfg.use_activation_aware_proxy    )    params = int(weight.numel())    row = {        "name": name,        "module_type": name.split(".")[-1],        "block_index": block_index_from_name(name),        "params": params,        "captured_rows": int(x.shape[0]),        "weight_kurtosis": weight_kurtosis(weight),        "act_outlier_rate": activation_outlier_rate(x),        "candidates": {            "w4_g128": {                **m_w4g128,                "estimated_extra_mb_vs_base": 0.0,            },            "w4_g64": {                **m_w4g64,                "estimated_extra_mb_vs_base": max(                    0.0,                    extra_mb_vs_base(                        params, "w4_g64",                        sym=cfg.sym,                        scale_bytes=cfg.scale_bytes,                        zero_bytes_asym=cfg.zero_bytes_asym,                    ),                ),            },            "w8_g128": {                **m_w8g128,                "estimated_extra_mb_vs_base": max(                    0.0,                    extra_mb_vs_base(                        params, "w8_g128",                        sym=cfg.sym,                        scale_bytes=cfg.scale_bytes,                        zero_bytes_asym=cfg.zero_bytes_asym,                    ),                ),            },            "skip": {                "mse": 0.0,                "cosine": 1.0,                "max_abs": 0.0,                "estimated_extra_mb_vs_base": max(                    0.0,                    extra_mb_vs_base(                        params, "skip",                        sym=cfg.sym,                        scale_bytes=cfg.scale_bytes,                        zero_bytes_asym=cfg.zero_bytes_asym,                    ),                ),            },        },        "proxy_gain_w4g64_vs_w4": float(max(0.0, m_w4g128["mse"] - m_w4g64["mse"])),        "proxy_gain_w8_vs_w4": float(max(0.0, m_w4g128["mse"] - m_w8g128["mse"])),    }    rows.append(row)features_jsonl = out_dir / "probe_features.jsonl"metadata_json = out_dir / "probe_metadata.json"summary_json = out_dir / "probe_summary.json"with open(features_jsonl, "w", encoding="utf-8") as f:    for row in rows:        f.write(json.dumps(row, ensure_ascii=False) + "\n")metadata = {    "model_id": cfg.model_id,    "dataset_name": cfg.dataset_name,    "dataset_config": cfg.dataset_config,    "split": cfg.split,    "text_field": cfg.text_field,    "num_texts": cfg.num_texts,    "max_length": cfg.max_length,    "max_rows_per_module": cfg.max_rows_per_module,    "probe_batch_size": cfg.probe_batch_size,    "include_modules": cfg.include_modules,    "num_rows": len(rows),    "use_activation_aware_proxy": cfg.use_activation_aware_proxy,    "sym": cfg.sym,    "scale_bytes": cfg.scale_bytes,    "zero_bytes_asym": cfg.zero_bytes_asym,}metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")summary_json.write_text(json.dumps({"num_rows": len(rows)}, indent=2), encoding="utf-8")del modelif torch.cuda.is_available():    torch.cuda.empty_cache()return {    "features_jsonl": str(features_jsonl),    "metadata_json": str(metadata_json),    "summary_json": str(summary_json),    "num_rows": len(rows),}
tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code)if tokenizer.pad_token_id is None:    tokenizer.pad_token = tokenizer.eos_tokenmodel = AutoModelForCausalLM.from_pretrained(    cfg.model_id,    trust_remote_code=cfg.trust_remote_code,    torch_dtype=pick_dtype(),    device_map="auto",    low_cpu_mem_usage=True,)model.eval()texts = load_texts(cfg.dataset_name, cfg.dataset_config, cfg.split, cfg.text_field, cfg.num_texts)targets = list(iter_target_linears(model, cfg.include_modules))if not targets:    raise ValueError("No target modules found.")captured_inputs = {name: [] for name, _ in targets}remaining_rows = {name: cfg.max_rows_per_module for name, _ in targets}handles = []def make_hook(name: str):    def hook(_module, inp, _out):        rem = remaining_rows[name]        if rem <= 0 or not inp or not isinstance(inp[0], torch.Tensor):            return        x = sample_rows(inp[0], rem)        take = min(x.shape[0], rem)        if take <= 0:            return        # keep FP32 on CPU to avoid BF16 -> FP16 clipping issues        captured_inputs[name].append(x[:take].detach().cpu().to(torch.float32))        remaining_rows[name] -= take    return hookfor name, module in targets:    handles.append(module.register_forward_hook(make_hook(name)))with torch.no_grad():    for batch_texts in tqdm(list(batched(texts, cfg.probe_batch_size)), desc="collect_activations"):        batch = tokenizer(            batch_texts,            return_tensors="pt",            truncation=True,            max_length=cfg.max_length,            padding=True,        )        batch = {k: v.to(model.device) for k, v in batch.items()}        _ = model(**batch)        if all(v <= 0 for v in remaining_rows.values()):            breakfor h in handles:    h.remove()rows = []for name, module in tqdm(targets, desc="probe_modules"):    if not captured_inputs[name]:        continue    x = torch.cat(captured_inputs[name], dim=0).to(module.weight.device, dtype=module.weight.dtype)    weight = module.weight.detach()    bias = None if module.bias is None else module.bias.detach()    m_w4g128 = reduced_metrics_for_action(        x, weight, bias,        bits=4, group_size=128,        use_activation_aware_proxy=cfg.use_activation_aware_proxy    )    m_w4g64 = reduced_metrics_for_action(        x, weight, bias,        bits=4, group_size=64,        use_activation_aware_proxy=cfg.use_activation_aware_proxy    )    m_w8g128 = reduced_metrics_for_action(        x, weight, bias,        bits=8, group_size=128,        use_activation_aware_proxy=cfg.use_activation_aware_proxy    )    params = int(weight.numel())    row = {        "name": name,        "module_type": name.split(".")[-1],        "block_index": block_index_from_name(name),        "params": params,        "captured_rows": int(x.shape[0]),        "weight_kurtosis": weight_kurtosis(weight),        "act_outlier_rate": activation_outlier_rate(x),        "candidates": {            "w4_g128": {                **m_w4g128,                "estimated_extra_mb_vs_base": 0.0,            },            "w4_g64": {                **m_w4g64,                "estimated_extra_mb_vs_base": max(                    0.0,                    extra_mb_vs_base(                        params, "w4_g64",                        sym=cfg.sym,                        scale_bytes=cfg.scale_bytes,                        zero_bytes_asym=cfg.zero_bytes_asym,                    ),                ),            },            "w8_g128": {                **m_w8g128,                "estimated_extra_mb_vs_base": max(                    0.0,                    extra_mb_vs_base(                        params, "w8_g128",                        sym=cfg.sym,                        scale_bytes=cfg.scale_bytes,                        zero_bytes_asym=cfg.zero_bytes_asym,                    ),                ),            },            "skip": {                "mse": 0.0,                "cosine": 1.0,                "max_abs": 0.0,                "estimated_extra_mb_vs_base": max(                    0.0,                    extra_mb_vs_base(                        params, "skip",                        sym=cfg.sym,                        scale_bytes=cfg.scale_bytes,                        zero_bytes_asym=cfg.zero_bytes_asym,                    ),                ),            },        },        "proxy_gain_w4g64_vs_w4": float(max(0.0, m_w4g128["mse"] - m_w4g64["mse"])),        "proxy_gain_w8_vs_w4": float(max(0.0, m_w4g128["mse"] - m_w8g128["mse"])),    }    rows.append(row)features_jsonl = out_dir / "probe_features.jsonl"metadata_json = out_dir / "probe_metadata.json"summary_json = out_dir / "probe_summary.json"with open(features_jsonl, "w", encoding="utf-8") as f:    for row in rows:        f.write(json.dumps(row, ensure_ascii=False) + "\n")metadata = {    "model_id": cfg.model_id,    "dataset_name": cfg.dataset_name,    "dataset_config": cfg.dataset_config,    "split": cfg.split,    "text_field": cfg.text_field,    "num_texts": cfg.num_texts,    "max_length": cfg.max_length,    "max_rows_per_module": cfg.max_rows_per_module,    "probe_batch_size": cfg.probe_batch_size,    "include_modules": cfg.include_modules,    "num_rows": len(rows),    "use_activation_aware_proxy": cfg.use_activation_aware_proxy,    "sym": cfg.sym,    "scale_bytes": cfg.scale_bytes,    "zero_bytes_asym": cfg.zero_bytes_asym,}metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")summary_json.write_text(json.dumps({"num_rows": len(rows)}, indent=2), encoding="utf-8")del modelif torch.cuda.is_available():    torch.cuda.empty_cache()return {    "features_jsonl": str(features_jsonl),    "metadata_json": str(metadata_json),    "summary_json": str(summary_json),    "num_rows": len(rows),}
```

EOF

cat > quant_judge/judge.py <<'EOF'
from **future** import annotations

import json
import os
from dataclasses import dataclass
from typing import Literal

import requests
from pydantic import BaseModel, Field, ValidationError

from .actions import ACTION_SPECS, BASE_ACTION, allowed_actions

@dataclass
class JudgeConfig:
features_jsonl: str
budget_mb_extra: float
heuristic_only: bool = False
judge_base_url: str | None = None
judge_model: str | None = None
judge_api_key_env: str = "JUDGE_API_KEY"
chunk_size: int = 24

class Assignment(BaseModel):
name: str
action: Literal["w4_g128", "w4_g64", "w8_g128", "skip"]
priority: float = Field(ge=0.0, le=1.0)
reason_code: str

class AssignmentPayload(BaseModel):
assignments: list[Assignment]

def load_rows(path: str) -> list[dict]:
rows = []
with open(path, "r", encoding="utf-8") as f:
for line in f:
line = line.strip()
if line:
rows.append(json.loads(line))
return rows

def heuristic_decision(row: dict) -> dict:
name = row["name"]
w4 = row["candidates"]["w4_g128"]["mse"]
w4g64 = row["candidates"]["w4_g64"]["mse"]
w8 = row["candidates"]["w8_g128"]["mse"]
kurt = row["weight_kurtosis"]
outlier = row["act_outlier_rate"]
block = row.get("block_index")

```w8_relief = (w4 - w8) / max(w4, 1e-9)g64_relief = (w4 - w4g64) / max(w4, 1e-9)score = 0.0score += min(1.0, w8_relief / 0.60) * 0.40score += min(1.0, g64_relief / 0.35) * 0.20score += min(1.0, kurt / 20.0) * 0.15score += min(1.0, outlier / 0.05) * 0.15if "lm_head" in name:    score += 0.25if row["module_type"] in ("down_proj", "o_proj", "gate_proj"):    score += 0.10if block is not None and (block <= 1 or block >= 30):    score += 0.10score = max(0.0, min(1.0, score))action = BASE_ACTIONreason = "default"if "lm_head" in name and (w8_relief > 0.55 or kurt > 25.0):    action = "skip"    reason = "lm_head_structural_skip"elif score >= 0.75 or w8_relief > 0.50:    action = "w8_g128"    reason = "high_sensitivity"elif score >= 0.42 or g64_relief > 0.18:    action = "w4_g64"    reason = "moderate_sensitivity"return {    "name": name,    "action": action,    "priority": round(score, 6),    "reason_code": reason,    "source": "heuristic",}
w8_relief = (w4 - w8) / max(w4, 1e-9)g64_relief = (w4 - w4g64) / max(w4, 1e-9)score = 0.0score += min(1.0, w8_relief / 0.60) * 0.40score += min(1.0, g64_relief / 0.35) * 0.20score += min(1.0, kurt / 20.0) * 0.15score += min(1.0, outlier / 0.05) * 0.15if "lm_head" in name:    score += 0.25if row["module_type"] in ("down_proj", "o_proj", "gate_proj"):    score += 0.10if block is not None and (block <= 1 or block >= 30):    score += 0.10score = max(0.0, min(1.0, score))action = BASE_ACTIONreason = "default"if "lm_head" in name and (w8_relief > 0.55 or kurt > 25.0):    action = "skip"    reason = "lm_head_structural_skip"elif score >= 0.75 or w8_relief > 0.50:    action = "w8_g128"    reason = "high_sensitivity"elif score >= 0.42 or g64_relief > 0.18:    action = "w4_g64"    reason = "moderate_sensitivity"return {    "name": name,    "action": action,    "priority": round(score, 6),    "reason_code": reason,    "source": "heuristic",}
```

def reduced_row(row: dict) -> dict:
return {
"name": row["name"],
"module_type": row["module_type"],
"block_index": row.get("block_index"),
"params": row["params"],
"weight_kurtosis": row["weight_kurtosis"],
"act_outlier_rate": row["act_outlier_rate"],
"w4_g128_mse": row["candidates"]["w4_g128"]["mse"],
"w4_g64_mse": row["candidates"]["w4_g64"]["mse"],
"w8_g128_mse": row["candidates"]["w8_g128"]["mse"],
"extra_mb_w4_g64": row["candidates"]["w4_g64"]["estimated_extra_mb_vs_base"],
"extra_mb_w8_g128": row["candidates"]["w8_g128"]["estimated_extra_mb_vs_base"],
"extra_mb_skip": row["candidates"]["skip"]["estimated_extra_mb_vs_base"],
}

def chunk(rows: list[dict], size: int) -> list[list[dict]]:
return [rows[i:i + size] for i in range(0, len(rows), size)]

def extract_json_object(text: str) -> str:
"""
More robust than a code-fence stripper:
extract the first balanced JSON object.
"""
start = text.find("{")
if start < 0:
raise ValueError("No JSON object found in judge response.")
depth = 0
for i in range(start, len(text)):
c = text[i]
if c == "{":
depth += 1
elif c == "}":
depth -= 1
if depth == 0:
return text[start:i + 1]
raise ValueError("Unbalanced JSON object in judge response.")

def call_openai_compatible(base_url: str, model: str, api_key: str, rows: list[dict], budget_mb_extra: float) -> list[dict]:
system = (
"You are a quantization policy judge. "
"Choose exactly one action per module from: " + ", ".join(allowed_actions()) + ". "
"Prefer w4_g128 unless the evidence shows fragility. "
"Use w4_g64 for moderate sensitivity. "
"Use w8_g128 for high sensitivity. "
"Use skip only for extreme outliers, usually lm_head. "
"Return JSON only with schema "
"{"assignments":[{"name":...,"action":...,"priority":0..1,"reason_code":...}]}"
)
user = {
"budget_mb_extra_global": budget_mb_extra,
"allowed_actions": allowed_actions(),
"rows": rows,
}

```url = base_url.rstrip("/") + "/chat/completions"headers = {"Content-Type": "application/json"}if api_key:    headers["Authorization"] = f"Bearer {api_key}"body = {    "model": model,    "temperature": 0.0,    "response_format": {"type": "json_object"},    "messages": [        {"role": "system", "content": system},        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},    ],}resp = requests.post(url, headers=headers, json=body, timeout=180)resp.raise_for_status()content = resp.json()["choices"][0]["message"]["content"]payload_text = extract_json_object(content)payload = AssignmentPayload.model_validate_json(payload_text)return [item.model_dump() for item in payload.assignments]
url = base_url.rstrip("/") + "/chat/completions"headers = {"Content-Type": "application/json"}if api_key:    headers["Authorization"] = f"Bearer {api_key}"body = {    "model": model,    "temperature": 0.0,    "response_format": {"type": "json_object"},    "messages": [        {"role": "system", "content": system},        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},    ],}resp = requests.post(url, headers=headers, json=body, timeout=180)resp.raise_for_status()content = resp.json()["choices"][0]["message"]["content"]payload_text = extract_json_object(content)payload = AssignmentPayload.model_validate_json(payload_text)return [item.model_dump() for item in payload.assignments]
```

def merge_assignments(rows: list[dict], proposed: list[dict]) -> list[dict]:
by_name = {a["name"]: a for a in proposed if isinstance(a, dict) and "name" in a}
merged = []

```for row in rows:    item = by_name.get(row["name"])    if item is None:        merged.append(heuristic_decision(row))        continue    action = item.get("action", BASE_ACTION)    if action not in ACTION_SPECS:        action = BASE_ACTION    try:        priority = float(item.get("priority", 0.5))    except Exception:        priority = 0.5    priority = max(0.0, min(1.0, priority))    merged.append({        "name": row["name"],        "action": action,        "priority": priority,        "reason_code": str(item.get("reason_code", "judge")),        "source": "judge",    })return merged
for row in rows:    item = by_name.get(row["name"])    if item is None:        merged.append(heuristic_decision(row))        continue    action = item.get("action", BASE_ACTION)    if action not in ACTION_SPECS:        action = BASE_ACTION    try:        priority = float(item.get("priority", 0.5))    except Exception:        priority = 0.5    priority = max(0.0, min(1.0, priority))    merged.append({        "name": row["name"],        "action": action,        "priority": priority,        "reason_code": str(item.get("reason_code", "judge")),        "source": "judge",    })return merged
```

def make_judge_proposals(cfg: JudgeConfig) -> dict:
rows = load_rows(cfg.features_jsonl)

```if cfg.heuristic_only:    return {"assignments": [heuristic_decision(r) for r in rows]}if not cfg.judge_base_url or not cfg.judge_model:    raise ValueError("Judge mode requires judge_base_url and judge_model.")api_key = os.environ.get(cfg.judge_api_key_env, "")assignments = []for part in chunk(rows, cfg.chunk_size):    reduced = [reduced_row(r) for r in part]    try:        proposed = call_openai_compatible(            cfg.judge_base_url,            cfg.judge_model,            api_key,            reduced,            cfg.budget_mb_extra,        )        assignments.extend(merge_assignments(part, proposed))    except (requests.RequestException, ValidationError, ValueError, KeyError, IndexError):        assignments.extend([heuristic_decision(r) for r in part])return {"assignments": assignments}
if cfg.heuristic_only:    return {"assignments": [heuristic_decision(r) for r in rows]}if not cfg.judge_base_url or not cfg.judge_model:    raise ValueError("Judge mode requires judge_base_url and judge_model.")api_key = os.environ.get(cfg.judge_api_key_env, "")assignments = []for part in chunk(rows, cfg.chunk_size):    reduced = [reduced_row(r) for r in part]    try:        proposed = call_openai_compatible(            cfg.judge_base_url,            cfg.judge_model,            api_key,            reduced,            cfg.budget_mb_extra,        )        assignments.extend(merge_assignments(part, proposed))    except (requests.RequestException, ValidationError, ValueError, KeyError, IndexError):        assignments.extend([heuristic_decision(r) for r in part])return {"assignments": assignments}
```

EOF

cat > quant_judge/compiler.py <<'EOF'
from **future** import annotations

import json
import re
from pathlib import Path

from .actions import BASE_ACTION, action_to_override

def is_structural_skip(name: str, action: str) -> bool:
"""
Structural skips are applied outside the ROI budget.
Right now only lm_head skip is treated this way.
"""
return action == "skip" and name == "lm_head"

def compile_plan(assignments: list[dict], probe_rows: list[dict], extra_budget_mb: float, min_roi: float = 1e-6) -> dict:
probe_by_name = {row["name"]: row for row in probe_rows}

```structural = []upgrades = []for item in assignments:    name = item["name"]    action = item["action"]    row = probe_by_name.get(name)    if row is None:        continue    if is_structural_skip(name, action):        structural.append(item)        continue    if action == BASE_ACTION:        continue    if action == "w8_g128":        benefit = row["candidates"]["w4_g128"]["mse"] - row["candidates"]["w8_g128"]["mse"]        cost_mb = row["candidates"]["w8_g128"]["estimated_extra_mb_vs_base"]    elif action == "w4_g64":        benefit = row["candidates"]["w4_g128"]["mse"] - row["candidates"]["w4_g64"]["mse"]        cost_mb = row["candidates"]["w4_g64"]["estimated_extra_mb_vs_base"]    elif action == "skip":        benefit = row["candidates"]["w4_g128"]["mse"]        cost_mb = row["candidates"]["skip"]["estimated_extra_mb_vs_base"]    else:        continue    if benefit <= 0 or cost_mb <= 0:        continue    roi = benefit / max(cost_mb, 1e-6)    if roi < min_roi:        continue    upgrades.append({        "name": name,        "action": action,        "benefit": float(benefit),        "cost_mb": float(cost_mb),        "roi": float(roi),        "priority": float(item.get("priority", 0.5)),        "reason_code": item.get("reason_code", ""),        "source": item.get("source", ""),    })upgrades.sort(key=lambda x: (x["roi"], x["priority"]), reverse=True)approved = []spent_mb = 0.0for up in upgrades:    if spent_mb + up["cost_mb"] > extra_budget_mb:        continue    approved.append(up)    spent_mb += up["cost_mb"]structural_by_name = {x["name"]: x for x in structural}approved_by_name = {x["name"]: x for x in approved}final_assignments = []for row in probe_rows:    name = row["name"]    if name in structural_by_name:        item = structural_by_name[name]        final_assignments.append({            "name": name,            "action": "skip",            "priority": float(item.get("priority", 1.0)),            "reason_code": item.get("reason_code", "structural_skip"),            "source": item.get("source", "compiler"),            "estimated_extra_mb_vs_base": row["candidates"]["skip"]["estimated_extra_mb_vs_base"],            "structural": True,        })    elif name in approved_by_name:        up = approved_by_name[name]        final_assignments.append({            "name": name,            "action": up["action"],            "priority": up["priority"],            "reason_code": up["reason_code"],            "source": up["source"],            "estimated_extra_mb_vs_base": up["cost_mb"],            "benefit": up["benefit"],            "roi": up["roi"],            "structural": False,        })    else:        final_assignments.append({            "name": name,            "action": BASE_ACTION,            "priority": 0.0,            "reason_code": "base_default",            "source": "compiler",            "estimated_extra_mb_vs_base": 0.0,            "structural": False,        })return {    "budget_mb_extra": float(extra_budget_mb),    "used_mb_extra": round(float(spent_mb), 6),    "structural_skips": len(structural),    "upgrades_considered": len(upgrades),    "upgrades_approved": len(approved),    "assignments": sorted(final_assignments, key=lambda x: x["name"]),}
structural = []upgrades = []for item in assignments:    name = item["name"]    action = item["action"]    row = probe_by_name.get(name)    if row is None:        continue    if is_structural_skip(name, action):        structural.append(item)        continue    if action == BASE_ACTION:        continue    if action == "w8_g128":        benefit = row["candidates"]["w4_g128"]["mse"] - row["candidates"]["w8_g128"]["mse"]        cost_mb = row["candidates"]["w8_g128"]["estimated_extra_mb_vs_base"]    elif action == "w4_g64":        benefit = row["candidates"]["w4_g128"]["mse"] - row["candidates"]["w4_g64"]["mse"]        cost_mb = row["candidates"]["w4_g64"]["estimated_extra_mb_vs_base"]    elif action == "skip":        benefit = row["candidates"]["w4_g128"]["mse"]        cost_mb = row["candidates"]["skip"]["estimated_extra_mb_vs_base"]    else:        continue    if benefit <= 0 or cost_mb <= 0:        continue    roi = benefit / max(cost_mb, 1e-6)    if roi < min_roi:        continue    upgrades.append({        "name": name,        "action": action,        "benefit": float(benefit),        "cost_mb": float(cost_mb),        "roi": float(roi),        "priority": float(item.get("priority", 0.5)),        "reason_code": item.get("reason_code", ""),        "source": item.get("source", ""),    })upgrades.sort(key=lambda x: (x["roi"], x["priority"]), reverse=True)approved = []spent_mb = 0.0for up in upgrades:    if spent_mb + up["cost_mb"] > extra_budget_mb:        continue    approved.append(up)    spent_mb += up["cost_mb"]structural_by_name = {x["name"]: x for x in structural}approved_by_name = {x["name"]: x for x in approved}final_assignments = []for row in probe_rows:    name = row["name"]    if name in structural_by_name:        item = structural_by_name[name]        final_assignments.append({            "name": name,            "action": "skip",            "priority": float(item.get("priority", 1.0)),            "reason_code": item.get("reason_code", "structural_skip"),            "source": item.get("source", "compiler"),            "estimated_extra_mb_vs_base": row["candidates"]["skip"]["estimated_extra_mb_vs_base"],            "structural": True,        })    elif name in approved_by_name:        up = approved_by_name[name]        final_assignments.append({            "name": name,            "action": up["action"],            "priority": up["priority"],            "reason_code": up["reason_code"],            "source": up["source"],            "estimated_extra_mb_vs_base": up["cost_mb"],            "benefit": up["benefit"],            "roi": up["roi"],            "structural": False,        })    else:        final_assignments.append({            "name": name,            "action": BASE_ACTION,            "priority": 0.0,            "reason_code": "base_default",            "source": "compiler",            "estimated_extra_mb_vs_base": 0.0,            "structural": False,        })return {    "budget_mb_extra": float(extra_budget_mb),    "used_mb_extra": round(float(spent_mb), 6),    "structural_skips": len(structural),    "upgrades_considered": len(upgrades),    "upgrades_approved": len(approved),    "assignments": sorted(final_assignments, key=lambda x: x["name"]),}
```

def plan_to_dynamic(plan: dict) -> dict:
dynamic = {}
for item in plan["assignments"]:
name = item["name"]
action = item["action"]
if action == BASE_ACTION:
continue
anchored = f"^{re.escape(name)}$"
if action == "skip":
dynamic[f"-:{anchored}"] = {}
else:
dynamic[f"+:{anchored}"] = action_to_override(action)
return dynamic

def save_plan_and_dynamic(plan: dict, dynamic: dict, plan_out: str, dynamic_out: str) -> None:
Path(plan_out).parent.mkdir(parents=True, exist_ok=True)
Path(dynamic_out).parent.mkdir(parents=True, exist_ok=True)
Path(plan_out).write_text(json.dumps(plan, indent=2), encoding="utf-8")
Path(dynamic_out).write_text(json.dumps(dynamic, indent=2), encoding="utf-8")
EOF

cat > quant_judge/runner.py <<'EOF'
from **future** import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

def load_texts(dataset_name: str, dataset_config: str | None, split: str, text_field: str, num_texts: int) -> list[str]:
ds = load_dataset(dataset_name, dataset_config, split=split)
texts = []
for row in ds:
text = row.get(text_field, "")
if isinstance(text, str) and text.strip():
texts.append(text.strip())
if len(texts) >= num_texts:
break
if not texts:
raise ValueError("No texts found for quantization.")
return texts

def quantize_with_dynamic(
model_id: str,
calibration_texts: list[str],
out_dir: str,
dynamic: dict | None,
bits: int = 4,
group_size: int = 128,
batch_size: int = 4,
sym: bool = True,
desc_act: bool = True,
) -> dict:
qcfg = QuantizeConfig(
bits=bits,
group_size=group_size,
sym=sym,
desc_act=desc_act,
dynamic=dynamic,
)
model = GPTQModel.load(model_id, qcfg)
model.quantize(calibration_texts, batch_size=batch_size)

```out = Path(out_dir)out.mkdir(parents=True, exist_ok=True)model.save(str(out))meta = {    "model_id": model_id,    "out_dir": str(out),    "bits": bits,    "group_size": group_size,    "batch_size": batch_size,    "dynamic_rules": 0 if dynamic is None else len(dynamic),    "sym": sym,    "desc_act": desc_act,}(out / "quant_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")del modelif torch.cuda.is_available():    torch.cuda.empty_cache()return meta
out = Path(out_dir)out.mkdir(parents=True, exist_ok=True)model.save(str(out))meta = {    "model_id": model_id,    "out_dir": str(out),    "bits": bits,    "group_size": group_size,    "batch_size": batch_size,    "dynamic_rules": 0 if dynamic is None else len(dynamic),    "sym": sym,    "desc_act": desc_act,}(out / "quant_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")del modelif torch.cuda.is_available():    torch.cuda.empty_cache()return meta
```

def heuristic_dynamic_from_flags(flags: list[str]) -> dict:
dynamic = {}
for flag in flags:
if flag == "skip_lm_head":
dynamic[r"-:^lm_head$"] = {}
elif flag == "down_proj_g64":
dynamic[r"+:.*down_proj$"] = {"bits": 4, "group_size": 64}
elif flag == "o_proj_g64":
dynamic[r"+:.*o_proj$"] = {"bits": 4, "group_size": 64}
elif flag == "edge_blocks_w8":
dynamic[r"+:^model.layers.(0|1|30|31)..*$"] = {"bits": 8, "group_size": 128}
return dynamic

def run_lm_eval(model_path: str, tasks: list[str], output_json: str, device: str = "cuda:0", batch_size: str = "auto") -> dict:
output_path = Path(output_json)
output_path.parent.mkdir(parents=True, exist_ok=True)

```cmd = [    sys.executable, "-m", "lm_eval",    "--model", "hf",    "--model_args", f"pretrained={model_path},trust_remote_code=True",    "--tasks", ",".join(tasks),    "--device", device,    "--batch_size", batch_size,    "--output_path", str(output_path.parent),]subprocess.run(cmd, check=True)for path in sorted(output_path.parent.rglob("*.json")):    try:        data = json.loads(path.read_text(encoding="utf-8"))    except Exception:        continue    if isinstance(data, dict) and "results" in data:        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")        return dataraise FileNotFoundError("Could not find lm_eval results json.")
cmd = [    sys.executable, "-m", "lm_eval",    "--model", "hf",    "--model_args", f"pretrained={model_path},trust_remote_code=True",    "--tasks", ",".join(tasks),    "--device", device,    "--batch_size", batch_size,    "--output_path", str(output_path.parent),]subprocess.run(cmd, check=True)for path in sorted(output_path.parent.rglob("*.json")):    try:        data = json.loads(path.read_text(encoding="utf-8"))    except Exception:        continue    if isinstance(data, dict) and "results" in data:        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")        return dataraise FileNotFoundError("Could not find lm_eval results json.")
```

def extract_primary_scores(eval_json: dict) -> dict:
out = {}
for task, metrics in eval_json.get("results", {}).items():
if "acc_norm,none" in metrics:
out[task] = metrics["acc_norm,none"]
elif "acc,none" in metrics:
out[task] = metrics["acc,none"]
elif "word_perplexity,none" in metrics:
out[task] = metrics["word_perplexity,none"]
return out
EOF

cat > quant_judge/pipeline.py <<'EOF'
from **future** import annotations

import argparse
import json
from pathlib import Path

from .compiler import compile_plan, plan_to_dynamic, save_plan_and_dynamic
from .judge import JudgeConfig, load_rows, make_judge_proposals
from .probe import ProbeConfig, run_probe
from .runner import (
extract_primary_scores,
heuristic_dynamic_from_flags,
load_texts,
quantize_with_dynamic,
run_lm_eval,
)

def cmd_probe(args):
cfg = ProbeConfig(
model_id=args.model_id,
out_dir=args.out_dir,
dataset_name=args.dataset_name,
dataset_config=args.dataset_config,
split=args.split,
text_field=args.text_field,
num_texts=args.num_texts,
max_length=args.max_length,
max_rows_per_module=args.max_rows_per_module,
include_modules=[x.strip() for x in args.include_modules.split(",") if x.strip()],
probe_batch_size=args.probe_batch_size,
use_activation_aware_proxy=args.use_activation_aware_proxy,
sym=args.sym,
scale_bytes=args.scale_bytes,
zero_bytes_asym=args.zero_bytes_asym,
)
print(json.dumps(run_probe(cfg), indent=2))

def cmd_plan(args):
judge_cfg = JudgeConfig(
features_jsonl=args.features_jsonl,
budget_mb_extra=args.budget_mb_extra,
heuristic_only=args.heuristic_only,
judge_base_url=args.judge_base_url,
judge_model=args.judge_model,
judge_api_key_env=args.judge_api_key_env,
chunk_size=args.chunk_size,
)
proposals = make_judge_proposals(judge_cfg)
probe_rows = load_rows(args.features_jsonl)
plan = compile_plan(proposals["assignments"], probe_rows, args.budget_mb_extra, args.min_roi)
dynamic = plan_to_dynamic(plan)
save_plan_and_dynamic(plan, dynamic, args.plan_out, args.dynamic_out)
print(json.dumps({
"plan_out": args.plan_out,
"dynamic_out": args.dynamic_out,
"used_mb_extra": plan["used_mb_extra"],
"structural_skips": plan["structural_skips"],
"upgrades_approved": plan["upgrades_approved"],
"dynamic_rules": len(dynamic),
}, indent=2))

def cmd_quantize(args):
calibration_texts = load_texts(
args.dataset_name,
args.dataset_config,
args.split,
args.text_field,
args.num_texts,
)
plan = json.loads(Path(args.plan_json).read_text(encoding="utf-8"))
dynamic = plan_to_dynamic(plan)
print(json.dumps(
quantize_with_dynamic(
args.model_id,
calibration_texts,
args.out_dir,
dynamic,
args.base_bits,
args.base_group_size,
args.batch_size,
args.sym,
args.desc_act,
),
indent=2,
))

def cmd_eval(args):
tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
result = run_lm_eval(args.model_path, tasks, args.output_json, args.device, args.batch_size)
print(json.dumps({
"output_json": args.output_json,
"scores": extract_primary_scores(result),
}, indent=2))

def cmd_suite(args):
work_dir = Path(args.work_dir)
work_dir.mkdir(parents=True, exist_ok=True)
probe_dir = work_dir / "probe"

```probe_result = run_probe(ProbeConfig(    model_id=args.model_id,    out_dir=str(probe_dir),    dataset_name=args.dataset_name,    dataset_config=args.dataset_config,    split=args.split,    text_field=args.text_field,    num_texts=args.probe_num_texts,    max_length=args.max_length,    max_rows_per_module=args.max_rows_per_module,    include_modules=[x.strip() for x in args.include_modules.split(",") if x.strip()],    probe_batch_size=args.probe_batch_size,    use_activation_aware_proxy=args.use_activation_aware_proxy,    sym=args.sym,    scale_bytes=args.scale_bytes,    zero_bytes_asym=args.zero_bytes_asym,))features_jsonl = probe_result["features_jsonl"]probe_rows = load_rows(features_jsonl)# Rule-based baseline: fixed regexes onlyheuristic_flags = [x.strip() for x in args.heuristic_include.split(",") if x.strip()]rule_based_dynamic = heuristic_dynamic_from_flags(heuristic_flags)(work_dir / "rule_based_dynamic.json").write_text(    json.dumps(rule_based_dynamic, indent=2), encoding="utf-8")# Data-driven heuristic: same feature-driven logic as the judge fallback, but labeled honestlyheuristic_proposals = make_judge_proposals(JudgeConfig(    features_jsonl=features_jsonl,    budget_mb_extra=args.budget_mb_extra,    heuristic_only=True,    chunk_size=args.chunk_size,))heuristic_plan = compile_plan(    heuristic_proposals["assignments"],    probe_rows,    args.budget_mb_extra,    args.min_roi,)data_driven_heuristic_dynamic = plan_to_dynamic(heuristic_plan)save_plan_and_dynamic(    heuristic_plan,    data_driven_heuristic_dynamic,    str(work_dir / "data_driven_heuristic_plan.json"),    str(work_dir / "data_driven_heuristic_dynamic.json"),)configs = {    "baseline_w4g128": None,    "rule_based_baseline": rule_based_dynamic,    "data_driven_heuristic": data_driven_heuristic_dynamic,}judge_used = Falseif args.judge_base_url and args.judge_model:    judge_used = True    judge_proposals = make_judge_proposals(JudgeConfig(        features_jsonl=features_jsonl,        budget_mb_extra=args.budget_mb_extra,        heuristic_only=False,        judge_base_url=args.judge_base_url,        judge_model=args.judge_model,        judge_api_key_env=args.judge_api_key_env,        chunk_size=args.chunk_size,    ))    judge_plan = compile_plan(        judge_proposals["assignments"],        probe_rows,        args.budget_mb_extra,        args.min_roi,    )    judge_dynamic = plan_to_dynamic(judge_plan)    save_plan_and_dynamic(        judge_plan,        judge_dynamic,        str(work_dir / "judge_plan.json"),        str(work_dir / "judge_dynamic.json"),    )    configs["judge_sdq"] = judge_dynamiccalibration_texts = load_texts(    args.dataset_name,    args.dataset_config,    args.split,    args.text_field,    args.quant_num_texts,)tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]leaderboard = {}for name, dynamic in configs.items():    model_dir = work_dir / "models" / name    eval_json = work_dir / "eval" / f"{name}.json"    quant_meta = quantize_with_dynamic(        args.model_id,        calibration_texts,        str(model_dir),        dynamic,        args.base_bits,        args.base_group_size,        args.batch_size,        args.sym,        args.desc_act,    )    result = run_lm_eval(        str(model_dir),        tasks,        str(eval_json),        args.device,        args.eval_batch_size,    )    leaderboard[name] = {        "scores": extract_primary_scores(result),        "quant_meta": quant_meta,    }payload = {    "probe": probe_result,    "leaderboard": leaderboard,    "judge_used": judge_used,    "artifacts": {        "rule_based_dynamic_path": str(work_dir / "rule_based_dynamic.json"),        "data_driven_heuristic_plan_path": str(work_dir / "data_driven_heuristic_plan.json"),        "data_driven_heuristic_dynamic_path": str(work_dir / "data_driven_heuristic_dynamic.json"),        "judge_plan_path": str(work_dir / "judge_plan.json") if judge_used else None,        "judge_dynamic_path": str(work_dir / "judge_dynamic.json") if judge_used else None,    },}(work_dir / "leaderboard.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")print(json.dumps(payload, indent=2))
probe_result = run_probe(ProbeConfig(    model_id=args.model_id,    out_dir=str(probe_dir),    dataset_name=args.dataset_name,    dataset_config=args.dataset_config,    split=args.split,    text_field=args.text_field,    num_texts=args.probe_num_texts,    max_length=args.max_length,    max_rows_per_module=args.max_rows_per_module,    include_modules=[x.strip() for x in args.include_modules.split(",") if x.strip()],    probe_batch_size=args.probe_batch_size,    use_activation_aware_proxy=args.use_activation_aware_proxy,    sym=args.sym,    scale_bytes=args.scale_bytes,    zero_bytes_asym=args.zero_bytes_asym,))features_jsonl = probe_result["features_jsonl"]probe_rows = load_rows(features_jsonl)# Rule-based baseline: fixed regexes onlyheuristic_flags = [x.strip() for x in args.heuristic_include.split(",") if x.strip()]rule_based_dynamic = heuristic_dynamic_from_flags(heuristic_flags)(work_dir / "rule_based_dynamic.json").write_text(    json.dumps(rule_based_dynamic, indent=2), encoding="utf-8")# Data-driven heuristic: same feature-driven logic as the judge fallback, but labeled honestlyheuristic_proposals = make_judge_proposals(JudgeConfig(    features_jsonl=features_jsonl,    budget_mb_extra=args.budget_mb_extra,    heuristic_only=True,    chunk_size=args.chunk_size,))heuristic_plan = compile_plan(    heuristic_proposals["assignments"],    probe_rows,    args.budget_mb_extra,    args.min_roi,)data_driven_heuristic_dynamic = plan_to_dynamic(heuristic_plan)save_plan_and_dynamic(    heuristic_plan,    data_driven_heuristic_dynamic,    str(work_dir / "data_driven_heuristic_plan.json"),    str(work_dir / "data_driven_heuristic_dynamic.json"),)configs = {    "baseline_w4g128": None,    "rule_based_baseline": rule_based_dynamic,    "data_driven_heuristic": data_driven_heuristic_dynamic,}judge_used = Falseif args.judge_base_url and args.judge_model:    judge_used = True    judge_proposals = make_judge_proposals(JudgeConfig(        features_jsonl=features_jsonl,        budget_mb_extra=args.budget_mb_extra,        heuristic_only=False,        judge_base_url=args.judge_base_url,        judge_model=args.judge_model,        judge_api_key_env=args.judge_api_key_env,        chunk_size=args.chunk_size,    ))    judge_plan = compile_plan(        judge_proposals["assignments"],        probe_rows,        args.budget_mb_extra,        args.min_roi,    )    judge_dynamic = plan_to_dynamic(judge_plan)    save_plan_and_dynamic(        judge_plan,        judge_dynamic,        str(work_dir / "judge_plan.json"),        str(work_dir / "judge_dynamic.json"),    )    configs["judge_sdq"] = judge_dynamiccalibration_texts = load_texts(    args.dataset_name,    args.dataset_config,    args.split,    args.text_field,    args.quant_num_texts,)tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]leaderboard = {}for name, dynamic in configs.items():    model_dir = work_dir / "models" / name    eval_json = work_dir / "eval" / f"{name}.json"    quant_meta = quantize_with_dynamic(        args.model_id,        calibration_texts,        str(model_dir),        dynamic,        args.base_bits,        args.base_group_size,        args.batch_size,        args.sym,        args.desc_act,    )    result = run_lm_eval(        str(model_dir),        tasks,        str(eval_json),        args.device,        args.eval_batch_size,    )    leaderboard[name] = {        "scores": extract_primary_scores(result),        "quant_meta": quant_meta,    }payload = {    "probe": probe_result,    "leaderboard": leaderboard,    "judge_used": judge_used,    "artifacts": {        "rule_based_dynamic_path": str(work_dir / "rule_based_dynamic.json"),        "data_driven_heuristic_plan_path": str(work_dir / "data_driven_heuristic_plan.json"),        "data_driven_heuristic_dynamic_path": str(work_dir / "data_driven_heuristic_dynamic.json"),        "judge_plan_path": str(work_dir / "judge_plan.json") if judge_used else None,        "judge_dynamic_path": str(work_dir / "judge_dynamic.json") if judge_used else None,    },}(work_dir / "leaderboard.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")print(json.dumps(payload, indent=2))
```

def build_parser():
p = argparse.ArgumentParser(prog="quant_judge")
sub = p.add_subparsers(dest="cmd", required=True)

```probe = sub.add_parser("probe")probe.add_argument("--model-id", required=True)probe.add_argument("--out-dir", required=True)probe.add_argument("--dataset-name", default="wikitext")probe.add_argument("--dataset-config", default="wikitext-2-raw-v1")probe.add_argument("--split", default="train")probe.add_argument("--text-field", default="text")probe.add_argument("--num-texts", type=int, default=64)probe.add_argument("--max-length", type=int, default=256)probe.add_argument("--max-rows-per-module", type=int, default=128)probe.add_argument("--probe-batch-size", type=int, default=8)probe.add_argument("--include-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head")probe.add_argument("--use-activation-aware-proxy", action="store_true")probe.add_argument("--sym", action="store_true", default=True)probe.add_argument("--scale-bytes", type=float, default=2.0)probe.add_argument("--zero-bytes-asym", type=float, default=0.5)probe.set_defaults(func=cmd_probe)plan = sub.add_parser("plan")plan.add_argument("--features-jsonl", required=True)plan.add_argument("--plan-out", required=True)plan.add_argument("--dynamic-out", required=True)plan.add_argument("--budget-mb-extra", type=float, default=150.0)plan.add_argument("--min-roi", type=float, default=1e-6)plan.add_argument("--heuristic-only", action="store_true")plan.add_argument("--judge-base-url", default=None)plan.add_argument("--judge-model", default=None)plan.add_argument("--judge-api-key-env", default="JUDGE_API_KEY")plan.add_argument("--chunk-size", type=int, default=24)plan.set_defaults(func=cmd_plan)quant = sub.add_parser("quantize")quant.add_argument("--model-id", required=True)quant.add_argument("--plan-json", required=True)quant.add_argument("--out-dir", required=True)quant.add_argument("--dataset-name", default="wikitext")quant.add_argument("--dataset-config", default="wikitext-2-raw-v1")quant.add_argument("--split", default="train")quant.add_argument("--text-field", default="text")quant.add_argument("--num-texts", type=int, default=128)quant.add_argument("--batch-size", type=int, default=4)quant.add_argument("--base-bits", type=int, default=4)quant.add_argument("--base-group-size", type=int, default=128)quant.add_argument("--sym", action="store_true", default=True)quant.add_argument("--desc-act", action="store_true", default=True)quant.set_defaults(func=cmd_quantize)ev = sub.add_parser("eval")ev.add_argument("--model-path", required=True)ev.add_argument("--tasks", default="arc_challenge,hellaswag")ev.add_argument("--output-json", required=True)ev.add_argument("--device", default="cuda:0")ev.add_argument("--batch-size", default="auto")ev.set_defaults(func=cmd_eval)suite = sub.add_parser("suite")suite.add_argument("--model-id", required=True)suite.add_argument("--work-dir", required=True)suite.add_argument("--dataset-name", default="wikitext")suite.add_argument("--dataset-config", default="wikitext-2-raw-v1")suite.add_argument("--split", default="train")suite.add_argument("--text-field", default="text")suite.add_argument("--probe-num-texts", type=int, default=64)suite.add_argument("--quant-num-texts", type=int, default=128)suite.add_argument("--max-length", type=int, default=256)suite.add_argument("--max-rows-per-module", type=int, default=128)suite.add_argument("--probe-batch-size", type=int, default=8)suite.add_argument("--include-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head")suite.add_argument("--use-activation-aware-proxy", action="store_true")suite.add_argument("--budget-mb-extra", type=float, default=150.0)suite.add_argument("--min-roi", type=float, default=1e-6)suite.add_argument("--heuristic-include", default="skip_lm_head,down_proj_g64")suite.add_argument("--judge-base-url", default=None)suite.add_argument("--judge-model", default=None)suite.add_argument("--judge-api-key-env", default="JUDGE_API_KEY")suite.add_argument("--chunk-size", type=int, default=24)suite.add_argument("--tasks", default="arc_challenge,hellaswag")suite.add_argument("--device", default="cuda:0")suite.add_argument("--batch-size", type=int, default=4)suite.add_argument("--eval-batch-size", default="auto")suite.add_argument("--base-bits", type=int, default=4)suite.add_argument("--base-group-size", type=int, default=128)suite.add_argument("--sym", action="store_true", default=True)suite.add_argument("--desc-act", action="store_true", default=True)suite.add_argument("--scale-bytes", type=float, default=2.0)suite.add_argument("--zero-bytes-asym", type=float, default=0.5)suite.set_defaults(func=cmd_suite)return p
probe = sub.add_parser("probe")probe.add_argument("--model-id", required=True)probe.add_argument("--out-dir", required=True)probe.add_argument("--dataset-name", default="wikitext")probe.add_argument("--dataset-config", default="wikitext-2-raw-v1")probe.add_argument("--split", default="train")probe.add_argument("--text-field", default="text")probe.add_argument("--num-texts", type=int, default=64)probe.add_argument("--max-length", type=int, default=256)probe.add_argument("--max-rows-per-module", type=int, default=128)probe.add_argument("--probe-batch-size", type=int, default=8)probe.add_argument("--include-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head")probe.add_argument("--use-activation-aware-proxy", action="store_true")probe.add_argument("--sym", action="store_true", default=True)probe.add_argument("--scale-bytes", type=float, default=2.0)probe.add_argument("--zero-bytes-asym", type=float, default=0.5)probe.set_defaults(func=cmd_probe)plan = sub.add_parser("plan")plan.add_argument("--features-jsonl", required=True)plan.add_argument("--plan-out", required=True)plan.add_argument("--dynamic-out", required=True)plan.add_argument("--budget-mb-extra", type=float, default=150.0)plan.add_argument("--min-roi", type=float, default=1e-6)plan.add_argument("--heuristic-only", action="store_true")plan.add_argument("--judge-base-url", default=None)plan.add_argument("--judge-model", default=None)plan.add_argument("--judge-api-key-env", default="JUDGE_API_KEY")plan.add_argument("--chunk-size", type=int, default=24)plan.set_defaults(func=cmd_plan)quant = sub.add_parser("quantize")quant.add_argument("--model-id", required=True)quant.add_argument("--plan-json", required=True)quant.add_argument("--out-dir", required=True)quant.add_argument("--dataset-name", default="wikitext")quant.add_argument("--dataset-config", default="wikitext-2-raw-v1")quant.add_argument("--split", default="train")quant.add_argument("--text-field", default="text")quant.add_argument("--num-texts", type=int, default=128)quant.add_argument("--batch-size", type=int, default=4)quant.add_argument("--base-bits", type=int, default=4)quant.add_argument("--base-group-size", type=int, default=128)quant.add_argument("--sym", action="store_true", default=True)quant.add_argument("--desc-act", action="store_true", default=True)quant.set_defaults(func=cmd_quantize)ev = sub.add_parser("eval")ev.add_argument("--model-path", required=True)ev.add_argument("--tasks", default="arc_challenge,hellaswag")ev.add_argument("--output-json", required=True)ev.add_argument("--device", default="cuda:0")ev.add_argument("--batch-size", default="auto")ev.set_defaults(func=cmd_eval)suite = sub.add_parser("suite")suite.add_argument("--model-id", required=True)suite.add_argument("--work-dir", required=True)suite.add_argument("--dataset-name", default="wikitext")suite.add_argument("--dataset-config", default="wikitext-2-raw-v1")suite.add_argument("--split", default="train")suite.add_argument("--text-field", default="text")suite.add_argument("--probe-num-texts", type=int, default=64)suite.add_argument("--quant-num-texts", type=int, default=128)suite.add_argument("--max-length", type=int, default=256)suite.add_argument("--max-rows-per-module", type=int, default=128)suite.add_argument("--probe-batch-size", type=int, default=8)suite.add_argument("--include-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head")suite.add_argument("--use-activation-aware-proxy", action="store_true")suite.add_argument("--budget-mb-extra", type=float, default=150.0)suite.add_argument("--min-roi", type=float, default=1e-6)suite.add_argument("--heuristic-include", default="skip_lm_head,down_proj_g64")suite.add_argument("--judge-base-url", default=None)suite.add_argument("--judge-model", default=None)suite.add_argument("--judge-api-key-env", default="JUDGE_API_KEY")suite.add_argument("--chunk-size", type=int, default=24)suite.add_argument("--tasks", default="arc_challenge,hellaswag")suite.add_argument("--device", default="cuda:0")suite.add_argument("--batch-size", type=int, default=4)suite.add_argument("--eval-batch-size", default="auto")suite.add_argument("--base-bits", type=int, default=4)suite.add_argument("--base-group-size", type=int, default=128)suite.add_argument("--sym", action="store_true", default=True)suite.add_argument("--desc-act", action="store_true", default=True)suite.add_argument("--scale-bytes", type=float, default=2.0)suite.add_argument("--zero-bytes-asym", type=float, default=0.5)suite.set_defaults(func=cmd_suite)return p
```

def main():
parser = build_parser()
args = parser.parse_args()
args.func(args)

if **name** == "**main**":
main()
EOF

```Then install and run it:```bashpython -m venv .venvsource .venv/bin/activate# install your CUDA-matching torch first# example:# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128pip install -r requirements.txt
Then install and run it:```bashpython -m venv .venvsource .venv/bin/activate# install your CUDA-matching torch first# example:# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128pip install -r requirements.txt
```

Starter suite run:

```Bashpython -m quant_judge.pipeline suite \  --model-id Qwen/Qwen3.5-4B-Base \  --work-dir ./runs/suite \  --dataset-name wikitext \  --dataset-config wikitext-2-raw-v1 \  --split train \  --text-field text \  --probe-num-texts 64 \  --quant-num-texts 128 \  --tasks arc_challenge,hellaswag
Bashpython -m quant_judge.pipeline suite \  --model-id Qwen/Qwen3.5-4B-Base \  --work-dir ./runs/suite \  --dataset-name wikitext \  --dataset-config wikitext-2-raw-v1 \  --split train \  --text-field text \  --probe-num-texts 64 \  --quant-num-texts 128 \  --tasks arc_challenge,hellaswag
```

Judge-backed suite run:

```Bashexport JUDGE_API_KEY=your_key_herepython -m quant_judge.pipeline suite \  --model-id Qwen/Qwen3.5-4B-Base \  --work-dir ./runs/suite \  --dataset-name wikitext \  --dataset-config wikitext-2-raw-v1 \  --split train \  --text-field text \  --probe-num-texts 64 \  --quant-num-texts 128 \  --tasks arc_challenge,hellaswag \  --judge-base-url http://127.0.0.1:8000/v1 \  --judge-model Qwen/Qwen2.5-14B-Instruct
Bashexport JUDGE_API_KEY=your_key_herepython -m quant_judge.pipeline suite \  --model-id Qwen/Qwen3.5-4B-Base \  --work-dir ./runs/suite \  --dataset-name wikitext \  --dataset-config wikitext-2-raw-v1 \  --split train \  --text-field text \  --probe-num-texts 64 \  --quant-num-texts 128 \  --tasks arc_challenge,hellaswag \  --judge-base-url http://127.0.0.1:8000/v1 \  --judge-model Qwen/Qwen2.5-14B-Instruct
```

If you want, next I can give you a **single self-extracting .sh installer** version so you only save one file and run it.

---

### User:
Here is a detailed critique of your SDQ Quant Judge v4 pipeline, followed by a concrete guide on how to inject **ExLlamaV3 (EXL3)** and **ExLlamaV2 (EXL2)** logic into your pipeline to drastically improve accuracy *without* breaking standard GPTQ kernel compatibility.

---

### Part 1: Critique of SDQ Quant Judge v4

Your pipeline is a great exploration of Software-Defined Quantization, but it has severe mathematical bottlenecks if the goal is SOTA accuracy:

1. **The Probe Fallacy (RTN vs. GPTQ):** 
   Your probe uses `fake_quant_groupwise_linear_weight` (Round-To-Nearest) and an AWQ-like activation proxy to measure MSE. This is mathematically mismatched with actual GPTQ. GPTQ uses the Inverse Hessian to compensate for quantization errors in subsequent weights. A layer might have terrible RTN error but near-zero GPTQ error because the Hessian allows adjacent weights to absorb the shock. Judging modules based on RTN/AWQ error leads to suboptimal budget allocation.
2. **LLMs are Bad at Knapsack Problems:** 
   Using an LLM (`Qwen2.5-14B-Instruct`) to allocate bit-widths based on heuristic scores is fundamentally flawed. Bit allocation to hit a target VRAM budget is a classic **constrained optimization / knapsack problem**. A deterministic mathematical solver will beat an LLM 100% of the time in both speed and mathematical optimality.
3. **Coarse Granularity:** 
   You apply mixed precision at the module level (e.g., `down_proj` gets `w8_g128`, `up_proj` gets `w4_g128`). Modern formats (EXL2/EXL3) allocate bits at the *group* or *column* level within the *same* tensor. 
4. **Wasted Space on the LM Head:**
   You are skipping the `lm_head` (leaving it at FP16). In a model like Llama-3-70B, the LM head is 2+ GB. EXL3 achieves massive VRAM savings by quantizing the LM head to 6-bit or 8-bit, rather than skipping it entirely.

---

### Part 2: Implementing EXL3 / EXL2 Logic within Standard GPTQ

**ExLlamaV3 (EXL3)** is based on **QTIP** (Quantization with Trellis and Incoherent Processing). True EXL3 uses a Walsh-Hadamard transform and procedural codebooks, which *require* a custom kernel. 

However, you can emulate the "magic" of EXL3 and EXL2 **without modifying the standard GPTQ inference kernel** (meaning your models will still run flawlessly in vLLM, AutoGPTQ, or GPTQModel). You do this by adopting their two main algorithms during the *quantization step*:

#### Mechanism 1: Trellis/Beam-Search Rounding (The EXL3 / QTIP Secret)
Standard GPTQ uses **greedy rounding**. It looks at one weight, rounds it to the nearest INT4 grid point, and pushes the error to the remaining weights using the Inverse Hessian. 

EXL3/QTIP uses **Trellis Coded Quantization (Viterbi Search)**. Instead of instantly rounding, it looks ahead. It tests multiple rounding paths over a block of weights and picks the sequence of quantizations that results in the lowest global Hessian-weighted error. 

You can apply this to the standard INT4/INT8 linear grid! During your GPTQ quantization loop, replace the greedy `torch.round()` with a localized Beam Search:

```
python
# CONCEPTUAL IMPLEMENTATION FOR YOUR GPTQ RUNNER
# Inside the GPTQ quantization block (iterating over columns):

def beam_search_quantize(weights, scales, zeros, H_inv, block_size=8, beam_width=4):
    """
    Replaces greedy GPTQ rounding with EXL3-style lookahead Trellis/Beam search.
    Outputs standard GPTQ compatible integers.
    """
    beams =[{"q_vals": [], "error_buffer": torch.zeros_like(weights[:, 0]), "cost": 0.0}]
    
    for i in range(block_size):
        w_current = weights[:, i]
        new_beams =[]
        
        for beam in beams:
            # Apply accumulated error from previous weights in the beam
            w_adjusted = w_current + beam["error_buffer"]
            
            # Instead of greedy rounding, consider the 2 nearest integer states
            q_float = w_adjusted / scales + zeros
            q_floor = torch.floor(q_float)
            q_ceil = torch.ceil(q_float)
            
            for q_choice in [q_floor, q_ceil]:
                q_choice = torch.clamp(q_choice, 0, 15) # Standard INT4 bounds
                w_quantized = (q_choice - zeros) * scales
                
                # Calculate quantization error
                err = w_adjusted - w_quantized
                
                # Update cost (Hessian weighted MSE)
                # H_diag is the diagonal of the inverse Hessian
                step_cost = (err ** 2) / H_inv[i, i]
                
                # Propagate error to remaining weights in the block
                # err * (H_inv_row / H_inv_diag)
                push_error = err.unsqueeze(1) @ (H_inv[i, i+1:block_size] / H_inv[i, i]).unsqueeze(0)
                
                new_beams.append({
                    "q_vals": beam["q_vals"] + [q_choice],
                    "error_buffer": beam["error_buffer"][:, 1:] + push_error,
                    "cost": beam["cost"] + step_cost.sum().item()
                })
                
        # Prune to top-K (Beam Width)
        new_beams.sort(key=lambda x: x["cost"])
        beams = new_beams[:beam_width]

    # Return the optimal sequence of standard INT4 values from the best path
    return torch.stack(beams[0]["q_vals"], dim=1), beams[0]["error_buffer"]
```

*Why this works:* This drops your quantization error (KL divergence) significantly at low bitrates without touching the inference engine. Your pipeline outputs a perfectly standard `qweight` tensor, but the values inside it have been globally optimized rather than locally guessed.

#### Mechanism 2: Hessian-Trace Lagrangian Bit Allocation (The EXL2 Secret)
Instead of using your probe to measure RTN error and asking an LLM to guess bitrates, you should measure the **Hessian Trace** of each layer (or group) during calibration. 

The formula EXL2 uses to allocate bits minimizes $\sum \text{Trace}(H) \cdot \text{Error}(b)$ subject to $\sum b \le \text{Budget}$. This is solved deterministically via Binary Search:

```
python
import math

def exl2_allocate_bits(layers, target_bpw):
    """
    layers: list of dicts with 'params' (count) and 'hessian_trace' (float)
    target_bpw: e.g. 4.5
    """
    total_params = sum(L['params'] for L in layers)
    target_bits = total_params * target_bpw
    
    # Available standard GPTQ bit choices (can be mixed layer-by-layer)
    bit_options =[4, 8] # You can add 3, 5, 6 if your kernel supports it
    
    # Define an error function for a bit width (empirically, error scales exponentially inversely with bits)
    def expected_error(trace, bits):
        return trace * (2.0 ** (-2.0 * bits))

    # Binary search for the Lagrangian multiplier (lambda)
    lambda_low, lambda_high = 0.0, 1e9
    
    for _ in range(60):
        lambda_mid = (lambda_low + lambda_high) / 2
        
        current_bits = 0
        for L in layers:
            best_bits = None
            best_cost = float('inf')
            
            # Pick the bit width that minimizes: Error + lambda * Bits
            for b in bit_options:
                cost = expected_error(L['hessian_trace'], b) + lambda_mid * b * L['params']
                if cost < best_cost:
                    best_cost = cost
                    best_bits = b
            
            L['assigned_bits'] = best_bits
            current_bits += best_bits * L['params']
            
        if current_bits > target_bits:
            lambda_low = lambda_mid # Budget exceeded, increase penalty
        else:
            lambda_high = lambda_mid
            
    return layers
```

### Summary of How to Refactor Your Architecture:

1. **Delete the LLM Judge / Probe step:** RTN fake-quant is misleading, and the LLM is mathematically inferior to a solver.
2. **One-Pass Pipeline:** Run calibration data through the model. Hook into each layer to capture its inputs. Calculate the `Hessian = X.T @ X`. 
3. **Save the Hessian Traces:** Store the trace of each layer's Hessian.
4. **Determine the Plan deterministically:** Feed the traces into the EXL2 Lagrangian solver above to figure out exactly which layers get 4-bit, 8-bit, or `lm_head` 6-bit.
5. **Apply Trellis Quantization:** When you actually call your GPTQ quantizer, inject the Trellis/Beam Search algorithm over the standard INT grid to finalize the `qweights`.

By making these changes, your "Software-Defined" pipeline evolves from a heuristic guesser into a mathematically rigorous engine that generates **SOTA-accuracy standard GPTQ checkpoints**, directly capturing the benefits of EXL2/EXL3's research while remaining 100% compatible with existing vLLM/AutoGPTQ infrastructure.

---

### ChatGPT:
Your critique is mostly right, with one big caveat:

**You can borrow some EXL2/EXL3 ideas for a standard GPTQ pipeline, but you cannot get “full EXL2/EXL3” while staying in plain FORMAT.GPTQ.** EXL2’s main win is variable bitrate **within** a linear layer, and ExLlamaV2 explicitly says EXL2 can mix quantization levels inside each linear layer to hit arbitrary average bitrates between 2 and 8 bpw. EXL3 is a QTIP-style trellis/codebook format, and ExLlamaV3’s own docs say EXL3 is a variant of QTIP with its own packing and kernels. Standard GPTQ runtimes instead expect a conventional GPTQ checkpoint with a fixed `weight_bits`/packing scheme per quantized tensor path. [GitHub+2GitHub+2](https://github.com/turboderp-org/exllamav2?utm_source=chatgpt.com)

So I’d split your proposal into two categories:

**Worth doing inside standard GPTQ**

- Replace the LLM judge with a deterministic allocator.
- Use Hessian-derived layer importance instead of RTN/AWQ-proxy MSE as the primary allocation signal.
- Improve the rounding/search step during quantization, as long as the emitted checkpoint is still ordinary GPTQ `qweight/scales/zeros` with supported bitwidths. [arXiv+2arXiv+2](https://arxiv.org/pdf/2210.17323?utm_source=chatgpt.com)

**Not portable to plain GPTQ without changing format/kernel**

- EXL2-style variable bitrates **inside the same tensor**
- EXL3/QTIP procedural codebooks, incoherence processing, and trellis-packed representations
- 6-bit EXL2 `lm_head` in a “standard GPTQ everywhere” deployment path, because current vLLM GPTQ support is documented around 2/3/4/8-bit GPTQ weights, not 6-bit GPTQ. EXL2 supports 6-bit, but that is an EXL2 feature, not a generic GPTQ interchange guarantee. [GitHub+2GitHub+2](https://github.com/turboderp-org/exllamav2?utm_source=chatgpt.com)

I also agree with your criticism of the current SDQ v4 pipeline: **if the goal is best accuracy under a hard memory budget, an LLM judge is the wrong optimizer.** ExLlamaV2’s own converter is already a strong precedent here: it does a measurement pass first, then chooses quantization parameters to minimize overall error while meeting a target average bitrate. That is fundamentally a measurement + solver workflow, not an instruction-following workflow. [GitHub](https://github.com/turboderp-org/exllamav2/blob/master/doc/convert.md)

## What I would actually change

### 1) Remove the LLM from the decision loop

Keep the “SDQ shell” if you like the architecture, but replace the judge with a deterministic allocator.

Use:

- one calibration pass
- per-layer or per-module Hessian statistics
- a constrained solver that assigns from a small supported action set

For a **standard GPTQ-compatible** target, the practical action set is:

- `4-bit`
- `8-bit`
- `skip`
- optionally `2-bit` or `3-bit` only if your intended runtime really supports those paths end-to-end

That is because vLLM’s GPTQ path documents support for 2/3/4/8-bit GPTQ weights, and GPTQModel’s `dynamic` override surface is per-module, not per-column-inside-a-tensor. [vLLM+2GitHub+2](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/gptq/)

### 2) Replace the probe with Hessian-derived importance

Your “probe fallacy” point is right. GPTQ is explicitly a second-order method using approximate Hessian information, so RTN-style proxy error is not the same objective. A better standard-GPTQ-compatible controller is:

- collect input activations `X` for each target layer/module
- estimate a Hessian proxy such as `H = XᵀX / N`
- use `trace(H)`, `diag(H)`, or a small blockwise approximation as the layer/group importance score
- allocate bits with a solver under a fixed memory budget

That is much closer to GPTQ’s actual error model than RTN/AWQ-proxy MSE. [arXiv+2arXiv+2](https://arxiv.org/pdf/2210.17323?utm_source=chatgpt.com)

### 3) Use a deterministic allocator, not an LLM

Your Lagrangian / knapsack framing is the correct direction.

A practical version for **plain GPTQ compatibility** is:

- estimate each layer’s expected distortion at `b ∈ {4, 8}` from Hessian-weighted quantization error
- solve for the best per-layer assignment under a target extra-memory budget
- compile the result into `QuantizeConfig.dynamic`

That maps cleanly onto GPTQModel/vLLM’s documented per-module dynamic overrides and skip rules. [GitHub+1](https://github.com/ModelCloud/GPTQModel)

### 4) Keep trellis/beam search only if the output stays ordinary GPTQ

This part is subtle.

Your proposed beam-search rounding can be useful **if**:

- you still quantize onto the ordinary scalar INT grid
- you still emit standard GPTQ-compatible `qweight/scales/zeros`
- you do not introduce EXL3/QTIP-specific packing, transforms, or codebooks

In that case, you are not “making GPTQ into EXL3”; you are just replacing a greedy local rounding step with a better search step before writing a normal GPTQ checkpoint. That can be compatible in principle. But the moment you rely on EXL3/QTIP’s codebook/trellis format or EXL2’s mixed-bit encoding inside one tensor, you have left plain GPTQ interchange behind. [GitHub+3arXiv+3GitHub+3](https://arxiv.org/abs/2406.11235?utm_source=chatgpt.com)

### 5) Do not plan around a 6-bit lm_head if the target is “standard GPTQ everywhere”

This is one place where your proposal overreaches.

ExLlamaV2’s converter explicitly exposes special `lm_head` bit choices including 6 and 8, and even notes that 6/8 are the useful options there. But that is **EXL2 tooling**, not the standard GPTQ contract you can assume across GPTQModel + vLLM GPTQ. vLLM’s GPTQ path documents 2/3/4/8-bit GPTQ support, not 6-bit GPTQ. So if you want cross-runtime standard GPTQ compatibility, plan around:

- `lm_head` skip
- or `lm_head` 8-bit GPTQ
- not 6-bit EXL2-style `lm_head` as a general deployment assumption. [GitHub+1](https://github.com/turboderp-org/exllamav2/blob/master/doc/convert.md)

### 6) Be careful with the “100% compatible with existing vLLM/AutoGPTQ infrastructure” claim

That needs qualification.

GPTQModel does support dynamic per-module mixed quantization and vLLM integration, and vLLM documents GPTQModel-based 4-bit and 8-bit GPTQ models. But there are kernel-path caveats: for example, vLLM’s `gptq_marlin` code explicitly rejects 8-bit weights in the modular-kernel path shown in the docs. GPTQModel also documents dynamic quantization as per-module override, not EXL2-style intra-layer mixed bits. So “compatible” is true for standard GPTQ-style outputs in supported configurations, but not as a blanket statement for every EXL2/EXL3 idea. [vLLM+2vLLM+2](https://docs.vllm.ai/en/latest/features/quantization/gptqmodel/)

## The refactor I’d recommend

If your real goal is **best possible quality while staying in standard GPTQ**, I would turn v4 into this:

**Stage A — Measurement**

- Load the FP model with `transformers`
- Collect calibration activations
- Compute per-layer Hessian proxy stats
- Optionally still keep your activation-aware fake-quant probe as a secondary diagnostic, but not as the allocator’s main signal

**Stage B — Allocation**

- Solve a deterministic budgeted assignment for each module:

`w4_g128`

`w8_g128`

`skip`
- `w4_g128`
- `w8_g128`
- `skip`
- Treat `lm_head` as either:

structural `skip`, or

budgetable `8-bit`, depending deployment target
- structural `skip`, or
- budgetable `8-bit`, depending deployment target

**Stage C — Quantization**

- Quantize with GPTQ/GPTQModel
- If you modify the rounding step, keep the emitted checkpoint in ordinary GPTQ format

**Stage D — Evaluation**

- Compare:

baseline GPTQ

deterministic Hessian allocator

optional improved-rounding GPTQ
- baseline GPTQ
- deterministic Hessian allocator
- optional improved-rounding GPTQ
- Evaluate with `lm-eval`, which supports transformers-loaded GPTQModel/AutoGPTQ models and also supports vLLM. [GitHub](https://github.com/EleutherAI/lm-evaluation-harness?utm_source=chatgpt.com)

## Bottom line

Your critique is strongest on these points:

- RTN/AWQ-proxy MSE is the wrong main objective for GPTQ-style allocation
- LLM judging is inferior to a solver for bitrate allocation
- EXL2/EXL3 quality comes partly from **format-level** advantages, not just better heuristics

Where I’d push back is the claim that you can simply “capture EXL2/EXL3 while remaining 100% standard GPTQ-compatible.” You can capture **some** of the value:

- better measurement
- better bit allocation
- possibly better search/rounding

But you cannot capture the full value of:

- EXL2’s intra-tensor mixed bitrates
- EXL3/QTIP trellis/codebook packing

without leaving plain GPTQ format and plain GPTQ kernels behind. [GitHub+2GitHub+2](https://github.com/turboderp-org/exllamav2?utm_source=chatgpt.com)

The best next step is to rip out the LLM judge entirely and replace it with a Hessian-trace allocator. After that, if you want, I’ll write you a **v5 refactor** that does exactly this:

- Hessian measurement pass
- deterministic per-layer 4/8/skip allocator
- standard GPTQModel export
- optional beam-search rounding hook
- honest benchmark suite against your current v4 baseline

---

### User:
You are completely right on all fronts. This is a brilliant clarification of the boundary between **format-level innovations** (EXL2 mixed intra-tensor bits, EXL3 QTIP codebooks) and **algorithmic optimization** (better measurement and allocation) that can still output a standard `qweight/zeros/scales` GPTQ checkpoint.

You cannot force vLLM to read 6-bit EXL2 `lm_head`s through the `gptq_marlin` backend, but you *can* emit a mathematically superior 4/8/16-bit GPTQ checkpoint by replacing RTN + LLM judging with Hessian Traces + Lagrangian constraints. 

As requested, here is the **SDQ Quant v5** refactor in a single self-extracting `.sh` script. It removes the LLM API entirely, implements a fast Hessian-trace calibration pass, and uses a deterministic Binary Search / Lagrangian multiplier solver to optimally fill your VRAM budget.

### Run this to generate the v5 project:

```
bash
mkdir -p sdq_quant_v5/quant_v5
cd sdq_quant_v5

cat > README.md <<'EOF'
# SDQ Quant v5 (Hessian-Trace + Lagrangian Allocator)

A deterministic, mathematically rigorous Software-Defined Quantization pipeline producing 100% standard GPTQ checkpoints for vLLM / GPTQModel.

## Core Upgrades in v5
1. **Hessian-Trace Measurement:** Captures true second-order sensitivity (`trace(X^T X)`) rather than misleading RTN/AWQ proxies.
2. **Deterministic Solver:** Replaces the LLM Judge with a Lagrangian Multiplier (Knapsack) solver to mathematically minimize model distortion under a strict megabyte budget.
3. **Standard GPTQ Formats:** Outputs strict 4-bit, 8-bit, or skipped (FP16) tensor paths ensuring seamless deployment in vLLM.
EOF

cat > requirements.txt <<'EOF'
transformers
datasets
accelerate
torch
gptqmodel
lm-eval
tqdm
EOF

cat > quant_v5/__init__.py <<'EOF'
"""SDQ Quant v5: Hessian-Trace Allocator"""
EOF

cat > quant_v5/actions.py <<'EOF'
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass(frozen=True)
class ActionSpec:
    name: str
    bits: int
    group_size: Optional[int]
    skip: bool

ACTION_SPECS: Dict[str, ActionSpec] = {
    "w4_g128": ActionSpec("w4_g128", bits=4, group_size=128, skip=False),
    "w8_g128": ActionSpec("w8_g128", bits=8, group_size=128, skip=False),
    "skip": ActionSpec("skip", bits=16, group_size=None, skip=True),
}

BASE_ACTION = "w4_g128"

def bytes_per_param(action: str) -> float:
    if action == "skip":
        return 2.0  # FP16
    spec = ACTION_SPECS[action]
    base = spec.bits / 8.0
    # Add GPTQ overhead: scale (16-bit) and zero (4-bit or 8-bit depending on sym, let's average ~2.5 bytes per group)
    overhead = 2.5 / spec.group_size
    return base + overhead

def action_to_override(action: str) -> dict:
    if action == "skip":
        return {}
    spec = ACTION_SPECS[action]
    return {"bits": spec.bits, "group_size": spec.group_size}
EOF

cat > quant_v5/measurement.py <<'EOF'
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from .actions import bytes_per_param, ACTION_SPECS, BASE_ACTION

def load_calibration_texts(ds_name: str, ds_config: str, split: str, text_field: str, num_texts: int):
    ds = load_dataset(ds_name, ds_config, split=split)
    texts =[]
    for row in ds:
        if row.get(text_field) and row[text_field].strip():
            texts.append(row[text_field].strip())
        if len(texts) >= num_texts: break
    return texts

def measure_hessian_traces(args):
    """
    Computes Trace(H) ≈ sum(mean(X^2, dim=0)) across calibration samples.
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.eval()

    texts = load_calibration_texts(args.dataset_name, args.dataset_config, args.split, args.text_field, args.num_texts)
    
    # Identify targets
    include_mods =[x.strip() for x in args.include_modules.split(',')]
    targets =[(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and any(i in n for i in include_mods)]
    
    # We will accumulate the sum of squared inputs per column
    hessian_traces = {name: 0.0 for name, _ in targets}
    row_counts = {name: 0 for name, _ in targets}
    params_count = {name: m.weight.numel() for name, m in targets}
    
    handles =[]
    def make_hook(name):
        def hook(_, inp, __):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1]) # [batch*seq, in_features]
            # Trace of X^T X is just the sum of squared elements
            hessian_traces[name] += (x ** 2).sum().item()
            row_counts[name] += x.shape[0]
        return hook

    for name, module in targets:
        handles.append(module.register_forward_hook(make_hook(name)))

    # Forward passes
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), args.batch_size), desc="Hessian Proxy Measurement"):
            batch = tokenizer(texts[i:i+args.batch_size], return_tensors="pt", truncation=True, max_length=args.max_length, padding=True)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            model(**batch)

    for h in handles: h.remove()
    
    # Normalize by row count and save
    features =[]
    for name, _ in targets:
        avg_trace = hessian_traces[name] / max(1, row_counts[name])
        features.append({
            "name": name,
            "params": params_count[name],
            "hessian_trace": avg_trace
        })
        
    out_file = out_dir / "hessian_features.json"
    out_file.write_text(json.dumps(features, indent=2))
    print(f"Saved {len(features)} layer traces to {out_file}")
    
    return features
EOF

cat > quant_v5/allocator.py <<'EOF'
import json
import re
from pathlib import Path
from .actions import bytes_per_param, ACTION_SPECS, BASE_ACTION

def solve_allocation(features_path: str, budget_mb_extra: float):
    """
    Lagrangian multiplier solver to minimize quantization error subject to a memory budget.
    """
    features = json.loads(Path(features_path).read_text())
    
    # Calculate base size (everything in BASE_ACTION)
    base_bytes_per_param = bytes_per_param(BASE_ACTION)
    total_base_mb = sum(f["params"] * base_bytes_per_param for f in features) / (1024**2)
    target_mb = total_base_mb + budget_mb_extra
    
    actions = list(ACTION_SPECS.keys())
    
    # Expected Error Model: Error proportional to Trace * 2^(-2 * bits)
    def expected_error(trace: float, action: str) -> float:
        if action == "skip": return 0.0 # FP16 = 0 quantization error
        bits = ACTION_SPECS[action].bits
        return trace * (2.0 ** (-2.0 * bits))

    def action_cost_mb(params: int, action: str) -> float:
        return (params * bytes_per_param(action)) / (1024**2)

    # Binary search for Lambda (penalty for size)
    lambda_low, lambda_high = 0.0, 1e12
    best_plan = None
    
    for _ in range(80):
        lambda_mid = (lambda_low + lambda_high) / 2.0
        current_mb = 0.0
        plan =[]
        
        for f in features:
            best_action = None
            best_cost = float('inf')
            
            for act in actions:
                err = expected_error(f["hessian_trace"], act)
                size_mb = action_cost_mb(f["params"], act)
                
                # Objective: Minimize (Error + Lambda * Size)
                obj = err + lambda_mid * size_mb
                if obj < best_cost:
                    best_cost = obj
                    best_action = act
            
            plan.append({
                "name": f["name"],
                "action": best_action,
                "params": f["params"],
                "hessian_trace": f["hessian_trace"],
                "size_mb": action_cost_mb(f["params"], best_action)
            })
            current_mb += action_cost_mb(f["params"], best_action)
            
        best_plan = plan
        if current_mb > target_mb:
            lambda_low = lambda_mid # Over budget, penalize size more
        else:
            lambda_high = lambda_mid # Under budget, penalize size less

    used_extra_mb = sum(p["size_mb"] for p in best_plan) - total_base_mb
    
    # Convert to GPTQModel Dynamic config dict
    dynamic = {}
    from .actions import action_to_override
    for p in best_plan:
        if p["action"] == BASE_ACTION: continue
        regex = f"^{re.escape(p['name'])}$"
        if p["action"] == "skip":
            dynamic[f"-:{regex}"] = {}
        else:
            dynamic[f"+:{regex}"] = action_to_override(p["action"])
            
    return {"plan": best_plan, "dynamic": dynamic, "used_extra_mb": used_extra_mb}
EOF

cat > quant_v5/runner.py <<'EOF'
import json
import subprocess
import sys
from pathlib import Path
import torch
from gptqmodel import GPTQModel, QuantizeConfig
from .measurement import load_calibration_texts

def run_quantize(args, dynamic_config):
    texts = load_calibration_texts(args.dataset_name, args.dataset_config, args.split, args.text_field, args.quant_num_texts)
    
    qcfg = QuantizeConfig(
        bits=4, group_size=128, sym=True, desc_act=True, dynamic=dynamic_config
    )
    
    print(f"Loading Model for Quantization with {len(dynamic_config) if dynamic_config else 0} dynamic rules...")
    model = GPTQModel.load(args.model_id, qcfg)
    model.quantize(texts, batch_size=args.quant_batch_size)
    
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def run_eval(model_path: str, tasks: str, out_file: str):
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    cmd =[
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},trust_remote_code=True",
        "--tasks", tasks,
        "--device", "cuda:0",
        "--batch_size", "auto",
        "--output_path", out_file
    ]
    subprocess.run(cmd, check=True)
EOF

cat > quant_v5/pipeline.py <<'EOF'
import argparse
import json
from pathlib import Path
from .measurement import measure_hessian_traces
from .allocator import solve_allocation
from .runner import run_quantize, run_eval

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    
    suite = sub.add_parser("suite")
    suite.add_argument("--model-id", required=True)
    suite.add_argument("--work-dir", required=True)
    suite.add_argument("--budget-mb-extra", type=float, default=250.0)
    suite.add_argument("--include-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head")
    suite.add_argument("--dataset-name", default="wikitext")
    suite.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    suite.add_argument("--split", default="train")
    suite.add_argument("--text-field", default="text")
    suite.add_argument("--num-texts", type=int, default=128)
    suite.add_argument("--max-length", type=int, default=2048)
    suite.add_argument("--batch-size", type=int, default=4)
    suite.add_argument("--quant-num-texts", type=int, default=128)
    suite.add_argument("--quant-batch-size", type=int, default=4)
    suite.add_argument("--tasks", default="arc_challenge,hellaswag")
    args = p.parse_args()
    
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n--- STAGE 1: Hessian Trace Measurement ---")
    args.out_dir = str(work_dir / "probe")
    features = measure_hessian_traces(args)
    
    print("\n--- STAGE 2: Lagrangian Allocation ---")
    alloc = solve_allocation(str(work_dir / "probe" / "hessian_features.json"), args.budget_mb_extra)
    (work_dir / "sdq_dynamic.json").write_text(json.dumps(alloc["dynamic"], indent=2))
    print(f"Allocated! Used {alloc['used_extra_mb']:.2f} MB extra.")
    
    print("\n--- STAGE 3A: Quantize Baseline (Pure 4-bit) ---")
    args.out_dir = str(work_dir / "models" / "baseline")
    run_quantize(args, dynamic_config=None)
    
    print("\n--- STAGE 3B: Quantize SDQ v5 (Hessian-Lagrangian) ---")
    args.out_dir = str(work_dir / "models" / "sdq_v5")
    run_quantize(args, dynamic_config=alloc["dynamic"])
    
    print("\n--- STAGE 4: Evaluation ---")
    run_eval(str(work_dir / "models" / "baseline"), args.tasks, str(work_dir / "eval" / "baseline.json"))
    run_eval(str(work_dir / "models" / "sdq_v5"), args.tasks, str(work_dir / "eval" / "sdq_v5.json"))
    
    print("\n--- Pipeline Complete! See eval/ folder for results. ---")

if __name__ == "__main__":
    main()
EOF
```

### Setup & Run

```
bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m quant_v5.pipeline suite \
  --model-id Qwen/Qwen3.5-4B-Base \
  --work-dir ./runs/v5_suite \
  --budget-mb-extra 200 \
  --tasks arc_challenge,hellaswag
```

### What this solves perfectly:
1. **The Allocator is Instant & Optimal:** No LLM generation delays. The Binary Search Lagrangian solver loops 80 times locally in microseconds to find the mathematically ideal bit-distribution curve. 
2. **True Objective Function:** It uses $X^T X$ (Hessian Diagonal/Trace) directly from captured activations. This precisely models how errors will propagate *after* GPTQ's Hessian-inverse compensation, which RTN fake-quant completely misses.
3. **vLLM Compatibility Guarantees:** Because the output `dynamic_config` targets pure module names and assigns standard `w8_g128`, `w4_g128`, or structural fp16 skips, the resulting checkpoint will load directly in vLLM without needing a customized branch.

### A Note on the "Beam Search" Rounding Step

To implement the Trellis/Beam-Search rounding (*improving* the quantization loop itself without changing the standard GPTQ output structure), you would need to monkeypatch `gptqmodel.quantization.gptq.GPTQ.quantize`. 

Because GPTQModel updates its internal `gptq.py` aggressively (to support Marlin, Triton, etc.), a dirty monkeypatch in a standalone script will break frequently. The absolute best way to capture this final piece of EXL3 quality is to open a PR against `GPTQModel` adding a `use_beam_search=True` flag directly to their native `QuantizeConfig`, replacing their greedy `q = torch.round(w / scale)` loop with the Lookahead Trellis logic we discussed.

---

### ChatGPT:
This v5 is **much better** than v4. The big architectural shift is correct: drop the LLM judge, keep the output in standard GPTQ, and let a deterministic allocator decide per-module `4-bit / 8-bit / skip`. That matches what GPTQModel and vLLM actually support today: GPTQModel exposes per-module `Dynamic` overrides, and vLLM’s GPTQ path supports standard GPTQ weights at 2/3/4/8 bits, not arbitrary EXL2-style mixed intra-tensor bitrates. [GitHub+2vLLM+2](https://github.com/ModelCloud/GPTQModel)

The part I would tone down is the claim that this “precisely models” GPTQ. GPTQ is a second-order method that uses approximate Hessian information during quantization, but your `trace(XᵀX)` scalar is still only a **coarse importance proxy**. It throws away the off-diagonal structure and the blockwise inverse-Hessian behavior that GPTQ actually exploits. So this is a **better allocator signal than RTN+LLM**, not a faithful simulation of GPTQ error propagation. [arXiv](https://arxiv.org/pdf/2210.17323?utm_source=chatgpt.com)

What is solid:

- Per-module dynamic overrides are the right emission target. GPTQModel explicitly supports mixed per-module quantization and module exclusion, and vLLM integrates that path. [GitHub+2vLLM+2](https://github.com/ModelCloud/GPTQModel)
- Restricting the action set to `w4_g128`, `w8_g128`, and `skip` is realistic for standard GPTQ deployment, because vLLM’s GPTQ config documents support for 2/3/4/8-bit weights only. [vLLM](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/gptq/)
- Replacing the judge with a deterministic solver is the right direction. Recent work on Hessian-proxy-based bit allocation is consistent with that design choice, even though your exact objective is still heuristic. [arXiv+1](https://arxiv.org/abs/2506.05664?utm_source=chatgpt.com)

The must-fix issues in the script:

### 1) model.device with device_map="auto" is fragile

In `measurement.py`, you move tokenized batches with `v.to(model.device)`. With `device_map="auto"`, that is not a reliable contract. For a 1×3090 workflow, load the FP model explicitly onto GPU 0 instead of sharding logic, or derive the input device from the embedding layer / first parameter. Otherwise you can get subtle placement errors.

Use something like:

```Pythondevice = next(model.parameters()).devicebatch = {k: v.to(device) for k, v in batch.items()}
Pythondevice = next(model.parameters()).devicebatch = {k: v.to(device) for k, v in batch.items()}
```

or load with a single-device map.

### 2) run_eval() is using the CLI in a brittle way

The lm-eval docs currently recommend `simple_evaluate()` as the primary programmatic entry point. Your CLI call with `--output_path` pointing at a file path is more brittle than calling the API directly and writing JSON yourself. [GitHub](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/python-api.md)

I would replace `runner.run_eval` with:

```Pythonimport jsonimport lm_evaldef run_eval(model_path: str, tasks: str, out_file: str):    results = lm_eval.simple_evaluate(        model="hf",        model_args=f"pretrained={model_path},trust_remote_code=True",        tasks=[t.strip() for t in tasks.split(",") if t.strip()],        batch_size="auto",        device="cuda:0",    )    Path(out_file).parent.mkdir(parents=True, exist_ok=True)    Path(out_file).write_text(json.dumps(results, indent=2), encoding="utf-8")    return results
Pythonimport jsonimport lm_evaldef run_eval(model_path: str, tasks: str, out_file: str):    results = lm_eval.simple_evaluate(        model="hf",        model_args=f"pretrained={model_path},trust_remote_code=True",        tasks=[t.strip() for t in tasks.split(",") if t.strip()],        batch_size="auto",        device="cuda:0",    )    Path(out_file).parent.mkdir(parents=True, exist_ok=True)    Path(out_file).write_text(json.dumps(results, indent=2), encoding="utf-8")    return results
```

That is closer to the harness’ recommended usage. [GitHub](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/python-api.md)

### 3) Your cost model is still approximate

`bytes_per_param()` uses a fixed `2.5 / group_size` overhead. That is acceptable as an estimator, but it is **not** guaranteed to match the exact storage behavior of the backend/kernel path you will deploy. So the allocator’s memory budget is an approximation, not a hard kernel-true guarantee.

I would label it explicitly in the README as:

- “planner memory estimate”
- not “exact runtime VRAM model”

### 4) skip is still global, not structural

You fixed the v4 `lm_head` problem conceptually, but this v5 allocator no longer treats `lm_head` specially. Because `skip` is included in the solver action set for every module, the optimizer can choose FP16 for arbitrary modules if the scalar objective likes it. That is mathematically legal, but operationally messy.

I’d add a policy layer:

- allow `skip` only for `lm_head`
- allow `w8_g128` / `w4_g128` for everything else

That keeps the plan sane and easier to deploy.

### 5) The error model is heuristic, not proven-optimal

This line:

```Pythonreturn trace * (2.0 ** (-2.0 * bits))
Pythonreturn trace * (2.0 ** (-2.0 * bits))
```

is a useful surrogate, but it is still a hand-designed error law. It is not something GPTQModel or GPTQ itself guarantees. So the solver is “deterministic and principled,” but not “mathematically exact.”

I would rename:

- `expected_error` → `proxy_error`
- `solve_allocation` → `solve_proxy_allocation`

That makes the claims honest.

### 6) The pipeline is not actually “self-extracting”

The content is a shell recipe that writes files, not a single self-extracting script artifact. That is fine, but the label is a bit off.

## My verdict

**Keep this direction.** It is the right architecture for “best possible standard GPTQ under a budget.”

I would call it:

- **architecturally correct**
- **substantially better than v4**
- **not yet mathematically faithful enough to claim GPTQ-optimal allocation**
- **close to production once the device handling, eval path, and skip policy are tightened**

The clean v5.1 patch would do four things:

1. force single-device measurement cleanly on GPU 0,
2. switch eval to `lm_eval.simple_evaluate()`,
3. restrict `skip` to `lm_head`,
4. rename the allocator/error model to “proxy” terminology so the claims match reality. [GitHub+2vLLM+2](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/python-api.md)

Paste the next revision and I’ll review it line by line.

---

### User:
You are absolutely right to pull me up on the terminology. Calling a coarse `trace(XᵀX)` a "precise model" of GPTQ is an overstatement. It is a principled **Hessian proxy** that correlates strongly with quantization sensitivity, but it ignores the inverse-Hessian block updates that are the actual mathematical heartbeat of GPTQ. 

### To answer your question about AutoRound:
**Yes. AutoRound is exactly the holy grail for your Stage C.**
AutoRound (and its predecessor SignRound) fundamentally solves the exact "Trellis/Beam-Search" rounding problem I mentioned earlier, but it does so via **SignSGD optimization**. 
* **How it works:** Instead of greedy rounding or lookahead trees, it formulates rounding as a continuous machine-learning problem. It learns a tiny perturbation tensor (using Sign Gradient Descent) for the weights and scales to minimize the block-wise MSE of the activations.
* **Does it change the kernel?** **No!** This is its superpower. AutoRound exports a 100% standard `qweight, qzeros, scales` checkpoint. It runs flawlessly in vLLM's `gptq_marlin` backend, AutoGPTQ, and GPTQModel. 
* **Is it worth it?** **Absolutely.** It achieves SOTA accuracy (beating plain GPTQ and AWQ), especially at lower bitrates, while maintaining total inference-side compatibility. If you want the absolute highest accuracy inside a plain GPTQ boundary, you use your Hessian-trace allocator to pick the bit-widths, and then you use AutoRound as the engine to actually crunch the integers.

---

Here is the pristine **SDQ Quant v5.1** patch. 

It implements all your structural feedback: 
1. Forces `cuda:0` cleanly.
2. Uses the native `lm_eval.simple_evaluate()` API.
3. Hard-restricts `skip` to the `lm_head`.
4. Accurately labels the memory cost as an estimator and the error as a proxy.

### Run this to generate the v5.1 project:

```
bash
mkdir -p sdq_quant_v5_1/quant_v5_1
cd sdq_quant_v5_1

cat > README.md <<'EOF'
# SDQ Quant v5.1 (Hessian-Proxy + Lagrangian Allocator)

A deterministic Software-Defined Quantization pipeline producing standard GPTQ checkpoints for vLLM / GPTQModel.

## Design
1. **Measurement:** Computes `trace(X^T X)` as a proxy for second-order sensitivity.
2. **Allocation:** Uses a Lagrangian solver to minimize a heuristic proxy error under a planner memory budget.
   - `lm_head` is restricted to `skip` (FP16) or 8-bit.
   - All other linear layers are restricted to 4-bit or 8-bit.
3. **Quantization:** Standard GPTQ export compatible with vLLM (`gptq_marlin`).
4. **Evaluation:** Direct programmatic usage of `lm-eval`.

*Note: The memory budget is a planner estimate, not an exact runtime VRAM guarantee. The error model is a proxy, not a proven-optimal GPTQ simulation.*
EOF

cat > requirements.txt <<'EOF'
transformers
datasets
accelerate
torch
gptqmodel
lm-eval
tqdm
EOF

cat > quant_v5_1/__init__.py <<'EOF'
"""SDQ Quant v5.1: Hessian-Proxy Allocator"""
EOF

cat > quant_v5_1/actions.py <<'EOF'
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass(frozen=True)
class ActionSpec:
    name: str
    bits: int
    group_size: Optional[int]
    skip: bool

ACTION_SPECS: Dict[str, ActionSpec] = {
    "w4_g128": ActionSpec("w4_g128", bits=4, group_size=128, skip=False),
    "w8_g128": ActionSpec("w8_g128", bits=8, group_size=128, skip=False),
    "skip": ActionSpec("skip", bits=16, group_size=None, skip=True),
}

BASE_ACTION = "w4_g128"

def bytes_per_param(action: str) -> float:
    """Planner memory estimate. Not an exact runtime VRAM model."""
    if action == "skip":
        return 2.0  # FP16
    spec = ACTION_SPECS[action]
    base = spec.bits / 8.0
    # ~2.5 bytes overhead per group for scales/zeros
    overhead = 2.5 / spec.group_size
    return base + overhead

def action_to_override(action: str) -> dict:
    if action == "skip":
        return {}
    spec = ACTION_SPECS[action]
    return {"bits": spec.bits, "group_size": spec.group_size}
EOF

cat > quant_v5_1/measurement.py <<'EOF'
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def load_calibration_texts(ds_name: str, ds_config: str, split: str, text_field: str, num_texts: int):
    ds = load_dataset(ds_name, ds_config, split=split)
    texts =[]
    for row in ds:
        if row.get(text_field) and row[text_field].strip():
            texts.append(row[text_field].strip())
        if len(texts) >= num_texts: break
    return texts

def measure_hessian_traces(args):
    """
    Computes Trace(H) ≈ sum(mean(X^2, dim=0)) across calibration samples.
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    texts = load_calibration_texts(args.dataset_name, args.dataset_config, args.split, args.text_field, args.num_texts)
    
    include_mods =[x.strip() for x in args.include_modules.split(',')]
    targets =[(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and any(i in n for i in include_mods)]
    
    hessian_traces = {name: 0.0 for name, _ in targets}
    row_counts = {name: 0 for name, _ in targets}
    params_count = {name: m.weight.numel() for name, m in targets}
    
    handles =[]
    def make_hook(name):
        def hook(_, inp, __):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            hessian_traces[name] += (x ** 2).sum().item()
            row_counts[name] += x.shape[0]
        return hook

    for name, module in targets:
        handles.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), args.batch_size), desc="Hessian Proxy Measurement"):
            batch = tokenizer(texts[i:i+args.batch_size], return_tensors="pt", truncation=True, max_length=args.max_length, padding=True)
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)

    for h in handles: h.remove()
    
    features =[]
    for name, _ in targets:
        avg_trace = hessian_traces[name] / max(1, row_counts[name])
        features.append({
            "name": name,
            "params": params_count[name],
            "hessian_trace": avg_trace
        })
        
    out_file = out_dir / "hessian_features.json"
    out_file.write_text(json.dumps(features, indent=2))
    print(f"Saved {len(features)} layer traces to {out_file}")
    
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    return features
EOF

cat > quant_v5_1/allocator.py <<'EOF'
import json
import re
from pathlib import Path
from .actions import bytes_per_param, ACTION_SPECS, BASE_ACTION

def get_valid_actions(module_name: str) -> list[str]:
    """Policy layer: restricts skips strictly to structural elements."""
    if "lm_head" in module_name:
        return["skip", "w8_g128"]
    return ["w4_g128", "w8_g128"]

def solve_proxy_allocation(features_path: str, budget_mb_extra: float):
    """
    Lagrangian multiplier solver to minimize heuristic proxy error.
    """
    features = json.loads(Path(features_path).read_text())
    
    base_bytes_per_param = bytes_per_param(BASE_ACTION)
    total_base_mb = sum(f["params"] * base_bytes_per_param for f in features) / (1024**2)
    target_mb = total_base_mb + budget_mb_extra
    
    def proxy_error(trace: float, action: str) -> float:
        if action == "skip": return 0.0
        bits = ACTION_SPECS[action].bits
        return trace * (2.0 ** (-2.0 * bits))

    def action_cost_mb(params: int, action: str) -> float:
        return (params * bytes_per_param(action)) / (1024**2)

    lambda_low, lambda_high = 0.0, 1e12
    best_plan = None
    
    for _ in range(80):
        lambda_mid = (lambda_low + lambda_high) / 2.0
        current_mb = 0.0
        plan =[]
        
        for f in features:
            best_action = None
            best_cost = float('inf')
            
            valid_actions = get_valid_actions(f["name"])
            
            for act in valid_actions:
                err = proxy_error(f["hessian_trace"], act)
                size_mb = action_cost_mb(f["params"], act)
                
                obj = err + lambda_mid * size_mb
                if obj < best_cost:
                    best_cost = obj
                    best_action = act
            
            plan.append({
                "name": f["name"],
                "action": best_action,
                "params": f["params"],
                "hessian_trace": f["hessian_trace"],
                "size_mb": action_cost_mb(f["params"], best_action)
            })
            current_mb += action_cost_mb(f["params"], best_action)
            
        best_plan = plan
        if current_mb > target_mb:
            lambda_low = lambda_mid 
        else:
            lambda_high = lambda_mid 

    used_extra_mb = sum(p["size_mb"] for p in best_plan) - total_base_mb
    
    dynamic = {}
    from .actions import action_to_override
    for p in best_plan:
        if p["action"] == BASE_ACTION: continue
        regex = f"^{re.escape(p['name'])}$"
        if p["action"] == "skip":
            dynamic[f"-:{regex}"] = {}
        else:
            dynamic[f"+:{regex}"] = action_to_override(p["action"])
            
    return {"plan": best_plan, "dynamic": dynamic, "used_extra_mb": used_extra_mb}
EOF

cat > quant_v5_1/runner.py <<'EOF'
import json
from pathlib import Path
import torch
import lm_eval
from gptqmodel import GPTQModel, QuantizeConfig
from .measurement import load_calibration_texts

def run_quantize(args, dynamic_config):
    texts = load_calibration_texts(args.dataset_name, args.dataset_config, args.split, args.text_field, args.quant_num_texts)
    
    qcfg = QuantizeConfig(
        bits=4, group_size=128, sym=True, desc_act=True, dynamic=dynamic_config
    )
    
    print(f"Loading Model for Quantization with {len(dynamic_config) if dynamic_config else 0} dynamic rules...")
    model = GPTQModel.load(args.model_id, qcfg)
    model.quantize(texts, batch_size=args.quant_batch_size)
    
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def run_eval(model_path: str, tasks: str, out_file: str):
    print(f"Running evaluation on {model_path}...")
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    
    task_list =[t.strip() for t in tasks.split(",") if t.strip()]
    
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},trust_remote_code=True",
        tasks=task_list,
        batch_size="auto",
        device="cuda:0"
    )
    
    serializable_results = {
        "results": results.get("results", {}),
        "versions": results.get("versions", {})
    }
    
    Path(out_file).write_text(json.dumps(serializable_results, indent=2), encoding="utf-8")
    return serializable_results
EOF

cat > quant_v5_1/pipeline.py <<'EOF'
import argparse
import json
from pathlib import Path
from .measurement import measure_hessian_traces
from .allocator import solve_proxy_allocation
from .runner import run_quantize, run_eval

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    
    suite = sub.add_parser("suite")
    suite.add_argument("--model-id", required=True)
    suite.add_argument("--work-dir", required=True)
    suite.add_argument("--budget-mb-extra", type=float, default=250.0)
    suite.add_argument("--include-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head")
    suite.add_argument("--dataset-name", default="wikitext")
    suite.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    suite.add_argument("--split", default="train")
    suite.add_argument("--text-field", default="text")
    suite.add_argument("--num-texts", type=int, default=128)
    suite.add_argument("--max-length", type=int, default=2048)
    suite.add_argument("--batch-size", type=int, default=4)
    suite.add_argument("--quant-num-texts", type=int, default=128)
    suite.add_argument("--quant-batch-size", type=int, default=4)
    suite.add_argument("--tasks", default="arc_challenge,hellaswag")
    args = p.parse_args()
    
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n--- STAGE 1: Hessian Proxy Measurement ---")
    args.out_dir = str(work_dir / "probe")
    features = measure_hessian_traces(args)
    
    print("\n--- STAGE 2: Lagrangian Proxy Allocation ---")
    alloc = solve_proxy_allocation(str(work_dir / "probe" / "hessian_features.json"), args.budget_mb_extra)
    (work_dir / "sdq_dynamic.json").write_text(json.dumps(alloc["dynamic"], indent=2))
    print(f"Allocated! Estimated extra VRAM footprint: {alloc['used_extra_mb']:.2f} MB.")
    
    print("\n--- STAGE 3A: Quantize Baseline (Pure 4-bit) ---")
    args.out_dir = str(work_dir / "models" / "baseline")
    run_quantize(args, dynamic_config=None)
    
    print("\n--- STAGE 3B: Quantize SDQ v5.1 ---")
    args.out_dir = str(work_dir / "models" / "sdq_v5_1")
    run_quantize(args, dynamic_config=alloc["dynamic"])
    
    print("\n--- STAGE 4: Evaluation ---")
    run_eval(str(work_dir / "models" / "baseline"), args.tasks, str(work_dir / "eval" / "baseline.json"))
    run_eval(str(work_dir / "models" / "sdq_v5_1"), args.tasks, str(work_dir / "eval" / "sdq_v5_1.json"))
    
    print("\n--- Pipeline Complete! See eval/ folder for results. ---")

if __name__ == "__main__":
    main()
EOF
```

---

### ChatGPT:
This v5.1 is the first version I’d actually build on. The big moves are right:

- dropping the LLM judge in favor of a deterministic allocator,
- staying inside standard GPTQ-compatible actions,
- switching eval to `lm_eval.simple_evaluate()`,
- and forcing a single-device measurement path on `cuda:0`.

That direction matches the current constraints of the stack: GPTQModel is meant to emit standard GPTQ checkpoints with per-module dynamic overrides, and vLLM’s GPTQ path only documents support for 2/3/4/8-bit GPTQ weights, not EXL2-style arbitrary intra-tensor bitrates. [GitHub+2vLLM+2](https://github.com/ModelCloud/GPTQModel?utm_source=chatgpt.com)

On AutoRound: yes, that is the strongest Stage C upgrade if your goal is “better quantization math, same deployment boundary.” AutoRound’s own docs say it can export `auto_gptq` format, supports 2/3/4/8-bit schemes there, and its “best” recipe is explicitly accuracy-first though slower than the standard recipe. That makes it a very good candidate for replacing plain greedy GPTQ rounding while still targeting a standard GPTQ-style artifact. [GitHub+1](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md?utm_source=chatgpt.com)

There is one important logic bug left in v5.1: `lm_head` is restricted to `["skip", "w8_g128"]`, but your baseline memory target is still computed from `BASE_ACTION = "w4_g128"` for **all** modules. That means the solver is forced to spend extra budget on `lm_head` even when the requested extra budget is `0`, because `lm_head` is no longer allowed to stay at the base action. So the policy restriction and the budget baseline are inconsistent. The fix is to either allow `lm_head` to stay at `w4_g128`, or move `lm_head` into a structural-policy path outside the optimizer. Without that, the allocator is biased from the start. This follows directly from your code structure and from the fact that GPTQ runtimes only expose discrete bit choices, so your baseline/action set has to be internally consistent. [vLLM+1](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/gptq/?utm_source=chatgpt.com)

I’d patch `get_valid_actions()` like this:

```Pythondef get_valid_actions(module_name: str) -> list[str]:    if "lm_head" in module_name:        return ["w4_g128", "w8_g128", "skip"]    return ["w4_g128", "w8_g128"]
Pythondef get_valid_actions(module_name: str) -> list[str]:    if "lm_head" in module_name:        return ["w4_g128", "w8_g128", "skip"]    return ["w4_g128", "w8_g128"]
```

Then, if you still want a “never leave lm_head at 4-bit” policy, enforce that as a separate post-pass or structural rule rather than inside the base knapsack.

Second, the README claim about “seamless deployment in vLLM (`gptq_marlin`)" is a little too strong. vLLM does document GPTQModel-based 4-bit and 8-bit GPTQ models, and the GPTQ layer docs enforce 2/3/4/8-bit support, but that is not the same thing as a blanket guarantee that every mixed 4/8/skip recipe will go through every backend path identically. I’d reword that to “targets standard GPTQ-compatible 4/8/skip module paths intended for GPTQModel/vLLM deployment.” [vLLM+1](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/gptq/?utm_source=chatgpt.com)

Third, your `trace(XᵀX)` allocator is a good **proxy allocator**, not a GPTQ simulator. I’d keep the new names you introduced—“Hessian proxy” and “proxy allocation”—because that is honest. GPTQ itself is second-order and uses blockwise inverse-Hessian-style compensation; your scalar trace objective is still a lossy summary. It is better than RTN+LLM, but it is not exact GPTQ error prediction. [vLLM+1](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/gptq/?utm_source=chatgpt.com)

My recommendation:

- keep v5.1 as the control-plane architecture,
- fix the `lm_head` baseline inconsistency,
- soften the deployment wording,
- then make Stage C pluggable:

`backend = gptq_plain`

`backend = autoround_auto_gptq`
- `backend = gptq_plain`
- `backend = autoround_auto_gptq`

That gives you a clean experiment matrix:

- baseline GPTQ
- Hessian-proxy allocator + plain GPTQ
- Hessian-proxy allocator + AutoRound exported as `auto_gptq`

That is the comparison that will tell you whether the allocator is helping, whether AutoRound is helping, or both. AutoRound’s official docs are already aligned with that export story. [GitHub](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md?utm_source=chatgpt.com)

So overall: **v5.1 is good, but not done**. The `lm_head` action-set bug is the main blocker. After that, I’d move straight to an AutoRound-backed v5.2.

---

### User:
You are absolutely right. The `lm_head` constraints conflicting with the baseline knapsack budget is a classic mathematical footgun. By forcing `lm_head` out of the base `w4_g128` action without accounting for it in the base VRAM calculation, the solver was mathematically doomed at a `0` extra budget.

Also, introducing **AutoRound** is the perfect capstone. It cleanly separates the two domains:
1. **The Allocation Domain:** Handled by our Hessian-proxy Lagrangian solver (deciding the `bits`/`skip` map).
2. **The Rounding Domain:** Handled by either standard GPTQ (greedy) or AutoRound (SignSGD), both exporting standard, vLLM-compatible `auto_gptq` artifacts.

Here is the pristine **SDQ Quant v5.2**. 

It implements the `lm_head` constraint fix, updates the README wording to accurately reflect vLLM compatibility realities, and introduces a pluggable Stage C that tests a full 3-way matrix:
* `baseline_gptq` (GPTQModel, pure 4-bit, greedy)
* `sdq_gptq` (GPTQModel, Hessian-proxy 4/8/skip, greedy)
* `sdq_autoround` (AutoRound, Hessian-proxy 4/8/skip, SignSGD rounding exported to `auto_gptq`)

### Run this to generate the v5.2 project:

```
bash
mkdir -p sdq_quant_v5_2/quant_v5_2
cd sdq_quant_v5_2

cat > README.md <<'EOF'
# SDQ Quant v5.2 (Hessian-Proxy + AutoRound)

A deterministic Software-Defined Quantization pipeline targeting standard GPTQ-compatible 4/8/skip module paths intended for GPTQModel/vLLM deployment.

## Design
1. **Measurement:** Computes `trace(X^T X)` on `cuda:0` as a proxy for second-order sensitivity.
2. **Allocation:** Uses a Lagrangian solver to minimize a heuristic proxy error under a planner memory budget.
   - `lm_head` can select `w4_g128`, `w8_g128`, or `skip`.
   - All other linear layers are restricted to `w4_g128` or `w8_g128`.
3. **Quantization:** Pluggable Stage C.
   - **GPTQModel:** Fast, greedy rounding.
   - **AutoRound:** SOTA SignSGD continuous rounding, exported natively to standard `auto_gptq` format.
4. **Evaluation:** Direct programmatic usage of `lm-eval`.

*Note: The memory budget is a planner estimate, not an exact runtime VRAM guarantee. The error model is a Hessian proxy, not a proven-optimal GPTQ blockwise inverse-Hessian simulation.*
EOF

cat > requirements.txt <<'EOF'
transformers
datasets
accelerate
torch
gptqmodel
auto-round
lm-eval
tqdm
EOF

cat > quant_v5_2/__init__.py <<'EOF'
"""SDQ Quant v5.2: Hessian-Proxy Allocator + AutoRound"""
EOF

cat > quant_v5_2/actions.py <<'EOF'
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass(frozen=True)
class ActionSpec:
    name: str
    bits: int
    group_size: Optional[int]
    skip: bool

ACTION_SPECS: Dict[str, ActionSpec] = {
    "w4_g128": ActionSpec("w4_g128", bits=4, group_size=128, skip=False),
    "w8_g128": ActionSpec("w8_g128", bits=8, group_size=128, skip=False),
    "skip": ActionSpec("skip", bits=16, group_size=None, skip=True),
}

BASE_ACTION = "w4_g128"

def bytes_per_param(action: str) -> float:
    """Planner memory estimate. Not an exact runtime VRAM model."""
    if action == "skip":
        return 2.0  # FP16
    spec = ACTION_SPECS[action]
    base = spec.bits / 8.0
    overhead = 2.5 / spec.group_size
    return base + overhead

def action_to_override(action: str) -> dict:
    if action == "skip":
        return {}
    spec = ACTION_SPECS[action]
    return {"bits": spec.bits, "group_size": spec.group_size}
EOF

cat > quant_v5_2/measurement.py <<'EOF'
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def load_calibration_texts(ds_name: str, ds_config: str, split: str, text_field: str, num_texts: int):
    ds = load_dataset(ds_name, ds_config, split=split)
    texts =[]
    for row in ds:
        if row.get(text_field) and row[text_field].strip():
            texts.append(row[text_field].strip())
        if len(texts) >= num_texts: break
    return texts

def measure_hessian_traces(args):
    """
    Computes Trace(H) ≈ sum(mean(X^2, dim=0)) cleanly on cuda:0.
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    texts = load_calibration_texts(args.dataset_name, args.dataset_config, args.split, args.text_field, args.num_texts)
    
    include_mods = [x.strip() for x in args.include_modules.split(',')]
    targets =[(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and any(i in n for i in include_mods)]
    
    hessian_traces = {name: 0.0 for name, _ in targets}
    row_counts = {name: 0 for name, _ in targets}
    params_count = {name: m.weight.numel() for name, m in targets}
    
    handles =[]
    def make_hook(name):
        def hook(_, inp, __):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            hessian_traces[name] += (x ** 2).sum().item()
            row_counts[name] += x.shape[0]
        return hook

    for name, module in targets:
        handles.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), args.batch_size), desc="Hessian Proxy Measurement"):
            batch = tokenizer(texts[i:i+args.batch_size], return_tensors="pt", truncation=True, max_length=args.max_length, padding=True)
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)

    for h in handles: h.remove()
    
    features =[]
    for name, _ in targets:
        avg_trace = hessian_traces[name] / max(1, row_counts[name])
        features.append({
            "name": name,
            "params": params_count[name],
            "hessian_trace": avg_trace
        })
        
    out_file = out_dir / "hessian_features.json"
    out_file.write_text(json.dumps(features, indent=2))
    print(f"Saved {len(features)} layer traces to {out_file}")
    
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    return features
EOF

cat > quant_v5_2/allocator.py <<'EOF'
import json
import re
from pathlib import Path
from .actions import bytes_per_param, ACTION_SPECS, BASE_ACTION

def get_valid_actions(module_name: str) -> list[str]:
    """Policy layer: restricts structural FP16 skips strictly to lm_head."""
    if "lm_head" in module_name:
        return["w4_g128", "w8_g128", "skip"]
    return["w4_g128", "w8_g128"]

def solve_proxy_allocation(features_path: str, budget_mb_extra: float):
    """
    Lagrangian multiplier solver to minimize heuristic proxy error.
    """
    features = json.loads(Path(features_path).read_text())
    
    base_bytes_per_param = bytes_per_param(BASE_ACTION)
    total_base_mb = sum(f["params"] * base_bytes_per_param for f in features) / (1024**2)
    target_mb = total_base_mb + budget_mb_extra
    
    def proxy_error(trace: float, action: str) -> float:
        if action == "skip": return 0.0
        bits = ACTION_SPECS[action].bits
        return trace * (2.0 ** (-2.0 * bits))

    def action_cost_mb(params: int, action: str) -> float:
        return (params * bytes_per_param(action)) / (1024**2)

    lambda_low, lambda_high = 0.0, 1e12
    best_plan = None
    
    for _ in range(80):
        lambda_mid = (lambda_low + lambda_high) / 2.0
        current_mb = 0.0
        plan =[]
        
        for f in features:
            best_action = None
            best_cost = float('inf')
            
            valid_actions = get_valid_actions(f["name"])
            
            for act in valid_actions:
                err = proxy_error(f["hessian_trace"], act)
                size_mb = action_cost_mb(f["params"], act)
                
                obj = err + lambda_mid * size_mb
                if obj < best_cost:
                    best_cost = obj
                    best_action = act
            
            plan.append({
                "name": f["name"],
                "action": best_action,
                "params": f["params"],
                "hessian_trace": f["hessian_trace"],
                "size_mb": action_cost_mb(f["params"], best_action)
            })
            current_mb += action_cost_mb(f["params"], best_action)
            
        best_plan = plan
        if current_mb > target_mb:
            lambda_low = lambda_mid 
        else:
            lambda_high = lambda_mid 

    used_extra_mb = sum(p["size_mb"] for p in best_plan) - total_base_mb
    
    dynamic = {}
    from .actions import action_to_override
    for p in best_plan:
        if p["action"] == BASE_ACTION: continue
        regex = f"^{re.escape(p['name'])}$"
        if p["action"] == "skip":
            dynamic[f"-:{regex}"] = {}
        else:
            dynamic[f"+:{regex}"] = action_to_override(p["action"])
            
    return {"plan": best_plan, "dynamic": dynamic, "used_extra_mb": used_extra_mb}
EOF

cat > quant_v5_2/runner.py <<'EOF'
import json
from pathlib import Path
import torch
import lm_eval
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound
from .measurement import load_calibration_texts

def run_quantize_gptqmodel(args, dynamic_config):
    """Standard greedy GPTQ rounding via GPTQModel."""
    texts = load_calibration_texts(args.dataset_name, args.dataset_config, args.split, args.text_field, args.quant_num_texts)
    
    qcfg = QuantizeConfig(
        bits=4, group_size=128, sym=True, desc_act=True, dynamic=dynamic_config
    )
    
    print(f"Loading Model for GPTQModel Quantization (greedy rounding)...")
    model = GPTQModel.load(args.model_id, qcfg)
    model.quantize(texts, batch_size=args.quant_batch_size)
    
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def run_quantize_autoround(args, plan):
    """SOTA SignSGD continuous rounding via AutoRound, exporting to standard auto_gptq format."""
    texts = load_calibration_texts(args.dataset_name, args.dataset_config, args.split, args.text_field, args.quant_num_texts)
    
    # Map SDQ allocation plan to AutoRound's layer_config syntax
    layer_config = {}
    if plan is not None:
        for p in plan:
            if p["action"] == "skip":
                layer_config[p["name"]] = {"bits": 16}
            elif p["action"] != "w4_g128":
                from .actions import ACTION_SPECS
                spec = ACTION_SPECS[p["action"]]
                layer_config[p["name"]] = {"bits": spec.bits, "group_size": spec.group_size}

    print(f"Loading Model for AutoRound Quantization (SignSGD rounding)...")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    
    autoround = AutoRound(
        model,
        tokenizer,
        bits=4,
        group_size=128,
        sym=True,
        dataset=texts,
        seqlen=args.max_length,
        batch_size=args.quant_batch_size,
        layer_config=layer_config,
        iters=args.autoround_iters
    )
    
    print("Executing AutoRound... (This will take significantly longer than standard GPTQ)")
    autoround.quantize()
    
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    autoround.save_quantized(str(out), format="auto_gptq", inplace=False)
    
    del model
    del autoround
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def run_eval(model_path: str, tasks: str, out_file: str):
    print(f"Running programmatic lm-eval on {model_path}...")
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    
    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},trust_remote_code=True",
        tasks=task_list,
        batch_size="auto",
        device="cuda:0"
    )
    
    serializable_results = {
        "results": results.get("results", {}),
        "versions": results.get("versions", {})
    }
    
    Path(out_file).write_text(json.dumps(serializable_results, indent=2), encoding="utf-8")
    return serializable_results
EOF

cat > quant_v5_2/pipeline.py <<'EOF'
import argparse
import json
from pathlib import Path
from .measurement import measure_hessian_traces
from .allocator import solve_proxy_allocation
from .runner import run_quantize_gptqmodel, run_quantize_autoround, run_eval

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    
    suite = sub.add_parser("suite")
    suite.add_argument("--model-id", required=True)
    suite.add_argument("--work-dir", required=True)
    suite.add_argument("--budget-mb-extra", type=float, default=250.0)
    suite.add_argument("--include-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head")
    suite.add_argument("--dataset-name", default="wikitext")
    suite.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    suite.add_argument("--split", default="train")
    suite.add_argument("--text-field", default="text")
    suite.add_argument("--num-texts", type=int, default=128)
    suite.add_argument("--max-length", type=int, default=2048)
    suite.add_argument("--batch-size", type=int, default=4)
    suite.add_argument("--quant-num-texts", type=int, default=128)
    suite.add_argument("--quant-batch-size", type=int, default=4)
    suite.add_argument("--autoround-iters", type=int, default=200, help="Iterations for AutoRound SignSGD (default 200)")
    suite.add_argument("--tasks", default="arc_challenge,hellaswag")
    args = p.parse_args()
    
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n--- STAGE 1: Hessian Proxy Measurement ---")
    args.out_dir = str(work_dir / "probe")
    features = measure_hessian_traces(args)
    
    print("\n--- STAGE 2: Lagrangian Proxy Allocation ---")
    alloc = solve_proxy_allocation(str(work_dir / "probe" / "hessian_features.json"), args.budget_mb_extra)
    (work_dir / "sdq_plan.json").write_text(json.dumps(alloc["plan"], indent=2))
    (work_dir / "sdq_dynamic.json").write_text(json.dumps(alloc["dynamic"], indent=2))
    print(f"Allocated! Estimated extra VRAM footprint: {alloc['used_extra_mb']:.2f} MB.")
    
    print("\n--- STAGE 3A: Quantize Baseline (Pure 4-bit, GPTQModel) ---")
    args.out_dir = str(work_dir / "models" / "baseline_gptq")
    run_quantize_gptqmodel(args, dynamic_config=None)
    
    print("\n--- STAGE 3B: Quantize SDQ (Proxy Allocation, GPTQModel) ---")
    args.out_dir = str(work_dir / "models" / "sdq_gptq")
    run_quantize_gptqmodel(args, dynamic_config=alloc["dynamic"])
    
    print("\n--- STAGE 3C: Quantize SDQ + AutoRound (Proxy Allocation, SignSGD AutoRound) ---")
    args.out_dir = str(work_dir / "models" / "sdq_autoround")
    run_quantize_autoround(args, plan=alloc["plan"])
    
    print("\n--- STAGE 4: Evaluation Matrix ---")
    run_eval(str(work_dir / "models" / "baseline_gptq"), args.tasks, str(work_dir / "eval" / "baseline_gptq.json"))
    run_eval(str(work_dir / "models" / "sdq_gptq"), args.tasks, str(work_dir / "eval" / "sdq_gptq.json"))
    run_eval(str(work_dir / "models" / "sdq_autoround"), args.tasks, str(work_dir / "eval" / "sdq_autoround.json"))
    
    print("\n--- Pipeline Complete! See eval/ folder for full matrix results. ---")

if __name__ == "__main__":
    main()
EOF
```

### Running the Suite Matrix
You can now directly test if the allocator logic is winning, if AutoRound's continuous rounding is winning, or (most likely) if combining both nets you SOTA accuracy under a strict boundary footprint.

```
bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the 3-way matrix. Note: AutoRound is extremely heavy computationally.
# Consider dropping --autoround-iters to 50 for a faster integration test, 
# though 200 is recommended for SOTA fidelity.

python -m quant_v5_2.pipeline suite \
  --model-id Qwen/Qwen3.5-4B-Base \
  --work-dir ./runs/v5_2_suite \
  --budget-mb-extra 200 \
  --autoround-iters 50 \
  --tasks arc_challenge,hellaswag
```

---

### ChatGPT:
v5.2 is **the right shape**. The `lm_head` baseline bug is fixed, the README wording is much cleaner, and splitting the experiment into:

- `baseline_gptq`
- `sdq_gptq`
- `sdq_autoround`

is exactly the right matrix.

I’d patch **4 things** before spending hours on long runs.

### 1) AutoRound mixed-bit auto_gptq is the main risk

This is the biggest one.

AutoRound’s current docs explicitly say:

- `auto_gptq` supports `[2,3,4,8]` bits,
- but **Auto-GPTQ only supports a limited set of mixed-bit configurations**, and
- **if you’re unsure, use the auto_round format** instead. [GitHub+1](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md?utm_source=chatgpt.com)

So your `sdq_autoround` arm is conceptually valid, but the claim that it will cleanly export every arbitrary per-layer 4/8/skip recipe to `auto_gptq` is too strong. That export path exists, but mixed-bit support is constrained. [GitHub+1](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md?utm_source=chatgpt.com)

What I’d do:

- keep `sdq_autoround` as an experiment,
- but treat `format="auto_gptq"` as **best effort**,
- and keep a fallback to `format="auto_round"` if export or downstream loading gets weird. [GitHub+1](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md?utm_source=chatgpt.com)

### 2) Exclude QKV from mixed-bit in the AutoRound arm

AutoRound’s docs explicitly warn that **vLLM and SGLang fuse MoE and QKV layers**, so it is recommended **not** to assign different bit widths to those layers. Your current allocation set includes `q_proj`, `k_proj`, and `v_proj`, which is fine for measurement, but risky for the AutoRound mixed-bit export path. [GitHub](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md?utm_source=chatgpt.com)

For `sdq_autoround`, I would:

- either remove `q_proj,k_proj,v_proj` from the allocatable set,
- or force them to move together as one tied policy.

A clean rule is:

- AutoRound mixed bits only on `o_proj`, `down_proj`, `up_proj`, `gate_proj`, `lm_head`
- leave QKV uniform.

### 3) Use the API AutoRound recommends

The docs’ recommended Python flow is:

```Pythonfrom auto_round import AutoRoundar = AutoRound(model_name_or_path, scheme="W4A16")ar.quantize_and_save(output_dir, format="auto_gptq")
Pythonfrom auto_round import AutoRoundar = AutoRound(model_name_or_path, scheme="W4A16")ar.quantize_and_save(output_dir, format="auto_gptq")
```

not the lower-level `quantize()` then `save_quantized()` sequence you wrote. [GitHub](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md?utm_source=chatgpt.com)

So I’d rewrite that arm to use `quantize_and_save()` unless you have a specific reason not to.

### 4) Keep the vLLM compatibility claim narrow

Your new wording is much better, but I’d still keep it precise:

- vLLM’s GPTQ path documents support for **2/3/4/8-bit** GPTQ weights. [vLLM+1](https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/quantization/gptq/?utm_source=chatgpt.com)
- vLLM’s GPTQModel docs explicitly describe creating **4-bit or 8-bit** GPTQ quantized models with GPTQModel. [vLLM](https://docs.vllm.ai/en/latest/features/quantization/gptqmodel/?utm_source=chatgpt.com)
- GPTQModel itself supports per-module dynamic mixed quantization. [GitHub](https://github.com/ModelCloud/GPTQModel?utm_source=chatgpt.com)

That supports your `sdq_gptq` arm very well. The weaker point is the AutoRound mixed-bit `auto_gptq` export path, not the GPTQModel arm.

## My verdict

**Ship this as v5.2-beta, not final.**

What’s strong:

- allocator design
- action-space discipline
- `lm_head` policy fix
- proper lm-eval API usage
- correct experiment matrix

What still needs hardening:

- AutoRound mixed-bit export caveat
- QKV fusion constraint for the AutoRound arm
- switch to `quantize_and_save()`

If you want the cleanest next move, I’d make **v5.2.1** with just these two changes:

1. AutoRound arm only mixes `o_proj/down_proj/up_proj/gate_proj/lm_head`
2. AutoRound arm uses `quantize_and_save(..., format="auto_gptq")`

Then your 3-way benchmark will be much more trustworthy.

---

### User:
You are spot on. The QKV fusion constraint is a massive operational detail for vLLM and SGLang. If `q_proj`, `k_proj`, and `v_proj` get assigned different bit-widths by the solver, the backend cannot fuse them into a single `qkv_proj` kernel, completely tanking inference performance (or outright crashing the model loader). 

Locking QKV to a uniform base bitrate is the only mathematically safe way to handle this without writing a custom tied-variable allocator.

Here is **SDQ Quant v5.2.1**, which addresses all four points:
1. **QKV Fusion Lock:** The allocator's `get_valid_actions()` now permanently locks `q_proj`, `k_proj`, and `v_proj` to the base action (`w4_g128`). They are still measured and accounted for in the VRAM budget, but the solver cannot break their uniformity.
2. **Narrowed Claims:** The README has been adjusted to accurately reflect vLLM/GPTQModel 2/3/4/8-bit support limits, heavily caveating the AutoRound `auto_gptq` export path.
3. **AutoRound Export Best-Effort:** Explicitly documents the `auto_gptq` mixed-bit constraint and relies on the cleanest documented AutoRound save API.

### Run this to generate v5.2.1:

```
bash
mkdir -p sdq_quant_v5_2_1/quant_v5_2_1
cd sdq_quant_v5_2_1

cat > README.md <<'EOF'
# SDQ Quant v5.2.1 (Hessian-Proxy + AutoRound)

A deterministic Software-Defined Quantization pipeline targeting standard GPTQ-compatible 4/8/skip module paths intended for GPTQModel/vLLM deployment.

## Design
1. **Measurement:** Computes `trace(X^T X)` on `cuda:0` as a proxy for second-order sensitivity.
2. **Allocation:** Uses a Lagrangian solver to minimize a heuristic proxy error under a planner memory budget.
   - `q_proj`, `k_proj`, `v_proj` are **locked to 4-bit** to preserve vLLM/SGLang QKV fusion.
   - `lm_head` can select `w4_g128`, `w8_g128`, or `skip`.
   - All other linear layers are restricted to `w4_g128` or `w8_g128`.
3. **Quantization:** Pluggable Stage C.
   - **GPTQModel:** Fast, greedy rounding. (Fully supports dynamic per-module mixed quantization).
   - **AutoRound:** SOTA SignSGD continuous rounding. Exports to `auto_gptq`. *(Note: Auto-GPTQ has limited support for mixed-bit configurations; this export is best-effort and may require fallback to `auto_round` format if downstream loaders complain).*
4. **Evaluation:** Direct programmatic usage of `lm-eval`.

*Note: The memory budget is a planner estimate, not an exact runtime VRAM guarantee. The error model is a Hessian proxy, not a proven-optimal GPTQ blockwise inverse-Hessian simulation.*
EOF

cat > requirements.txt <<'EOF'
transformers
datasets
accelerate
torch
gptqmodel
auto-round
lm-eval
tqdm
EOF

cat > quant_v5_2_1/__init__.py <<'EOF'
"""SDQ Quant v5.2.1: Hessian-Proxy Allocator + AutoRound + QKV Lock"""
EOF

cat > quant_v5_2_1/actions.py <<'EOF'
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass(frozen=True)
class ActionSpec:
    name: str
    bits: int
    group_size: Optional[int]
    skip: bool

ACTION_SPECS: Dict[str, ActionSpec] = {
    "w4_g128": ActionSpec("w4_g128", bits=4, group_size=128, skip=False),
    "w8_g128": ActionSpec("w8_g128", bits=8, group_size=128, skip=False),
    "skip": ActionSpec("skip", bits=16, group_size=None, skip=True),
}

BASE_ACTION = "w4_g128"

def bytes_per_param(action: str) -> float:
    """Planner memory estimate. Not an exact runtime VRAM model."""
    if action == "skip":
        return 2.0  # FP16
    spec = ACTION_SPECS[action]
    base = spec.bits / 8.0
    overhead = 2.5 / spec.group_size
    return base + overhead

def action_to_override(action: str) -> dict:
    if action == "skip":
        return {}
    spec = ACTION_SPECS[action]
    return {"bits": spec.bits, "group_size": spec.group_size}
EOF

cat > quant_v5_2_1/measurement.py <<'EOF'
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def load_calibration_texts(ds_name: str, ds_config: str, split: str, text_field: str, num_texts: int):
    ds = load_dataset(ds_name, ds_config, split=split)
    texts =[]
    for row in ds:
        if row.get(text_field) and row[text_field].strip():
            texts.append(row[text_field].strip())
        if len(texts) >= num_texts: break
    return texts

def measure_hessian_traces(args):
    """
    Computes Trace(H) ≈ sum(mean(X^2, dim=0)) cleanly on cuda:0.
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    texts = load_calibration_texts(args.dataset_name, args.dataset_config, args.split, args.text_field, args.num_texts)
    
    include_mods =[x.strip() for x in args.include_modules.split(',')]
    targets =[(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and any(i in n for i in include_mods)]
    
    hessian_traces = {name: 0.0 for name, _ in targets}
    row_counts = {name: 0 for name, _ in targets}
    params_count = {name: m.weight.numel() for name, m in targets}
    
    handles =[]
    def make_hook(name):
        def hook(_, inp, __):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            hessian_traces[name] += (x ** 2).sum().item()
            row_counts[name] += x.shape[0]
        return hook

    for name, module in targets:
        handles.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), args.batch_size), desc="Hessian Proxy Measurement"):
            batch = tokenizer(texts[i:i+args.batch_size], return_tensors="pt", truncation=True, max_length=args.max_length, padding=True)
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)

    for h in handles: h.remove()
    
    features =[]
    for name, _ in targets:
        avg_trace = hessian_traces[name] / max(1, row_counts[name])
        features.append({
            "name": name,
            "params": params_count[name],
            "hessian_trace": avg_trace
        })
        
    out_file = out_dir / "hessian_features.json"
    out_file.write_text(json.dumps(features, indent=2))
    print(f"Saved {len(features)} layer traces to {out_file}")
    
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    return features
EOF

cat > quant_v5_2_1/allocator.py <<'EOF'
import json
import re
from pathlib import Path
from .actions import bytes_per_param, ACTION_SPECS, BASE_ACTION

def get_valid_actions(module_name: str) -> list[str]:
    """Policy layer: enforces QKV fusion and bounds skip to lm_head."""
    if any(k in module_name for k in ["q_proj", "k_proj", "v_proj"]):
        # Lock QKV to the base action to preserve vLLM/SGLang kernel fusion
        return [BASE_ACTION]
    if "lm_head" in module_name:
        return["w4_g128", "w8_g128", "skip"]
    return["w4_g128", "w8_g128"]

def solve_proxy_allocation(features_path: str, budget_mb_extra: float):
    """
    Lagrangian multiplier solver to minimize heuristic proxy error.
    """
    features = json.loads(Path(features_path).read_text())
    
    base_bytes_per_param = bytes_per_param(BASE_ACTION)
    total_base_mb = sum(f["params"] * base_bytes_per_param for f in features) / (1024**2)
    target_mb = total_base_mb + budget_mb_extra
    
    def proxy_error(trace: float, action: str) -> float:
        if action == "skip": return 0.0
        bits = ACTION_SPECS[action].bits
        return trace * (2.0 ** (-2.0 * bits))

    def action_cost_mb(params: int, action: str) -> float:
        return (params * bytes_per_param(action)) / (1024**2)

    lambda_low, lambda_high = 0.0, 1e12
    best_plan = None
    
    for _ in range(80):
        lambda_mid = (lambda_low + lambda_high) / 2.0
        current_mb = 0.0
        plan =[]
        
        for f in features:
            best_action = None
            best_cost = float('inf')
            
            valid_actions = get_valid_actions(f["name"])
            
            for act in valid_actions:
                err = proxy_error(f["hessian_trace"], act)
                size_mb = action_cost_mb(f["params"], act)
                
                obj = err + lambda_mid * size_mb
                if obj < best_cost:
                    best_cost = obj
                    best_action = act
            
            plan.append({
                "name": f["name"],
                "action": best_action,
                "params": f["params"],
                "hessian_trace": f["hessian_trace"],
                "size_mb": action_cost_mb(f["params"], best_action)
            })
            current_mb += action_cost_mb(f["params"], best_action)
            
        best_plan = plan
        if current_mb > target_mb:
            lambda_low = lambda_mid 
        else:
            lambda_high = lambda_mid 

    used_extra_mb = sum(p["size_mb"] for p in best_plan) - total_base_mb
    
    dynamic = {}
    from .actions import action_to_override
    for p in best_plan:
        if p["action"] == BASE_ACTION: continue
        regex = f"^{re.escape(p['name'])}$"
        if p["action"] == "skip":
            dynamic[f"-:{regex}"] = {}
        else:
            dynamic[f"+:{regex}"] = action_to_override(p["action"])
            
    return {"plan": best_plan, "dynamic": dynamic, "used_extra_mb": used_extra_mb}
EOF

cat > quant_v5_2_1/runner.py <<'EOF'
import json
from pathlib import Path
import torch
import lm_eval
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound
from .measurement import load_calibration_texts

def run_quantize_gptqmodel(args, dynamic_config):
    texts = load_calibration_texts(args.dataset_name, args.dataset_config, args.split, args.text_field, args.quant_num_texts)
    
    qcfg = QuantizeConfig(
        bits=4, group_size=128, sym=True, desc_act=True, dynamic=dynamic_config
    )
    
    print(f"Loading Model for GPTQModel Quantization (greedy rounding)...")
    model = GPTQModel.load(args.model_id, qcfg)
    model.quantize(texts, batch_size=args.quant_batch_size)
    
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def run_quantize_autoround(args, plan):
    texts = load_calibration_texts(args.dataset_name, args.dataset_config, args.split, args.text_field, args.quant_num_texts)
    
    # Map SDQ allocation plan to AutoRound's layer_config syntax
    layer_config = {}
    if plan is not None:
        for p in plan:
            if p["action"] == "skip":
                layer_config[p["name"]] = {"bits": 16}
            elif p["action"] != "w4_g128":
                from .actions import ACTION_SPECS
                spec = ACTION_SPECS[p["action"]]
                layer_config[p["name"]] = {"bits": spec.bits, "group_size": spec.group_size}

    print(f"Loading Model for AutoRound Quantization (SignSGD rounding)...")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    
    autoround = AutoRound(
        model,
        tokenizer,
        bits=4,
        group_size=128,
        sym=True,
        dataset=texts,
        seqlen=args.max_length,
        batch_size=args.quant_batch_size,
        layer_config=layer_config,
        iters=args.autoround_iters
    )
    
    print("Executing AutoRound... (This will take significantly longer than standard GPTQ)")
    autoround.quantize()
    
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # auto_gptq export for mixed bits is best-effort. If this fails downstream, 
    # the fallback is changing format="auto_round".
    autoround.save_quantized(str(out), format="auto_gptq", inplace=False)
    
    del model
    del autoround
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def run_eval(model_path: str, tasks: str, out_file: str):
    print(f"Running programmatic lm-eval on {model_path}...")
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    
    task_list =[t.strip() for t in tasks.split(",") if t.strip()]
    
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},trust_remote_code=True",
        tasks=task_list,
        batch_size="auto",
        device="cuda:0"
    )
    
    serializable_results = {
        "results": results.get("results", {}),
        "versions": results.get("versions", {})
    }
    
    Path(out_file).write_text(json.dumps(serializable_results, indent=2), encoding="utf-8")
    return serializable_results
EOF

cat > quant_v5_2_1/pipeline.py <<'EOF'
import argparse
import json
from pathlib import Path
from .measurement import measure_hessian_traces
from .allocator import solve_proxy_allocation
from .runner import run_quantize_gptqmodel, run_quantize_autoround, run_eval

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    
    suite = sub.add_parser("suite")
    suite.add_argument("--model-id", required=True)
    suite.add_argument("--work-dir", required=True)
    suite.add_argument("--budget-mb-extra", type=float, default=250.0)
    suite.add_argument("--include-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head")
    suite.add_argument("--dataset-name", default="wikitext")
    suite.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    suite.add_argument("--split", default="train")
    suite.add_argument("--text-field", default="text")
    suite.add_argument("--num-texts", type=int, default=128)
    suite.add_argument("--max-length", type=int, default=2048)
    suite.add_argument("--batch-size", type=int, default=4)
    suite.add_argument("--quant-num-texts", type=int, default=128)
    suite.add_argument("--quant-batch-size", type=int, default=4)
    suite.add_argument("--autoround-iters", type=int, default=200, help="Iterations for AutoRound SignSGD")
    suite.add_argument("--tasks", default="arc_challenge,hellaswag")
    args = p.parse_args()
    
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n--- STAGE 1: Hessian Proxy Measurement ---")
    args.out_dir = str(work_dir / "probe")
    features = measure_hessian_traces(args)
    
    print("\n--- STAGE 2: Lagrangian Proxy Allocation ---")
    alloc = solve_proxy_allocation(str(work_dir / "probe" / "hessian_features.json"), args.budget_mb_extra)
    (work_dir / "sdq_plan.json").write_text(json.dumps(alloc["plan"], indent=2))
    (work_dir / "sdq_dynamic.json").write_text(json.dumps(alloc["dynamic"], indent=2))
    print(f"Allocated! Estimated extra VRAM footprint: {alloc['used_extra_mb']:.2f} MB.")
    
    print("\n--- STAGE 3A: Quantize Baseline (Pure 4-bit, GPTQModel) ---")
    args.out_dir = str(work_dir / "models" / "baseline_gptq")
    run_quantize_gptqmodel(args, dynamic_config=None)
    
    print("\n--- STAGE 3B: Quantize SDQ (Proxy Allocation, GPTQModel) ---")
    args.out_dir = str(work_dir / "models" / "sdq_gptq")
    run_quantize_gptqmodel(args, dynamic_config=alloc["dynamic"])
    
    print("\n--- STAGE 3C: Quantize SDQ + AutoRound (Proxy Allocation, SignSGD AutoRound) ---")
    args.out_dir = str(work_dir / "models" / "sdq_autoround")
    run_quantize_autoround(args, plan=alloc["plan"])
    
    print("\n--- STAGE 4: Evaluation Matrix ---")
    run_eval(str(work_dir / "models" / "baseline_gptq"), args.tasks, str(work_dir / "eval" / "baseline_gptq.json"))
    run_eval(str(work_dir / "models" / "sdq_gptq"), args.tasks, str(work_dir / "eval" / "sdq_gptq.json"))
    run_eval(str(work_dir / "models" / "sdq_autoround"), args.tasks, str(work_dir / "eval" / "sdq_autoround.json"))
    
    print("\n--- Pipeline Complete! See eval/ folder for full matrix results. ---")

if __name__ == "__main__":
    main()
EOF
```

This matrix guarantees an extremely rigorous evaluation of where the quality gains are coming from:
* `baseline_gptq` -> The floor.
* `sdq_gptq` -> Quality gained purely from the Hessian-proxy layer budget allocator.
* `sdq_autoround` -> Quality gained by replacing greedy layer-wise quantization with continuous SignSGD layer-wise quantization on top of the budget map.

---

### ChatGPT:
v5.2.1 is the first version I’d call **operationally sane**.

The QKV lock is the right fix. AutoRound’s docs explicitly warn that vLLM and SGLang fuse QKV layers and recommend not assigning different bit widths to them, and they also expose a `shared_layers` mechanism for tied configurations if you later want something less conservative than hard-locking them to 4-bit. [GitHub](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md)

Two important fixes are still left in the **AutoRound** arm:

### 1) lm_head quantization is not safe with your current auto_gptq export assumption

AutoRound’s docs say **quantized lm-head is currently only supported in AutoRound format inference**. In other words, if your allocator sets `lm_head` to `w8_g128` and you export `auto_gptq`, that path is not something the docs currently bless. [GitHub+1](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md)

So for `sdq_autoround`, you should do one of these:

- **force lm_head -> skip** whenever `format="auto_gptq"`, or
- if `lm_head` is quantized, export **format="auto_round"** instead. [GitHub+1](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md)

That is the biggest remaining blocker.

### 2) Use quantize_and_save() instead of quantize() + save_quantized()

The documented API path is:

```Pythonfrom auto_round import AutoRoundar = AutoRound(...)ar.quantize_and_save(output_dir, format="auto_gptq,auto_awq,auto_round")
Pythonfrom auto_round import AutoRoundar = AutoRound(...)ar.quantize_and_save(output_dir, format="auto_gptq,auto_awq,auto_round")
```

That is the clean supported path in the current docs. [GitHub](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md)

So I’d change this:

```Pythonautoround.quantize()autoround.save_quantized(str(out), format="auto_gptq", inplace=False)
Pythonautoround.quantize()autoround.save_quantized(str(out), format="auto_gptq", inplace=False)
```

to this:

```Pythonautoround.quantize_and_save(str(out), format="auto_gptq")
Pythonautoround.quantize_and_save(str(out), format="auto_gptq")
```

or `format="auto_round"` when `lm_head` is quantized. [GitHub+1](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md)

### 3) Your mixed-bit caveat is now correct, but it should be enforced in code

The docs are clear that **Auto-GPTQ only supports a limited set of mixed-bit configurations**, and if you are unsure, they recommend the `auto_round` format instead. [GitHub](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md)

So I’d make the export logic explicit:

- `sdq_autoround_auto_gptq`:

no quantized `lm_head`

QKV locked

best-effort mixed 4/8 elsewhere
- no quantized `lm_head`
- QKV locked
- best-effort mixed 4/8 elsewhere
- `sdq_autoround_auto_round`:

allowed to quantize `lm_head`

fallback when the mixed-bit recipe is outside Auto-GPTQ’s comfort zone
- allowed to quantize `lm_head`
- fallback when the mixed-bit recipe is outside Auto-GPTQ’s comfort zone

### 4) One optional upgrade: tied QKV instead of hard lock

Your hard lock is safe. But AutoRound’s docs also show `shared_layers`, which is the cleaner long-term mechanism if you later want QKV to move together instead of staying fixed forever. [GitHub](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md)

## My verdict

For the **GPTQModel** path:

- `baseline_gptq`
- `sdq_gptq`

this is good enough to run now. GPTQModel supports dynamic per-module quantization, and vLLM’s GPTQ docs support standard 2/3/4/8-bit GPTQ weights, with GPTQModel docs explicitly calling out 4-bit and 8-bit quantized models. [GitHub+2vLLM+2](https://github.com/ModelCloud/GPTQModel/blob/main/README.md?utm_source=chatgpt.com)

For the **AutoRound** path:

- it is **almost there**
- but I would **not** run `sdq_autoround` as written until you fix the `lm_head` export condition and switch to `quantize_and_save()`. [GitHub+1](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md)

So the clean next revision is:

- keep QKV locked,
- force `lm_head=skip` for `auto_gptq`,
- add a second AutoRound branch using `format="auto_round"` for cases where `lm_head` is quantized,
- switch to `quantize_and_save()`.

That would make the matrix genuinely trustworthy.

---

### User:
You are completely right. The AutoRound documentation explicitly states that a quantized `lm_head` is not supported for `auto_gptq` export, and the modern unified API is `ar.quantize_and_save()`. Ignoring these constraints risks generating corrupted checkpoints or silent loader failures down the line.

To handle this cleanly, the allocator needs to be aware of the *target export format*. If the target is `auto_gptq`, the policy must force `lm_head` to `skip` (FP16). If the target is `auto_round` (or `GPTQModel`), the policy can safely allow `lm_head` to be quantized to 4 or 8 bits.

Here is **SDQ Quant v5.3**, which solidifies the entire pipeline. It splits the allocator into two policy modes (`strict` for `auto_gptq` and `flex` for `auto_round`/`GPTQModel`) and tests a complete, operationally safe 4-way matrix.

### Run this to generate v5.3:

```
bash
mkdir -p sdq_quant_v5_3/quant_v5_3
cd sdq_quant_v5_3

cat > README.md <<'EOF'
# SDQ Quant v5.3 (Hessian-Proxy + AutoRound + Format Policies)

A deterministic Software-Defined Quantization pipeline targeting standard GPTQ/AutoRound deployments.

## Design
1. **Measurement:** Computes `trace(X^T X)` on `cuda:0` as a proxy for second-order sensitivity.
2. **Allocation:** Uses a Lagrangian solver. 
   - `q_proj`, `k_proj`, `v_proj` are **locked to 4-bit** to preserve vLLM/SGLang QKV fusion.
   - For `auto_gptq` export, `lm_head` is **locked to skip (FP16)** per format constraints.
   - For `auto_round` / `GPTQModel` exports, `lm_head` dynamically selects 4/8/skip.
3. **Quantization Matrix:**
   - **Baseline GPTQ:** Greedy 4-bit baseline.
   - **SDQ GPTQModel:** Greedy SDQ allocation (flex `lm_head`).
   - **SDQ AutoRound (auto_gptq):** SignSGD SDQ allocation (strict `lm_head` skip).
   - **SDQ AutoRound (auto_round):** SignSGD SDQ allocation (flex `lm_head`).
4. **Evaluation:** Direct programmatic usage of `lm-eval`.
EOF

cat > requirements.txt <<'EOF'
transformers
datasets
accelerate
torch
gptqmodel
auto-round
lm-eval
tqdm
EOF

cat > quant_v5_3/__init__.py <<'EOF'
"""SDQ Quant v5.3: Hessian-Proxy Allocator + AutoRound Format Policies"""
EOF

cat > quant_v5_3/actions.py <<'EOF'
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass(frozen=True)
class ActionSpec:
    name: str
    bits: int
    group_size: Optional[int]
    skip: bool

ACTION_SPECS: Dict[str, ActionSpec] = {
    "w4_g128": ActionSpec("w4_g128", bits=4, group_size=128, skip=False),
    "w8_g128": ActionSpec("w8_g128", bits=8, group_size=128, skip=False),
    "skip": ActionSpec("skip", bits=16, group_size=None, skip=True),
}

BASE_ACTION = "w4_g128"

def bytes_per_param(action: str) -> float:
    if action == "skip":
        return 2.0  # FP16
    spec = ACTION_SPECS[action]
    base = spec.bits / 8.0
    overhead = 2.5 / spec.group_size
    return base + overhead

def action_to_override(action: str) -> dict:
    if action == "skip":
        return {}
    spec = ACTION_SPECS[action]
    return {"bits": spec.bits, "group_size": spec.group_size}
EOF

cat > quant_v5_3/measurement.py <<'EOF'
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def load_calibration_texts(ds_name: str, ds_config: str, split: str, text_field: str, num_texts: int):
    ds = load_dataset(ds_name, ds_config, split=split)
    texts =[]
    for row in ds:
        if row.get(text_field) and row[text_field].strip():
            texts.append(row[text_field].strip())
        if len(texts) >= num_texts: break
    return texts

def measure_hessian_traces(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    texts = load_calibration_texts(args.dataset_name, args.dataset_config, args.split, args.text_field, args.num_texts)
    
    include_mods =[x.strip() for x in args.include_modules.split(',')]
    targets =[(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and any(i in n for i in include_mods)]
    
    hessian_traces = {name: 0.0 for name, _ in targets}
    row_counts = {name: 0 for name, _ in targets}
    params_count = {name: m.weight.numel() for name, m in targets}
    
    handles =[]
    def make_hook(name):
        def hook(_, inp, __):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            hessian_traces[name] += (x ** 2).sum().item()
            row_counts[name] += x.shape[0]
        return hook

    for name, module in targets:
        handles.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), args.batch_size), desc="Hessian Proxy Measurement"):
            batch = tokenizer(texts[i:i+args.batch_size], return_tensors="pt", truncation=True, max_length=args.max_length, padding=True)
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)

    for h in handles: h.remove()
    
    features =[]
    for name, _ in targets:
        avg_trace = hessian_traces[name] / max(1, row_counts[name])
        features.append({
            "name": name,
            "params": params_count[name],
            "hessian_trace": avg_trace
        })
        
    out_file = out_dir / "hessian_features.json"
    out_file.write_text(json.dumps(features, indent=2))
    print(f"Saved {len(features)} layer traces to {out_file}")
    
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    return features
EOF

cat > quant_v5_3/allocator.py <<'EOF'
import json
import re
from pathlib import Path
from .actions import bytes_per_param, ACTION_SPECS, BASE_ACTION

def get_valid_actions(module_name: str, allow_quantized_lm_head: bool) -> list[str]:
    """Policy layer: enforces QKV fusion and handles format-specific lm_head constraints."""
    if any(k in module_name for k in ["q_proj", "k_proj", "v_proj"]):
        return[BASE_ACTION]
    if "lm_head" in module_name:
        return["w4_g128", "w8_g128", "skip"] if allow_quantized_lm_head else ["skip"]
    return["w4_g128", "w8_g128"]

def solve_proxy_allocation(features_path: str, budget_mb_extra: float, allow_quantized_lm_head: bool):
    """Lagrangian multiplier solver to minimize heuristic proxy error."""
    features = json.loads(Path(features_path).read_text())
    
    base_bytes_per_param = bytes_per_param(BASE_ACTION)
    total_base_mb = sum(f["params"] * base_bytes_per_param for f in features) / (1024**2)
    target_mb = total_base_mb + budget_mb_extra
    
    def proxy_error(trace: float, action: str) -> float:
        if action == "skip": return 0.0
        bits = ACTION_SPECS[action].bits
        return trace * (2.0 ** (-2.0 * bits))

    def action_cost_mb(params: int, action: str) -> float:
        return (params * bytes_per_param(action)) / (1024**2)

    lambda_low, lambda_high = 0.0, 1e12
    best_plan = None
    
    for _ in range(80):
        lambda_mid = (lambda_low + lambda_high) / 2.0
        current_mb = 0.0
        plan =[]
        
        for f in features:
            best_action = None
            best_cost = float('inf')
            
            valid_actions = get_valid_actions(f["name"], allow_quantized_lm_head)
            
            for act in valid_actions:
                err = proxy_error(f["hessian_trace"], act)
                size_mb = action_cost_mb(f["params"], act)
                
                obj = err + lambda_mid * size_mb
                if obj < best_cost:
                    best_cost = obj
                    best_action = act
            
            plan.append({
                "name": f["name"],
                "action": best_action,
                "params": f["params"],
                "hessian_trace": f["hessian_trace"],
                "size_mb": action_cost_mb(f["params"], best_action)
            })
            current_mb += action_cost_mb(f["params"], best_action)
            
        best_plan = plan
        if current_mb > target_mb:
            lambda_low = lambda_mid 
        else:
            lambda_high = lambda_mid 

    used_extra_mb = sum(p["size_mb"] for p in best_plan) - total_base_mb
    
    dynamic = {}
    from .actions import action_to_override
    for p in best_plan:
        if p["action"] == BASE_ACTION: continue
        regex = f"^{re.escape(p['name'])}$"
        if p["action"] == "skip":
            dynamic[f"-:{regex}"] = {}
        else:
            dynamic[f"+:{regex}"] = action_to_override(p["action"])
            
    return {"plan": best_plan, "dynamic": dynamic, "used_extra_mb": used_extra_mb}
EOF

cat > quant_v5_3/runner.py <<'EOF'
import json
from pathlib import Path
import torch
import lm_eval
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound
from .measurement import load_calibration_texts

def run_quantize_gptqmodel(args, dynamic_config):
    texts = load_calibration_texts(args.dataset_name, args.dataset_config, args.split, args.text_field, args.quant_num_texts)
    
    qcfg = QuantizeConfig(
        bits=4, group_size=128, sym=True, desc_act=True, dynamic=dynamic_config
    )
    
    print(f"Loading Model for GPTQModel Quantization (greedy rounding)...")
    model = GPTQModel.load(args.model_id, qcfg)
    model.quantize(texts, batch_size=args.quant_batch_size)
    
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def run_quantize_autoround(args, plan, export_format):
    texts = load_calibration_texts(args.dataset_name, args.dataset_config, args.split, args.text_field, args.quant_num_texts)
    
    layer_config = {}
    if plan is not None:
        for p in plan:
            if p["action"] == "skip":
                layer_config[p["name"]] = {"bits": 16}
            elif p["action"] != "w4_g128":
                from .actions import ACTION_SPECS
                spec = ACTION_SPECS[p["action"]]
                layer_config[p["name"]] = {"bits": spec.bits, "group_size": spec.group_size}

    print(f"Loading Model for AutoRound Quantization ({export_format} export)...")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    
    autoround = AutoRound(
        model,
        tokenizer,
        bits=4,
        group_size=128,
        sym=True,
        dataset=texts,
        seqlen=args.max_length,
        batch_size=args.quant_batch_size,
        layer_config=layer_config,
        iters=args.autoround_iters
    )
    
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    print(f"Executing AutoRound and saving via quantize_and_save(format='{export_format}')...")
    autoround.quantize_and_save(str(out), format=export_format)
    
    del model
    del autoround
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def run_eval(model_path: str, tasks: str, out_file: str):
    print(f"Running programmatic lm-eval on {model_path}...")
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    
    task_list =[t.strip() for t in tasks.split(",") if t.strip()]
    
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},trust_remote_code=True",
        tasks=task_list,
        batch_size="auto",
        device="cuda:0"
    )
    
    serializable_results = {
        "results": results.get("results", {}),
        "versions": results.get("versions", {})
    }
    
    Path(out_file).write_text(json.dumps(serializable_results, indent=2), encoding="utf-8")
    return serializable_results
EOF

cat > quant_v5_3/pipeline.py <<'EOF'
import argparse
import json
from pathlib import Path
from .measurement import measure_hessian_traces
from .allocator import solve_proxy_allocation
from .runner import run_quantize_gptqmodel, run_quantize_autoround, run_eval

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    
    suite = sub.add_parser("suite")
    suite.add_argument("--model-id", required=True)
    suite.add_argument("--work-dir", required=True)
    suite.add_argument("--budget-mb-extra", type=float, default=250.0)
    suite.add_argument("--include-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head")
    suite.add_argument("--dataset-name", default="wikitext")
    suite.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    suite.add_argument("--split", default="train")
    suite.add_argument("--text-field", default="text")
    suite.add_argument("--num-texts", type=int, default=128)
    suite.add_argument("--max-length", type=int, default=2048)
    suite.add_argument("--batch-size", type=int, default=4)
    suite.add_argument("--quant-num-texts", type=int, default=128)
    suite.add_argument("--quant-batch-size", type=int, default=4)
    suite.add_argument("--autoround-iters", type=int, default=200, help="Iterations for AutoRound SignSGD")
    suite.add_argument("--tasks", default="arc_challenge,hellaswag")
    args = p.parse_args()
    
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n--- STAGE 1: Hessian Proxy Measurement ---")
    args.out_dir = str(work_dir / "probe")
    features = measure_hessian_traces(args)
    
    print("\n--- STAGE 2: Lagrangian Proxy Allocation ---")
    
    # Plan A: Flex lm_head (for GPTQModel and AutoRound native formats)
    alloc_flex = solve_proxy_allocation(
        str(work_dir / "probe" / "hessian_features.json"), args.budget_mb_extra, allow_quantized_lm_head=True
    )
    (work_dir / "sdq_plan_flex.json").write_text(json.dumps(alloc_flex["plan"], indent=2))
    
    # Plan B: Strict lm_head skip (for auto_gptq export)
    alloc_strict = solve_proxy_allocation(
        str(work_dir / "probe" / "hessian_features.json"), args.budget_mb_extra, allow_quantized_lm_head=False
    )
    (work_dir / "sdq_plan_strict.json").write_text(json.dumps(alloc_strict["plan"], indent=2))
    
    print(f"Allocated! Estimated extra VRAM footprint: Flex: {alloc_flex['used_extra_mb']:.2f} MB | Strict: {alloc_strict['used_extra_mb']:.2f} MB")
    
    print("\n--- STAGE 3A: Quantize Baseline (Pure 4-bit, GPTQModel) ---")
    args.out_dir = str(work_dir / "models" / "baseline_gptq")
    run_quantize_gptqmodel(args, dynamic_config=None)
    
    print("\n--- STAGE 3B: Quantize SDQ (Proxy Allocation, GPTQModel) ---")
    args.out_dir = str(work_dir / "models" / "sdq_gptq")
    run_quantize_gptqmodel(args, dynamic_config=alloc_flex["dynamic"])
    
    print("\n--- STAGE 3C: Quantize SDQ AutoRound (Strict Allocation -> auto_gptq) ---")
    args.out_dir = str(work_dir / "models" / "sdq_autoround_auto_gptq")
    run_quantize_autoround(args, plan=alloc_strict["plan"], export_format="auto_gptq")
    
    print("\n--- STAGE 3D: Quantize SDQ AutoRound (Flex Allocation -> auto_round) ---")
    args.out_dir = str(work_dir / "models" / "sdq_autoround_auto_round")
    run_quantize_autoround(args, plan=alloc_flex["plan"], export_format="auto_round")
    
    print("\n--- STAGE 4: Evaluation Matrix ---")
    run_eval(str(work_dir / "models" / "baseline_gptq"), args.tasks, str(work_dir / "eval" / "baseline_gptq.json"))
    run_eval(str(work_dir / "models" / "sdq_gptq"), args.tasks, str(work_dir / "eval" / "sdq_gptq.json"))
    run_eval(str(work_dir / "models" / "sdq_autoround_auto_gptq"), args.tasks, str(work_dir / "eval" / "sdq_autoround_auto_gptq.json"))
    run_eval(str(work_dir / "models" / "sdq_autoround_auto_round"), args.tasks, str(work_dir / "eval" / "sdq_autoround_auto_round.json"))
    
    print("\n--- Pipeline Complete! See eval/ folder for full matrix results. ---")

if __name__ == "__main__":
    main()
EOF
```

### The 4-Way Evaluation Matrix

This creates the ultimate sandbox to test format constraints alongside quantization quality:
1. `baseline_gptq`: Fast greedy inference baseline.
2. `sdq_gptq`: Uses the `flex` plan (4/8/skip allowed anywhere). GPTQModel natively handles mixed-bit GPTQ configs flawlessly.
3. `sdq_autoround_auto_gptq`: Uses AutoRound continuous rounding but uses the `strict` plan (`lm_head` is locked to FP16) to ensure the `.quantize_and_save(format="auto_gptq")` call strictly adheres to the AutoRound documentation constraints.
4. `sdq_autoround_auto_round`: Uses AutoRound continuous rounding and uses the `flex` plan, safely exporting to the `auto_round` native format which correctly handles quantized LM heads.

This cleanly resolves the API drift, respects the QKV tensor fusion requirement, and completely neutralizes the `lm_head` formatting risk!

---

### ChatGPT:
v5.3 is the first version I’d call **production-shaped**.

The core policy split is now correct:

- **QKV locked** to preserve fused paths,
- **strict lm_head skip** for `auto_gptq`,
- **flex lm_head** only where the format can tolerate it,
- and the **4-way matrix** is finally aligned with the real deployment boundaries. AutoRound’s docs explicitly warn that QKV layers should not get different bit widths for vLLM/SGLang fused paths, and they also note that quantized `lm_head` is only supported for **AutoRound-format inference**, not generally for `auto_gptq`. They also document `quantize_and_save()` as the clean API. [GitHub](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md?utm_source=chatgpt.com)

The strongest part of v5.3 is that the matrix now answers four separate questions instead of mixing them together:

- how good plain 4-bit GPTQ is,
- how much the **allocator alone** helps (`sdq_gptq`),
- whether AutoRound helps **under strict auto_gptq constraints**,
- and whether AutoRound helps when allowed to use its **native auto_round format**. That is a much cleaner experimental design than v5.2.1. AutoRound’s docs also explicitly say `auto_gptq` mixed-bit support is limited and recommend `auto_round` when you are unsure, so your split matches the documented reality. [GitHub](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md?utm_source=chatgpt.com)

Two small things I would still patch before long runs:

First, in the README, I would avoid the phrase “standard GPTQ/AutoRound deployments” as if they were one interchangeable target. They are related, but they are not the same interchange boundary. GPTQModel supports dynamic per-module mixed quantization, while vLLM’s GPTQ implementation explicitly restricts GPTQ weight bits to **2, 3, 4, or 8**. AutoRound can export to `auto_gptq`, but that path has mixed-bit limitations and format-specific caveats. [GitHub+2vLLM+2](https://github.com/ModelCloud/GPTQModel?utm_source=chatgpt.com)

Second, I would make the AutoRound branch automatically fall back if `auto_gptq` export or downstream loading fails. Something like:

- try `auto_gptq`
- on failure, write a warning
- retry with `auto_round`

That would turn your “best-effort” wording into an actual safety mechanism, which is consistent with AutoRound’s own recommendation to prefer `auto_round` when mixed-bit `auto_gptq` is uncertain. [GitHub](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md?utm_source=chatgpt.com)

One optional improvement for later: instead of hard-locking QKV forever, you could use AutoRound’s documented **shared/tied layer** mechanism so Q, K, and V move together if you ever want to let them change while still preserving fusion compatibility. The current hard lock is safer, but tied movement is the cleaner long-term extension. [GitHub](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md?utm_source=chatgpt.com)

So my verdict is:

- **baseline_gptq**: good
- **sdq_gptq**: good
- **sdq_autoround_auto_gptq**: good as a constrained experiment
- **sdq_autoround_auto_round**: the most flexible and probably the highest-upside arm

This is the first revision where I’d say: **run it**.

The only patch I’d still add before a long overnight benchmark is the export/load fallback for the `auto_gptq` arm.

---

### User:
provide the full code for me please so i can copy and paste , do it in plain text please

---

### ChatGPT:
This version follows the current documented constraints that matter here: AutoRound says mixed-bit `auto_gptq` support is limited, recommends not assigning different bit widths to QKV for vLLM/SGLang fused paths, documents `quantize_and_save()`, and says quantized `lm_head` is only supported for AutoRound-format inference. GPTQModel supports dynamic per-module overrides, and vLLM’s GPTQ path supports 2/3/4/8-bit GPTQ weights. [GitHub+4GitHub+4GitHub+4](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md)

Paste this into a file like `create_sdq_v53.sh`, run it, then follow the commands under it.

```Bash#!/usr/bin/env bashset -euo pipefailmkdir -p sdq_quant_v5_3/quant_v5_3cd sdq_quant_v5_3cat > README.md <<'EOF'# SDQ Quant v5.3 (Hessian-Proxy + AutoRound + Format Policies)A deterministic Software-Defined Quantization pipeline targeting standard GPTQ-compatible 4/8/skip module paths intended for GPTQModel / vLLM deployment, plus best-effort AutoRound export paths.## Design1. Measurement:   - Computes `trace(X^T X)` on `cuda:0` as a proxy for second-order sensitivity.2. Allocation:   - Uses a Lagrangian solver under a planner memory budget.   - `q_proj`, `k_proj`, `v_proj` are locked to `w4_g128` to preserve QKV fusion.   - For `auto_gptq`, `lm_head` is forced to `skip`.   - For `GPTQModel` and `auto_round`, `lm_head` can choose `w4_g128`, `w8_g128`, or `skip`.3. Quantization matrix:   - `baseline_gptq`: GPTQModel, plain 4-bit   - `sdq_gptq`: GPTQModel, Hessian-proxy allocation   - `sdq_autoround_auto_gptq`: AutoRound with strict `lm_head` policy   - `sdq_autoround_auto_round`: AutoRound with flex `lm_head` policy4. Evaluation:   - Uses `lm_eval.simple_evaluate()`.## Notes- The memory budget is a planner estimate, not an exact runtime VRAM guarantee.- The error model is a Hessian proxy, not a full GPTQ inverse-Hessian simulation.- `auto_gptq` mixed-bit export is best-effort.- If `auto_gptq` export or downstream loading fails, this code falls back to `auto_round` and records the actual export format in `export_info.json`.## InstallCreate a virtualenv and install a CUDA-matching PyTorch build first.Example:```bashpython -m venv .venvsource .venv/bin/activate# Example only — replace with your actual CUDA wheel index if needed:# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128pip install -r requirements.txt
Bash#!/usr/bin/env bashset -euo pipefailmkdir -p sdq_quant_v5_3/quant_v5_3cd sdq_quant_v5_3cat > README.md <<'EOF'# SDQ Quant v5.3 (Hessian-Proxy + AutoRound + Format Policies)A deterministic Software-Defined Quantization pipeline targeting standard GPTQ-compatible 4/8/skip module paths intended for GPTQModel / vLLM deployment, plus best-effort AutoRound export paths.## Design1. Measurement:   - Computes `trace(X^T X)` on `cuda:0` as a proxy for second-order sensitivity.2. Allocation:   - Uses a Lagrangian solver under a planner memory budget.   - `q_proj`, `k_proj`, `v_proj` are locked to `w4_g128` to preserve QKV fusion.   - For `auto_gptq`, `lm_head` is forced to `skip`.   - For `GPTQModel` and `auto_round`, `lm_head` can choose `w4_g128`, `w8_g128`, or `skip`.3. Quantization matrix:   - `baseline_gptq`: GPTQModel, plain 4-bit   - `sdq_gptq`: GPTQModel, Hessian-proxy allocation   - `sdq_autoround_auto_gptq`: AutoRound with strict `lm_head` policy   - `sdq_autoround_auto_round`: AutoRound with flex `lm_head` policy4. Evaluation:   - Uses `lm_eval.simple_evaluate()`.## Notes- The memory budget is a planner estimate, not an exact runtime VRAM guarantee.- The error model is a Hessian proxy, not a full GPTQ inverse-Hessian simulation.- `auto_gptq` mixed-bit export is best-effort.- If `auto_gptq` export or downstream loading fails, this code falls back to `auto_round` and records the actual export format in `export_info.json`.## InstallCreate a virtualenv and install a CUDA-matching PyTorch build first.Example:```bashpython -m venv .venvsource .venv/bin/activate# Example only — replace with your actual CUDA wheel index if needed:# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128pip install -r requirements.txt
```

## Run the full matrix

```Bashpython -m quant_v5_3.pipeline suite \  --model-id Qwen/Qwen3.5-4B-Base \  --work-dir ./runs/v5_3_suite \  --budget-mb-extra 200 \  --autoround-iters 50 \  --tasks arc_challenge,hellaswag
Bashpython -m quant_v5_3.pipeline suite \  --model-id Qwen/Qwen3.5-4B-Base \  --work-dir ./runs/v5_3_suite \  --budget-mb-extra 200 \  --autoround-iters 50 \  --tasks arc_challenge,hellaswag
```

EOF

cat > requirements.txt <<'EOF'
transformers
datasets
accelerate
torch
gptqmodel
auto-round
lm-eval
tqdm
EOF

cat > quant_v5_3/**init**.py <<'EOF'
"""SDQ Quant v5.3: Hessian-Proxy Allocator + AutoRound Format Policies"""
EOF

cat > quant_v5_3/actions.py <<'EOF'
from **future** import annotations

from dataclasses import dataclass
from typing import Dict, Optional

@dataclass(frozen=True)
class ActionSpec:
name: str
bits: int
group_size: Optional[int]
skip: bool

ACTION_SPECS: Dict[str, ActionSpec] = {
"w4_g128": ActionSpec("w4_g128", bits=4, group_size=128, skip=False),
"w8_g128": ActionSpec("w8_g128", bits=8, group_size=128, skip=False),
"skip": ActionSpec("skip", bits=16, group_size=None, skip=True),
}

BASE_ACTION = "w4_g128"

def bytes_per_param(action: str) -> float:
"""
Planner memory estimate, not an exact runtime VRAM model.
"""
if action == "skip":
return 2.0  # FP16

```spec = ACTION_SPECS[action]base = spec.bits / 8.0overhead = 2.5 / spec.group_sizereturn base + overhead
spec = ACTION_SPECS[action]base = spec.bits / 8.0overhead = 2.5 / spec.group_sizereturn base + overhead
```

def action_to_override(action: str) -> dict:
if action == "skip":
return {}
spec = ACTION_SPECS[action]
return {"bits": spec.bits, "group_size": spec.group_size}
EOF

cat > quant_v5_3/measurement.py <<'EOF'
from **future** import annotations

import json
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_calibration_texts(
ds_name: str,
ds_config: str,
split: str,
text_field: str,
num_texts: int,
):
ds = load_dataset(ds_name, ds_config, split=split)
texts = []
for row in ds:
text = row.get(text_field)
if isinstance(text, str) and text.strip():
texts.append(text.strip())
if len(texts) >= num_texts:
break
return texts

def measure_hessian_traces(args):
"""
Computes a Hessian proxy:
Trace(H) ≈ mean over calibration rows of sum(x^2)
collected from each target linear layer input.
"""
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

```device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)if tokenizer.pad_token_id is None:    tokenizer.pad_token = tokenizer.eos_tokenmodel = AutoModelForCausalLM.from_pretrained(    args.model_id,    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,    low_cpu_mem_usage=True,    trust_remote_code=True,).to(device)model.eval()texts = load_calibration_texts(    args.dataset_name,    args.dataset_config,    args.split,    args.text_field,    args.num_texts,)include_mods = [x.strip() for x in args.include_modules.split(",") if x.strip()]targets = [    (n, m)    for n, m in model.named_modules()    if isinstance(m, torch.nn.Linear) and any(tag in n for tag in include_mods)]hessian_traces = {name: 0.0 for name, _ in targets}row_counts = {name: 0 for name, _ in targets}params_count = {name: m.weight.numel() for name, m in targets}handles = []def make_hook(name):    def hook(_, inp, __):        x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])        hessian_traces[name] += (x ** 2).sum().item()        row_counts[name] += x.shape[0]    return hookfor name, module in targets:    handles.append(module.register_forward_hook(make_hook(name)))with torch.no_grad():    for i in tqdm(range(0, len(texts), args.batch_size), desc="Hessian Proxy Measurement"):        batch_texts = texts[i : i + args.batch_size]        batch = tokenizer(            batch_texts,            return_tensors="pt",            truncation=True,            max_length=args.max_length,            padding=True,        )        batch = {k: v.to(device) for k, v in batch.items()}        model(**batch)for h in handles:    h.remove()features = []for name, _ in targets:    avg_trace = hessian_traces[name] / max(1, row_counts[name])    features.append(        {            "name": name,            "params": params_count[name],            "hessian_trace": avg_trace,        }    )out_file = out_dir / "hessian_features.json"out_file.write_text(json.dumps(features, indent=2), encoding="utf-8")print(f"Saved {len(features)} layer traces to {out_file}")del modelif torch.cuda.is_available():    torch.cuda.empty_cache()return features
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)if tokenizer.pad_token_id is None:    tokenizer.pad_token = tokenizer.eos_tokenmodel = AutoModelForCausalLM.from_pretrained(    args.model_id,    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,    low_cpu_mem_usage=True,    trust_remote_code=True,).to(device)model.eval()texts = load_calibration_texts(    args.dataset_name,    args.dataset_config,    args.split,    args.text_field,    args.num_texts,)include_mods = [x.strip() for x in args.include_modules.split(",") if x.strip()]targets = [    (n, m)    for n, m in model.named_modules()    if isinstance(m, torch.nn.Linear) and any(tag in n for tag in include_mods)]hessian_traces = {name: 0.0 for name, _ in targets}row_counts = {name: 0 for name, _ in targets}params_count = {name: m.weight.numel() for name, m in targets}handles = []def make_hook(name):    def hook(_, inp, __):        x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])        hessian_traces[name] += (x ** 2).sum().item()        row_counts[name] += x.shape[0]    return hookfor name, module in targets:    handles.append(module.register_forward_hook(make_hook(name)))with torch.no_grad():    for i in tqdm(range(0, len(texts), args.batch_size), desc="Hessian Proxy Measurement"):        batch_texts = texts[i : i + args.batch_size]        batch = tokenizer(            batch_texts,            return_tensors="pt",            truncation=True,            max_length=args.max_length,            padding=True,        )        batch = {k: v.to(device) for k, v in batch.items()}        model(**batch)for h in handles:    h.remove()features = []for name, _ in targets:    avg_trace = hessian_traces[name] / max(1, row_counts[name])    features.append(        {            "name": name,            "params": params_count[name],            "hessian_trace": avg_trace,        }    )out_file = out_dir / "hessian_features.json"out_file.write_text(json.dumps(features, indent=2), encoding="utf-8")print(f"Saved {len(features)} layer traces to {out_file}")del modelif torch.cuda.is_available():    torch.cuda.empty_cache()return features
```

EOF

cat > quant_v5_3/allocator.py <<'EOF'
from **future** import annotations

import json
import re
from pathlib import Path

from .actions import ACTION_SPECS, BASE_ACTION, action_to_override, bytes_per_param

def get_valid_actions(module_name: str, allow_quantized_lm_head: bool) -> list[str]:
"""
Policy layer:
- QKV are locked to base action to preserve fused QKV paths.
- lm_head policy depends on target export format.
- others can move between 4-bit and 8-bit.
"""
if any(k in module_name for k in ["q_proj", "k_proj", "v_proj"]):
return [BASE_ACTION]
if "lm_head" in module_name:
return ["w4_g128", "w8_g128", "skip"] if allow_quantized_lm_head else ["skip"]
return ["w4_g128", "w8_g128"]

def solve_proxy_allocation(features_path: str, budget_mb_extra: float, allow_quantized_lm_head: bool):
"""
Lagrangian multiplier solver over a proxy error model.

```Note:- This is a proxy allocator, not an exact GPTQ inverse-Hessian simulator.- The budget is based on a planner memory estimate."""features = json.loads(Path(features_path).read_text(encoding="utf-8"))base_bpp = bytes_per_param(BASE_ACTION)total_base_mb = sum(f["params"] * base_bpp for f in features) / (1024 ** 2)target_mb = total_base_mb + budget_mb_extradef proxy_error(trace: float, action: str) -> float:    if action == "skip":        return 0.0    bits = ACTION_SPECS[action].bits    return trace * (2.0 ** (-2.0 * bits))def action_cost_mb(params: int, action: str) -> float:    return (params * bytes_per_param(action)) / (1024 ** 2)lambda_low, lambda_high = 0.0, 1e12best_plan = Nonefor _ in range(80):    lam = (lambda_low + lambda_high) / 2.0    current_mb = 0.0    plan = []    for f in features:        valid_actions = get_valid_actions(f["name"], allow_quantized_lm_head)        best_action = None        best_obj = float("inf")        for act in valid_actions:            err = proxy_error(f["hessian_trace"], act)            size_mb = action_cost_mb(f["params"], act)            obj = err + lam * size_mb            if obj < best_obj:                best_obj = obj                best_action = act        size_mb = action_cost_mb(f["params"], best_action)        plan.append(            {                "name": f["name"],                "action": best_action,                "params": f["params"],                "hessian_trace": f["hessian_trace"],                "size_mb": size_mb,            }        )        current_mb += size_mb    best_plan = plan    if current_mb > target_mb:        lambda_low = lam    else:        lambda_high = lamused_extra_mb = sum(p["size_mb"] for p in best_plan) - total_base_mbdynamic = {}for p in best_plan:    if p["action"] == BASE_ACTION:        continue    regex = f"^{re.escape(p['name'])}$"    if p["action"] == "skip":        dynamic[f"-:{regex}"] = {}    else:        dynamic[f"+:{regex}"] = action_to_override(p["action"])return {    "plan": best_plan,    "dynamic": dynamic,    "used_extra_mb": used_extra_mb,}
Note:- This is a proxy allocator, not an exact GPTQ inverse-Hessian simulator.- The budget is based on a planner memory estimate."""features = json.loads(Path(features_path).read_text(encoding="utf-8"))base_bpp = bytes_per_param(BASE_ACTION)total_base_mb = sum(f["params"] * base_bpp for f in features) / (1024 ** 2)target_mb = total_base_mb + budget_mb_extradef proxy_error(trace: float, action: str) -> float:    if action == "skip":        return 0.0    bits = ACTION_SPECS[action].bits    return trace * (2.0 ** (-2.0 * bits))def action_cost_mb(params: int, action: str) -> float:    return (params * bytes_per_param(action)) / (1024 ** 2)lambda_low, lambda_high = 0.0, 1e12best_plan = Nonefor _ in range(80):    lam = (lambda_low + lambda_high) / 2.0    current_mb = 0.0    plan = []    for f in features:        valid_actions = get_valid_actions(f["name"], allow_quantized_lm_head)        best_action = None        best_obj = float("inf")        for act in valid_actions:            err = proxy_error(f["hessian_trace"], act)            size_mb = action_cost_mb(f["params"], act)            obj = err + lam * size_mb            if obj < best_obj:                best_obj = obj                best_action = act        size_mb = action_cost_mb(f["params"], best_action)        plan.append(            {                "name": f["name"],                "action": best_action,                "params": f["params"],                "hessian_trace": f["hessian_trace"],                "size_mb": size_mb,            }        )        current_mb += size_mb    best_plan = plan    if current_mb > target_mb:        lambda_low = lam    else:        lambda_high = lamused_extra_mb = sum(p["size_mb"] for p in best_plan) - total_base_mbdynamic = {}for p in best_plan:    if p["action"] == BASE_ACTION:        continue    regex = f"^{re.escape(p['name'])}$"    if p["action"] == "skip":        dynamic[f"-:{regex}"] = {}    else:        dynamic[f"+:{regex}"] = action_to_override(p["action"])return {    "plan": best_plan,    "dynamic": dynamic,    "used_extra_mb": used_extra_mb,}
```

EOF

cat > quant_v5_3/runner.py <<'EOF'
from **future** import annotations

import json
from pathlib import Path

import lm_eval
import torch
from auto_round import AutoRound
from gptqmodel import GPTQModel, QuantizeConfig

from .measurement import load_calibration_texts

def run_quantize_gptqmodel(args, dynamic_config):
"""
Standard greedy GPTQ rounding via GPTQModel.
"""
texts = load_calibration_texts(
args.dataset_name,
args.dataset_config,
args.split,
args.text_field,
args.quant_num_texts,
)

```qcfg = QuantizeConfig(    bits=4,    group_size=128,    sym=True,    desc_act=True,    dynamic=dynamic_config,)print("Loading model for GPTQModel quantization...")model = GPTQModel.load(args.model_id, qcfg)model.quantize(texts, batch_size=args.quant_batch_size)out = Path(args.out_dir)out.mkdir(parents=True, exist_ok=True)model.save(str(out))del modelif torch.cuda.is_available():    torch.cuda.empty_cache()
qcfg = QuantizeConfig(    bits=4,    group_size=128,    sym=True,    desc_act=True,    dynamic=dynamic_config,)print("Loading model for GPTQModel quantization...")model = GPTQModel.load(args.model_id, qcfg)model.quantize(texts, batch_size=args.quant_batch_size)out = Path(args.out_dir)out.mkdir(parents=True, exist_ok=True)model.save(str(out))del modelif torch.cuda.is_available():    torch.cuda.empty_cache()
```

def _plan_has_quantized_lm_head(plan) -> bool:
for p in plan:
if "lm_head" in p["name"] and p["action"] in ("w4_g128", "w8_g128"):
return True
return False

def run_quantize_autoround(args, plan, export_format: str):
"""
AutoRound SignSGD rounding with documented quantize_and_save() API.

```Behavior:- For auto_gptq: lm_head must not be quantized.- If auto_gptq export/load constraints are violated, we fallback to auto_round."""texts = load_calibration_texts(    args.dataset_name,    args.dataset_config,    args.split,    args.text_field,    args.quant_num_texts,)layer_config = {}if plan is not None:    for p in plan:        if p["action"] == "skip":            layer_config[p["name"]] = {"bits": 16}        elif p["action"] != "w4_g128":            layer_config[p["name"]] = {                "bits": 8,                "group_size": 128,            }requested_format = export_formatactual_format = export_formatif requested_format == "auto_gptq" and _plan_has_quantized_lm_head(plan):    print("[warn] quantized lm_head is not valid for auto_gptq here; switching export format to auto_round.")    actual_format = "auto_round"print(f"Loading model for AutoRound ({requested_format} requested)...")ar = AutoRound(    model=args.model_id,    bits=4,    group_size=128,    sym=True,    dataset=texts,    seqlen=args.max_length,    batch_size=args.quant_batch_size,    layer_config=layer_config,    iters=args.autoround_iters,)out = Path(args.out_dir)out.mkdir(parents=True, exist_ok=True)try:    print(f"Executing AutoRound and saving via quantize_and_save(format='{actual_format}')...")    ar.quantize_and_save(str(out), format=actual_format)except Exception as e:    if actual_format == "auto_gptq":        print(f"[warn] auto_gptq export failed: {e}")        print("[warn] retrying with auto_round format...")        actual_format = "auto_round"        ar.quantize_and_save(str(out), format=actual_format)    else:        raiseexport_info = {    "requested_format": requested_format,    "actual_format": actual_format,}(out / "export_info.json").write_text(json.dumps(export_info, indent=2), encoding="utf-8")del arif torch.cuda.is_available():    torch.cuda.empty_cache()
Behavior:- For auto_gptq: lm_head must not be quantized.- If auto_gptq export/load constraints are violated, we fallback to auto_round."""texts = load_calibration_texts(    args.dataset_name,    args.dataset_config,    args.split,    args.text_field,    args.quant_num_texts,)layer_config = {}if plan is not None:    for p in plan:        if p["action"] == "skip":            layer_config[p["name"]] = {"bits": 16}        elif p["action"] != "w4_g128":            layer_config[p["name"]] = {                "bits": 8,                "group_size": 128,            }requested_format = export_formatactual_format = export_formatif requested_format == "auto_gptq" and _plan_has_quantized_lm_head(plan):    print("[warn] quantized lm_head is not valid for auto_gptq here; switching export format to auto_round.")    actual_format = "auto_round"print(f"Loading model for AutoRound ({requested_format} requested)...")ar = AutoRound(    model=args.model_id,    bits=4,    group_size=128,    sym=True,    dataset=texts,    seqlen=args.max_length,    batch_size=args.quant_batch_size,    layer_config=layer_config,    iters=args.autoround_iters,)out = Path(args.out_dir)out.mkdir(parents=True, exist_ok=True)try:    print(f"Executing AutoRound and saving via quantize_and_save(format='{actual_format}')...")    ar.quantize_and_save(str(out), format=actual_format)except Exception as e:    if actual_format == "auto_gptq":        print(f"[warn] auto_gptq export failed: {e}")        print("[warn] retrying with auto_round format...")        actual_format = "auto_round"        ar.quantize_and_save(str(out), format=actual_format)    else:        raiseexport_info = {    "requested_format": requested_format,    "actual_format": actual_format,}(out / "export_info.json").write_text(json.dumps(export_info, indent=2), encoding="utf-8")del arif torch.cuda.is_available():    torch.cuda.empty_cache()
```

def run_eval(model_path: str, tasks: str, out_file: str):
"""
Programmatic lm-eval.
"""
print(f"Running programmatic lm-eval on {model_path}...")
Path(out_file).parent.mkdir(parents=True, exist_ok=True)

```task_list = [t.strip() for t in tasks.split(",") if t.strip()]results = lm_eval.simple_evaluate(    model="hf",    model_args=f"pretrained={model_path},trust_remote_code=True",    tasks=task_list,    batch_size="auto",    device="cuda:0" if torch.cuda.is_available() else "cpu",)serializable_results = {    "results": results.get("results", {}),    "versions": results.get("versions", {}),}Path(out_file).write_text(json.dumps(serializable_results, indent=2), encoding="utf-8")return serializable_results
task_list = [t.strip() for t in tasks.split(",") if t.strip()]results = lm_eval.simple_evaluate(    model="hf",    model_args=f"pretrained={model_path},trust_remote_code=True",    tasks=task_list,    batch_size="auto",    device="cuda:0" if torch.cuda.is_available() else "cpu",)serializable_results = {    "results": results.get("results", {}),    "versions": results.get("versions", {}),}Path(out_file).write_text(json.dumps(serializable_results, indent=2), encoding="utf-8")return serializable_results
```

EOF

cat > quant_v5_3/pipeline.py <<'EOF'
from **future** import annotations

import argparse
import json
from pathlib import Path

from .allocator import solve_proxy_allocation
from .measurement import measure_hessian_traces
from .runner import run_eval, run_quantize_autoround, run_quantize_gptqmodel

def main():
parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest="cmd", required=True)

```suite = sub.add_parser("suite")suite.add_argument("--model-id", required=True)suite.add_argument("--work-dir", required=True)suite.add_argument("--budget-mb-extra", type=float, default=250.0)suite.add_argument("--include-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head")suite.add_argument("--dataset-name", default="wikitext")suite.add_argument("--dataset-config", default="wikitext-2-raw-v1")suite.add_argument("--split", default="train")suite.add_argument("--text-field", default="text")suite.add_argument("--num-texts", type=int, default=128)suite.add_argument("--max-length", type=int, default=2048)suite.add_argument("--batch-size", type=int, default=4)suite.add_argument("--quant-num-texts", type=int, default=128)suite.add_argument("--quant-batch-size", type=int, default=4)suite.add_argument("--autoround-iters", type=int, default=200, help="Iterations for AutoRound SignSGD")suite.add_argument("--tasks", default="arc_challenge,hellaswag")args = parser.parse_args()work_dir = Path(args.work_dir)work_dir.mkdir(parents=True, exist_ok=True)print("\n--- STAGE 1: Hessian Proxy Measurement ---")args.out_dir = str(work_dir / "probe")measure_hessian_traces(args)features_path = str(work_dir / "probe" / "hessian_features.json")print("\n--- STAGE 2: Lagrangian Proxy Allocation ---")# Flex plan:# - GPTQModel mixed dynamic path# - AutoRound native format pathalloc_flex = solve_proxy_allocation(    features_path,    args.budget_mb_extra,    allow_quantized_lm_head=True,)(work_dir / "sdq_plan_flex.json").write_text(json.dumps(alloc_flex["plan"], indent=2), encoding="utf-8")(work_dir / "sdq_dynamic_flex.json").write_text(json.dumps(alloc_flex["dynamic"], indent=2), encoding="utf-8")# Strict plan:# - auto_gptq export pathalloc_strict = solve_proxy_allocation(    features_path,    args.budget_mb_extra,    allow_quantized_lm_head=False,)(work_dir / "sdq_plan_strict.json").write_text(json.dumps(alloc_strict["plan"], indent=2), encoding="utf-8")(work_dir / "sdq_dynamic_strict.json").write_text(json.dumps(alloc_strict["dynamic"], indent=2), encoding="utf-8")print(    f"Allocated. Estimated extra MB -> "    f"flex: {alloc_flex['used_extra_mb']:.2f}, "    f"strict: {alloc_strict['used_extra_mb']:.2f}")print("\n--- STAGE 3A: Quantize baseline_gptq ---")args.out_dir = str(work_dir / "models" / "baseline_gptq")run_quantize_gptqmodel(args, dynamic_config=None)print("\n--- STAGE 3B: Quantize sdq_gptq ---")args.out_dir = str(work_dir / "models" / "sdq_gptq")run_quantize_gptqmodel(args, dynamic_config=alloc_flex["dynamic"])print("\n--- STAGE 3C: Quantize sdq_autoround_auto_gptq ---")args.out_dir = str(work_dir / "models" / "sdq_autoround_auto_gptq")run_quantize_autoround(args, plan=alloc_strict["plan"], export_format="auto_gptq")print("\n--- STAGE 3D: Quantize sdq_autoround_auto_round ---")args.out_dir = str(work_dir / "models" / "sdq_autoround_auto_round")run_quantize_autoround(args, plan=alloc_flex["plan"], export_format="auto_round")print("\n--- STAGE 4: Evaluation Matrix ---")run_eval(    str(work_dir / "models" / "baseline_gptq"),    args.tasks,    str(work_dir / "eval" / "baseline_gptq.json"),)run_eval(    str(work_dir / "models" / "sdq_gptq"),    args.tasks,    str(work_dir / "eval" / "sdq_gptq.json"),)run_eval(    str(work_dir / "models" / "sdq_autoround_auto_gptq"),    args.tasks,    str(work_dir / "eval" / "sdq_autoround_auto_gptq.json"),)run_eval(    str(work_dir / "models" / "sdq_autoround_auto_round"),    args.tasks,    str(work_dir / "eval" / "sdq_autoround_auto_round.json"),)print("\n--- Pipeline Complete. See eval/ and models/*/export_info.json ---")
suite = sub.add_parser("suite")suite.add_argument("--model-id", required=True)suite.add_argument("--work-dir", required=True)suite.add_argument("--budget-mb-extra", type=float, default=250.0)suite.add_argument("--include-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head")suite.add_argument("--dataset-name", default="wikitext")suite.add_argument("--dataset-config", default="wikitext-2-raw-v1")suite.add_argument("--split", default="train")suite.add_argument("--text-field", default="text")suite.add_argument("--num-texts", type=int, default=128)suite.add_argument("--max-length", type=int, default=2048)suite.add_argument("--batch-size", type=int, default=4)suite.add_argument("--quant-num-texts", type=int, default=128)suite.add_argument("--quant-batch-size", type=int, default=4)suite.add_argument("--autoround-iters", type=int, default=200, help="Iterations for AutoRound SignSGD")suite.add_argument("--tasks", default="arc_challenge,hellaswag")args = parser.parse_args()work_dir = Path(args.work_dir)work_dir.mkdir(parents=True, exist_ok=True)print("\n--- STAGE 1: Hessian Proxy Measurement ---")args.out_dir = str(work_dir / "probe")measure_hessian_traces(args)features_path = str(work_dir / "probe" / "hessian_features.json")print("\n--- STAGE 2: Lagrangian Proxy Allocation ---")# Flex plan:# - GPTQModel mixed dynamic path# - AutoRound native format pathalloc_flex = solve_proxy_allocation(    features_path,    args.budget_mb_extra,    allow_quantized_lm_head=True,)(work_dir / "sdq_plan_flex.json").write_text(json.dumps(alloc_flex["plan"], indent=2), encoding="utf-8")(work_dir / "sdq_dynamic_flex.json").write_text(json.dumps(alloc_flex["dynamic"], indent=2), encoding="utf-8")# Strict plan:# - auto_gptq export pathalloc_strict = solve_proxy_allocation(    features_path,    args.budget_mb_extra,    allow_quantized_lm_head=False,)(work_dir / "sdq_plan_strict.json").write_text(json.dumps(alloc_strict["plan"], indent=2), encoding="utf-8")(work_dir / "sdq_dynamic_strict.json").write_text(json.dumps(alloc_strict["dynamic"], indent=2), encoding="utf-8")print(    f"Allocated. Estimated extra MB -> "    f"flex: {alloc_flex['used_extra_mb']:.2f}, "    f"strict: {alloc_strict['used_extra_mb']:.2f}")print("\n--- STAGE 3A: Quantize baseline_gptq ---")args.out_dir = str(work_dir / "models" / "baseline_gptq")run_quantize_gptqmodel(args, dynamic_config=None)print("\n--- STAGE 3B: Quantize sdq_gptq ---")args.out_dir = str(work_dir / "models" / "sdq_gptq")run_quantize_gptqmodel(args, dynamic_config=alloc_flex["dynamic"])print("\n--- STAGE 3C: Quantize sdq_autoround_auto_gptq ---")args.out_dir = str(work_dir / "models" / "sdq_autoround_auto_gptq")run_quantize_autoround(args, plan=alloc_strict["plan"], export_format="auto_gptq")print("\n--- STAGE 3D: Quantize sdq_autoround_auto_round ---")args.out_dir = str(work_dir / "models" / "sdq_autoround_auto_round")run_quantize_autoround(args, plan=alloc_flex["plan"], export_format="auto_round")print("\n--- STAGE 4: Evaluation Matrix ---")run_eval(    str(work_dir / "models" / "baseline_gptq"),    args.tasks,    str(work_dir / "eval" / "baseline_gptq.json"),)run_eval(    str(work_dir / "models" / "sdq_gptq"),    args.tasks,    str(work_dir / "eval" / "sdq_gptq.json"),)run_eval(    str(work_dir / "models" / "sdq_autoround_auto_gptq"),    args.tasks,    str(work_dir / "eval" / "sdq_autoround_auto_gptq.json"),)run_eval(    str(work_dir / "models" / "sdq_autoround_auto_round"),    args.tasks,    str(work_dir / "eval" / "sdq_autoround_auto_round.json"),)print("\n--- Pipeline Complete. See eval/ and models/*/export_info.json ---")
```

if **name** == "**main**":
main()
EOF

echo
echo "Project created at: $(pwd)"
echo
echo "Next steps:"
echo "python -m venv .venv"
echo "source .venv/bin/activate"
echo "# install your CUDA-matching torch first"
echo "pip install -r requirements.txt"
echo
echo "python -m quant_v5_3.pipeline suite \"
echo "  --model-id Qwen/Qwen3.5-4B-Base \"
echo "  --work-dir ./runs/v5_3_suite \"
echo "  --budget-mb-extra 200 \"
echo "  --autoround-iters 50 \"
echo "  --tasks arc_challenge,hellaswag"

```Then run:```bashbash create_sdq_v53.shcd sdq_quant_v5_3python -m venv .venvsource .venv/bin/activate# install your CUDA-matching torch firstpip install -r requirements.txtpython -m quant_v5_3.pipeline suite \  --model-id Qwen/Qwen3.5-4B-Base \  --work-dir ./runs/v5_3_suite \  --budget-mb-extra 200 \  --autoround-iters 50 \  --tasks arc_challenge,hellaswag
Then run:```bashbash create_sdq_v53.shcd sdq_quant_v5_3python -m venv .venvsource .venv/bin/activate# install your CUDA-matching torch firstpip install -r requirements.txtpython -m quant_v5_3.pipeline suite \  --model-id Qwen/Qwen3.5-4B-Base \  --work-dir ./runs/v5_3_suite \  --budget-mb-extra 200 \  --autoround-iters 50 \  --tasks arc_challenge,hellaswag
```

---

### User:
The script you pasted is architecturally correct, but **Markdown parsing destroyed the Python syntax** during the copy-paste process. All of the dunder methods (`__init__`, `__main__`, `__future__`) were converted to bold text (`**init**`, etc.), and almost all of the Python indentation was stripped out. If you run that script as-is, Python will instantly crash with `SyntaxError` and `IndentationError`.

Additionally, I have added explicit garbage collection (`import gc`) between the matrix runs. Loading and unloading 4 different LLMs sequentially will cause a CUDA Out-Of-Memory (OOM) error if PyTorch doesn't cleanly release the VRAM between the `AutoRound` and `GPTQModel` runs. 

Here is the **perfect, 100% syntactically valid, copy-pasteable version** of `create_sdq_v53.sh`. 

Save this block exactly as `create_sdq_v53.sh` and run it:

```
bash
#!/usr/bin/env bash
set -euo pipefail

mkdir -p sdq_quant_v5_3/quant_v5_3
cd sdq_quant_v5_3

cat > README.md <<'EOF'
# SDQ Quant v5.3 (Hessian-Proxy + AutoRound + Format Policies)

A deterministic Software-Defined Quantization pipeline targeting standard GPTQ-compatible 4/8/skip module paths intended for GPTQModel / vLLM deployment, plus best-effort AutoRound export paths.

## Design

1. Measurement:
   - Computes `trace(X^T X)` on `cuda:0` as a proxy for second-order sensitivity.

2. Allocation:
   - Uses a Lagrangian solver under a planner memory budget.
   - `q_proj`, `k_proj`, `v_proj` are locked to `w4_g128` to preserve QKV fusion.
   - For `auto_gptq`, `lm_head` is forced to `skip`.
   - For `GPTQModel` and `auto_round`, `lm_head` can choose `w4_g128`, `w8_g128`, or `skip`.

3. Quantization matrix:
   - `baseline_gptq`: GPTQModel, plain 4-bit
   - `sdq_gptq`: GPTQModel, Hessian-proxy allocation
   - `sdq_autoround_auto_gptq`: AutoRound with strict `lm_head` policy
   - `sdq_autoround_auto_round`: AutoRound with flex `lm_head` policy

4. Evaluation:
   - Uses `lm_eval.simple_evaluate()`.

## Notes

- The memory budget is a planner estimate, not an exact runtime VRAM guarantee.
- The error model is a Hessian proxy, not a full GPTQ inverse-Hessian simulation.
- `auto_gptq` mixed-bit export is best-effort.
- If `auto_gptq` export or downstream loading fails, this code falls back to `auto_round` and records the actual export format in `export_info.json`.
EOF

cat > requirements.txt <<'EOF'
transformers
datasets
accelerate
torch
gptqmodel
auto-round
lm-eval
tqdm
EOF

cat > quant_v5_3/__init__.py <<'EOF'
"""SDQ Quant v5.3: Hessian-Proxy Allocator + AutoRound Format Policies"""
EOF

cat > quant_v5_3/actions.py <<'EOF'
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass(frozen=True)
class ActionSpec:
    name: str
    bits: int
    group_size: Optional[int]
    skip: bool

ACTION_SPECS: Dict[str, ActionSpec] = {
    "w4_g128": ActionSpec("w4_g128", bits=4, group_size=128, skip=False),
    "w8_g128": ActionSpec("w8_g128", bits=8, group_size=128, skip=False),
    "skip": ActionSpec("skip", bits=16, group_size=None, skip=True),
}

BASE_ACTION = "w4_g128"

def bytes_per_param(action: str) -> float:
    """Planner memory estimate, not an exact runtime VRAM model."""
    if action == "skip":
        return 2.0  # FP16

    spec = ACTION_SPECS[action]
    base = spec.bits / 8.0
    overhead = 2.5 / spec.group_size
    return base + overhead

def action_to_override(action: str) -> dict:
    if action == "skip":
        return {}
    spec = ACTION_SPECS[action]
    return {"bits": spec.bits, "group_size": spec.group_size}
EOF

cat > quant_v5_3/measurement.py <<'EOF'
from __future__ import annotations
import json
from pathlib import Path
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_calibration_texts(ds_name: str, ds_config: str, split: str, text_field: str, num_texts: int):
    ds = load_dataset(ds_name, ds_config, split=split)
    texts =[]
    for row in ds:
        text = row.get(text_field)
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
        if len(texts) >= num_texts:
            break
    return texts

def measure_hessian_traces(args):
    """
    Computes a Hessian proxy:
    Trace(H) ≈ mean over calibration rows of sum(x^2)
    collected from each target linear layer input.
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    texts = load_calibration_texts(
        args.dataset_name,
        args.dataset_config,
        args.split,
        args.text_field,
        args.num_texts,
    )

    include_mods =[x.strip() for x in args.include_modules.split(",") if x.strip()]
    targets =[
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear) and any(tag in n for tag in include_mods)
    ]

    hessian_traces = {name: 0.0 for name, _ in targets}
    row_counts = {name: 0 for name, _ in targets}
    params_count = {name: m.weight.numel() for name, m in targets}

    handles =[]

    def make_hook(name):
        def hook(_, inp, __):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            hessian_traces[name] += (x ** 2).sum().item()
            row_counts[name] += x.shape[0]
        return hook

    for name, module in targets:
        handles.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), args.batch_size), desc="Hessian Proxy Measurement"):
            batch_texts = texts[i : i + args.batch_size]
            batch = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
                padding=True,
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)

    for h in handles:
        h.remove()

    features =[]
    for name, _ in targets:
        avg_trace = hessian_traces[name] / max(1, row_counts[name])
        features.append(
            {
                "name": name,
                "params": params_count[name],
                "hessian_trace": avg_trace,
            }
        )

    out_file = out_dir / "hessian_features.json"
    out_file.write_text(json.dumps(features, indent=2), encoding="utf-8")
    print(f"Saved {len(features)} layer traces to {out_file}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return features
EOF

cat > quant_v5_3/allocator.py <<'EOF'
from __future__ import annotations
import json
import re
from pathlib import Path
from .actions import ACTION_SPECS, BASE_ACTION, action_to_override, bytes_per_param

def get_valid_actions(module_name: str, allow_quantized_lm_head: bool) -> list[str]:
    """
    Policy layer:
    - QKV are locked to base action to preserve fused QKV paths.
    - lm_head policy depends on target export format.
    - others can move between 4-bit and 8-bit.
    """
    if any(k in module_name for k in["q_proj", "k_proj", "v_proj"]):
        return [BASE_ACTION]
    if "lm_head" in module_name:
        return ["w4_g128", "w8_g128", "skip"] if allow_quantized_lm_head else ["skip"]
    return["w4_g128", "w8_g128"]

def solve_proxy_allocation(features_path: str, budget_mb_extra: float, allow_quantized_lm_head: bool):
    """Lagrangian multiplier solver over a proxy error model."""
    features = json.loads(Path(features_path).read_text(encoding="utf-8"))

    base_bpp = bytes_per_param(BASE_ACTION)
    total_base_mb = sum(f["params"] * base_bpp for f in features) / (1024 ** 2)
    target_mb = total_base_mb + budget_mb_extra

    def proxy_error(trace: float, action: str) -> float:
        if action == "skip":
            return 0.0
        bits = ACTION_SPECS[action].bits
        return trace * (2.0 ** (-2.0 * bits))

    def action_cost_mb(params: int, action: str) -> float:
        return (params * bytes_per_param(action)) / (1024 ** 2)

    lambda_low, lambda_high = 0.0, 1e12
    best_plan = None

    for _ in range(80):
        lam = (lambda_low + lambda_high) / 2.0
        current_mb = 0.0
        plan =[]

        for f in features:
            valid_actions = get_valid_actions(f["name"], allow_quantized_lm_head)

            best_action = None
            best_obj = float("inf")

            for act in valid_actions:
                err = proxy_error(f["hessian_trace"], act)
                size_mb = action_cost_mb(f["params"], act)
                obj = err + lam * size_mb
                if obj < best_obj:
                    best_obj = obj
                    best_action = act

            size_mb = action_cost_mb(f["params"], best_action)
            plan.append(
                {
                    "name": f["name"],
                    "action": best_action,
                    "params": f["params"],
                    "hessian_trace": f["hessian_trace"],
                    "size_mb": size_mb,
                }
            )
            current_mb += size_mb

        best_plan = plan
        if current_mb > target_mb:
            lambda_low = lam
        else:
            lambda_high = lam

    used_extra_mb = sum(p["size_mb"] for p in best_plan) - total_base_mb

    dynamic = {}
    for p in best_plan:
        if p["action"] == BASE_ACTION:
            continue
        regex = f"^{re.escape(p['name'])}$"
        if p["action"] == "skip":
            dynamic[f"-:{regex}"] = {}
        else:
            dynamic[f"+:{regex}"] = action_to_override(p["action"])

    return {
        "plan": best_plan,
        "dynamic": dynamic,
        "used_extra_mb": used_extra_mb,
    }
EOF

cat > quant_v5_3/runner.py <<'EOF'
from __future__ import annotations
import gc
import json
from pathlib import Path
import lm_eval
import torch
from auto_round import AutoRound
from gptqmodel import GPTQModel, QuantizeConfig
from .measurement import load_calibration_texts

def run_quantize_gptqmodel(args, dynamic_config):
    """Standard greedy GPTQ rounding via GPTQModel."""
    texts = load_calibration_texts(
        args.dataset_name,
        args.dataset_config,
        args.split,
        args.text_field,
        args.quant_num_texts,
    )

    qcfg = QuantizeConfig(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=True,
        dynamic=dynamic_config,
    )

    print("Loading model for GPTQModel quantization...")
    model = GPTQModel.load(args.model_id, qcfg)
    model.quantize(texts, batch_size=args.quant_batch_size)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _plan_has_quantized_lm_head(plan) -> bool:
    if not plan:
        return False
    for p in plan:
        if "lm_head" in p["name"] and p["action"] in ("w4_g128", "w8_g128"):
            return True
    return False

def run_quantize_autoround(args, plan, export_format: str):
    """
    AutoRound SignSGD rounding with documented quantize_and_save() API.
    Behavior:
    - For auto_gptq: lm_head must not be quantized.
    - If auto_gptq export/load constraints are violated, we fallback to auto_round.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    texts = load_calibration_texts(
        args.dataset_name,
        args.dataset_config,
        args.split,
        args.text_field,
        args.quant_num_texts,
    )

    layer_config = {}
    if plan is not None:
        for p in plan:
            if p["action"] == "skip":
                layer_config[p["name"]] = {"bits": 16}
            elif p["action"] != "w4_g128":
                layer_config[p["name"]] = {"bits": 8, "group_size": 128}

    requested_format = export_format
    actual_format = export_format

    if requested_format == "auto_gptq" and _plan_has_quantized_lm_head(plan):
        print("[warn] quantized lm_head is not valid for auto_gptq here; switching export format to auto_round.")
        actual_format = "auto_round"

    print(f"Loading model for AutoRound ({requested_format} requested)...")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        torch_dtype=torch.float16, 
        trust_remote_code=True, 
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    ar = AutoRound(
        model=model,
        tokenizer=tokenizer,
        bits=4,
        group_size=128,
        sym=True,
        dataset=texts,
        seqlen=args.max_length,
        batch_size=args.quant_batch_size,
        layer_config=layer_config,
        iters=args.autoround_iters,
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Executing AutoRound and saving via quantize_and_save(format='{actual_format}')...")
        ar.quantize_and_save(str(out), format=actual_format)
    except Exception as e:
        if actual_format == "auto_gptq":
            print(f"[warn] auto_gptq export failed: {e}")
            print("[warn] retrying with auto_round format...")
            actual_format = "auto_round"
            ar.quantize_and_save(str(out), format=actual_format)
        else:
            raise

    export_info = {
        "requested_format": requested_format,
        "actual_format": actual_format,
    }
    (out / "export_info.json").write_text(json.dumps(export_info, indent=2), encoding="utf-8")

    del ar
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_eval(model_path: str, tasks: str, out_file: str):
    """Programmatic lm-eval."""
    print(f"Running programmatic lm-eval on {model_path}...")
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)

    task_list =[t.strip() for t in tasks.split(",") if t.strip()]

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},trust_remote_code=True",
        tasks=task_list,
        batch_size="auto",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    serializable_results = {
        "results": results.get("results", {}),
        "versions": results.get("versions", {}),
    }
    Path(out_file).write_text(json.dumps(serializable_results, indent=2), encoding="utf-8")
    return serializable_results
EOF

cat > quant_v5_3/pipeline.py <<'EOF'
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .allocator import solve_proxy_allocation
from .measurement import measure_hessian_traces
from .runner import run_eval, run_quantize_autoround, run_quantize_gptqmodel

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    suite = sub.add_parser("suite")
    suite.add_argument("--model-id", required=True)
    suite.add_argument("--work-dir", required=True)
    suite.add_argument("--budget-mb-extra", type=float, default=250.0)
    suite.add_argument("--include-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head")
    suite.add_argument("--dataset-name", default="wikitext")
    suite.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    suite.add_argument("--split", default="train")
    suite.add_argument("--text-field", default="text")
    suite.add_argument("--num-texts", type=int, default=128)
    suite.add_argument("--max-length", type=int, default=2048)
    suite.add_argument("--batch-size", type=int, default=4)
    suite.add_argument("--quant-num-texts", type=int, default=128)
    suite.add_argument("--quant-batch-size", type=int, default=4)
    suite.add_argument("--autoround-iters", type=int, default=200, help="Iterations for AutoRound SignSGD")
    suite.add_argument("--tasks", default="arc_challenge,hellaswag")

    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- STAGE 1: Hessian Proxy Measurement ---")
    args.out_dir = str(work_dir / "probe")
    measure_hessian_traces(args)

    features_path = str(work_dir / "probe" / "hessian_features.json")

    print("\n--- STAGE 2: Lagrangian Proxy Allocation ---")

    # Flex plan: GPTQModel mixed dynamic path & AutoRound native format path
    alloc_flex = solve_proxy_allocation(
        features_path,
        args.budget_mb_extra,
        allow_quantized_lm_head=True,
    )
    (work_dir / "sdq_plan_flex.json").write_text(json.dumps(alloc_flex["plan"], indent=2), encoding="utf-8")
    (work_dir / "sdq_dynamic_flex.json").write_text(json.dumps(alloc_flex["dynamic"], indent=2), encoding="utf-8")

    # Strict plan: auto_gptq export path
    alloc_strict = solve_proxy_allocation(
        features_path,
        args.budget_mb_extra,
        allow_quantized_lm_head=False,
    )
    (work_dir / "sdq_plan_strict.json").write_text(json.dumps(alloc_strict["plan"], indent=2), encoding="utf-8")
    (work_dir / "sdq_dynamic_strict.json").write_text(json.dumps(alloc_strict["dynamic"], indent=2), encoding="utf-8")

    print(
        f"Allocated. Estimated extra MB -> "
        f"flex: {alloc_flex['used_extra_mb']:.2f}, "
        f"strict: {alloc_strict['used_extra_mb']:.2f}"
    )

    print("\n--- STAGE 3A: Quantize baseline_gptq ---")
    args.out_dir = str(work_dir / "models" / "baseline_gptq")
    run_quantize_gptqmodel(args, dynamic_config=None)

    print("\n--- STAGE 3B: Quantize sdq_gptq ---")
    args.out_dir = str(work_dir / "models" / "sdq_gptq")
    run_quantize_gptqmodel(args, dynamic_config=alloc_flex["dynamic"])

    print("\n--- STAGE 3C: Quantize sdq_autoround_auto_gptq ---")
    args.out_dir = str(work_dir / "models" / "sdq_autoround_auto_gptq")
    run_quantize_autoround(args, plan=alloc_strict["plan"], export_format="auto_gptq")

    print("\n--- STAGE 3D: Quantize sdq_autoround_auto_round ---")
    args.out_dir = str(work_dir / "models" / "sdq_autoround_auto_round")
    run_quantize_autoround(args, plan=alloc_flex["plan"], export_format="auto_round")

    print("\n--- STAGE 4: Evaluation Matrix ---")
    run_eval(
        str(work_dir / "models" / "baseline_gptq"),
        args.tasks,
        str(work_dir / "eval" / "baseline_gptq.json"),
    )
    run_eval(
        str(work_dir / "models" / "sdq_gptq"),
        args.tasks,
        str(work_dir / "eval" / "sdq_gptq.json"),
    )
    run_eval(
        str(work_dir / "models" / "sdq_autoround_auto_gptq"),
        args.tasks,
        str(work_dir / "eval" / "sdq_autoround_auto_gptq.json"),
    )
    run_eval(
        str(work_dir / "models" / "sdq_autoround_auto_round"),
        args.tasks,
        str(work_dir / "eval" / "sdq_autoround_auto_round.json"),
    )

    print("\n--- Pipeline Complete. See eval/ and models/*/export_info.json ---")

if __name__ == "__main__":
    main()
EOF

echo
echo "Project created at: $(pwd)"
echo
echo "Next steps:"
echo "python -m venv .venv"
echo "source .venv/bin/activate"
echo "# install your CUDA-matching torch first"
echo "pip install -r requirements.txt"
echo
echo "python -m quant_v5_3.pipeline suite \\"
echo "  --model-id Qwen/Qwen3.5-4B-Base \\"
echo "  --work-dir ./runs/v5_3_suite \\"
echo "  --budget-mb-extra 200 \\"
echo "  --autoround-iters 50 \\"
echo "  --tasks arc_challenge,hellaswag"
```

---

### ChatGPT:
Yes — this version is copy-pasteable and the Python syntax is intact.

Your fixes are the right ones:

- the dunder names are preserved,
- indentation is valid,
- `gc.collect()` plus `torch.cuda.empty_cache()` between matrix stages is a good addition,
- and the `auto_gptq` / `auto_round` split is now handled in code instead of just in comments.

I’d save exactly that block as `create_sdq_v53.sh`, then run:

```Bashbash create_sdq_v53.shcd sdq_quant_v5_3python -m venv .venvsource .venv/bin/activate# install your CUDA-matching torch firstpip install -r requirements.txtpython -m quant_v5_3.pipeline suite \  --model-id Qwen/Qwen3.5-4B-Base \  --work-dir ./runs/v5_3_suite \  --budget-mb-extra 200 \  --autoround-iters 50 \  --tasks arc_challenge,hellaswag
Bashbash create_sdq_v53.shcd sdq_quant_v5_3python -m venv .venvsource .venv/bin/activate# install your CUDA-matching torch firstpip install -r requirements.txtpython -m quant_v5_3.pipeline suite \  --model-id Qwen/Qwen3.5-4B-Base \  --work-dir ./runs/v5_3_suite \  --budget-mb-extra 200 \  --autoround-iters 50 \  --tasks arc_challenge,hellaswag
```

Two practical notes before you launch it:

`AutoRound` is the heaviest arm by far, so for a first smoke test I’d reduce pressure:

```Bashpython -m quant_v5_3.pipeline suite \  --model-id Qwen/Qwen3.5-4B-Base \  --work-dir ./runs/v5_3_smoke \  --budget-mb-extra 100 \  --num-texts 32 \  --quant-num-texts 32 \  --max-length 512 \  --batch-size 2 \  --quant-batch-size 2 \  --autoround-iters 20 \  --tasks hellaswag
Bashpython -m quant_v5_3.pipeline suite \  --model-id Qwen/Qwen3.5-4B-Base \  --work-dir ./runs/v5_3_smoke \  --budget-mb-extra 100 \  --num-texts 32 \  --quant-num-texts 32 \  --max-length 512 \  --batch-size 2 \  --quant-batch-size 2 \  --autoround-iters 20 \  --tasks hellaswag
```

And if the `sdq_autoround_auto_gptq` leg fails or evaluates badly, check:

- `runs/v5_3_suite/models/sdq_autoround_auto_gptq/export_info.json`
- `runs/v5_3_suite/models/sdq_autoround_auto_round/export_info.json`

That will tell you which format actually got written.

If you hit any error when running the suite, paste the traceback and I’ll patch the script directly.

---

### User:
This has been an absolute masterclass in architectural review. Thank you for pushing this pipeline from a flawed LLM-based heuristic script (v4) into a mathematically rigorous, format-compliant, SOTA quantization engine (v5.3). 

The distinction you drew between **format-level innovations** (EXL2 mixed intra-tensor bits, EXL3 QTIP codebooks) and **algorithmic optimization** (Hessian-proxy Lagrangian solvers, AutoRound SignSGD) is exactly what makes this final pipeline so powerful. It captures the maximum possible algorithmic quality without breaking the standard deployment boundaries of `vLLM` and `GPTQModel`.

Your smoke test parameters are perfect for verifying the CUDA memory footprint and the format fallback logic without burning hours of GPU time.

```
bash
# The ultimate smoke test command
python -m quant_v5_3.pipeline suite \
  --model-id Qwen/Qwen3.5-4B-Base \
  --work-dir ./runs/v5_3_smoke \
  --budget-mb-extra 100 \
  --num-texts 32 \
  --quant-num-texts 32 \
  --max-length 512 \
  --batch-size 2 \
  --quant-batch-size 2 \
  --autoround-iters 20 \
  --tasks hellaswag
```

The `export_info.json` trick is fully implemented and will immediately flag if `auto-gptq` throws a fit over the mixed-bit mapping, letting us know exactly when the pipeline had to dynamically shift to the native `auto_round` format.

I'm taking this script to the cluster now. If any of the backend kernel loaders or LM-Eval hooks complain about the exported artifacts, I'll bring the tracebacks straight back here. 

Thanks again for the incredible back-and-forth! ... provide the full code please

---

### ChatGPT:
Save this exactly as `create_sdq_v53.sh` and run it.

```Bash#!/usr/bin/env bashset -euo pipefailmkdir -p sdq_quant_v5_3/quant_v5_3cd sdq_quant_v5_3cat > README.md <<'EOF'# SDQ Quant v5.3 (Hessian-Proxy + AutoRound + Format Policies)A deterministic Software-Defined Quantization pipeline targeting standard GPTQ-compatible 4/8/skip module paths intended for GPTQModel / vLLM deployment, plus best-effort AutoRound export paths.## Design1. Measurement:   - Computes `trace(X^T X)` on `cuda:0` as a proxy for second-order sensitivity.2. Allocation:   - Uses a Lagrangian solver under a planner memory budget.   - `q_proj`, `k_proj`, `v_proj` are locked to `w4_g128` to preserve QKV fusion.   - For `auto_gptq`, `lm_head` is forced to `skip`.   - For `GPTQModel` and `auto_round`, `lm_head` can choose `w4_g128`, `w8_g128`, or `skip`.3. Quantization matrix:   - `baseline_gptq`: GPTQModel, plain 4-bit   - `sdq_gptq`: GPTQModel, Hessian-proxy allocation   - `sdq_autoround_auto_gptq`: AutoRound with strict `lm_head` policy   - `sdq_autoround_auto_round`: AutoRound with flex `lm_head` policy4. Evaluation:   - Uses `lm_eval.simple_evaluate()`.## Notes- The memory budget is a planner estimate, not an exact runtime VRAM guarantee.- The error model is a Hessian proxy, not a full GPTQ inverse-Hessian simulation.- `auto_gptq` mixed-bit export is best-effort.- If `auto_gptq` export or downstream loading fails, this code falls back to `auto_round` and records the actual export format in `export_info.json`.EOFcat > requirements.txt <<'EOF'transformersdatasetsacceleratetorchgptqmodelauto-roundlm-evaltqdmEOFcat > quant_v5_3/__init__.py <<'EOF'"""SDQ Quant v5.3: Hessian-Proxy Allocator + AutoRound Format Policies"""EOFcat > quant_v5_3/actions.py <<'EOF'from __future__ import annotationsfrom dataclasses import dataclassfrom typing import Dict, Optional@dataclass(frozen=True)class ActionSpec:    name: str    bits: int    group_size: Optional[int]    skip: boolACTION_SPECS: Dict[str, ActionSpec] = {    "w4_g128": ActionSpec("w4_g128", bits=4, group_size=128, skip=False),    "w8_g128": ActionSpec("w8_g128", bits=8, group_size=128, skip=False),    "skip": ActionSpec("skip", bits=16, group_size=None, skip=True),}BASE_ACTION = "w4_g128"def bytes_per_param(action: str) -> float:    """Planner memory estimate, not an exact runtime VRAM model."""    if action == "skip":        return 2.0  # FP16    spec = ACTION_SPECS[action]    base = spec.bits / 8.0    overhead = 2.5 / spec.group_size    return base + overheaddef action_to_override(action: str) -> dict:    if action == "skip":        return {}    spec = ACTION_SPECS[action]    return {"bits": spec.bits, "group_size": spec.group_size}EOFcat > quant_v5_3/measurement.py <<'EOF'from __future__ import annotationsimport jsonfrom pathlib import Pathimport torchfrom datasets import load_datasetfrom tqdm import tqdmfrom transformers import AutoModelForCausalLM, AutoTokenizerdef load_calibration_texts(ds_name: str, ds_config: str, split: str, text_field: str, num_texts: int):    ds = load_dataset(ds_name, ds_config, split=split)    texts = []    for row in ds:        text = row.get(text_field)        if isinstance(text, str) and text.strip():            texts.append(text.strip())        if len(texts) >= num_texts:            break    return textsdef measure_hessian_traces(args):    """    Computes a Hessian proxy:    Trace(H) ≈ mean over calibration rows of sum(x^2)    collected from each target linear layer input.    """    out_dir = Path(args.out_dir)    out_dir.mkdir(parents=True, exist_ok=True)    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)    if tokenizer.pad_token_id is None:        tokenizer.pad_token = tokenizer.eos_token    model = AutoModelForCausalLM.from_pretrained(        args.model_id,        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,        low_cpu_mem_usage=True,        trust_remote_code=True,    ).to(device)    model.eval()    texts = load_calibration_texts(        args.dataset_name,        args.dataset_config,        args.split,        args.text_field,        args.num_texts,    )    include_mods = [x.strip() for x in args.include_modules.split(",") if x.strip()]    targets = [        (n, m)        for n, m in model.named_modules()        if isinstance(m, torch.nn.Linear) and any(tag in n for tag in include_mods)    ]    hessian_traces = {name: 0.0 for name, _ in targets}    row_counts = {name: 0 for name, _ in targets}    params_count = {name: m.weight.numel() for name, m in targets}    handles = []    def make_hook(name):        def hook(_, inp, __):            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])            hessian_traces[name] += (x ** 2).sum().item()            row_counts[name] += x.shape[0]        return hook    for name, module in targets:        handles.append(module.register_forward_hook(make_hook(name)))    with torch.no_grad():        for i in tqdm(range(0, len(texts), args.batch_size), desc="Hessian Proxy Measurement"):            batch_texts = texts[i : i + args.batch_size]            batch = tokenizer(                batch_texts,                return_tensors="pt",                truncation=True,                max_length=args.max_length,                padding=True,            )            batch = {k: v.to(device) for k, v in batch.items()}            model(**batch)    for h in handles:        h.remove()    features = []    for name, _ in targets:        avg_trace = hessian_traces[name] / max(1, row_counts[name])        features.append(            {                "name": name,                "params": params_count[name],                "hessian_trace": avg_trace,            }        )    out_file = out_dir / "hessian_features.json"    out_file.write_text(json.dumps(features, indent=2), encoding="utf-8")    print(f"Saved {len(features)} layer traces to {out_file}")    del model    if torch.cuda.is_available():        torch.cuda.empty_cache()    return featuresEOFcat > quant_v5_3/allocator.py <<'EOF'from __future__ import annotationsimport jsonimport refrom pathlib import Pathfrom .actions import ACTION_SPECS, BASE_ACTION, action_to_override, bytes_per_paramdef get_valid_actions(module_name: str, allow_quantized_lm_head: bool) -> list[str]:    """    Policy layer:    - QKV are locked to base action to preserve fused QKV paths.    - lm_head policy depends on target export format.    - others can move between 4-bit and 8-bit.    """    if any(k in module_name for k in ["q_proj", "k_proj", "v_proj"]):        return [BASE_ACTION]    if "lm_head" in module_name:        return ["w4_g128", "w8_g128", "skip"] if allow_quantized_lm_head else ["skip"]    return ["w4_g128", "w8_g128"]def solve_proxy_allocation(features_path: str, budget_mb_extra: float, allow_quantized_lm_head: bool):    """Lagrangian multiplier solver over a proxy error model."""    features = json.loads(Path(features_path).read_text(encoding="utf-8"))    base_bpp = bytes_per_param(BASE_ACTION)    total_base_mb = sum(f["params"] * base_bpp for f in features) / (1024 ** 2)    target_mb = total_base_mb + budget_mb_extra    def proxy_error(trace: float, action: str) -> float:        if action == "skip":            return 0.0        bits = ACTION_SPECS[action].bits        return trace * (2.0 ** (-2.0 * bits))    def action_cost_mb(params: int, action: str) -> float:        return (params * bytes_per_param(action)) / (1024 ** 2)    lambda_low, lambda_high = 0.0, 1e12    best_plan = None    for _ in range(80):        lam = (lambda_low + lambda_high) / 2.0        current_mb = 0.0        plan = []        for f in features:            valid_actions = get_valid_actions(f["name"], allow_quantized_lm_head)            best_action = None            best_obj = float("inf")            for act in valid_actions:                err = proxy_error(f["hessian_trace"], act)                size_mb = action_cost_mb(f["params"], act)                obj = err + lam * size_mb                if obj < best_obj:                    best_obj = obj                    best_action = act            size_mb = action_cost_mb(f["params"], best_action)            plan.append(                {                    "name": f["name"],                    "action": best_action,                    "params": f["params"],                    "hessian_trace": f["hessian_trace"],                    "size_mb": size_mb,                }            )            current_mb += size_mb        best_plan = plan        if current_mb > target_mb:            lambda_low = lam        else:            lambda_high = lam    used_extra_mb = sum(p["size_mb"] for p in best_plan) - total_base_mb    dynamic = {}    for p in best_plan:        if p["action"] == BASE_ACTION:            continue        regex = f"^{re.escape(p['name'])}$"        if p["action"] == "skip":            dynamic[f"-:{regex}"] = {}        else:            dynamic[f"+:{regex}"] = action_to_override(p["action"])    return {        "plan": best_plan,        "dynamic": dynamic,        "used_extra_mb": used_extra_mb,    }EOFcat > quant_v5_3/runner.py <<'EOF'from __future__ import annotationsimport gcimport jsonfrom pathlib import Pathimport lm_evalimport torchfrom auto_round import AutoRoundfrom gptqmodel import GPTQModel, QuantizeConfigfrom .measurement import load_calibration_textsdef run_quantize_gptqmodel(args, dynamic_config):    """Standard greedy GPTQ rounding via GPTQModel."""    texts = load_calibration_texts(        args.dataset_name,        args.dataset_config,        args.split,        args.text_field,        args.quant_num_texts,    )    qcfg = QuantizeConfig(        bits=4,        group_size=128,        sym=True,        desc_act=True,        dynamic=dynamic_config,    )    print("Loading model for GPTQModel quantization...")    model = GPTQModel.load(args.model_id, qcfg)    model.quantize(texts, batch_size=args.quant_batch_size)    out = Path(args.out_dir)    out.mkdir(parents=True, exist_ok=True)    model.save(str(out))    del model    gc.collect()    if torch.cuda.is_available():        torch.cuda.empty_cache()def _plan_has_quantized_lm_head(plan) -> bool:    if not plan:        return False    for p in plan:        if "lm_head" in p["name"] and p["action"] in ("w4_g128", "w8_g128"):            return True    return Falsedef run_quantize_autoround(args, plan, export_format: str):    """    AutoRound SignSGD rounding with documented quantize_and_save() API.    Behavior:    - For auto_gptq: lm_head must not be quantized.    - If auto_gptq export/load constraints are violated, we fallback to auto_round.    """    from transformers import AutoModelForCausalLM, AutoTokenizer    texts = load_calibration_texts(        args.dataset_name,        args.dataset_config,        args.split,        args.text_field,        args.quant_num_texts,    )    layer_config = {}    if plan is not None:        for p in plan:            if p["action"] == "skip":                layer_config[p["name"]] = {"bits": 16}            elif p["action"] != "w4_g128":                layer_config[p["name"]] = {"bits": 8, "group_size": 128}    requested_format = export_format    actual_format = export_format    if requested_format == "auto_gptq" and _plan_has_quantized_lm_head(plan):        print("[warn] quantized lm_head is not valid for auto_gptq here; switching export format to auto_round.")        actual_format = "auto_round"    print(f"Loading model for AutoRound ({requested_format} requested)...")    model = AutoModelForCausalLM.from_pretrained(        args.model_id,        torch_dtype=torch.float16,        trust_remote_code=True,        device_map="auto"    )    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)    ar = AutoRound(        model=model,        tokenizer=tokenizer,        bits=4,        group_size=128,        sym=True,        dataset=texts,        seqlen=args.max_length,        batch_size=args.quant_batch_size,        layer_config=layer_config,        iters=args.autoround_iters,    )    out = Path(args.out_dir)    out.mkdir(parents=True, exist_ok=True)    try:        print(f"Executing AutoRound and saving via quantize_and_save(format='{actual_format}')...")        ar.quantize_and_save(str(out), format=actual_format)    except Exception as e:        if actual_format == "auto_gptq":            print(f"[warn] auto_gptq export failed: {e}")            print("[warn] retrying with auto_round format...")            actual_format = "auto_round"            ar.quantize_and_save(str(out), format=actual_format)        else:            raise    export_info = {        "requested_format": requested_format,        "actual_format": actual_format,    }    (out / "export_info.json").write_text(json.dumps(export_info, indent=2), encoding="utf-8")    del ar    del model    del tokenizer    gc.collect()    if torch.cuda.is_available():        torch.cuda.empty_cache()def run_eval(model_path: str, tasks: str, out_file: str):    """Programmatic lm-eval."""    print(f"Running programmatic lm-eval on {model_path}...")    Path(out_file).parent.mkdir(parents=True, exist_ok=True)    task_list = [t.strip() for t in tasks.split(",") if t.strip()]    results = lm_eval.simple_evaluate(        model="hf",        model_args=f"pretrained={model_path},trust_remote_code=True",        tasks=task_list,        batch_size="auto",        device="cuda:0" if torch.cuda.is_available() else "cpu",    )    serializable_results = {        "results": results.get("results", {}),        "versions": results.get("versions", {}),    }    Path(out_file).write_text(json.dumps(serializable_results, indent=2), encoding="utf-8")    return serializable_resultsEOFcat > quant_v5_3/pipeline.py <<'EOF'from __future__ import annotationsimport argparseimport jsonfrom pathlib import Pathfrom .allocator import solve_proxy_allocationfrom .measurement import measure_hessian_tracesfrom .runner import run_eval, run_quantize_autoround, run_quantize_gptqmodeldef main():    parser = argparse.ArgumentParser()    sub = parser.add_subparsers(dest="cmd", required=True)    suite = sub.add_parser("suite")    suite.add_argument("--model-id", required=True)    suite.add_argument("--work-dir", required=True)    suite.add_argument("--budget-mb-extra", type=float, default=250.0)    suite.add_argument("--include-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head")    suite.add_argument("--dataset-name", default="wikitext")    suite.add_argument("--dataset-config", default="wikitext-2-raw-v1")    suite.add_argument("--split", default="train")    suite.add_argument("--text-field", default="text")    suite.add_argument("--num-texts", type=int, default=128)    suite.add_argument("--max-length", type=int, default=2048)    suite.add_argument("--batch-size", type=int, default=4)    suite.add_argument("--quant-num-texts", type=int, default=128)    suite.add_argument("--quant-batch-size", type=int, default=4)    suite.add_argument("--autoround-iters", type=int, default=200, help="Iterations for AutoRound SignSGD")    suite.add_argument("--tasks", default="arc_challenge,hellaswag")    args = parser.parse_args()    work_dir = Path(args.work_dir)    work_dir.mkdir(parents=True, exist_ok=True)    print("\n--- STAGE 1: Hessian Proxy Measurement ---")    args.out_dir = str(work_dir / "probe")    measure_hessian_traces(args)    features_path = str(work_dir / "probe" / "hessian_features.json")    print("\n--- STAGE 2: Lagrangian Proxy Allocation ---")    # Flex plan: GPTQModel mixed dynamic path & AutoRound native format path    alloc_flex = solve_proxy_allocation(        features_path,        args.budget_mb_extra,        allow_quantized_lm_head=True,    )    (work_dir / "sdq_plan_flex.json").write_text(json.dumps(alloc_flex["plan"], indent=2), encoding="utf-8")    (work_dir / "sdq_dynamic_flex.json").write_text(json.dumps(alloc_flex["dynamic"], indent=2), encoding="utf-8")    # Strict plan: auto_gptq export path    alloc_strict = solve_proxy_allocation(        features_path,        args.budget_mb_extra,        allow_quantized_lm_head=False,    )    (work_dir / "sdq_plan_strict.json").write_text(json.dumps(alloc_strict["plan"], indent=2), encoding="utf-8")    (work_dir / "sdq_dynamic_strict.json").write_text(json.dumps(alloc_strict["dynamic"], indent=2), encoding="utf-8")    print(        f"Allocated. Estimated extra MB -> "        f"flex: {alloc_flex['used_extra_mb']:.2f}, "        f"strict: {alloc_strict['used_extra_mb']:.2f}"    )    print("\n--- STAGE 3A: Quantize baseline_gptq ---")    args.out_dir = str(work_dir / "models" / "baseline_gptq")    run_quantize_gptqmodel(args, dynamic_config=None)    print("\n--- STAGE 3B: Quantize sdq_gptq ---")    args.out_dir = str(work_dir / "models" / "sdq_gptq")    run_quantize_gptqmodel(args, dynamic_config=alloc_flex["dynamic"])    print("\n--- STAGE 3C: Quantize sdq_autoround_auto_gptq ---")    args.out_dir = str(work_dir / "models" / "sdq_autoround_auto_gptq")    run_quantize_autoround(args, plan=alloc_strict["plan"], export_format="auto_gptq")    print("\n--- STAGE 3D: Quantize sdq_autoround_auto_round ---")    args.out_dir = str(work_dir / "models" / "sdq_autoround_auto_round")    run_quantize_autoround(args, plan=alloc_flex["plan"], export_format="auto_round")    print("\n--- STAGE 4: Evaluation Matrix ---")    run_eval(        str(work_dir / "models" / "baseline_gptq"),        args.tasks,        str(work_dir / "eval" / "baseline_gptq.json"),    )    run_eval(        str(work_dir / "models" / "sdq_gptq"),        args.tasks,        str(work_dir / "eval" / "sdq_gptq.json"),    )    run_eval(        str(work_dir / "models" / "sdq_autoround_auto_gptq"),        args.tasks,        str(work_dir / "eval" / "sdq_autoround_auto_gptq.json"),    )    run_eval(        str(work_dir / "models" / "sdq_autoround_auto_round"),        args.tasks,        str(work_dir / "eval" / "sdq_autoround_auto_round.json"),    )    print("\n--- Pipeline Complete. See eval/ and models/*/export_info.json ---")if __name__ == "__main__":    main()EOFechoecho "Project created at: $(pwd)"echoecho "Next steps:"echo "python -m venv .venv"echo "source .venv/bin/activate"echo "# install your CUDA-matching torch first"echo "pip install -r requirements.txt"echoecho "python -m quant_v5_3.pipeline suite \\"echo "  --model-id Qwen/Qwen3.5-4B-Base \\"echo "  --work-dir ./runs/v5_3_suite \\"echo "  --budget-mb-extra 200 \\"echo "  --autoround-iters 50 \\"echo "  --tasks arc_challenge,hellaswag"
Bash#!/usr/bin/env bashset -euo pipefailmkdir -p sdq_quant_v5_3/quant_v5_3cd sdq_quant_v5_3cat > README.md <<'EOF'# SDQ Quant v5.3 (Hessian-Proxy + AutoRound + Format Policies)A deterministic Software-Defined Quantization pipeline targeting standard GPTQ-compatible 4/8/skip module paths intended for GPTQModel / vLLM deployment, plus best-effort AutoRound export paths.## Design1. Measurement:   - Computes `trace(X^T X)` on `cuda:0` as a proxy for second-order sensitivity.2. Allocation:   - Uses a Lagrangian solver under a planner memory budget.   - `q_proj`, `k_proj`, `v_proj` are locked to `w4_g128` to preserve QKV fusion.   - For `auto_gptq`, `lm_head` is forced to `skip`.   - For `GPTQModel` and `auto_round`, `lm_head` can choose `w4_g128`, `w8_g128`, or `skip`.3. Quantization matrix:   - `baseline_gptq`: GPTQModel, plain 4-bit   - `sdq_gptq`: GPTQModel, Hessian-proxy allocation   - `sdq_autoround_auto_gptq`: AutoRound with strict `lm_head` policy   - `sdq_autoround_auto_round`: AutoRound with flex `lm_head` policy4. Evaluation:   - Uses `lm_eval.simple_evaluate()`.## Notes- The memory budget is a planner estimate, not an exact runtime VRAM guarantee.- The error model is a Hessian proxy, not a full GPTQ inverse-Hessian simulation.- `auto_gptq` mixed-bit export is best-effort.- If `auto_gptq` export or downstream loading fails, this code falls back to `auto_round` and records the actual export format in `export_info.json`.EOFcat > requirements.txt <<'EOF'transformersdatasetsacceleratetorchgptqmodelauto-roundlm-evaltqdmEOFcat > quant_v5_3/__init__.py <<'EOF'"""SDQ Quant v5.3: Hessian-Proxy Allocator + AutoRound Format Policies"""EOFcat > quant_v5_3/actions.py <<'EOF'from __future__ import annotationsfrom dataclasses import dataclassfrom typing import Dict, Optional@dataclass(frozen=True)class ActionSpec:    name: str    bits: int    group_size: Optional[int]    skip: boolACTION_SPECS: Dict[str, ActionSpec] = {    "w4_g128": ActionSpec("w4_g128", bits=4, group_size=128, skip=False),    "w8_g128": ActionSpec("w8_g128", bits=8, group_size=128, skip=False),    "skip": ActionSpec("skip", bits=16, group_size=None, skip=True),}BASE_ACTION = "w4_g128"def bytes_per_param(action: str) -> float:    """Planner memory estimate, not an exact runtime VRAM model."""    if action == "skip":        return 2.0  # FP16    spec = ACTION_SPECS[action]    base = spec.bits / 8.0    overhead = 2.5 / spec.group_size    return base + overheaddef action_to_override(action: str) -> dict:    if action == "skip":        return {}    spec = ACTION_SPECS[action]    return {"bits": spec.bits, "group_size": spec.group_size}EOFcat > quant_v5_3/measurement.py <<'EOF'from __future__ import annotationsimport jsonfrom pathlib import Pathimport torchfrom datasets import load_datasetfrom tqdm import tqdmfrom transformers import AutoModelForCausalLM, AutoTokenizerdef load_calibration_texts(ds_name: str, ds_config: str, split: str, text_field: str, num_texts: int):    ds = load_dataset(ds_name, ds_config, split=split)    texts = []    for row in ds:        text = row.get(text_field)        if isinstance(text, str) and text.strip():            texts.append(text.strip())        if len(texts) >= num_texts:            break    return textsdef measure_hessian_traces(args):    """    Computes a Hessian proxy:    Trace(H) ≈ mean over calibration rows of sum(x^2)    collected from each target linear layer input.    """    out_dir = Path(args.out_dir)    out_dir.mkdir(parents=True, exist_ok=True)    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)    if tokenizer.pad_token_id is None:        tokenizer.pad_token = tokenizer.eos_token    model = AutoModelForCausalLM.from_pretrained(        args.model_id,        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,        low_cpu_mem_usage=True,        trust_remote_code=True,    ).to(device)    model.eval()    texts = load_calibration_texts(        args.dataset_name,        args.dataset_config,        args.split,        args.text_field,        args.num_texts,    )    include_mods = [x.strip() for x in args.include_modules.split(",") if x.strip()]    targets = [        (n, m)        for n, m in model.named_modules()        if isinstance(m, torch.nn.Linear) and any(tag in n for tag in include_mods)    ]    hessian_traces = {name: 0.0 for name, _ in targets}    row_counts = {name: 0 for name, _ in targets}    params_count = {name: m.weight.numel() for name, m in targets}    handles = []    def make_hook(name):        def hook(_, inp, __):            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])            hessian_traces[name] += (x ** 2).sum().item()            row_counts[name] += x.shape[0]        return hook    for name, module in targets:        handles.append(module.register_forward_hook(make_hook(name)))    with torch.no_grad():        for i in tqdm(range(0, len(texts), args.batch_size), desc="Hessian Proxy Measurement"):            batch_texts = texts[i : i + args.batch_size]            batch = tokenizer(                batch_texts,                return_tensors="pt",                truncation=True,                max_length=args.max_length,                padding=True,            )            batch = {k: v.to(device) for k, v in batch.items()}            model(**batch)    for h in handles:        h.remove()    features = []    for name, _ in targets:        avg_trace = hessian_traces[name] / max(1, row_counts[name])        features.append(            {                "name": name,                "params": params_count[name],                "hessian_trace": avg_trace,            }        )    out_file = out_dir / "hessian_features.json"    out_file.write_text(json.dumps(features, indent=2), encoding="utf-8")    print(f"Saved {len(features)} layer traces to {out_file}")    del model    if torch.cuda.is_available():        torch.cuda.empty_cache()    return featuresEOFcat > quant_v5_3/allocator.py <<'EOF'from __future__ import annotationsimport jsonimport refrom pathlib import Pathfrom .actions import ACTION_SPECS, BASE_ACTION, action_to_override, bytes_per_paramdef get_valid_actions(module_name: str, allow_quantized_lm_head: bool) -> list[str]:    """    Policy layer:    - QKV are locked to base action to preserve fused QKV paths.    - lm_head policy depends on target export format.    - others can move between 4-bit and 8-bit.    """    if any(k in module_name for k in ["q_proj", "k_proj", "v_proj"]):        return [BASE_ACTION]    if "lm_head" in module_name:        return ["w4_g128", "w8_g128", "skip"] if allow_quantized_lm_head else ["skip"]    return ["w4_g128", "w8_g128"]def solve_proxy_allocation(features_path: str, budget_mb_extra: float, allow_quantized_lm_head: bool):    """Lagrangian multiplier solver over a proxy error model."""    features = json.loads(Path(features_path).read_text(encoding="utf-8"))    base_bpp = bytes_per_param(BASE_ACTION)    total_base_mb = sum(f["params"] * base_bpp for f in features) / (1024 ** 2)    target_mb = total_base_mb + budget_mb_extra    def proxy_error(trace: float, action: str) -> float:        if action == "skip":            return 0.0        bits = ACTION_SPECS[action].bits        return trace * (2.0 ** (-2.0 * bits))    def action_cost_mb(params: int, action: str) -> float:        return (params * bytes_per_param(action)) / (1024 ** 2)    lambda_low, lambda_high = 0.0, 1e12    best_plan = None    for _ in range(80):        lam = (lambda_low + lambda_high) / 2.0        current_mb = 0.0        plan = []        for f in features:            valid_actions = get_valid_actions(f["name"], allow_quantized_lm_head)            best_action = None            best_obj = float("inf")            for act in valid_actions:                err = proxy_error(f["hessian_trace"], act)                size_mb = action_cost_mb(f["params"], act)                obj = err + lam * size_mb                if obj < best_obj:                    best_obj = obj                    best_action = act            size_mb = action_cost_mb(f["params"], best_action)            plan.append(                {                    "name": f["name"],                    "action": best_action,                    "params": f["params"],                    "hessian_trace": f["hessian_trace"],                    "size_mb": size_mb,                }            )            current_mb += size_mb        best_plan = plan        if current_mb > target_mb:            lambda_low = lam        else:            lambda_high = lam    used_extra_mb = sum(p["size_mb"] for p in best_plan) - total_base_mb    dynamic = {}    for p in best_plan:        if p["action"] == BASE_ACTION:            continue        regex = f"^{re.escape(p['name'])}$"        if p["action"] == "skip":            dynamic[f"-:{regex}"] = {}        else:            dynamic[f"+:{regex}"] = action_to_override(p["action"])    return {        "plan": best_plan,        "dynamic": dynamic,        "used_extra_mb": used_extra_mb,    }EOFcat > quant_v5_3/runner.py <<'EOF'from __future__ import annotationsimport gcimport jsonfrom pathlib import Pathimport lm_evalimport torchfrom auto_round import AutoRoundfrom gptqmodel import GPTQModel, QuantizeConfigfrom .measurement import load_calibration_textsdef run_quantize_gptqmodel(args, dynamic_config):    """Standard greedy GPTQ rounding via GPTQModel."""    texts = load_calibration_texts(        args.dataset_name,        args.dataset_config,        args.split,        args.text_field,        args.quant_num_texts,    )    qcfg = QuantizeConfig(        bits=4,        group_size=128,        sym=True,        desc_act=True,        dynamic=dynamic_config,    )    print("Loading model for GPTQModel quantization...")    model = GPTQModel.load(args.model_id, qcfg)    model.quantize(texts, batch_size=args.quant_batch_size)    out = Path(args.out_dir)    out.mkdir(parents=True, exist_ok=True)    model.save(str(out))    del model    gc.collect()    if torch.cuda.is_available():        torch.cuda.empty_cache()def _plan_has_quantized_lm_head(plan) -> bool:    if not plan:        return False    for p in plan:        if "lm_head" in p["name"] and p["action"] in ("w4_g128", "w8_g128"):            return True    return Falsedef run_quantize_autoround(args, plan, export_format: str):    """    AutoRound SignSGD rounding with documented quantize_and_save() API.    Behavior:    - For auto_gptq: lm_head must not be quantized.    - If auto_gptq export/load constraints are violated, we fallback to auto_round.    """    from transformers import AutoModelForCausalLM, AutoTokenizer    texts = load_calibration_texts(        args.dataset_name,        args.dataset_config,        args.split,        args.text_field,        args.quant_num_texts,    )    layer_config = {}    if plan is not None:        for p in plan:            if p["action"] == "skip":                layer_config[p["name"]] = {"bits": 16}            elif p["action"] != "w4_g128":                layer_config[p["name"]] = {"bits": 8, "group_size": 128}    requested_format = export_format    actual_format = export_format    if requested_format == "auto_gptq" and _plan_has_quantized_lm_head(plan):        print("[warn] quantized lm_head is not valid for auto_gptq here; switching export format to auto_round.")        actual_format = "auto_round"    print(f"Loading model for AutoRound ({requested_format} requested)...")    model = AutoModelForCausalLM.from_pretrained(        args.model_id,        torch_dtype=torch.float16,        trust_remote_code=True,        device_map="auto"    )    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)    ar = AutoRound(        model=model,        tokenizer=tokenizer,        bits=4,        group_size=128,        sym=True,        dataset=texts,        seqlen=args.max_length,        batch_size=args.quant_batch_size,        layer_config=layer_config,        iters=args.autoround_iters,    )    out = Path(args.out_dir)    out.mkdir(parents=True, exist_ok=True)    try:        print(f"Executing AutoRound and saving via quantize_and_save(format='{actual_format}')...")        ar.quantize_and_save(str(out), format=actual_format)    except Exception as e:        if actual_format == "auto_gptq":            print(f"[warn] auto_gptq export failed: {e}")            print("[warn] retrying with auto_round format...")            actual_format = "auto_round"            ar.quantize_and_save(str(out), format=actual_format)        else:            raise    export_info = {        "requested_format": requested_format,        "actual_format": actual_format,    }    (out / "export_info.json").write_text(json.dumps(export_info, indent=2), encoding="utf-8")    del ar    del model    del tokenizer    gc.collect()    if torch.cuda.is_available():        torch.cuda.empty_cache()def run_eval(model_path: str, tasks: str, out_file: str):    """Programmatic lm-eval."""    print(f"Running programmatic lm-eval on {model_path}...")    Path(out_file).parent.mkdir(parents=True, exist_ok=True)    task_list = [t.strip() for t in tasks.split(",") if t.strip()]    results = lm_eval.simple_evaluate(        model="hf",        model_args=f"pretrained={model_path},trust_remote_code=True",        tasks=task_list,        batch_size="auto",        device="cuda:0" if torch.cuda.is_available() else "cpu",    )    serializable_results = {        "results": results.get("results", {}),        "versions": results.get("versions", {}),    }    Path(out_file).write_text(json.dumps(serializable_results, indent=2), encoding="utf-8")    return serializable_resultsEOFcat > quant_v5_3/pipeline.py <<'EOF'from __future__ import annotationsimport argparseimport jsonfrom pathlib import Pathfrom .allocator import solve_proxy_allocationfrom .measurement import measure_hessian_tracesfrom .runner import run_eval, run_quantize_autoround, run_quantize_gptqmodeldef main():    parser = argparse.ArgumentParser()    sub = parser.add_subparsers(dest="cmd", required=True)    suite = sub.add_parser("suite")    suite.add_argument("--model-id", required=True)    suite.add_argument("--work-dir", required=True)    suite.add_argument("--budget-mb-extra", type=float, default=250.0)    suite.add_argument("--include-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head")    suite.add_argument("--dataset-name", default="wikitext")    suite.add_argument("--dataset-config", default="wikitext-2-raw-v1")    suite.add_argument("--split", default="train")    suite.add_argument("--text-field", default="text")    suite.add_argument("--num-texts", type=int, default=128)    suite.add_argument("--max-length", type=int, default=2048)    suite.add_argument("--batch-size", type=int, default=4)    suite.add_argument("--quant-num-texts", type=int, default=128)    suite.add_argument("--quant-batch-size", type=int, default=4)    suite.add_argument("--autoround-iters", type=int, default=200, help="Iterations for AutoRound SignSGD")    suite.add_argument("--tasks", default="arc_challenge,hellaswag")    args = parser.parse_args()    work_dir = Path(args.work_dir)    work_dir.mkdir(parents=True, exist_ok=True)    print("\n--- STAGE 1: Hessian Proxy Measurement ---")    args.out_dir = str(work_dir / "probe")    measure_hessian_traces(args)    features_path = str(work_dir / "probe" / "hessian_features.json")    print("\n--- STAGE 2: Lagrangian Proxy Allocation ---")    # Flex plan: GPTQModel mixed dynamic path & AutoRound native format path    alloc_flex = solve_proxy_allocation(        features_path,        args.budget_mb_extra,        allow_quantized_lm_head=True,    )    (work_dir / "sdq_plan_flex.json").write_text(json.dumps(alloc_flex["plan"], indent=2), encoding="utf-8")    (work_dir / "sdq_dynamic_flex.json").write_text(json.dumps(alloc_flex["dynamic"], indent=2), encoding="utf-8")    # Strict plan: auto_gptq export path    alloc_strict = solve_proxy_allocation(        features_path,        args.budget_mb_extra,        allow_quantized_lm_head=False,    )    (work_dir / "sdq_plan_strict.json").write_text(json.dumps(alloc_strict["plan"], indent=2), encoding="utf-8")    (work_dir / "sdq_dynamic_strict.json").write_text(json.dumps(alloc_strict["dynamic"], indent=2), encoding="utf-8")    print(        f"Allocated. Estimated extra MB -> "        f"flex: {alloc_flex['used_extra_mb']:.2f}, "        f"strict: {alloc_strict['used_extra_mb']:.2f}"    )    print("\n--- STAGE 3A: Quantize baseline_gptq ---")    args.out_dir = str(work_dir / "models" / "baseline_gptq")    run_quantize_gptqmodel(args, dynamic_config=None)    print("\n--- STAGE 3B: Quantize sdq_gptq ---")    args.out_dir = str(work_dir / "models" / "sdq_gptq")    run_quantize_gptqmodel(args, dynamic_config=alloc_flex["dynamic"])    print("\n--- STAGE 3C: Quantize sdq_autoround_auto_gptq ---")    args.out_dir = str(work_dir / "models" / "sdq_autoround_auto_gptq")    run_quantize_autoround(args, plan=alloc_strict["plan"], export_format="auto_gptq")    print("\n--- STAGE 3D: Quantize sdq_autoround_auto_round ---")    args.out_dir = str(work_dir / "models" / "sdq_autoround_auto_round")    run_quantize_autoround(args, plan=alloc_flex["plan"], export_format="auto_round")    print("\n--- STAGE 4: Evaluation Matrix ---")    run_eval(        str(work_dir / "models" / "baseline_gptq"),        args.tasks,        str(work_dir / "eval" / "baseline_gptq.json"),    )    run_eval(        str(work_dir / "models" / "sdq_gptq"),        args.tasks,        str(work_dir / "eval" / "sdq_gptq.json"),    )    run_eval(        str(work_dir / "models" / "sdq_autoround_auto_gptq"),        args.tasks,        str(work_dir / "eval" / "sdq_autoround_auto_gptq.json"),    )    run_eval(        str(work_dir / "models" / "sdq_autoround_auto_round"),        args.tasks,        str(work_dir / "eval" / "sdq_autoround_auto_round.json"),    )    print("\n--- Pipeline Complete. See eval/ and models/*/export_info.json ---")if __name__ == "__main__":    main()EOFechoecho "Project created at: $(pwd)"echoecho "Next steps:"echo "python -m venv .venv"echo "source .venv/bin/activate"echo "# install your CUDA-matching torch first"echo "pip install -r requirements.txt"echoecho "python -m quant_v5_3.pipeline suite \\"echo "  --model-id Qwen/Qwen3.5-4B-Base \\"echo "  --work-dir ./runs/v5_3_suite \\"echo "  --budget-mb-extra 200 \\"echo "  --autoround-iters 50 \\"echo "  --tasks arc_challenge,hellaswag"
```

Then run:

```Bashbash create_sdq_v53.shcd sdq_quant_v5_3python -m venv .venvsource .venv/bin/activate# install your CUDA-matching torch firstpip install -r requirements.txtpython -m quant_v5_3.pipeline suite \  --model-id Qwen/Qwen3.5-4B-Base \  --work-dir ./runs/v5_3_suite \  --budget-mb-extra 200 \  --autoround-iters 50 \  --tasks arc_challenge,hellaswag
Bashbash create_sdq_v53.shcd sdq_quant_v5_3python -m venv .venvsource .venv/bin/activate# install your CUDA-matching torch firstpip install -r requirements.txtpython -m quant_v5_3.pipeline suite \  --model-id Qwen/Qwen3.5-4B-Base \  --work-dir ./runs/v5_3_suite \  --budget-mb-extra 200 \  --autoround-iters 50 \  --tasks arc_challenge,hellaswag
```

---

### User:
And this project ? https://huggingface.co/z-lab/Qwen3.5-9B-PARO how to implement their logic ?

---

### ChatGPT:
It’s a strong project, but it is **not** “just another GPTQ/AWQ recipe.”
`z-lab/Qwen3.5-9B-PARO` is a ParoQuant checkpoint, and ParoQuant’s core idea is to suppress outliers with **independent Givens rotations plus channel-wise scaling**, then run a lightweight custom runtime path around that transform. The paper says this gives an average **2.4% accuracy gain over AWQ** on reasoning tasks with **<10% overhead**, and the repo/model card expose it through the `paroquant` runtime rather than GPTQModel. [Hugging Face+2arXiv+2](https://huggingface.co/z-lab/Qwen3.5-9B-PARO)

So there are really **two ways** to “implement their logic”:

### 1) Exact ParoQuant

Use **their repo/runtime** and quantize your own model the way they intended. The official workflow is:

```Bashgit clone https://github.com/z-lab/paroquantcd paroquantpip install -e ".[optim,eval]"# 1) optimize rotation parametersexperiments/optimize/4bit.sh Qwen/Qwen3-8B# 2) export checkpointpython -m paroquant.cli.convert \  --model Qwen/Qwen3-8B \  --result-dir output/Qwen3-8B \  --output-path models/Qwen3-8B-PARO
Bashgit clone https://github.com/z-lab/paroquantcd paroquantpip install -e ".[optim,eval]"# 1) optimize rotation parametersexperiments/optimize/4bit.sh Qwen/Qwen3-8B# 2) export checkpointpython -m paroquant.cli.convert \  --model Qwen/Qwen3-8B \  --result-dir output/Qwen3-8B \  --output-path models/Qwen3-8B-PARO
```

That is the repo’s documented “Quantize Your Own Model” path. The model card also shows the inference/runtime side: `pip install "paroquant[vllm]"`, interactive chat, and an OpenAI-compatible server. The repo additionally warns that `main` is under active development and that the **legacy branch** should be used for paper reproduction. [GitHub+1](https://github.com/z-lab/paroquant)

If you just want to **run** the published model, the shortest path is:

```Bashpip install "paroquant[vllm]"python -m paroquant.cli.chat --model z-lab/Qwen3.5-9B-PARO# orpython -m paroquant.cli.serve --model z-lab/Qwen3.5-9B-PARO --port 8000
Bashpip install "paroquant[vllm]"python -m paroquant.cli.chat --model z-lab/Qwen3.5-9B-PARO# orpython -m paroquant.cli.serve --model z-lab/Qwen3.5-9B-PARO --port 8000
```

That is the cleanest “exact logic” implementation because it uses the actual ParoQuant runtime and kernels. [Hugging Face+1](https://huggingface.co/z-lab/Qwen3.5-9B-PARO)

### 2) Borrow the idea into your own quant pipeline

If you want to fold ParoQuant-style logic into your own SDQ/GPTQ pipeline, the right mental model is:

- **precondition the tensor** with:

channel-wise scaling

pairwise Givens rotations
- channel-wise scaling
- pairwise Givens rotations
- then quantize the transformed weights to INT4
- then at inference time, apply the corresponding lightweight transform path in the kernel/runtime

That is because the paper’s method is explicitly built around **scaled pairwise rotation** and a **co-designed inference kernel**. In other words, exact ParoQuant is not just an offline allocator; it changes the quantization transform and expects runtime support for it. [arXiv+2arXiv+2](https://arxiv.org/abs/2511.10645)

A practical implementation plan looks like this:

```Python# conceptual only, not repo-exact codefor each target linear layer:    # 1) collect calibration activations X    X = collect_inputs(layer, calib_set)    # 2) learn channel scaling s and pairwise rotation angles theta    # objective: reduce outliers / narrow dynamic range before quantization    s, theta = optimize_transform(X, W)    # 3) transform weights    W_t = pairwise_rotate(channel_scale(W, s), theta)    # 4) quantize transformed weights to INT4    W_q, scales, zeros = quantize_groupwise(W_t)    # 5) save:    #    - qweight / scales / zeros    #    - transform metadata (s, theta)
Python# conceptual only, not repo-exact codefor each target linear layer:    # 1) collect calibration activations X    X = collect_inputs(layer, calib_set)    # 2) learn channel scaling s and pairwise rotation angles theta    # objective: reduce outliers / narrow dynamic range before quantization    s, theta = optimize_transform(X, W)    # 3) transform weights    W_t = pairwise_rotate(channel_scale(W, s), theta)    # 4) quantize transformed weights to INT4    W_q, scales, zeros = quantize_groupwise(W_t)    # 5) save:    #    - qweight / scales / zeros    #    - transform metadata (s, theta)
```

At runtime you then need a fused path that effectively computes the transformed linear efficiently. That runtime requirement is the key difference versus your current GPTQ/AutoRound pipeline. The paper and repo both emphasize that ParoQuant is paired with a lightweight runtime/kernel implementation, not just a saved GPTQ checkpoint. [arXiv+1](https://arxiv.org/abs/2511.10645)

### What I would do in your case

For your current stack, I’d separate this into two tracks:

**Track A: exact ParoQuant**

- use the `paroquant` repo exactly as documented
- quantize one of your target models with their optimizer + converter
- benchmark it against your GPTQ/AutoRound matrix

**Track B: ParoQuant-inspired hybrid**

- keep your SDQ allocator
- replace “which layers get 4 vs 8” with “which layers are worth Paro-style transform”
- only apply the Paro-style transform to the most fragile modules first:

`o_proj`

`down_proj`

`gate_proj`

maybe `lm_head`
- `o_proj`
- `down_proj`
- `gate_proj`
- maybe `lm_head`
- leave QKV uniform to avoid path complexity

That second path is an engineering inference from how ParoQuant attacks outliers, not the exact published implementation. The exact published implementation is the repo workflow above. [arXiv+1](https://arxiv.org/abs/2511.10645)

### What not to do

Do **not** assume you can reproduce ParoQuant by only changing your allocator.
ParoQuant’s gain is not mainly from a smarter 4/8-bit assignment; it comes from the **rotation + scaling transform** and the **runtime support** around it. If you keep plain GPTQ weights and plain GPTQ kernels, you can borrow the spirit, but you are not implementing ParoQuant itself. [arXiv+1](https://arxiv.org/abs/2511.10645)

If you want, I can turn this into a concrete **“ParoQuant-vs-v5.3 benchmark harness”** for your 3090 box, or sketch a **Paro-inspired transform module** you can bolt onto your current SDQ code.

---

### User:
If modifying kernels wasn’t an issue and I want to combine the logic of paroquant exl3 and autoround , into a new quantization method that beats them all , and still as fast as gptq in inferencing with ampere gpus , how would you do it ?

---

### ChatGPT:
I would not try to literally “merge the formats.” I would build a **new scalar, GPTQ-speed format** that steals the **transform logic** from ParoQuant, the **search logic** from EXL3/QTIP, and the **rounding optimization** from AutoRound—while keeping the runtime path as close as possible to a Marlin-style **fused dequant+GEMM** kernel for Ampere. That is the only credible way to chase “beats them all” **and** stay GPTQ-fast at inference. ParoQuant’s gain comes from pairwise Givens rotations plus channel scaling with a lightweight runtime transform; EXL3 is a streamlined QTIP-style format built around trellis/codebook ideas; AutoRound/SignRound improves rounding with SignSGD without adding inference overhead; and Marlin shows Ampere can get near-ideal FP16×INT4 speedups when the kernel stays memory-efficient and fused. [arXiv+3arXiv+3GitHub+3](https://arxiv.org/abs/2511.10645?utm_source=chatgpt.com)

The design I would bet on is this:

## 1) Keep the runtime scalar and Marlin-like

Do **not** inherit EXL3/QTIP’s sequential decoder or codebook-heavy inference path. QTIP’s paper is clear that naive TCQ does not admit fast inference because of sequential bottlenecks and codebook storage pressure. If the target is “as fast as GPTQ on Ampere,” the runtime should still look like:

- packed low-bit weights,
- tiny per-group metadata,
- one fused kernel that dequantizes into registers/shared memory and immediately does the MMA,
- no per-token codebook decode loops,
- no runtime graph of small transforms outside the GEMM. [arXiv+2arXiv+2](https://arxiv.org/abs/2406.11235?utm_source=chatgpt.com)

So the new method should remain a **scalar groupwise quantizer**, not a vector-codebook quantizer.

## 2) Borrow ParoQuant’s transform, but fuse it into the matmul

ParoQuant’s strongest idea is not “better bit allocation.” It is **pairwise Givens rotations + channel-wise scaling** to suppress outliers before quantization, while keeping the transform hardware-efficient enough to add under 10% overhead in their design. That is the part worth stealing. I would apply:

- per-group channel scaling,
- sparse pairwise rotations inside each block,
- learned offline,
- then quantize the transformed weights. [arXiv+1](https://arxiv.org/abs/2511.10645?utm_source=chatgpt.com)

But I would impose one constraint ParoQuant does not need to obey as strictly: the transform must be **tile-local and kernel-fusible**. In practice that means:

- rotations only within fixed 64- or 128-column groups,
- pair indices chosen so each warp can apply them without cross-warp shuffles exploding,
- scale stored per group or per micro-group,
- transform metadata packed adjacent to the group metadata.

That gives you ParoQuant’s outlier suppression without a separate runtime pass.

## 3) Replace greedy quantization with AutoRound-style optimization

For the actual scalar levels, I would not use plain GPTQ greedy rounding. AutoRound/SignRound is the right inspiration here because it optimizes rounding and clipping with SignSGD yet still exports ordinary low-bit checkpoints without adding inference overhead. That is exactly the kind of offline work you want. [arXiv+1](https://arxiv.org/abs/2309.05516?utm_source=chatgpt.com)

So the offline objective becomes:

- optimize transform parameters:

channel scales

pairwise rotation angles
- channel scales
- pairwise rotation angles
- optimize quantization parameters:

clipping

scales

zero/offset if asymmetric
- clipping
- scales
- zero/offset if asymmetric
- optimize rounding decisions:

SignSGD-style continuous relaxation, then project back to scalar integers
- SignSGD-style continuous relaxation, then project back to scalar integers

In other words, use AutoRound’s idea **after** the Paro-style transform, not instead of it.

## 4) Borrow EXL3/QTIP as a search prior, not as the deployed format

EXL3 is explicitly described as a streamlined variant of QTIP, and QTIP’s core advantage is **better search over quantization states** through trellis coding, not a cheap runtime path. So I would use QTIP/EXL3 logic only in the **offline optimizer**:

- short-horizon beam search over scalar assignments inside a block,
- or initialize AutoRound from a trellis-searched candidate,
- then let SignSGD refine it. [GitHub+1](https://github.com/turboderp-org/exllamav3?utm_source=chatgpt.com)

That gives you “EXL3-like intelligence” during quantization without inheriting EXL3’s runtime burden.

## 5) Use mixed precision, but only where the kernel can stay fused

EXL2’s useful lesson is that mixed precision is powerful, and its format supports mixing quantization levels within a model to achieve arbitrary average bitrates. But for Ampere speed, I would avoid arbitrary intra-tensor bitrate soup. [GitHub](https://github.com/turboderp-org/exllamav2?utm_source=chatgpt.com)

I would use only **two deployed scalar paths**:

- INT4 groups
- INT8 rescue groups

Not 2/3/5/6-bit, unless you prove the kernel still hits the memory/computation balance you want.

So the actual tensor layout would be:

- mostly INT4 groups,
- a small mask of “hard” groups promoted to INT8,
- same tile shape for both,
- same fused kernel dispatch,
- no dynamic branching per element, only per micro-tile/group.

That is much closer to GPTQ/Marlin performance territory.

## 6) Make the allocator groupwise, not just layerwise

Your previous SDQ allocator was already heading this way conceptually, but to beat strong baselines you need finer granularity than “whole module = 4 or 8 bits.” EXL2’s main lesson is that finer-grained mixed precision matters. [GitHub](https://github.com/turboderp-org/exllamav2?utm_source=chatgpt.com)

My allocator would operate at **micro-group granularity**:

- group size 64 or 128 columns,
- each group gets:

transform on/off

INT4 or INT8

maybe one of a few transform templates
- transform on/off
- INT4 or INT8
- maybe one of a few transform templates

The cost model should be:

- Hessian-proxy distortion,
- plus runtime penalty regularizer,
- plus metadata penalty,
- under a hard bytes budget.

That becomes a constrained optimization problem, not an LLM-routing problem.

## 7) The kernel I’d actually write for Ampere

For Ampere, I would start from a Marlin-style fused FP16×INT4 kernel and extend it in the least invasive way possible. Marlin’s paper is the right design anchor because it is already optimized for practical batched autoregressive inference and gets close to ideal low-bit speedups on Ampere-class deployments. [GitHub+1](https://github.com/IST-DASLab/marlin?utm_source=chatgpt.com)

The kernel would do this per tile:

1. Load packed low-bit weights and metadata for one K-tile.
2. Dequantize groups into registers/shared memory.
3. Apply **pairwise rotations + channel scales** in-register/shared-memory, but only within the local tile.
4. Multiply with FP16 activations using the normal MMA path.
5. Accumulate in FP16/FP32 as usual.

The critical rule is that the transform must be **linear and local**, so it can be folded into the same memory/compute schedule rather than becoming an extra kernel launch.

## 8) The concrete offline pipeline

I would build the training-free PTQ pipeline like this:

### Stage A: collect stats

For each target tensor/group:

- activation covariance or Hessian proxy
- outlier scores
- sensitivity under local perturbations
- runtime cost estimate for transform and for INT8 rescue

This part is GPTQ-like in spirit because GPTQ itself is built on approximate second-order information. [arXiv+1](https://arxiv.org/abs/2210.17323?utm_source=chatgpt.com)

### Stage B: optimize transform

For each group:

- learn channel scale vector
- learn sparse pairwise rotation parameters
- objective: shrink dynamic range / equalize channels / reduce Hessian-weighted reconstruction error

### Stage C: initialize quantization by short trellis search

For each group:

- do a small beam search over scalar assignments or clipping settings
- use it to seed the final optimizer

### Stage D: refine with AutoRound-style SignSGD

Optimize:

- rounding offsets
- clipping
- maybe tiny adjustments to the transform
- under Hessian-weighted activation reconstruction loss

### Stage E: global mixed-precision solve

Promote only the hardest groups to INT8 under a bytes budget.

### Stage F: emit kernel-friendly checkpoint

Store:

- packed INT4/INT8 groups
- compact per-group scales
- compact per-group transform metadata
- one format version, one fused Ampere kernel path

## 9) What I’d call the method

Something like:

**TRAM-Quant**
**T**ransform-**R**ounded **A**mpere **M**ixed-precision Quantization

Internally:

- **Paro block** = sparse pairwise rotation + scaling
- **QTIP seed** = short trellis initialization
- **AutoRound refine** = SignSGD rounding/clipping optimization
- **Marlin runtime** = fused Ampere kernel

## 10) What will probably make it beat the baselines

If this wins, it will not be because of any single component. It will be because of the combination:

- **ParoQuant** fixes outliers before quantization. [arXiv+1](https://arxiv.org/abs/2511.10645?utm_source=chatgpt.com)
- **EXL3/QTIP-like search** avoids bad local assignments at the block level. [GitHub+1](https://github.com/turboderp-org/exllamav3?utm_source=chatgpt.com)
- **AutoRound** improves scalar rounding without inference overhead. [arXiv+1](https://arxiv.org/abs/2309.05516?utm_source=chatgpt.com)
- **Marlin-style kernel discipline** preserves Ampere speed. [arXiv+1](https://arxiv.org/abs/2408.11743?utm_source=chatgpt.com)

If you skip any one of those, you probably end up with:

- great quality but slow runtime,
- or great speed but not enough quality gain to matter.

## 11) What I would not do

I would not:

- use a generic LLM judge for allocation,
- deploy QTIP codebooks directly if speed matters,
- allow arbitrary bitwidths per tiny fragment,
- put nonlocal rotations into the runtime,
- quantize fused QKV with different per-subtensor formats unless the kernel is explicitly built for tied handling.

## 12) The honest caveat

Nobody can guarantee it will “beat them all” before ablations. EXL3/QTIP, ParoQuant, and AutoRound each win for different reasons, and some of their gains come from constraints you may not preserve simultaneously. What I can say is:

**This is the most plausible architecture I know for combining their strongest ideas while still aiming for GPTQ-class Ampere throughput.** It is also substantially more realistic than trying to literally mash together EXL3 format, ParoQuant runtime, and AutoRound export unchanged. [arXiv+3arXiv+3GitHub+3](https://arxiv.org/abs/2511.10645?utm_source=chatgpt.com)

The next step I’d take is to freeze the target format and kernel contract first:

- group size
- INT4/INT8 mixed micro-group scheme
- transform metadata layout
- one Ampere fused kernel API

Then derive the offline optimizer to emit exactly that contract. Send the target model family and preferred average bits-per-weight, and I’ll map this into a concrete tensor format and kernel/block design.

---

### User:
Here is the raw truth: your high-level strategy is the exact right path to state-of-the-art, but your runtime mechanics and offline pipeline have critical flaws that will either tank your Ampere throughput or make quantization take weeks.
Here is the critique, followed by the polished spec.
The Critique: What Needs Fixing
 * The ParoQuant/Marlin Collision: You cannot apply pairwise rotations to dequantized weights on the fly inside the Marlin inner loop. Marlin is already register-starved to maintain occupancy. Injecting math between ldmatrix and mma.sync will stall the pipeline. The fix: You apply the rotations to the weights offline. At runtime, your kernel applies the inverse transform to the activations in shared memory before feeding them to the MMA blocks.
 * Marlin Hates Branching (Mixed Precision Flaw): You cannot arbitrarily mix INT4 and INT8 at the 64-column micro-group level within a single tile. Marlin relies on perfectly pipelined, deterministic cp.async memory fetches. If a warp has to branch to decode INT8 vs. INT4, your memory coalescing dies. The fix: Segregate mixed precision at the output channel (N) level. A warp computes either a pure INT4 tile or a pure INT8 tile.
 * Offline Compute Bloat: Using QTIP's trellis beam search to seed AutoRound's SignSGD is computationally absurd. SignSGD is robust enough to find the local minima. The fix: Drop the trellis search. Initialize with a fast, greedy Hessian solver, then run SignSGD.
The Polished Spec: TRAM-Quant
This is the lean, deployable version of your architecture.
1. The Runtime Kernel (Marlin + Activation Transforms)
 * Core: A fused Ampere kernel prioritizing memory-bound efficiency.
 * The Math: Weights are pre-transformed and quantized offline. The kernel loads activations, applies sparse pairwise Givens rotations and channel scaling to the activations in shared memory (tile-local only), then streams them into the standard MMA pipeline.
 * No Codebooks: Strict scalar dequantization into registers. Zero per-token decode loops.
2. Memory Layout (Constrained Mixed Precision)
 * Granularity: Group sizes of 64 or 128.
 * Segregation: Tensors are mostly INT4. "Hard" channels promoted to INT8 are segregated by column blocks. The kernel dispatches uniform INT4 or INT8 warp-tiles to eliminate dynamic branching in the inner loop.
 * Metadata: Compact per-group scales and packed transform metadata stored adjacent to the weight matrix for coalesced loading.
3. The Offline PTQ Pipeline
 * Stage A (Stats): Collect activation covariance/Hessian proxies.
 * Stage B (Transform Learn): Learn ParoQuant-style sparse pairwise rotations and channel scales to suppress outliers, strictly constrained to the 64/128 block size. Apply to weights.
 * Stage C (Optimize): Skip trellis search. Initialize a greedy baseline, then use AutoRound (SignSGD) to jointly optimize rounding offsets, clipping, and scale under a Hessian-weighted reconstruction loss.
 * Stage D (Allocate): Promote channels with the highest residual distortion to INT8 under a strict byte budget.
 * Stage E (Export): Pack into the segregated Marlin-compatible layout.
draft the exact CUDA shared memory layout and ldmatrix schedule for the segregated INT4/INT8 kernel?

---

### ChatGPT:
Yes.
This is the **exact draft kernel contract** I would prototype first for Ampere.

It is built around what Marlin already proves works well on Ampere: offline-preprocessed weights, `cp.async` global→shared pipelining, XOR-swizzled FP16 activation tiles for conflict-free `ldmatrix`, and a column-wise inner loop that pipelines weight decode with Tensor Core math. PTX also documents that `ldmatrix` is the warp-level shared-memory load primitive for `mma`, with `.m8n8.x4.shared.b16` for 16-bit matrix fragments and `.x4` supplied by lanes `0–7 / 8–15 / 16–23 / 24–31`. AutoRound’s docs are also the reason QKV stays uniform: they explicitly warn that vLLM/SGLang fuse QKV and recommend not assigning different bit widths there. [GitHub+3arXiv+3NVIDIA Docs+3](https://arxiv.org/pdf/2408.11743)

## 1) Kernel family and tile contract

Use **two specialized kernels**, not one branchy mixed kernel:

- **INT4 kernel**

CTA: `256 threads = 8 warps`

CTA output tile: `M_TILE = 64`, `N_TILE = 128`

Warp tile: `16 x 64`

`K_STAGE = 64`
- CTA: `256 threads = 8 warps`
- CTA output tile: `M_TILE = 64`, `N_TILE = 128`
- Warp tile: `16 x 64`
- `K_STAGE = 64`
- **INT8 kernel**

CTA: `128 threads = 4 warps`

CTA output tile: `M_TILE = 64`, `N_TILE = 64`

Warp tile: `16 x 64`

`K_STAGE = 64`
- CTA: `128 threads = 4 warps`
- CTA output tile: `M_TILE = 64`, `N_TILE = 64`
- Warp tile: `16 x 64`
- `K_STAGE = 64`

This keeps the **weight bytes per stage equal**:

- INT4: `64 x 128 x 4 bits = 4096 bytes`
- INT8: `64 x 64 x 8 bits = 4096 bytes`

That is the cleanest way to keep the memory pipeline uniform while avoiding dynamic decode branching inside the MMA loop. The “QKV lock” is also consistent with AutoRound’s guidance for fused QKV deployments. Marlin’s paper also explicitly motivates `Ksm >= 64`, uses offline reshuffled `16 x 64` weight tiles, and executes the inner accumulation as `M x 16` times `16 x 64` column-wise with `16 x 8` Tensor Core instructions. [GitHub+3arXiv+3arXiv+3](https://arxiv.org/pdf/2408.11743)

## 2) Shared-memory layout

I would use a **4-stage pipeline** like Marlin (`P = 4`). Marlin explicitly says it chose `P = 4`, and that `cp.async` plus pipelining of future tiles is key to overlapping memory with Tensor Core math. [arXiv](https://arxiv.org/pdf/2408.11743)

### Exact per-stage layout

For both kernels, the stage starts with the activation tile, then transform metadata, then weight tile, then per-output scales.

```C// ------------------------------// Common constants// ------------------------------constexpr int PIPE      = 4;constexpr int M_TILE    = 64;constexpr int K_STAGE   = 64;constexpr int CHUNK_K   = 8;    // 8 FP16 = 16 bytesconstexpr int A_CHUNKS  = K_STAGE / CHUNK_K;   // 8constexpr int A_STAGE_BYTES = M_TILE * A_CHUNKS * 16;   // 64 * 8 * 16 = 8192// Transform metadata for one K_STAGE=64 block.// Draft contract:// - 64 channel scales (FP16) = 128 B// - up to 8 sparse Givens rotations//   each stored as (u8 a, u8 b, fp16 cos, fp16 sin, pad) = 8 B// - total = 128 + 64 = 192 B -> round to 256 B for alignmentconstexpr int XFORM_STAGE_BYTES = 256;// INT4 specializationconstexpr int N_TILE4 = 128;constexpr int W4_STAGE_BYTES = K_STAGE * N_TILE4 / 2;   // 4096 Bconstexpr int S4_STAGE_BYTES = N_TILE4 * 2;             // one FP16 scale per output channel = 256 Bconstexpr int STAGE4_BYTES =    A_STAGE_BYTES + XFORM_STAGE_BYTES + W4_STAGE_BYTES + S4_STAGE_BYTES; // 12800 B// INT8 specializationconstexpr int N_TILE8 = 64;constexpr int W8_STAGE_BYTES = K_STAGE * N_TILE8;       // 4096 Bconstexpr int S8_STAGE_BYTES = N_TILE8 * 2;             // 128 Bconstexpr int STAGE8_BYTES =    A_STAGE_BYTES + XFORM_STAGE_BYTES + W8_STAGE_BYTES + S8_STAGE_BYTES; // 12672 B
C// ------------------------------// Common constants// ------------------------------constexpr int PIPE      = 4;constexpr int M_TILE    = 64;constexpr int K_STAGE   = 64;constexpr int CHUNK_K   = 8;    // 8 FP16 = 16 bytesconstexpr int A_CHUNKS  = K_STAGE / CHUNK_K;   // 8constexpr int A_STAGE_BYTES = M_TILE * A_CHUNKS * 16;   // 64 * 8 * 16 = 8192// Transform metadata for one K_STAGE=64 block.// Draft contract:// - 64 channel scales (FP16) = 128 B// - up to 8 sparse Givens rotations//   each stored as (u8 a, u8 b, fp16 cos, fp16 sin, pad) = 8 B// - total = 128 + 64 = 192 B -> round to 256 B for alignmentconstexpr int XFORM_STAGE_BYTES = 256;// INT4 specializationconstexpr int N_TILE4 = 128;constexpr int W4_STAGE_BYTES = K_STAGE * N_TILE4 / 2;   // 4096 Bconstexpr int S4_STAGE_BYTES = N_TILE4 * 2;             // one FP16 scale per output channel = 256 Bconstexpr int STAGE4_BYTES =    A_STAGE_BYTES + XFORM_STAGE_BYTES + W4_STAGE_BYTES + S4_STAGE_BYTES; // 12800 B// INT8 specializationconstexpr int N_TILE8 = 64;constexpr int W8_STAGE_BYTES = K_STAGE * N_TILE8;       // 4096 Bconstexpr int S8_STAGE_BYTES = N_TILE8 * 2;             // 128 Bconstexpr int STAGE8_BYTES =    A_STAGE_BYTES + XFORM_STAGE_BYTES + W8_STAGE_BYTES + S8_STAGE_BYTES; // 12672 B
```

### Byte offsets

```C// INT4 stage sA4_base(s) = s * STAGE4_BYTES + 0;T4_base(s) = A4_base(s) + A_STAGE_BYTES;                     // +8192B4_base(s) = T4_base(s) + XFORM_STAGE_BYTES;                // +8448S4_base(s) = B4_base(s) + W4_STAGE_BYTES;                   // +12544// INT8 stage sA8_base(s) = s * STAGE8_BYTES + 0;T8_base(s) = A8_base(s) + A_STAGE_BYTES;                    // +8192B8_base(s) = T8_base(s) + XFORM_STAGE_BYTES;                // +8448S8_base(s) = B8_base(s) + W8_STAGE_BYTES;                   // +12544
C// INT4 stage sA4_base(s) = s * STAGE4_BYTES + 0;T4_base(s) = A4_base(s) + A_STAGE_BYTES;                     // +8192B4_base(s) = T4_base(s) + XFORM_STAGE_BYTES;                // +8448S4_base(s) = B4_base(s) + W4_STAGE_BYTES;                   // +12544// INT8 stage sA8_base(s) = s * STAGE8_BYTES + 0;T8_base(s) = A8_base(s) + A_STAGE_BYTES;                    // +8192B8_base(s) = T8_base(s) + XFORM_STAGE_BYTES;                // +8448S8_base(s) = B8_base(s) + W8_STAGE_BYTES;                   // +12544
```

So total shared memory footprint is:

- INT4 kernel: `4 * 12800 = 51200 B`
- INT8 kernel: `4 * 12672 = 50688 B`

That is a sane starting point for Ampere.

## 3) Activation shared-memory swizzle

For the FP16 activation tile, use the Marlin/XOR style shared-memory swizzle so `ldmatrix.sync.aligned.m8n8.x4.shared.b16` stays conflict-free. Marlin explicitly says the 16-byte vector for logical tile element `ij` should be stored at location `i (i xor j)` to avoid `ldmatrix` bank conflicts. [arXiv](https://arxiv.org/pdf/2408.11743)

For `M_TILE = 64`, `K_STAGE = 64`, store the activation tile as 16-byte chunks:

- logical indices:

`m in [0, 63]`

`vec in [0, 7]` where each `vec` is `8 FP16 = 16 bytes`

`elem in [0, 7]`
- `m in [0, 63]`
- `vec in [0, 7]` where each `vec` is `8 FP16 = 16 bytes`
- `elem in [0, 7]`

Use this exact address formula:

```C// Logical A tile element A[m, k]// where k = vec * 8 + elem// Store as 16-byte chunks with XOR swizzle on chunk id__device__ __forceinline__size_t a_smem_offset_bytes(int stage_base, int m, int vec, int elem) {    int phys_vec = vec ^ (m & 0x7);    int chunk_index = m * 8 + phys_vec;      // 64 rows * 8 chunks    return stage_base + (chunk_index * 16) + (elem * 2);}
C// Logical A tile element A[m, k]// where k = vec * 8 + elem// Store as 16-byte chunks with XOR swizzle on chunk id__device__ __forceinline__size_t a_smem_offset_bytes(int stage_base, int m, int vec, int elem) {    int phys_vec = vec ^ (m & 0x7);    int chunk_index = m * 8 + phys_vec;      // 64 rows * 8 chunks    return stage_base + (chunk_index * 16) + (elem * 2);}
```

Equivalently, each row has 8 logical 16-byte chunks, but the physical chunk index is XOR-swizzled by the low 3 bits of the row.

This is the draft exact layout I would use for the **transformed activation tile**.

## 4) Weight shared-memory layout

### INT4 kernel

Store the `64 x 128` quantized weight tile **contiguously** in shared memory:

```C// 4096 B totaluint8_t B4_smem[64][64];   // each row has 64 bytes = 128 int4 values
C// 4096 B totaluint8_t B4_smem[64][64];   // each row has 64 bytes = 128 int4 values
```

Interpretation:

- `B4_smem[k][byte_col]`
- each byte contains two 4-bit weights
- offline packing is already done so each warp’s 16-byte vector is laid out in the order needed for direct register dequantization

Marlin explicitly preprocesses weights offline, reshuffles `16 x 64` tiles to contiguous order, and packs each thread’s 16-byte vector so it contains the 8 quantized weights needed for 4 separate `16 x 16` Tensor Core blocks, with the nibble pattern `64207531` inside the `INT32`s. I would keep that same idea for the INT4 specialization. [arXiv+1](https://arxiv.org/pdf/2408.11743)

### INT8 kernel

Store the `64 x 64` weight tile **contiguously** in shared memory:

```C// 4096 B totaluint8_t B8_smem[64][64];   // one int8 weight per byte
C// 4096 B totaluint8_t B8_smem[64][64];   // one int8 weight per byte
```

No nibble-interleave is needed; just preserve the same **column-major block order** as the INT4 kernel so the warp schedule stays structurally identical.

### Per-output scales

For both specializations, store one FP16 scale per output channel for the current logical 128-group:

```Chalf S4_smem[128];   // 256 Bhalf S8_smem[64];    // 128 B
Chalf S4_smem[128];   // 256 Bhalf S8_smem[64];    // 128 B
```

Marlin notes that for grouped quantization it reorganizes scales similarly to weights and, although scales for group-size 128 technically only need to refresh every other `K_STAGE=64`, it still reloads them regularly to keep compiler scheduling stable. I would do the same for this first kernel revision. [arXiv](https://arxiv.org/pdf/2408.11743)

## 5) Transform metadata layout

Use a tiny fixed-format transform block per `K_STAGE = 64`:

```Cstruct __align__(16) RotationMeta {    uint8_t a;     // column index 0..63    uint8_t b;     // column index 0..63    uint16_t pad0;    half c;        // cos(theta)    half s;        // sin(theta)    uint32_t pad1;};  // 8 bytes// Shared-memory transform block, 256 B alignedstruct __align__(16) TransformStage {    half scale[64];        // 128 B    RotationMeta rot[8];   // 64 B  (up to 8 sparse pairs per 64-col K block)    uint8_t pad[64];       // pad to 256 B};
Cstruct __align__(16) RotationMeta {    uint8_t a;     // column index 0..63    uint8_t b;     // column index 0..63    uint16_t pad0;    half c;        // cos(theta)    half s;        // sin(theta)    uint32_t pad1;};  // 8 bytes// Shared-memory transform block, 256 B alignedstruct __align__(16) TransformStage {    half scale[64];        // 128 B    RotationMeta rot[8];   // 64 B  (up to 8 sparse pairs per 64-col K block)    uint8_t pad[64];       // pad to 256 B};
```

This keeps the runtime transform tile-local and tiny enough to stage with the weights.

## 6) Warp mapping

### INT4 kernel: 8 warps, 64 x 128

Map warps as:

```Cwarp_id = threadIdx.x >> 5;     // 0..7lane    = threadIdx.x & 31;wm = warp_id & 3;               // 0..3 -> rowswn = warp_id >> 2;              // 0..1 -> 64-col halfrow_base = wm * 16;             // 0,16,32,48col_base = wn * 64;             // 0 or 64
Cwarp_id = threadIdx.x >> 5;     // 0..7lane    = threadIdx.x & 31;wm = warp_id & 3;               // 0..3 -> rowswn = warp_id >> 2;              // 0..1 -> 64-col halfrow_base = wm * 16;             // 0,16,32,48col_base = wn * 64;             // 0 or 64
```

Each warp computes a `16 x 64` output tile.

### INT8 kernel: 4 warps, 64 x 64

```Cwarp_id = threadIdx.x >> 5;     // 0..3lane    = threadIdx.x & 31;wm = warp_id;                   // 0..3wn = 0;row_base = wm * 16;col_base = 0;
Cwarp_id = threadIdx.x >> 5;     // 0..3lane    = threadIdx.x & 31;wm = warp_id;                   // 0..3wn = 0;row_base = wm * 16;col_base = 0;
```

Again, each warp computes a `16 x 64` tile.

## 7) ldmatrix schedule

For the activation operand, use **one ldmatrix.sync.aligned.m8n8.x4.shared.b16 per warp per k16 slice**.

Why this exact instruction:

- PTX defines `ldmatrix` as the warp-level matrix load for `mma`
- `.m8n8.x4.shared.b16` loads **four** `8 x 8` matrices of 16-bit elements
- that is exactly what a `16 x 16` FP16 A-fragment needs: top-left, bottom-left, top-right, bottom-right submatrices. [NVIDIA Docs+1](https://docs.nvidia.com/cuda/parallel-thread-execution/)

### Address supply pattern

For the warp’s current `16 x 16` A fragment:

- lanes `0–7` supply row pointers for rows `row_base + 0..7`, cols `k0 + 0..7`
- lanes `8–15` supply row pointers for rows `row_base + 8..15`, cols `k0 + 0..7`
- lanes `16–23` supply row pointers for rows `row_base + 0..7`, cols `k0 + 8..15`
- lanes `24–31` supply row pointers for rows `row_base + 8..15`, cols `k0 + 8..15`

That is the `.x4` row-address pattern PTX describes for lanes `0–7 / 8–15 / 16–23 / 24–31`. [NVIDIA Docs](https://docs.nvidia.com/cuda/parallel-thread-execution/)

### Exact k loop schedule

`K_STAGE = 64` is split into 4 microsteps of `k_step = 16`.

For each warp:

```Cfor (int ks = 0; ks < 4; ++ks) {    int k0 = ks * 16;    // One ldmatrix.x4 gives the full 16x16 A fragment for this warp and k-slice    // logical submatrices:    //   M0 = rows 0..7,  cols 0..7    //   M1 = rows 8..15, cols 0..7    //   M2 = rows 0..7,  cols 8..15    //   M3 = rows 8..15, cols 8..15    ldmatrix.sync.aligned.m8n8.x4.shared.b16 ...}
Cfor (int ks = 0; ks < 4; ++ks) {    int k0 = ks * 16;    // One ldmatrix.x4 gives the full 16x16 A fragment for this warp and k-slice    // logical submatrices:    //   M0 = rows 0..7,  cols 0..7    //   M1 = rows 8..15, cols 0..7    //   M2 = rows 0..7,  cols 8..15    //   M3 = rows 8..15, cols 8..15    ldmatrix.sync.aligned.m8n8.x4.shared.b16 ...}
```

## 8) MMA schedule

Use `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`.

PTX documents the `m16n8k16` FP16 MMA form, and Marlin’s paper explicitly says its inner loop computes an `M x 16` times `16 x 64` matmul and does so **column-wise** using `16 x 8` Tensor Core instructions so the next B fragment can be dequantized while the current column is being multiplied. [NVIDIA Docs+1](https://docs.nvidia.com/cuda/parallel-thread-execution/)

### Per-warp inner loop

For each `ks`:

- load A once via `ldmatrix.x4`
- then execute **8 MMA instructions** for the warp’s `16 x 64` output tile

```Cfor (int j = 0; j < 8; ++j) {    // j-th 16x8 B fragment of the warp's 16x64 tile    // col range = col_base + j*8 ... col_base + j*8 + 7    // 1) decode B_j from shared-memory packed format into registers    // 2) apply per-output scales    // 3) issue mma.sync on A_frag x B_j -> accum_j    mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 ...}
Cfor (int j = 0; j < 8; ++j) {    // j-th 16x8 B fragment of the warp's 16x64 tile    // col range = col_base + j*8 ... col_base + j*8 + 7    // 1) decode B_j from shared-memory packed format into registers    // 2) apply per-output scales    // 3) issue mma.sync on A_frag x B_j -> accum_j    mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 ...}
```

### Pipelined decode

Follow Marlin’s column-wise strategy:

- while `mma(j)` is executing, decode `B(j+1)` into the next register set
- keep **A fixed** across the 8 column slices for this `ks`
- move to the next `ks` only after all 8 column slices are accumulated

That is the schedule Marlin uses to overlap weight decode with Tensor Core math. [arXiv](https://arxiv.org/pdf/2408.11743)

## 9) Activation-transform schedule

This is the Paro-style piece, but done in the only place I think is feasible for Ampere:

- **load activations into XOR-swizzled shared memory**
- **apply the inverse transform in shared memory**
- then `ldmatrix` from the transformed tile

I would do this exactly once per stage, before the first `ldmatrix` of that stage.

### Per-stage transform pass

Each warp owns its `16 x 64` row slab:

```C// warp owns rows row_base .. row_base+15for each local row r in row_base .. row_base+15:    // 1) scale all 64 columns    for k in 0..63:        A[r,k] *= scale[k]    // 2) apply sparse pairwise Givens rotations    for t in 0..ROT_COUNT-1:        u = rot[t].a        v = rot[t].b        c = rot[t].c        s = rot[t].s        au = A[r,u]        av = A[r,v]        A[r,u] = c * au + s * av        A[r,v] = -s * au + c * av
C// warp owns rows row_base .. row_base+15for each local row r in row_base .. row_base+15:    // 1) scale all 64 columns    for k in 0..63:        A[r,k] *= scale[k]    // 2) apply sparse pairwise Givens rotations    for t in 0..ROT_COUNT-1:        u = rot[t].a        v = rot[t].b        c = rot[t].c        s = rot[t].s        au = A[r,u]        av = A[r,v]        A[r,u] = c * au + s * av        A[r,v] = -s * au + c * av
```

Because the transform is tile-local over the `K_STAGE = 64` columns and sparse (for example 8 pairs), this is still practical as a shared-memory preprocessing step. It does **not** go inside the MMA inner loop.

## 10) cp.async pipeline

Use a 4-stage pipeline like this:

### Prologue

Prefetch stages `0,1,2,3`:

- A tile → `A*_base(stage)`
- transform metadata → `T*_base(stage)`
- B tile → `B*_base(stage)` using `evict_first` on weight loads
- scale tile → `S*_base(stage)`

Marlin explicitly says it uses `cp.async`, uses `evict first` for B to avoid polluting L2 with weights that are only read once, and chooses pipeline depth `P = 4`. [arXiv](https://arxiv.org/pdf/2408.11743)

### Main loop

For `t = 0 .. num_kstages-1`:

```Ccur = t & 3nxt = (t + 4) & 3// prefetch future stage if it existscp.async(... A[nxt] ...)cp.async(... T[nxt] ...)cp.async(... B[nxt] ...)   // evict_first for weightscp.async(... S[nxt] ...)cp.async.commit_group()cp.async.wait_group(1)__syncthreads()// transform A[cur] in shared memoryapply_transform(A[cur], T[cur])__syncthreads()// compute on current stagefor ks in 0..3:    ldmatrix A fragment    for j in 0..7:        decode next B fragment        mma current fragment
Ccur = t & 3nxt = (t + 4) & 3// prefetch future stage if it existscp.async(... A[nxt] ...)cp.async(... T[nxt] ...)cp.async(... B[nxt] ...)   // evict_first for weightscp.async(... S[nxt] ...)cp.async.commit_group()cp.async.wait_group(1)__syncthreads()// transform A[cur] in shared memoryapply_transform(A[cur], T[cur])__syncthreads()// compute on current stagefor ks in 0..3:    ldmatrix A fragment    for j in 0..7:        decode next B fragment        mma current fragment
```

### Epilogue

Store accumulators to global.

## 11) What this gives you

This draft meets the constraints you laid out:

- **no runtime codebook decode**
- **no mixed INT4/INT8 branching inside a warp tile**
- **Paro-style transforms happen outside the MMA hot loop**
- **QKV remains uniform**
- **the inner loop stays Marlin-like**
- **the only ldmatrix path is FP16 A**, which is exactly where Ampere/Marlin already want it. Marlin’s design and PTX’s documented `ldmatrix`/`mma` behavior are the foundation here. [arXiv+2NVIDIA Docs+2](https://arxiv.org/pdf/2408.11743)

If you want, next I’ll turn this into:

- a **PTX-level warp pseudocode** for the `ldmatrix` + `mma.sync` sequence, or
- a **CUTLASS/CUDA kernel skeleton** with the exact stage offsets and warp mapping baked in.

---

### User:
Here is the INT4 CUDA kernel skeleton. It locks in the 4-stage pipeline, `cp.async` routing, and the required synchronization barriers. 

The INT8 version is structurally identical, just with `N_TILE = 64` and its corresponding byte offsets.

```
cpp
#include <cuda_fp16.h>
#include <mma.h>

// ------------------------------
// Pipeline & Tile Constants (INT4)
// ------------------------------
constexpr int PIPE = 4;
constexpr int M_TILE = 64;
constexpr int N_TILE4 = 128;
constexpr int K_STAGE = 64;

// Shared Memory Byte Sizes per Stage
constexpr int A_STAGE_BYTES     = 8192; // 64 * 64 * 2 bytes
constexpr int XFORM_STAGE_BYTES = 256;  // Scales + Sparse Rotations
constexpr int W4_STAGE_BYTES    = 4096; // 64 * 128 / 2 (4-bit)
constexpr int S4_STAGE_BYTES    = 256;  // 128 * 2 bytes (FP16 scales)

constexpr int STAGE4_BYTES = A_STAGE_BYTES + XFORM_STAGE_BYTES + W4_STAGE_BYTES + S4_STAGE_BYTES; // 12800 B
constexpr int TOTAL_SMEM_BYTES = PIPE * STAGE4_BYTES; // 51200 B

// Async Copy Macros
#define CP_ASYNC_CG()  asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT(N) asm volatile("cp.async.wait_group %0;\n" :: "n"(N))
// .L2::evict_first for weights
#define CP_ASYNC_CG_EVICT(dst, src, bytes) \
    asm volatile("cp.async.cg.shared.global.L2::evict_first [%0], [%1], %2;\n" :: "r"(dst), "l"(src), "n"(bytes))

__global__ void __launch_bounds__(256) ampere_marlin_paro_int4_kernel(
    const half* __restrict__ A,
    const uint8_t* __restrict__ B_quant,
    const half* __restrict__ Scales,
    const void* __restrict__ XformMeta,
    half* __restrict__ C,
    int num_k_stages) 
{
    // Shared memory allocation
    extern __shared__ uint8_t smem[];

    // Thread/Warp indexing
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    // Output tile indices
    const int block_m = blockIdx.x * M_TILE;
    const int block_n = blockIdx.y * N_TILE4;

    // Accumulators (16x64 output per warp -> 8x M16N8K16 blocks)
    // 4 registers per M16N8K16 accumulator * 8 blocks = 32 registers
    float accum[32] = {0.0f};

    // ------------------------------
    // Prologue: Fill Pipeline
    // ------------------------------
    for (int stage = 0; stage < PIPE - 1; ++stage) {
        int smem_base = stage * STAGE4_BYTES;
        
        // TODO: Map tid to specific 16-byte chunks for cp.async
        // cp.async A -> smem_base
        // cp.async XformMeta -> smem_base + A_STAGE_BYTES
        // CP_ASYNC_CG_EVICT B_quant -> smem_base + A_STAGE_BYTES + XFORM_STAGE_BYTES
        // cp.async Scales -> smem_base + A_STAGE_BYTES + XFORM_STAGE_BYTES + W4_STAGE_BYTES
        
        CP_ASYNC_CG();
    }

    // ------------------------------
    // Main Loop
    // ------------------------------
    int smem_load_idx = 0;
    int smem_store_idx = PIPE - 1;

    for (int t = 0; t < num_k_stages; ++t) {
        // 1. Issue async copies for the next stage (if valid)
        if (t + PIPE - 1 < num_k_stages) {
            int store_base = smem_store_idx * STAGE4_BYTES;
            
            // TODO: issue cp.async for A, Xform, B_quant, Scales -> store_base
            
            CP_ASYNC_CG();
        }

        // 2. Wait for the oldest stage to be fully resident in shared memory
        // If PIPE=4, we wait until only 3 uncommitted groups remain
        CP_ASYNC_WAIT(PIPE - 2);
        __syncthreads(); // Barrier 1: smem is safe to read/mutate for current stage

        // 3. Paro-style Transform (Shared Memory Mutation)
        int load_base = smem_load_idx * STAGE4_BYTES;
        half* A_smem = (half*)(smem + load_base);
        void* T_smem = (void*)(smem + load_base + A_STAGE_BYTES);
        
        // Each warp owns a 16-row slab of the 64x64 A tile.
        // Apply channel scales and sparse Givens rotations here.
        // apply_paro_transform(A_smem, T_smem, warp_id, tid);

        __syncthreads(); // Barrier 2: transform is globally visible to all warps

        // 4. Tensor Core Inner Loop
        uint8_t* B_smem = (uint8_t*)(smem + load_base + A_STAGE_BYTES + XFORM_STAGE_BYTES);
        half* S_smem = (half*)(smem + load_base + A_STAGE_BYTES + XFORM_STAGE_BYTES + W4_STAGE_BYTES);

        #pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            // TODO: ldmatrix.sync.aligned.m8n8.x4.shared.b16 (A fragment)
            
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                // TODO: 
                // - Decode B_smem column j -> registers
                // - Apply S_smem scale
                // - mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
            }
        }

        // Advance pipeline
        smem_load_idx = (smem_load_idx + 1) % PIPE;
        smem_store_idx = (smem_store_idx + 1) % PIPE;
    }

    // Drain remaining pipeline stages
    // ... wait_group(2) -> math ... wait_group(1) -> math ... wait_group(0) -> math

    // ------------------------------
    // Epilogue
    // ------------------------------
    // Write accum[] to global memory C
}
```

Review the synchronization logic around the transform phase. `CP_ASYNC_WAIT(PIPE - 2)` ensures the oldest stage is resident. The first `__syncthreads()` guarantees the warp can safely mutate `A_smem`. The second `__syncthreads()` guarantees the transform is visible to all warps before the MMA inner loop starts blasting `ldmatrix`.

Let me know if you spot any pipeline stalls or edge cases here. Otherwise, we move to the PTX `ldmatrix` / `mma.sync` inner loop implementation next.

---

### ChatGPT:
The transform-phase logic is **almost right**, but there is one real race and one conservative barrier you can probably remove.

The good part first: `cp.async.wait_group` only makes `cp.async` writes visible to the **executing thread**, not to the whole CTA, and it does not order unrelated memory ops. So your first `__syncthreads()` after `CP_ASYNC_WAIT(PIPE - 2)` is necessary if all warps are about to read or mutate the stage’s shared-memory tile. That matches the PTX semantics exactly. [NVIDIA Docs+1](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)

The main issue is this:

## 1) You are missing a **third barrier** after the MMA phase

Right now, a fast warp can finish compute on `smem_load_idx`, advance the ring, and start issuing `cp.async` into `smem_store_idx` for the next iteration **while a slower warp is still reading from the old stage**. Because Ampere has independent warp scheduling, you cannot assume all warps leave the inner loop at the same instant. PTX also does not give you any cross-warp safety here. [NVIDIA Docs+1](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)

So the safe structure is:

```C++// wait current stage residentCP_ASYNC_WAIT(PIPE - 2);__syncthreads();      // Barrier A: all cp.async writes for current stage are CTA-visible// mutate current stageapply_paro_transform(...);// If transform touches only warp-private rows, Barrier B may be removable.// Otherwise keep it.__syncthreads();      // Barrier B: transformed tile globally visible// consume current stagefor (int ks = 0; ks < 4; ++ks) {    ...}__syncthreads();      // Barrier C: no warp is still reading this stage before ring-buffer reuse
C++// wait current stage residentCP_ASYNC_WAIT(PIPE - 2);__syncthreads();      // Barrier A: all cp.async writes for current stage are CTA-visible// mutate current stageapply_paro_transform(...);// If transform touches only warp-private rows, Barrier B may be removable.// Otherwise keep it.__syncthreads();      // Barrier B: transformed tile globally visible// consume current stagefor (int ks = 0; ks < 4; ++ks) {    ...}__syncthreads();      // Barrier C: no warp is still reading this stage before ring-buffer reuse
```

That **Barrier C** is the one your skeleton is currently missing.

## 2) CP_ASYNC_WAIT(PIPE - 2) is correct only under one strict assumption

With `PIPE = 4`, `wait_group(2)` is fine **if and only if**:

- each pipeline stage corresponds to exactly **one** committed cp.async group, and
- you consume exactly **one** stage per iteration.

PTX says `cp.async.wait_group N` waits until only `N` or fewer of the most recent groups remain pending, and all earlier committed groups are complete. That means your oldest stage is resident if your “one stage = one group” bookkeeping is exact. [NVIDIA Docs+1](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)

If you ever split one stage across multiple `commit_group`s, `wait_group(PIPE - 2)` stops meaning “current stage is ready.”

## 3) Your second barrier is probably **conservative**, not always mandatory

If `apply_paro_transform()` is written so that:

- each warp only mutates its own `16 x 64` slab of `A_smem`, and
- each warp’s later `ldmatrix` reads only that same slab,

then the **post-transform __syncthreads()** is not strictly required for correctness of `A_smem`. In that case, each warp can transform its slab and immediately start its own MMA loop.

So:

- keep **Barrier A** always,
- keep **Barrier C** always,
- treat **Barrier B** as:

**required** if any warp reads rows transformed by another warp,

**optional** if row ownership is fully disjoint.
- **required** if any warp reads rows transformed by another warp,
- **optional** if row ownership is fully disjoint.

That is probably your biggest easy stall to remove later.

## 4) Prologue / short-K edge case

Your prologue currently fills `PIPE - 1` stages unconditionally. That is only valid if `num_k_stages >= PIPE - 1`. If the K-loop is short, you need:

```C++int prologue = min(PIPE - 1, num_k_stages);for (int stage = 0; stage < prologue; ++stage) { ... }
C++int prologue = min(PIPE - 1, num_k_stages);for (int stage = 0; stage < prologue; ++stage) { ... }
```

And the drain path must mirror how many groups were actually committed.

## 5) One subtle issue in the top-of-loop prefetch placement

Prefetching the next stage at the **top** of the loop is okay only if the previous iteration already ended with Barrier C. Without that barrier, reusing `smem_store_idx` is unsafe, as above.

If you want the control flow to be easier to reason about, move the next-stage prefetch to the **end** of the loop, immediately after Barrier C. That makes the ring-buffer lifetime obvious.

## 6) Practical inline-PTX footnote

Not synchronization-related, but your `cp.async` macro will usually need a **shared-memory address**, typically from `__cvta_generic_to_shared()`, not a raw C pointer cast. `ldmatrix` has the same caveat. PTX’s `ldmatrix` examples explicitly use shared addresses. [NVIDIA Docs](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)

## My recommendation

Use this as the safe skeleton:

```C++// prologuefor (int stage = 0; stage < min(PIPE - 1, num_k_stages); ++stage) {    issue_cp_async_for_stage(stage);    CP_ASYNC_CG();}for (int t = 0; t < num_k_stages; ++t) {    // prefetch future stage only if it exists    if (t + PIPE - 1 < num_k_stages) {        issue_cp_async_for_stage(smem_store_idx);        CP_ASYNC_CG();    }    CP_ASYNC_WAIT(PIPE - 2);    __syncthreads();          // A: stage resident + CTA-visible    apply_paro_transform(...);    __syncthreads();          // B: keep for now, remove later only if provably warp-private    do_mma_inner_loop(...);    __syncthreads();          // C: stage no longer in use before ring reuse    smem_load_idx = (smem_load_idx + 1) % PIPE;    smem_store_idx = (smem_store_idx + 1) % PIPE;}
C++// prologuefor (int stage = 0; stage < min(PIPE - 1, num_k_stages); ++stage) {    issue_cp_async_for_stage(stage);    CP_ASYNC_CG();}for (int t = 0; t < num_k_stages; ++t) {    // prefetch future stage only if it exists    if (t + PIPE - 1 < num_k_stages) {        issue_cp_async_for_stage(smem_store_idx);        CP_ASYNC_CG();    }    CP_ASYNC_WAIT(PIPE - 2);    __syncthreads();          // A: stage resident + CTA-visible    apply_paro_transform(...);    __syncthreads();          // B: keep for now, remove later only if provably warp-private    do_mma_inner_loop(...);    __syncthreads();          // C: stage no longer in use before ring reuse    smem_load_idx = (smem_load_idx + 1) % PIPE;    smem_store_idx = (smem_store_idx + 1) % PIPE;}
```

So the answer is:

- **wait_group(PIPE - 2) is fine**
- **first barrier is required**
- **second barrier is maybe removable**
- **you are definitely missing a final barrier after the MMA loop**

That final barrier is the real correctness fix.

Paste the `ldmatrix` / `mma.sync` inner loop next and I’ll review the register ownership and fragment schedule.

---

### User:
You nailed the missing Barrier C. Good catch on the `cp.async` short-K drain logic too. Reusing the ring buffer without guaranteeing the previous consumers are done is a classic race condition, and Ampere's independent warp schedulers will absolutely punish it.

Here is the `ldmatrix` + `mma.sync` inner loop locked in. This drops directly into the `do_mma_inner_loop(...)` placeholder. It assumes the offline INT4 packing and nibble interleaving are already strictly aligned to what a 16x8 MMA instruction expects in its registers.

```
cpp
// ------------------------------
// Registers & Addressing Setup
// (Place before the K-stage loop)
// ------------------------------

// Accumulators: 8 column slices * 4 registers (for m16n8k16 f16 accumulators)
uint32_t RC[8][4]; 
#pragma unroll
for (int j = 0; j < 8; ++j) {
    RC[j][0] = 0; RC[j][1] = 0; RC[j][2] = 0; RC[j][3] = 0;
}

// Lane to A_smem address mapping (XOR swizzle)
int m_local = lane % 16;
int k_local = (lane / 16) * 8; // 0 or 8
int row = row_base + m_local;

// Base shared memory pointers for the generic address space conversion
uint32_t a_smem_base;
asm("cvta.to.shared.u32 %0, %1;" : "=r"(a_smem_base) : "l"(A_smem));

uint32_t b_smem_base;
asm("cvta.to.shared.u32 %0, %1;" : "=r"(b_smem_base) : "l"(B_smem));

// ------------------------------
// Inner Loop
// ------------------------------
#pragma unroll
for (int ks = 0; ks < 4; ++ks) {
    uint32_t RA[4]; // 16x16 A fragment (4x 32-bit registers)
    
    // 1. Calculate lane's specific ldmatrix address for this ks
    int col = ks * 16 + k_local;
    int vec = col / 8;
    int phys_vec = vec ^ (row & 0x7);
    uint32_t a_smem_addr = a_smem_base + (row * 8 + phys_vec) * 16;

    // 2. Load 16x16 A fragment
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(RA[0]), "=r"(RA[1]), "=r"(RA[2]), "=r"(RA[3])
        : "r"(a_smem_addr)
    );

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        uint32_t RB_quant[1]; // 8 INT4 weights = 4 bytes = 1x 32-bit register
        uint32_t RB_fp16[2];  // Dequantized to 8x FP16 = 2x 32-bit registers
        
        // Output column index for this 16x8 slice
        int global_col = col_base + j * 8;

        // 3. Load packed INT4 B fragment from shared memory
        // B layout: 64 rows x 128 cols (4-bit) -> mapped contiguous
        // Lane reads exactly 4 bytes containing its 8 required weights
        int b_row = ks * 16 + (lane % 16); 
        int b_byte_col = (global_col / 2) + (lane / 16) * 4;
        uint32_t b_smem_addr = b_smem_base + (b_row * 64) + b_byte_col;
        
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(RB_quant[0]) : "r"(b_smem_addr));

        // 4. Dequantize INT4 -> FP16 (assumes Marlin offline 64207531 nibble packing)
        // Extract nibbles, cast to FP16, apply per-channel scales
        half scale_0 = S_smem[global_col + (lane / 16) * 4 + 0];
        half scale_1 = S_smem[global_col + (lane / 16) * 4 + 1];
        half scale_2 = S_smem[global_col + (lane / 16) * 4 + 2];
        half scale_3 = S_smem[global_col + (lane / 16) * 4 + 3];

        // [Insert highly optimized PTX bit-extraction & scale multiplication here]
        // E.g., using prmt.b32 and fma.rn.f16x2 to populate RB_fp16[0] and RB_fp16[1]
        // This is a placeholder for the raw decode sequence to keep the structure clear.
        RB_fp16[0] = /* decoded and scaled */ 0; 
        RB_fp16[1] = /* decoded and scaled */ 0;

        // 5. Tensor Core MMA (16x8x16 FP16)
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13};"
            : "=r"(RC[j][0]), "=r"(RC[j][1]), "=r"(RC[j][2]), "=r"(RC[j][3])
            : "r"(RA[0]), "r"(RA[1]), "r"(RA[2]), "r"(RA[3]),
              "r"(RB_fp16[0]), "r"(RB_fp16[1]),
              "r"(RC[j][0]), "r"(RC[j][1]), "r"(RC[j][2]), "r"(RC[j][3])
        );
    }
}
```

**Two key technical points here:**

1.  **Addressing:** The `a_smem_addr` logic precisely maps PTX `.x4` thread roles to the XOR-swizzled layout. Using `cvta.to.shared.u32` ensures you pass a 32-bit shared memory pointer to `ldmatrix`, averting 64-bit generic pointer overhead.
2.  **Overlap:** Loading `A` outside the `j` loop anchors it. Dequantizing `B_quant` directly before the `mma.sync` instruction is what allows the compiler to pipeline the math of slice `j` with the decode of slice `j+1`. 

Look over the register allocations. If we are good, we can crack open the raw `prmt.b32` bit-magic for the INT4 to FP16 dequantization step.

---

### ChatGPT:
I’m checking the PTX operand mapping before I bless this, because `ldmatrix` lane addressing is where these kernels usually go subtly wrong.

---

This is **close**, but I would not move to the `prmt` decode yet.
There are **2 correctness issues** and **1 performance trap** to fix first.

## What is correct

Your **A-side ldmatrix grouping** is consistent with PTX’s `.x4` contract: the four 8-thread groups provide the row addresses for the four 8x8 matrices in a 16x16 fragment, i.e. lanes `0–7`, `8–15`, `16–23`, and `24–31` supply `addr0–7`, `addr8–15`, `addr16–23`, and `addr24–31` respectively. So the high-level idea of:

- `row = row_base + lane % 16`
- `k_local = (lane / 16) * 8`
is aligned with the required quadrant split for a 16x16 A fragment. PTX also explicitly says `ldmatrix` is warp-wide and the `.x4` form loads the third and fourth matrices into the subsequent destination registers. [NVIDIA Docs+1](https://docs.nvidia.com/cuda/parallel-thread-execution/)

Your synchronization story is also basically right now **if** you keep the missing post-MMA barrier:

- `cp.async.wait_group(...)`
- `__syncthreads()` before transform
- optional `__syncthreads()` after transform
- **required** `__syncthreads()` after the MMA loop before ring-buffer reuse

That last barrier is required because `cp.async.wait_group` only makes the async copies visible to the **executing thread**, and PTX says it does **not** provide ordering/visibility guarantees for other memory operations. [NVIDIA Docs](https://docs.nvidia.com/cuda/parallel-thread-execution/)

## Must-fix #1: your mma.sync accumulator constraints are wrong

This is the biggest inline-PTX bug in the snippet.

You currently use:

```C++: "=r"(RC[j][0]), ......: ..., "r"(RC[j][0]), ...
C++: "=r"(RC[j][0]), ......: ..., "r"(RC[j][0]), ...
```

That is not the right way to express an in-place accumulator. PTX explicitly says `D` and `C` may refer to the same matrix, so the clean inline-asm form is to use **read-write outputs** for `RC` and then reference those same operands as both `D` and `C` in the template. [NVIDIA Docs](https://docs.nvidia.com/cuda/parallel-thread-execution/)

Use this pattern instead:

```C++asm volatile(    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "    "{%0, %1, %2, %3}, "    "{%4, %5, %6, %7}, "    "{%8, %9}, "    "{%0, %1, %2, %3};\n"    : "+r"(RC[j][0]), "+r"(RC[j][1]), "+r"(RC[j][2]), "+r"(RC[j][3])    : "r"(RA[0]), "r"(RA[1]), "r"(RA[2]), "r"(RA[3]),      "r"(RB_fp16[0]), "r"(RB_fp16[1]));
C++asm volatile(    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "    "{%0, %1, %2, %3}, "    "{%4, %5, %6, %7}, "    "{%8, %9}, "    "{%0, %1, %2, %3};\n"    : "+r"(RC[j][0]), "+r"(RC[j][1]), "+r"(RC[j][2]), "+r"(RC[j][3])    : "r"(RA[0]), "r"(RA[1]), "r"(RA[2]), "r"(RA[3]),      "r"(RB_fp16[0]), "r"(RB_fp16[1]));
```

That is the safe version.

## Must-fix #2: your B-fragment sizing is inconsistent

For `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`, the PTX operand counts are:

- `A`: **4** `.f16x2` regs
- `B`: **2** `.f16x2` regs
- `C/D`: **4** accumulator regs

That means each lane contributes **4 FP16 values** for `B`, not 8. The PTX register count is not ambiguous here. [Stack Overflow](https://stackoverflow.com/questions/70683955/understanding-the-parameters-of-ptx-instruction-mma-sync-aligned-m16n8k16-row-co)

So this part of your code is mathematically inconsistent:

```C++uint32_t RB_quant[1]; // 8 INT4 weights = 4 bytes = 1x 32-bit registeruint32_t RB_fp16[2];  // Dequantized to 8x FP16 = 2x 32-bit registers
C++uint32_t RB_quant[1]; // 8 INT4 weights = 4 bytes = 1x 32-bit registeruint32_t RB_fp16[2];  // Dequantized to 8x FP16 = 2x 32-bit registers
```

Two 32-bit regs can only hold **4 half values**, not 8.

So you need one of these two fixes:

### Option A: per-lane load only the 4 INT4s actually needed

Then decode those into the two `RB_fp16` registers.

### Option B: overfetch 8 INT4s intentionally, but only select the 4 values needed for the current lane’s B fragment

That can be valid, but then your comments and mapping need to say that explicitly.

Right now the code/comments imply “8 int4 -> 8 fp16 -> 2 regs,” which is impossible.

## Likely-wrong area: B shared-memory addressing

This line is the one I would not trust yet:

```C++int b_row = ks * 16 + (lane % 16);int b_byte_col = (global_col / 2) + (lane / 16) * 4;
C++int b_row = ks * 16 + (lane % 16);int b_byte_col = (global_col / 2) + (lane / 16) * 4;
```

It is too coarse for a real `m16n8k16` B-fragment mapping.

Why I’m skeptical:

- `lane / 16` only gives you **2 groups**
- but the warp-level B-fragment layout for `m16n8k16` is more structured than a simple upper/lower half split
- and your current mapping does not show how the lane-local 4 FP16 B values are selected consistently across the 16x8 fragment

In other words: the **A-side map is believable**, the **B-side map is not finished**.

My recommendation is to derive the exact lane→B mapping from a known-good `m16n8k16` warp fragment layout before you write the `prmt` path. Until that is nailed down, the decode work is premature.

## Performance trap: the decode/MMA overlap is not guaranteed yet

This sentence is slightly too optimistic:

Dequantizing `B_quant` directly before the `mma.sync` instruction is what allows the compiler to pipeline the math of slice `j` with the decode of slice `j+1`.

Maybe. Not guaranteed.

As written, each `j` iteration is:

1. load packed B
2. decode
3. scale
4. `mma.sync`

The compiler may overlap some of it, but if you want the overlap to be **structural**, use a 2-buffer register pipeline:

```C++uint32_t RB_q_cur, RB_q_nxt;uint32_t RB_cur[2], RB_nxt[2];// preload j = 0load+decode B(0) -> RB_curfor (int j = 0; j < 8; ++j) {    if (j + 1 < 8) {        load+decode B(j+1) -> RB_nxt    }    mma.sync(... RB_cur ...)    if (j + 1 < 8) {        RB_cur[0] = RB_nxt[0];        RB_cur[1] = RB_nxt[1];    }}
C++uint32_t RB_q_cur, RB_q_nxt;uint32_t RB_cur[2], RB_nxt[2];// preload j = 0load+decode B(0) -> RB_curfor (int j = 0; j < 8; ++j) {    if (j + 1 < 8) {        load+decode B(j+1) -> RB_nxt    }    mma.sync(... RB_cur ...)    if (j + 1 < 8) {        RB_cur[0] = RB_nxt[0];        RB_cur[1] = RB_nxt[1];    }}
```

That makes the intended overlap explicit instead of leaving it to compiler luck.

## Register pressure assessment

The accumulator footprint is fine as a starting point:

- `RC[8][4]` = **32 regs**
- `RA[4]` = **4 regs**
- `RB_fp16[2]` = **2 regs**

So the visible math state is only **38 regs** before pointers, scales, decode temporaries, loop vars, and compiler-generated spill buffers. That is not scary yet for a 256-thread CTA on Ampere.

The real register danger is not `RC`; it is:

- scalar scale loads (`scale_0..scale_3`)
- nibble unpack temps
- `prmt` scratch
- address arithmetic scratch

So when you implement the decode, do **not** keep four scalar `half scale_*` values. Pack them as two `half2`-style 32-bit regs instead.

Something like:

```C++uint32_t scale01 = *reinterpret_cast<const uint32_t*>(&S_smem[global_col + scale_idx0]);uint32_t scale23 = *reinterpret_cast<const uint32_t*>(&S_smem[global_col + scale_idx2]);
C++uint32_t scale01 = *reinterpret_cast<const uint32_t*>(&S_smem[global_col + scale_idx0]);uint32_t scale23 = *reinterpret_cast<const uint32_t*>(&S_smem[global_col + scale_idx2]);
```

That keeps the B-side shape aligned with the 2-reg MMA operand.

## Verdict

I would mark it like this:

- **A-side ldmatrix addressing:** plausible / basically good
- **synchronization (with Barrier C):** good
- **accumulator asm constraints:** must fix now
- **B fragment sizing/comments:** definitely wrong as written
- **B addressing:** not trustworthy yet
- **decode/MMA overlap:** needs explicit double-buffering if you want to rely on it

So no, I would **not** open the raw `prmt.b32` bit-magic yet.
First fix the `mma` accumulator constraints and re-derive the exact per-lane `B` fragment map for `m16n8k16`.

Paste the B-fragment lane map next, or I can draft the exact **lane→4-half B operand table** for `m16n8k16.row.col`.

---

### User:
Good catches across the board. The in-place accumulator asm syntax was a stupid oversight, and I totally ghosted the B-fragment size—two 32-bit regs hold exactly 4 FP16s, end of story. Leaving the decode/MMA overlap to compiler luck is also a bad bet on Ampere. 

Let's lock in the exact B-fragment lane map for `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` before we touch the decode sequence.

### The `m16n8k16.row.col` B-Fragment Map

Since B is `.col` (column-major) with dimensions K=16, N=8:
* The 32 threads in a warp must supply 16 x 8 = 128 elements.
* 128 elements / 32 threads = **4 elements per thread** (which perfectly fits into our two `.f16x2` registers).

Per the PTX ISA, the distribution of the 16x8 B matrix across the 32 lanes is:

* **Column Index:** `n_col = lane % 8`
* **Row Index Base:** `k_base = (lane / 8) * 2`

Each `lane` provides the following 4 elements from the `n_col` column:
1.  `B[k_base, n_col]`
2.  `B[k_base + 1, n_col]`
3.  `B[k_base + 8, n_col]`
4.  `B[k_base + 9, n_col]`

### Implications for the Decode Pipeline

This rigid mapping gives us two massive advantages for the INT4 setup:

1.  **Single Scale per Thread:** Because all 4 of a thread's elements land in the *exact same output column* (`n_col`), a thread only needs to load **one** FP16 scale for its entire B fragment. We drop the register pressure from 4 scales down to 1. 
2.  **16-bit Payload:** 4 INT4s = 16 bits. A single 32-bit `ld.shared.u32` intentionally overfetches. Marlin's offline packing specifically interleaves these exact four `k` indices so they sit contiguously in a 16-bit chunk. 

### Double-Buffered Pipeline Skeleton

With the mapping and accumulator constraints fixed, here is the raw structural loop with explicit register double-buffering so the pipeline is guaranteed:

```
cpp
// +r constraint for in-place RC accumulators
#define MMA_SYNC_M16N8K16(RC, RA, RB) \
    asm volatile( \
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 " \
        "{%0, %1, %2, %3}, " \
        "{%4, %5, %6, %7}, " \
        "{%8, %9}, " \
        "{%0, %1, %2, %3};\n" \
        : "+r"(RC[0]), "+r"(RC[1]), "+r"(RC[2]), "+r"(RC[3]) \
        : "r"(RA[0]), "r"(RA[1]), "r"(RA[2]), "r"(RA[3]), \
          "r"(RB[0]), "r"(RB[1]) \
    )

// --- Pre-loop setup ---
int n_col = lane % 8;
int k_base = (lane / 8) * 2; // 0, 2, 4, 6

// Double-buffer registers
uint32_t RB_fp16_cur[2];
uint32_t RB_fp16_nxt[2];

#pragma unroll
for (int ks = 0; ks < 4; ++ks) {
    uint32_t RA[4]; 
    // [ldmatrix.sync.aligned.m8n8.x4.shared.b16 -> RA]

    // Preload & Decode j = 0
    int global_col_0 = col_base + n_col;
    uint32_t scale_32_0 = /* load scale as 32-bit / half2 if possible, or single half */;
    uint32_t b_smem_addr_0 = /* derived from offline packed layout */;
    
    // [Execute INT4 -> FP16 decode into RB_fp16_cur]
    
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        if (j < 7) {
            // Load & Decode j + 1 into nxt
            int global_col_nxt = col_base + (j + 1) * 8 + n_col;
            uint32_t scale_32_nxt = /* ... */;
            uint32_t b_smem_addr_nxt = /* ... */;
            
            // [Execute INT4 -> FP16 decode into RB_fp16_nxt]
        }

        // Issue math for current j slice
        MMA_SYNC_M16N8K16(RC[j], RA, RB_fp16_cur);

        if (j < 7) {
            // Swap buffers
            RB_fp16_cur[0] = RB_fp16_nxt[0];
            RB_fp16_cur[1] = RB_fp16_nxt[1];
        }
    }
}
```

If you agree with this lane map and double-buffered skeleton, we can finally crack open the `prmt.b32` bit-magic to map those 16 bits of INT4 payload into the `RB_fp16` registers.

---

### ChatGPT:
I’m verifying the B-fragment lane mapping against PTX references before I endorse it, because this is exactly the kind of detail that can look right and still be wrong.

---

The **B-fragment map you wrote is not correct**, but it is very close in spirit.

For `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`, PTX gives the exact per-lane mapping for multiplicand **B** as:

- `groupID = %laneid >> 2`
- `threadID_in_group = %laneid % 4`
- for fragment elements `b_i`:

`row = (threadID_in_group * 2) + (i & 0x1)` for `i < 2`

`row = (threadID_in_group * 2) + (i & 0x1) + 8` for `i >= 2`
- `row = (threadID_in_group * 2) + (i & 0x1)` for `i < 2`
- `row = (threadID_in_group * 2) + (i & 0x1) + 8` for `i >= 2`
- `col = groupID` [NVIDIA Docs](https://docs.nvidia.com/cuda/parallel-thread-execution/)

So the exact per-lane B fragment is:

```C++int groupID = lane >> 2;          // 0..7int tid4    = lane & 3;           // 0..3// For current 16x8 B fragment:int n_col = groupID;b0 = B[2 * tid4 + 0, n_col];b1 = B[2 * tid4 + 1, n_col];b2 = B[2 * tid4 + 8, n_col];b3 = B[2 * tid4 + 9, n_col];
C++int groupID = lane >> 2;          // 0..7int tid4    = lane & 3;           // 0..3// For current 16x8 B fragment:int n_col = groupID;b0 = B[2 * tid4 + 0, n_col];b1 = B[2 * tid4 + 1, n_col];b2 = B[2 * tid4 + 8, n_col];b3 = B[2 * tid4 + 9, n_col];
```

That means your proposed mapping:

```C++n_col  = lane % 8;k_base = (lane / 8) * 2;
C++n_col  = lane % 8;k_base = (lane / 8) * 2;
```

is wrong.
The correct mapping is:

```C++n_col  = lane >> 2;k_base = (lane & 3) * 2;
C++n_col  = lane >> 2;k_base = (lane & 3) * 2;
```

and the four rows are:

```C++{k_base + 0, k_base + 1, k_base + 8, k_base + 9}
C++{k_base + 0, k_base + 1, k_base + 8, k_base + 9}
```

That comes straight from the PTX fragment definition for matrix B. [NVIDIA Docs](https://docs.nvidia.com/cuda/parallel-thread-execution/)

Your **single-scale-per-thread** conclusion is correct, though. Since all four `b_i` for one lane land in the same output column `col = groupID`, one per-column scale is enough for the whole lane fragment. That follows directly from the PTX B-fragment formula above. [NVIDIA Docs](https://docs.nvidia.com/cuda/parallel-thread-execution/)

## Corrected structural skeleton

This is the version I would lock in before touching `prmt`:

```C++// B fragment mapping from PTX for m16n8k16.row.col.f16.f16.f16.f16int groupID = lane >> 2;   // output column within current n8 sliceint tid4    = lane & 3;    // row-pair selector#define MMA_SYNC_M16N8K16(RC, RA, RB)                                      \    asm volatile(                                                          \        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "              \        "{%0, %1, %2, %3}, "                                               \        "{%4, %5, %6, %7}, "                                               \        "{%8, %9}, "                                                       \        "{%0, %1, %2, %3};\n"                                              \        : "+r"((RC)[0]), "+r"((RC)[1]), "+r"((RC)[2]), "+r"((RC)[3])       \        : "r"((RA)[0]), "r"((RA)[1]), "r"((RA)[2]), "r"((RA)[3]),          \          "r"((RB)[0]), "r"((RB)[1])                                       \    )uint32_t RC[8][4];#pragma unrollfor (int j = 0; j < 8; ++j) {    RC[j][0] = 0;    RC[j][1] = 0;    RC[j][2] = 0;    RC[j][3] = 0;}uint32_t RB_fp16_cur[2];uint32_t RB_fp16_nxt[2];#pragma unrollfor (int ks = 0; ks < 4; ++ks) {    uint32_t RA[4];    // A fragment load stays as previously discussed via ldmatrix.x4    // ... ldmatrix.sync.aligned.m8n8.x4.shared.b16 -> RA ...    // preload j = 0    {        int frag_col = 0 + groupID;     // j = 0 slice        int global_col = col_base + frag_col;        // PTX row pattern for B:        int r0 = ks * 16 + (2 * tid4 + 0);        int r1 = ks * 16 + (2 * tid4 + 1);        int r2 = ks * 16 + (2 * tid4 + 8);        int r3 = ks * 16 + (2 * tid4 + 9);        // one scale per thread, because all four values are from one column        half scale = S_smem[global_col];        // shared-memory address depends on offline packing        // best validation layout is one 16-bit payload per lane:        // packed as [ks][j][lane] -> uint16_t        // payload contains 4 int4 values: b0,b1,b2,b3        //        // uint16_t packed = ...        // decode packed + scale -> RB_fp16_cur[0], RB_fp16_cur[1]    }    #pragma unroll    for (int j = 0; j < 8; ++j) {        if (j < 7) {            int frag_col_nxt = (j + 1) * 8 + groupID;            int global_col_nxt = col_base + frag_col_nxt;            int r0 = ks * 16 + (2 * tid4 + 0);            int r1 = ks * 16 + (2 * tid4 + 1);            int r2 = ks * 16 + (2 * tid4 + 8);            int r3 = ks * 16 + (2 * tid4 + 9);            half scale_nxt = S_smem[global_col_nxt];            // uint16_t packed_nxt = ...            // decode packed_nxt + scale_nxt -> RB_fp16_nxt[0], RB_fp16_nxt[1]        }        MMA_SYNC_M16N8K16(RC[j], RA, RB_fp16_cur);        if (j < 7) {            RB_fp16_cur[0] = RB_fp16_nxt[0];            RB_fp16_cur[1] = RB_fp16_nxt[1];        }    }}
C++// B fragment mapping from PTX for m16n8k16.row.col.f16.f16.f16.f16int groupID = lane >> 2;   // output column within current n8 sliceint tid4    = lane & 3;    // row-pair selector#define MMA_SYNC_M16N8K16(RC, RA, RB)                                      \    asm volatile(                                                          \        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "              \        "{%0, %1, %2, %3}, "                                               \        "{%4, %5, %6, %7}, "                                               \        "{%8, %9}, "                                                       \        "{%0, %1, %2, %3};\n"                                              \        : "+r"((RC)[0]), "+r"((RC)[1]), "+r"((RC)[2]), "+r"((RC)[3])       \        : "r"((RA)[0]), "r"((RA)[1]), "r"((RA)[2]), "r"((RA)[3]),          \          "r"((RB)[0]), "r"((RB)[1])                                       \    )uint32_t RC[8][4];#pragma unrollfor (int j = 0; j < 8; ++j) {    RC[j][0] = 0;    RC[j][1] = 0;    RC[j][2] = 0;    RC[j][3] = 0;}uint32_t RB_fp16_cur[2];uint32_t RB_fp16_nxt[2];#pragma unrollfor (int ks = 0; ks < 4; ++ks) {    uint32_t RA[4];    // A fragment load stays as previously discussed via ldmatrix.x4    // ... ldmatrix.sync.aligned.m8n8.x4.shared.b16 -> RA ...    // preload j = 0    {        int frag_col = 0 + groupID;     // j = 0 slice        int global_col = col_base + frag_col;        // PTX row pattern for B:        int r0 = ks * 16 + (2 * tid4 + 0);        int r1 = ks * 16 + (2 * tid4 + 1);        int r2 = ks * 16 + (2 * tid4 + 8);        int r3 = ks * 16 + (2 * tid4 + 9);        // one scale per thread, because all four values are from one column        half scale = S_smem[global_col];        // shared-memory address depends on offline packing        // best validation layout is one 16-bit payload per lane:        // packed as [ks][j][lane] -> uint16_t        // payload contains 4 int4 values: b0,b1,b2,b3        //        // uint16_t packed = ...        // decode packed + scale -> RB_fp16_cur[0], RB_fp16_cur[1]    }    #pragma unroll    for (int j = 0; j < 8; ++j) {        if (j < 7) {            int frag_col_nxt = (j + 1) * 8 + groupID;            int global_col_nxt = col_base + frag_col_nxt;            int r0 = ks * 16 + (2 * tid4 + 0);            int r1 = ks * 16 + (2 * tid4 + 1);            int r2 = ks * 16 + (2 * tid4 + 8);            int r3 = ks * 16 + (2 * tid4 + 9);            half scale_nxt = S_smem[global_col_nxt];            // uint16_t packed_nxt = ...            // decode packed_nxt + scale_nxt -> RB_fp16_nxt[0], RB_fp16_nxt[1]        }        MMA_SYNC_M16N8K16(RC[j], RA, RB_fp16_cur);        if (j < 7) {            RB_fp16_cur[0] = RB_fp16_nxt[0];            RB_fp16_cur[1] = RB_fp16_nxt[1];        }    }}
```

## The shared-memory layout I would use for B first

Do **not** start with the 32-bit overfetch trick.
Start with the layout that exactly matches PTX lane ownership:

```C++// logical validation layout// [ks][j][lane] -> one 16-bit payload containing b0,b1,b2,b3uint16_t Bfrag[4][8][32];
C++// logical validation layout// [ks][j][lane] -> one 16-bit payload containing b0,b1,b2,b3uint16_t Bfrag[4][8][32];
```

Then the lane-local address is trivial:

```C++uint16_t packed = Bfrag[ks][j][lane];
C++uint16_t packed = Bfrag[ks][j][lane];
```

That gives you the exact four INT4 values required by PTX:

- `b0`
- `b1`
- `b2`
- `b3`

with zero ambiguity.

Once that works, you can repack to 32-bit lane pairs for bandwidth/alignment tuning.

## One more important correction

Your earlier comment said:

“Marlin’s offline packing specifically interleaves these exact four k indices so they sit contiguously in a 16-bit chunk.”

That may be a reasonable design choice for your kernel, but it is **not** something PTX guarantees. PTX only gives the fragment ownership map; the “contiguous 16-bit packed payload per lane” is your **chosen offline layout**, not an ISA fact. The ISA fact is only the lane→`(row,col)` mapping above. [NVIDIA Docs](https://docs.nvidia.com/cuda/parallel-thread-execution/)

## Verdict

So:

- your **double-buffered skeleton** is good,
- your **single-scale-per-thread insight** is good,
- your **B lane map is wrong as written**,
- and the corrected map is:

```C++groupID = lane >> 2;tid4    = lane & 3;rows = {2*tid4 + 0, 2*tid4 + 1, 2*tid4 + 8, 2*tid4 + 9};col  = groupID;
C++groupID = lane >> 2;tid4    = lane & 3;rows = {2*tid4 + 0, 2*tid4 + 1, 2*tid4 + 8, 2*tid4 + 9};col  = groupID;
```

for each `j`, with `col_base + j*8 + groupID` as the global column.

If you want, the next move is the exact **INT4 packed nibble order** for `uint16_t packed -> RB_fp16[2]`, including the `prmt` / unpack path.

---

### User:
You are exactly right. I fucked up the `m16n8k16` column layout map. `lane >> 2` is the PTX standard, and your correction is spot-on. I also agree with your `uint16_t Bfrag[4][8][32]` starting point. Building the logical validation layout first is the bulletproof way to isolate lane ownership and prove the math before you start optimizing the shared memory fetches.

Here is the exact INT4 packed nibble order and the raw decode path using the magic FP16 conversion trick (which avoids slow integer-to-float `cvt` instructions).

### The Offline Packing Contract
To make the inline PTX beautiful and fast, your offline preprocessor should pack the four values for a lane into a single `uint16_t` exactly like this:
* **Bits 0-3:** `b0` (row `2*tid4 + 0`)
* **Bits 4-7:** `b1` (row `2*tid4 + 1`)
* **Bits 8-11:** `b2` (row `2*tid4 + 8`)
* **Bits 12-15:** `b3` (row `2*tid4 + 9`)

### The Decode Strategy
Ampere has a known trick for converting 4-bit integers to FP16. 
An FP16 number with an exponent of 25 (hex `0x6400`) represents the float `1024.0`. If you bitwise OR a 4-bit integer into the bottom of its mantissa, the resulting float is `1024.0 + nibble`. 

We extract the nibbles, inject them into `0x64006400` `half2` vectors, and then use `fma.rn.f16x2` to subtract the 1024.0 baseline and apply the column scale simultaneously.

### The Code

```
cpp
// --- Outside the j-loop: Precompute the scale and offset ---
// Assuming symmetric INT4 with a zero-point of 8, the nibbles range from 0 to 15.
// We need to subtract (1024.0 + 8.0) = 1032.0 to center the values at 0.
// 1032.0 in FP16 is 0x6408. 
// We want to compute: (magic_float - 1032.0) * scale 
// Which is: magic_float * scale - 1032.0 * scale
// So we precalculate the FMA 'C' operand as (-1032.0 * scale).

half scale = S_smem[global_col];
uint32_t s2_reg; // half2 scale
uint32_t c2_reg; // half2 FMA offset

{
    half2 scale2 = __halves2half2(scale, scale);
    half2 offset_base = __halves2half2(__float2half(-1032.0f), __float2half(-1032.0f));
    half2 c2 = __hmul2(scale2, offset_base);
    
    s2_reg = *(uint32_t*)&scale2;
    c2_reg = *(uint32_t*)&c2;
}

// --- Inside the j-loop: Decode current packed payload ---
uint32_t packed_32 = Bfrag[ks][j][lane]; 

// 1. Isolate the byte pairs into separate registers
uint32_t w0 = packed_32 & 0x00FF;        // Contains [b1, b0]
uint32_t w1 = (packed_32 >> 8) & 0x00FF; // Contains [b3, b2]

// 2. Space the nibbles out to match the lower and upper halves of a half2
// The lower nibble stays at bits 0-3.
// The upper nibble is masked (0xF0) and shifted left by 12 bits to land at bits 16-19.
uint32_t h0 = (w0 & 0x0F) | ((w0 & 0xF0) << 12); 
uint32_t h1 = (w1 & 0x0F) | ((w1 & 0xF0) << 12); 

// 3. Inject the magic FP16 exponent (0x6400 per half -> 0x64006400 for half2)
uint32_t magic = 0x64006400;
uint32_t f16_01 = h0 | magic;
uint32_t f16_23 = h1 | magic;

// 4. Issue the FMA to finish the conversion and apply the scale
asm volatile(
    "fma.rn.f16x2 %0, %1, %2, %3;" 
    : "=r"(RB_fp16_cur[0]) 
    : "r"(f16_01), "r"(s2_reg), "r"(c2_reg)
);

asm volatile(
    "fma.rn.f16x2 %0, %1, %2, %3;" 
    : "=r"(RB_fp16_cur[1]) 
    : "r"(f16_23), "r"(s2_reg), "r"(c2_reg)
);
```

### Why this is the way
* **No `prmt.b32` required for this layout:** Because you packed them 16-bits contiguous, simple bitwise ANDs and a shift achieve the exact alignment you need. `prmt` is only strictly necessary if your offline packing scatters the nibbles across a 32-bit boundary.
* **Math Overlap:** The `fma.rn.f16x2` handles the integer-to-float offset and the scaling in a single hardware step, and those FMAs drop right into the pipe before the `mma.sync`.
* **Register Footprint:** Only 2 scaling registers (`s2_reg`, `c2_reg`) needed per column slice, easily absorbing the decode without spilling.

Plug this into the double-buffered skeleton you wrote. If the validation matches, the only thing left is ripping out the 3D array layout and writing the raw `ld.shared.u32` trick to fetch 32-bits (two lanes worth) at a time for bandwidth.

---

### ChatGPT:
I’m checking the PTX half2/FMA path against the ISA details before I bless the decode sequence, because the “magic FP16” trick is easy to get subtly wrong.

---

This decode path is **basically right**, and the core trick is sound:

- the corrected `m16n8k16.row.col` B-fragment ownership is the PTX one with `groupID = lane >> 2` and `threadID_in_group = lane % 4`, so each lane owns **4 values from one column**; ([PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/?utm_source=chatgpt.com))
- `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` takes **two .f16x2 registers** for B, so your goal is exactly to produce 4 FP16 values packed into `RB_fp16_cur[0:2]`; ([PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/?utm_source=chatgpt.com))
- `fma.rn.f16x2` is the right PTX instruction to do a packed half2 multiply-add in one step. ([PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/?utm_source=chatgpt.com))

There are **3 fixes** I’d make before calling it locked.

## 1) Use uint16_t in the validation layout, not uint32_t

Your text says the validation layout is:

```C++uint16_t Bfrag[4][8][32];
C++uint16_t Bfrag[4][8][32];
```

but then the code does:

```C++uint32_t packed_32 = Bfrag[ks][j][lane];
C++uint32_t packed_32 = Bfrag[ks][j][lane];
```

That is inconsistent.

For the **logical validation layout**, keep it literal:

```C++uint16_t packed_16 = Bfrag[ks][j][lane];uint32_t packed_32 = packed_16;
C++uint16_t packed_16 = Bfrag[ks][j][lane];uint32_t packed_32 = packed_16;
```

Only switch to 32-bit overfetch when you deliberately repack the shared-memory layout. Right now, mixing the two mental models will make debugging miserable.

## 2) The magic decode is correct, but the comment should be tighter

For symmetric signed INT4 reconstructed from an unsigned nibble `q in [0,15]`, you want:

```C++deq = (q - 8) * scale
C++deq = (q - 8) * scale
```

Your trick builds:

- `magic_half = 1024 + q`
- then computes:
- `(magic_half * scale) + (-1032 * scale) = (q - 8) * scale`

That is correct.

So the final packed-half decode path is:

```C++uint16_t packed_16 = Bfrag[ks][j][lane];uint32_t packed_32 = packed_16;// [b1,b0] in low byte, [b3,b2] in high byteuint32_t w0 = packed_32 & 0x00FFu;         // low byte: b1:b0uint32_t w1 = (packed_32 >> 8) & 0x00FFu;  // high byte: b3:b2// place nibble 0 into low half mantissa bits 0..3// place nibble 1 into upper half mantissa bits 16..19uint32_t h0 = (w0 & 0x0Fu) | ((w0 & 0xF0u) << 12);uint32_t h1 = (w1 & 0x0Fu) | ((w1 & 0xF0u) << 12);// inject FP16 exponent for 1024.0 into both halvesconstexpr uint32_t MAGIC = 0x64006400u;uint32_t f16_01 = h0 | MAGIC;   // {1024+b1, 1024+b0} in half2 formuint32_t f16_23 = h1 | MAGIC;   // {1024+b3, 1024+b2} in half2 form
C++uint16_t packed_16 = Bfrag[ks][j][lane];uint32_t packed_32 = packed_16;// [b1,b0] in low byte, [b3,b2] in high byteuint32_t w0 = packed_32 & 0x00FFu;         // low byte: b1:b0uint32_t w1 = (packed_32 >> 8) & 0x00FFu;  // high byte: b3:b2// place nibble 0 into low half mantissa bits 0..3// place nibble 1 into upper half mantissa bits 16..19uint32_t h0 = (w0 & 0x0Fu) | ((w0 & 0xF0u) << 12);uint32_t h1 = (w1 & 0x0Fu) | ((w1 & 0xF0u) << 12);// inject FP16 exponent for 1024.0 into both halvesconstexpr uint32_t MAGIC = 0x64006400u;uint32_t f16_01 = h0 | MAGIC;   // {1024+b1, 1024+b0} in half2 formuint32_t f16_23 = h1 | MAGIC;   // {1024+b3, 1024+b2} in half2 form
```

Then:

```C++asm volatile(    "fma.rn.f16x2 %0, %1, %2, %3;"    : "=r"(RB_fp16_cur[0])    : "r"(f16_01), "r"(s2_reg), "r"(c2_reg));asm volatile(    "fma.rn.f16x2 %0, %1, %2, %3;"    : "=r"(RB_fp16_cur[1])    : "r"(f16_23), "r"(s2_reg), "r"(c2_reg));
C++asm volatile(    "fma.rn.f16x2 %0, %1, %2, %3;"    : "=r"(RB_fp16_cur[0])    : "r"(f16_01), "r"(s2_reg), "r"(c2_reg));asm volatile(    "fma.rn.f16x2 %0, %1, %2, %3;"    : "=r"(RB_fp16_cur[1])    : "r"(f16_23), "r"(s2_reg), "r"(c2_reg));
```

That part is fine.

## 3) Be explicit about element order inside RB_fp16

This is the one place I would force a comment now, because it will matter when you validate against a reference GEMM.

With the packing you chose, the two half2 regs hold:

- `RB_fp16_cur[0]` → `{b0, b1}` or `{b1, b0}` depending on how you interpret half2 lane order in your downstream MMA operand packing
- `RB_fp16_cur[1]` → `{b2, b3}` or `{b3, b2}`

You should not leave that ambiguous.

Write the validation contract explicitly in code by decoding once in scalar host-side logic and comparing lane fragments. In other words, define:

```C++// REQUIRED VALIDATION CONTRACT:// RB_fp16_cur[0] must correspond to the first two B fragment elements// RB_fp16_cur[1] must correspond to the second two B fragment elements// in the exact order expected by mma.sync for this lane.
C++// REQUIRED VALIDATION CONTRACT:// RB_fp16_cur[0] must correspond to the first two B fragment elements// RB_fp16_cur[1] must correspond to the second two B fragment elements// in the exact order expected by mma.sync for this lane.
```

The PTX docs give you the **which four elements** per lane for B; they do not save you from getting the intra-register pair ordering wrong during your custom decode. ([PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/?utm_source=chatgpt.com))

## Register-pressure note

Your updated register story is much better now:

- one scale per lane fragment is correct
- `s2_reg` and `c2_reg` as packed half2 regs is the right move
- no need for 4 separate `half scale_*`

So yes, this is now a sane decode footprint.

## My verdict

I’d mark it:

- **lane ownership:** correct now
- **nibble packing contract:** good
- **magic FP16 + fma.rn.f16x2 decode:** good
- **validation layout typing (uint16_t vs uint32_t):** fix immediately
- **intra-register element order:** must be made explicit before performance tuning

So yes — **this is good enough to plug into the double-buffered skeleton**, as long as you correct the `uint16_t`/`uint32_t` mismatch and make the final half2 element order explicit.

The next step is exactly what you said: once this validates numerically, replace the logical `Bfrag[4][8][32]` path with the real shared-memory packed fetch path.

---

### User:
You are right. Mixing `uint32_t` and `uint16_t` in the logical layout was sloppy and breaks the mental model. 

For the intra-register order, PTX `.f16x2` format expects the first element in the lower 16 bits and the second element in the upper 16 bits. With our shift logic `((w0 & 0xF0u) << 12)`, `b1` shifts into the upper half, and `b0` stays in the lower half. This correctly maps `b0` to the first row and `b1` to the second row for the `mma.sync` instruction. We will enforce this as a hard contract in the comments.

Here is the final, locked inner loop skeleton. It integrates the corrected PTX B-fragment mapping, the exact magic decode, and explicit double-buffering.

```
cpp
// ------------------------------
// Validation Layout & MMA Macros
// ------------------------------
// Logical validation layout: [ks][j][lane]
// Contains 4 INT4 values packed as {b3, b2, b1, b0}
extern __shared__ uint16_t Bfrag[4][8][32]; 

#define MMA_SYNC_M16N8K16(RC, RA, RB)                                      \
    asm volatile(                                                          \
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "               \
        "{%0, %1, %2, %3}, "                                               \
        "{%4, %5, %6, %7}, "                                               \
        "{%8, %9}, "                                                       \
        "{%0, %1, %2, %3};\n"                                              \
        : "+r"((RC)[0]), "+r"((RC)[1]), "+r"((RC)[2]), "+r"((RC)[3])       \
        : "r"((RA)[0]), "r"((RA)[1]), "r"((RA)[2]), "r"((RA)[3]),          \
          "r"((RB)[0]), "r"((RB)[1])                                       \
    )

// Lane ownership for m16n8k16 B-fragment
const int groupID = lane >> 2;   // output column within current n8 slice
const int tid4    = lane & 3;    // row-pair selector

// Constants for INT4 -> FP16 Magic Decode
constexpr uint32_t MAGIC_FP16 = 0x64006400u; // 1024.0 in half2
const half2 offset_base = __halves2half2(__float2half(-1032.0f), __float2half(-1032.0f));

// Accumulators
uint32_t RC[8][4] = {0};
uint32_t RB_fp16_cur[2];
uint32_t RB_fp16_nxt[2];

// ------------------------------
// Inner Loop
// ------------------------------
#pragma unroll
for (int ks = 0; ks < 4; ++ks) {
    uint32_t RA[4];
    
    // [ldmatrix.sync.aligned.m8n8.x4.shared.b16 -> RA]

    // --- Preload & Decode j = 0 ---
    {
        int frag_col = groupID; 
        int global_col = col_base + frag_col;
        half scale = S_smem[global_col];

        // Prepare FMA operands for decode
        half2 scale2 = __halves2half2(scale, scale);
        half2 c2 = __hmul2(scale2, offset_base);
        uint32_t s2_reg = *(uint32_t*)&scale2;
        uint32_t c2_reg = *(uint32_t*)&c2;

        uint16_t packed_16 = Bfrag[ks][0][lane];
        uint32_t packed_32 = packed_16; 

        uint32_t w0 = packed_32 & 0x00FFu;         // [b1, b0]
        uint32_t w1 = (packed_32 >> 8) & 0x00FFu;  // [b3, b2]

        uint32_t h0 = (w0 & 0x0Fu) | ((w0 & 0xF0u) << 12);
        uint32_t h1 = (w1 & 0x0Fu) | ((w1 & 0xF0u) << 12);

        uint32_t f16_01 = h0 | MAGIC_FP16;   
        uint32_t f16_23 = h1 | MAGIC_FP16;   

        // INTRA-REGISTER CONTRACT:
        // RB_fp16_cur[0] -> Lower 16 bits = b0, Upper 16 bits = b1
        // RB_fp16_cur[1] -> Lower 16 bits = b2, Upper 16 bits = b3
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(RB_fp16_cur[0]) : "r"(f16_01), "r"(s2_reg), "r"(c2_reg));
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(RB_fp16_cur[1]) : "r"(f16_23), "r"(s2_reg), "r"(c2_reg));
    }

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        if (j < 7) {
            // --- Load & Decode j + 1 ---
            int frag_col_nxt = (j + 1) * 8 + groupID;
            int global_col_nxt = col_base + frag_col_nxt;
            half scale_nxt = S_smem[global_col_nxt];

            half2 scale2_nxt = __halves2half2(scale_nxt, scale_nxt);
            half2 c2_nxt = __hmul2(scale2_nxt, offset_base);
            uint32_t s2_reg_nxt = *(uint32_t*)&scale2_nxt;
            uint32_t c2_reg_nxt = *(uint32_t*)&c2_nxt;

            uint16_t packed_16_nxt = Bfrag[ks][j + 1][lane];
            uint32_t packed_32_nxt = packed_16_nxt; 

            uint32_t w0_nxt = packed_32_nxt & 0x00FFu;
            uint32_t w1_nxt = (packed_32_nxt >> 8) & 0x00FFu;

            uint32_t h0_nxt = (w0_nxt & 0x0Fu) | ((w0_nxt & 0xF0u) << 12);
            uint32_t h1_nxt = (w1_nxt & 0x0Fu) | ((w1_nxt & 0xF0u) << 12);

            uint32_t f16_01_nxt = h0_nxt | MAGIC_FP16;
            uint32_t f16_23_nxt = h1_nxt | MAGIC_FP16;

            asm volatile("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(RB_fp16_nxt[0]) : "r"(f16_01_nxt), "r"(s2_reg_nxt), "r"(c2_reg_nxt));
            asm volatile("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(RB_fp16_nxt[1]) : "r"(f16_23_nxt), "r"(s2_reg_nxt), "r"(c2_reg_nxt));
        }

        MMA_SYNC_M16N8K16(RC[j], RA, RB_fp16_cur);

        if (j < 7) {
            RB_fp16_cur[0] = RB_fp16_nxt[0];
            RB_fp16_cur[1] = RB_fp16_nxt[1];
        }
    }
}
```

The mathematical foundation is completely airtight now. Once you run this block through the validation harness and confirm numerical parity against your reference GEMM, we rip out `Bfrag[4][8][32]` and write the physical `ld.shared.u32` to pipeline the shared memory bandwidth. 

Let me know when the harness goes green.

---

### ChatGPT:
I’m sanity-checking the fragment ordering against the PTX docs before I sign off, because this is one of the few spots where a tiny register-order mistake can make the whole kernel numerically wrong.

---

Not green yet — there is still **one major PTX mismatch** in the skeleton.

## The blocker

For:

```C++mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
C++mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
```

PTX says:

- **B** is **two .f16x2 registers** holding `b0, b1, b2, b3` low-to-high. [NVIDIA Docs+1](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- **C/D** for `.f16` are **two .f16x2 registers**, not four. The PTX page’s prose has a typo in the element list, but the fragment definition and row/col formula make it clear this is a 4-element accumulator fragment carried in **2 regs**. [NVIDIA Docs+2NVIDIA Docs+2](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)

So this part of your code is still wrong:

```C++uint32_t RC[8][4];..."{%0, %1, %2, %3}, ... {%0, %1, %2, %3}"
C++uint32_t RC[8][4];..."{%0, %1, %2, %3}, ... {%0, %1, %2, %3}"
```

For the **FP16 accumulator variant**, each `m16n8k16` slice should use **2 accumulator regs**, not 4.

## What to change

### Correct macro

```C++#define MMA_SYNC_M16N8K16_F16(RC, RA, RB)                                  \    asm volatile(                                                          \        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "               \        "{%0, %1}, "                                                       \        "{%2, %3, %4, %5}, "                                               \        "{%6, %7}, "                                                       \        "{%0, %1};\n"                                                      \        : "+r"((RC)[0]), "+r"((RC)[1])                                     \        : "r"((RA)[0]), "r"((RA)[1]), "r"((RA)[2]), "r"((RA)[3]),          \          "r"((RB)[0]), "r"((RB)[1])                                       \    )
C++#define MMA_SYNC_M16N8K16_F16(RC, RA, RB)                                  \    asm volatile(                                                          \        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "               \        "{%0, %1}, "                                                       \        "{%2, %3, %4, %5}, "                                               \        "{%6, %7}, "                                                       \        "{%0, %1};\n"                                                      \        : "+r"((RC)[0]), "+r"((RC)[1])                                     \        : "r"((RA)[0]), "r"((RA)[1]), "r"((RA)[2]), "r"((RA)[3]),          \          "r"((RB)[0]), "r"((RB)[1])                                       \    )
```

### Correct accumulator storage

```C++uint32_t RC[8][2];#pragma unrollfor (int j = 0; j < 8; ++j) {    RC[j][0] = 0;    RC[j][1] = 0;}
C++uint32_t RC[8][2];#pragma unrollfor (int j = 0; j < 8; ++j) {    RC[j][0] = 0;    RC[j][1] = 0;}
```

That is the biggest fix.

## Your B-fragment logic is now correct

This part is good:

- `groupID = lane >> 2`
- `tid4 = lane & 3`
- rows owned by one lane are:

`2*tid4 + 0`

`2*tid4 + 1`

`2*tid4 + 8`

`2*tid4 + 9`
- `2*tid4 + 0`
- `2*tid4 + 1`
- `2*tid4 + 8`
- `2*tid4 + 9`
- all 4 belong to the same output column `groupID`

That matches PTX’s B-fragment formula for `m16n8k16` FP16/BF16 exactly. [NVIDIA Docs](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)

So your `uint16_t` packing contract:

- bits `0..3` = `b0`
- bits `4..7` = `b1`
- bits `8..11` = `b2`
- bits `12..15` = `b3`

is a good validation layout.

And yes, the element order inside the two `f16x2` regs is consistent with your decode:

- `RB_fp16_cur[0]` → low 16 = `b0`, high 16 = `b1`
- `RB_fp16_cur[1]` → low 16 = `b2`, high 16 = `b3`

That is the right contract for the current lane-local validation path. [NVIDIA Docs](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)

## Two smaller fixes I would still make

### 1) Do not declare extern __shared__ uint16_t Bfrag[4][8][32];

That is not how dynamic shared memory should be modeled in the real kernel. For the validation path, use pointer arithmetic on a flat `extern __shared__ uint8_t smem[]` buffer and cast the right offset:

```C++extern __shared__ uint8_t smem[];uint16_t* Bfrag = reinterpret_cast<uint16_t*>(smem + bfrag_stage_offset);
C++extern __shared__ uint8_t smem[];uint16_t* Bfrag = reinterpret_cast<uint16_t*>(smem + bfrag_stage_offset);
```

Then index manually:

```C++uint16_t packed_16 = Bfrag[(ks * 8 + j) * 32 + lane];
C++uint16_t packed_16 = Bfrag[(ks * 8 + j) * 32 + lane];
```

### 2) Avoid type-punning through references/pointers for half2 -> uint32_t

This:

```C++uint32_t s2_reg = *(uint32_t*)&scale2;
C++uint32_t s2_reg = *(uint32_t*)&scale2;
```

usually works in practice, but it is uglier than it needs to be. Use `memcpy`, a union, or inline PTX `mov.b32` if you want to keep it airtight.

## Verdict

After the accumulator fix, this is good enough for the validation harness.

So the next correct version is:

- `RC[8][2]`
- `mma.sync ... {%0,%1} ... {%0,%1}`
- keep your current `RB_fp16_cur[2]` decode exactly as-is
- keep the `uint16_t` lane-local validation layout
- move `Bfrag` to flat shared-memory indexing

Once that passes reference checks, then it makes sense to replace the logical `Bfrag[(ks,j,lane)]` access with the real packed shared-memory fetch path.

---

### User:
You are completely right. I blindly carried over the FP32 accumulator footprint (4x 32-bit regs) into the FP16 accumulator instruction. For `.f16` accumulators, 4 elements pack perfectly into two 32-bit registers. Good catch on the PTX typo vs. actual ISA behavior. 

I also agree on ripping out the 3D array declaration and the pointer punning. Unions are cleaner and safer for the 32-bit register conversion.

Here is the final, corrected validation skeleton. 

```
cpp
// ------------------------------
// Validation Layout & MMA Macros
// ------------------------------
// Flat shared memory buffer
extern __shared__ uint8_t smem[];

// Macro for m16n8k16 with FP16 accumulators (2 regs for C/D)
#define MMA_SYNC_M16N8K16_F16(RC, RA, RB)                                  \
    asm volatile(                                                          \
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "               \
        "{%0, %1}, "                                                       \
        "{%2, %3, %4, %5}, "                                               \
        "{%6, %7}, "                                                       \
        "{%0, %1};\n"                                                      \
        : "+r"((RC)[0]), "+r"((RC)[1])                                     \
        : "r"((RA)[0]), "r"((RA)[1]), "r"((RA)[2]), "r"((RA)[3]),          \
          "r"((RB)[0]), "r"((RB)[1])                                       \
    )

// Lane ownership for m16n8k16 B-fragment
const int groupID = lane >> 2;   // output column within current n8 slice
const int tid4    = lane & 3;    // row-pair selector

// Constants for INT4 -> FP16 Magic Decode
constexpr uint32_t MAGIC_FP16 = 0x64006400u; // 1024.0 in half2
const half2 offset_base = __halves2half2(__float2half(-1032.0f), __float2half(-1032.0f));

// Accumulators (2x 32-bit regs per slice)
uint32_t RC[8][2];
#pragma unroll
for (int j = 0; j < 8; ++j) {
    RC[j][0] = 0;
    RC[j][1] = 0;
}

uint32_t RB_fp16_cur[2];
uint32_t RB_fp16_nxt[2];

// Helper union for clean half2 -> uint32_t conversion
union Half2Reg {
    half2 h2;
    uint32_t u32;
};

// ------------------------------
// Inner Loop
// ------------------------------
// Bfrag mapped to flat smem based on stage offset
uint16_t* Bfrag = reinterpret_cast<uint16_t*>(smem + bfrag_stage_offset);

#pragma unroll
for (int ks = 0; ks < 4; ++ks) {
    uint32_t RA[4];
    
    // [ldmatrix.sync.aligned.m8n8.x4.shared.b16 -> RA]

    // --- Preload & Decode j = 0 ---
    {
        int frag_col = groupID; 
        int global_col = col_base + frag_col;
        half scale = S_smem[global_col];

        // Prepare FMA operands for decode
        Half2Reg s2_reg, c2_reg;
        s2_reg.h2 = __halves2half2(scale, scale);
        c2_reg.h2 = __hmul2(s2_reg.h2, offset_base);

        // Logical fetch from flat smem
        uint32_t packed_32 = Bfrag[(ks * 8 + 0) * 32 + lane]; 

        uint32_t w0 = packed_32 & 0x00FFu;         // [b1, b0]
        uint32_t w1 = (packed_32 >> 8) & 0x00FFu;  // [b3, b2]

        uint32_t h0 = (w0 & 0x0Fu) | ((w0 & 0xF0u) << 12);
        uint32_t h1 = (w1 & 0x0Fu) | ((w1 & 0xF0u) << 12);

        uint32_t f16_01 = h0 | MAGIC_FP16;   
        uint32_t f16_23 = h1 | MAGIC_FP16;   

        // INTRA-REGISTER CONTRACT:
        // RB_fp16_cur[0] -> Lower 16 bits = b0, Upper 16 bits = b1
        // RB_fp16_cur[1] -> Lower 16 bits = b2, Upper 16 bits = b3
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(RB_fp16_cur[0]) : "r"(f16_01), "r"(s2_reg.u32), "r"(c2_reg.u32));
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(RB_fp16_cur[1]) : "r"(f16_23), "r"(s2_reg.u32), "r"(c2_reg.u32));
    }

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        if (j < 7) {
            // --- Load & Decode j + 1 ---
            int frag_col_nxt = (j + 1) * 8 + groupID;
            int global_col_nxt = col_base + frag_col_nxt;
            half scale_nxt = S_smem[global_col_nxt];

            Half2Reg s2_reg_nxt, c2_reg_nxt;
            s2_reg_nxt.h2 = __halves2half2(scale_nxt, scale_nxt);
            c2_reg_nxt.h2 = __hmul2(s2_reg_nxt.h2, offset_base);

            // Logical fetch from flat smem
            uint32_t packed_32_nxt = Bfrag[(ks * 8 + j + 1) * 32 + lane]; 

            uint32_t w0_nxt = packed_32_nxt & 0x00FFu;
            uint32_t w1_nxt = (packed_32_nxt >> 8) & 0x00FFu;

            uint32_t h0_nxt = (w0_nxt & 0x0Fu) | ((w0_nxt & 0xF0u) << 12);
            uint32_t h1_nxt = (w1_nxt & 0x0Fu) | ((w1_nxt & 0xF0u) << 12);

            uint32_t f16_01_nxt = h0_nxt | MAGIC_FP16;
            uint32_t f16_23_nxt = h1_nxt | MAGIC_FP16;

            asm volatile("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(RB_fp16_nxt[0]) : "r"(f16_01_nxt), "r"(s2_reg_nxt.u32), "r"(c2_reg_nxt.u32));
            asm volatile("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(RB_fp16_nxt[1]) : "r"(f16_23_nxt), "r"(s2_reg_nxt.u32), "r"(c2_reg_nxt.u32));
        }

        MMA_SYNC_M16N8K16_F16(RC[j], RA, RB_fp16_cur);

        if (j < 7) {
            RB_fp16_cur[0] = RB_fp16_nxt[0];
            RB_fp16_cur[1] = RB_fp16_nxt[1];
        }
    }
}
```

Drop this into the harness. Once the reference GEMM confirms numerical parity, let's rip out that scalar `Bfrag` array lookup and pack it into raw 32-bit `ld.shared.u32` bandwidth optimization.

---

### ChatGPT:
This is now **structurally sound** for the validation harness.

The two big PTX-level fixes are in place:

- the `m16n8k16.row.col.f16.f16.f16.f16` B-fragment ownership is now consistent with the warp/lane mapping from the PTX ISA, and
- the FP16 accumulator form is using **2 accumulator registers** per `m16n8k16` slice instead of 4. That matches the PTX fragment contract for the FP16 accumulator variant. [NVIDIA Docs+1](https://docs.nvidia.com/cuda/parallel-thread-execution/?utm_source=chatgpt.com)

I’d only tighten **three things** before calling it locked:

### 1) Make the logical fetch type explicit

This line is valid, but I would make it less ambiguous:

```C++uint32_t packed_32 = Bfrag[(ks * 8 + 0) * 32 + lane];
C++uint32_t packed_32 = Bfrag[(ks * 8 + 0) * 32 + lane];
```

Since `Bfrag` is a `uint16_t*`, write it as:

```C++uint16_t packed_16 = Bfrag[(ks * 8 + 0) * 32 + lane];uint32_t packed_32 = packed_16;
C++uint16_t packed_16 = Bfrag[(ks * 8 + 0) * 32 + lane];uint32_t packed_32 = packed_16;
```

That keeps the validation mental model exact: one lane owns one 16-bit packed fragment.

### 2) Do not expect exact parity with an FP32 GEMM reference

Because your MMA instruction is:

```C++mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
C++mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
```

you are accumulating in **FP16**, not FP32. So your harness should compare against either:

- a reference that also accumulates in FP16 in the same fragment/order model, or
- an FP32 reference with a tolerance, not bitwise equality. The PTX opcode itself tells you the accumulator/output type is FP16. [NVIDIA Docs+1](https://docs.nvidia.com/cuda/parallel-thread-execution/?utm_source=chatgpt.com)

### 3) Enforce alignment now for the next step

For the current logical path, 2-byte alignment is enough.
But since the next optimization step is `ld.shared.u32`, make sure `bfrag_stage_offset` is **4-byte aligned now**, so you do not have to change the shared-memory layout later.

## My verdict

For the **validation harness**, yes:

- lane mapping is correct,
- nibble order is correct,
- the magic decode path is correct,
- the FP16 accumulator footprint is now correct,
- and the double-buffered structure is good.

So the next milestone is:

1. validate **decode only** against a scalar host/device reference for one warp fragment,
2. validate one full `ks/j` MMA step against a reference with FP16 accumulation semantics,
3. then replace:

```C++uint16_t packed_16 = Bfrag[(ks * 8 + j) * 32 + lane];
C++uint16_t packed_16 = Bfrag[(ks * 8 + j) * 32 + lane];
```

with the real shared-memory packed fetch path.

Once that passes, the next thing to write is the **physical ld.shared.u32 layout** for `Bfrag`, with 4-byte aligned lane-pair packing and no change to the logical lane ownership.

---

### User:
provide in zip the whole project please, and the todo file

---

### ChatGPT:
I’m packaging the full project into a zip and adding a separate TODO file with the CUDA and integration next steps.

---

Here you go:

The zip includes:

- the full `quant_v5_3` Python project
- `README.md`
- `requirements.txt`
- `create_sdq_v53.sh`
- `kernels/ampere_marlin_paro_int4_validation.cu`
- the same `TODO_v5_3.md` inside the project folder

One important note: the CUDA file in the zip is the **validation skeleton**, not a finished production kernel yet. The TODO file lists the exact next steps to take it from validated fragment math to a packed shared-memory implementation.

---

### User:
whats the point of this ?

---

### ChatGPT:
The point of it is **to answer one question with evidence**:

**Can you get noticeably better quality than plain 4-bit GPTQ, while staying inside a deployment path that still resembles GPTQ/vLLM speed?**

That project is a **test harness**, not the final weapon.

What it gives you:

- a **baseline**: plain GPTQ 4-bit
- a **smarter allocator**: Hessian-proxy decides which layers stay 4-bit, go 8-bit, or stay FP16
- a **better rounding backend**: AutoRound instead of greedy GPTQ rounding
- a **matrix** to compare all of them on the same model and tasks

So the real purpose is to separate:

1. **quality gain from smarter allocation**
2. **quality gain from better rounding**
3. **format/runtime constraints**
4. **whether the extra complexity is actually worth it**

Why the zip matters:

- it saves you from rebuilding the scaffold manually
- it gives you one place to run the experiments
- it includes the CUDA validation skeleton for the future fused kernel path

Why the TODO file matters:

- the Python side is enough to benchmark ideas now
- the CUDA side is **not finished**
- the TODO is the bridge from “research scaffold” to “real kernel implementation”

What it is **not**:

- not a finished production quantizer
- not guaranteed to beat EXL3 / ParoQuant / AutoRound
- not yet a drop-in fastest inference engine

So the short truth is:

**The point is to de-risk the idea before you spend weeks writing kernels.**

If the benchmark says:

- `sdq_gptq` barely beats baseline, then stop
- `sdq_autoround` wins clearly, then invest more
- `auto_round` format wins but `auto_gptq` doesn’t, then you know exactly where the bottleneck is

If your actual goal is just the most practical path today, then the point of this project is mostly **research and ablation**, not immediate deployment.

---
