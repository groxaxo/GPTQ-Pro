# Qwopus3.8-27B Flash V2 / GPTQ-Pro / three RTX 3090s

Status: **design and read-only capacity preflight. No Qwopus model has yet been
quantized or benchmarked in GPTQ-Pro.** Do not treat a resource check as a
successful build, deployment, or permission to evict a running proxy model.

## Reference and source

- BF16: https://huggingface.co/Jackrong/Qwopus3.8-27B-Flash-V2
  (27.7814B params, qwen3_5 multimodal wrapper, 64 decoder layers, 5120
  hidden width, 17408 FFN, 3 linear-attention + 1 full-attention repeated).
- Published benchmark: https://huggingface.co/JRamirez-UAB/Qwopus3.8-27B-Flash-V2-GPTQ-W4A16-embed-int4-24GB
  The author's card reports 14.21 GiB in the mtp-int8 revision, 222k text
  context on one RTX 3090 and KL divergence 0.0357 on the author's internal
  BF16 comparison. These are *their* figures, not GPTQ-Pro test results.
- GPTQ-Pro target: symmetric W4A16 linears (g128 for a reproduction arm;
  g64 for its validated-quality candidate), INT4 token embeddings and
  lm_head, separate INT8 g128 MTP projections, BF16 protected/recurrent
  GatedDeltaNet and vision modules. Do not change protected tensor types
  just to make a checkpoint load.

The current GPTQ-Pro dense Qwen3.8 driver quantizes decoder linears and
leaves embeddings, lm_head, mtp and vision in original precision. A reference
match requires additional selective quantization and loader work. The
current native GPTQ-Pro backend is not a general compressed-tensors loader.

## Admission before running anything

Read-only preflight:

    python3 scripts/plan_qwopus3gpu.py --workspace /home/op/quant-jobs --json

Planning budget: approximately 51.75 GiB BF16 source + 18 GiB output +
24 GiB scratch + 20 GiB env/cache + 25 GiB headroom; the script uses the
rounded-up total as its minimum disk admission gate, not a measured upper
bound. Target >=150 GiB free before downloading weights. Also require >=72
GiB MemAvailable and >=20 GiB free VRAM on EACH of three 3090s. These are
conservative admission heuristics, not proof of ability to finish, especially
if three workers load three full CPU models.

On 29 October 2026 on workstation 12700, disk had ~65 GB free, ~13 GiB
RAM was available, and all three GPUs had ~0.5 GiB VRAM free. The resident
strata-dsv41 service was occupying the three cards. No service was stopped
and no checkpoints were deleted during the initial inspection.

## Implementation plan (not yet built)

1. Pin upstream source revision and hashes. Validate the source tensor index
   for one MTP head, source precision, and proper module coverage. The config
   check alone cannot validate MTP tensor preservation.
2. Produce representative long-context English/Spanish coding, real agent
   and tool-use calibration data; keep a held-out evaluation set. Compare
   against the external reference's 512 x 2048 general-chat calibration in
   a separate experiment.
3. Quantize standard linears in GPTQ-Pro with g128 and g64 recipes, recording
   the effective solver, its group size and exact packed module inventory.
   Keep sensitive projections and recurrent helpers in source precision.
4. Implement genuinely concurrent 3-GPU scheduling, preferably parallel
   independent projection groups inside a layer with a barrier before the
   next layer's calibration, or another verified pipeline preserving true
   sequential activations. **CUDA_VISIBLE_DEVICES=0,1,2 alone does not prove
   all GPUs are computing.** Existing independent four-layer chunk results
   are not numerically equivalent to a single true-sequential quantization.
   Their full mixed-precision checkpoints can also consume much more disk.
5. Add selective INT4 symmetric RTN embeddings and lm_head without breaking
   any shared/tied-weight semantics. Add independent INT8 g128 MTP processing,
   with byte-level tensor verification and actual speculative decoding tests.
6. Implement and separately verify the required consumer/loader contract,
   whether GPTQ-Pro native first or an explicitly supported
   compressed-tensors/vLLM export. A saved checkpoint is not a passing load.
7. Compare to BF16 and the reference on held-out KL/NLL, real tool calls,
   8-fact retrieval at high context, images, reasoning, MTP tokens/step,
   time-to-first-token and 1/2/8-request throughput.
8. During the pilot log each 3090's useful GPU utilisation, max VRAM,
   system RAM, disk bytes and PCIe transfers. Do not call it a "3-GPU
   simultaneous build" without this instrumentation and execution evidence.

## Safety and acceptance

- Never automatically unload an inference service or delete a model to
  satisfy these resource gates.
- Do not run a full download/quantization until capacity is sufficient and
  the project owner has an explicit maintenance/compute window.
- No claim of model-card-equivalent output until embeddings, lm_head and
  MTP are implemented, reloaded and scored.
- First delivered increment: this capacity guard and architecture/quality
  implementation plan. Multi-GPU quantization, weights, and benchmarks remain
  separate uncompleted deliverables.
