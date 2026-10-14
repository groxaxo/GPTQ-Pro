#!/usr/bin/env python3
"""Bounded real first-block Qwen3.5-MoE expert reconstruction experiment.

Reads original BF16 tensors and existing GPTQ packed tensors; never changes either
checkpoint. CPU mode hides CUDA. CUDA mode requires an assigned gpuq reservation.
Captures real first-block post-attention activations without materializing the full
35B model. It does NOT test downstream layers, language accuracy, or long context.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VALIDATION = [
    (
        "Explain idempotent API design.",
        "An idempotent operation can be repeated without changing the outcome beyond the first successful application. A request identifier can connect retries to a stored result. The server should validate the request body and reject reuse of the same identifier with different content. This prevents a network retry from charging an account twice. Store the identifier and the state change in one transaction. Expire old identifiers according to the service retention policy.",
    ),
    (
        "Explica cómo funciona una cola de trabajos.",
        "Una cola separa la recepción de una tarea de su ejecución. El productor guarda el trabajo y un consumidor lo procesa cuando hay recursos. Cada intento necesita un identificador y un estado claro. Los errores temporales pueden reintentarse con una espera creciente. Los errores permanentes deben registrarse sin crear un bucle infinito. Si un trabajador se detiene, otro puede continuar únicamente después de comprobar que no hay una ejecución activa del mismo trabajo.",
    ),
    (
        "Design a safe database migration.",
        "First add the new nullable column without removing the original field. Deploy code that can read both representations and writes the new representation. Backfill existing records in bounded batches. Verify row counts, value distributions and application behavior. Switch reads after checking consistency. Keep the old representation until rollback is no longer required. Remove obsolete fields in a separate change after all consumers have migrated.",
    ),
    (
        "Describe a deterministic software test.",
        "A deterministic test controls its input data, random seed and external dependencies. Its assertions compare behavior rather than incidental formatting. Temporary files belong in a unique directory. Network requests should be replaced by explicit fixtures unless the test is intentionally an integration test. A failing test must preserve enough diagnostics to reproduce the failure. Passing on one computer is not proof that timing-dependent behavior is correct.",
    ),
    (
        "Explain how a hash table handles collisions.",
        "Different keys can produce the same bucket index. A hash table must still compare the original keys before returning a value. Separate chaining stores multiple entries in a bucket. Open addressing probes other slots according to a defined rule. Resizing reduces the load factor but costs additional memory and work. A good implementation tests deletion, duplicate keys and adversarial collisions independently.",
    ),
    (
        "Explain why training and testing need separate documents.",
        "A model can memorize the phrasing of a document rather than learn a general rule. Splitting adjacent tokens from the same conversation across training and testing leaks context. Assign each complete document to one split before tokenization. Use validation to choose settings, and reserve a separate test set for the final report. Do not adjust hyperparameters after inspecting test errors and then present the same test as independent.",
    ),
    (
        "Describe a safe file upload service.",
        "Limit upload size and validate the declared format against the actual bytes. Store files under generated identifiers rather than trusting a client path. Apply authorization when both writing and retrieving content. Do not execute uploaded files. Record checksums to detect incomplete transfers. A temporary upload should become visible only after integrity checks succeed. Keep operational errors separate from content-validation failures.",
    ),
    (
        "Explica el valor de las pruebas de regresión.",
        "Las pruebas de regresión conservan ejemplos de fallos anteriores para detectar su reaparición. Deben ejecutarse junto con casos normales y entradas inválidas. No basta con comprobar que una función devuelve algo: hay que verificar el resultado esperado y los efectos secundarios. Un cambio que mejora una métrica puede romper otra parte del sistema. Por eso conviene mantener invariantes y medir las diferencias con una referencia estable.",
    ),
]
TEST = [
    (
        "Explain optimistic concurrency control.",
        "A record carries a version number. A writer reads the record, prepares a change, and submits the expected version with the update. The database accepts the update only when that version still matches. A conflict means another writer changed the record and the operation must be reconsidered. Retrying without rereading can overwrite valid work. Version checks complement transactions but do not replace all application-level constraints.",
    ),
    (
        "Explica cómo diseñar una copia de seguridad verificable.",
        "Una copia de seguridad debe incluir los datos necesarios para restaurar el servicio y una descripción de sus versiones. Los archivos necesitan comprobaciones de integridad. Guardar una copia no demuestra que pueda recuperarse: hay que ensayar la restauración en un entorno separado. Conviene registrar cuánto tarda y qué información falta. La prueba no debe sobrescribir el sistema original ni depender de credenciales que desaparecerían durante una emergencia.",
    ),
    (
        "Explain bounded memory processing.",
        "A bounded-memory pipeline reads data in chunks and discards intermediate values after accumulating the required statistics. Chunk boundaries must not change the meaning of the computation. Running totals should use appropriate numerical precision. The program needs explicit limits for caches and queues. It should report an admission failure instead of silently changing the algorithm when memory becomes scarce. Resource use and output correctness are different measurements.",
    ),
    (
        "Explain a reproducible performance benchmark.",
        "Record the hardware, software revision, inputs and configuration before measuring. Warm up the actual code path separately from timed iterations. Report the number of repetitions and variation, not only the fastest result. Make sure competing jobs are accounted for. Compare equivalent output quality and resource limits. A result from a synthetic kernel should not be labeled as full-application throughput.",
    ),
    (
        "Explain binary search prerequisites.",
        "Binary search requires an ordered search space and a comparison rule consistent with that ordering. It repeatedly removes the half that cannot contain the target. The implementation must handle empty inputs, duplicate values and boundaries. Use an interval convention consistently to avoid missing the final candidate. A logarithmic number of comparisons does not imply logarithmic total cost when indexing the underlying container is expensive.",
    ),
    (
        "Describe graceful service shutdown.",
        "Stop accepting new work before terminating existing workers. Mark the instance unavailable to the load balancer and let in-flight requests finish within a bounded period. Persist the state needed to retry unfinished jobs safely. Release locks only when ownership is verified. Shutdown logs should distinguish a planned exit from a crash. The next startup must reconcile abandoned attempts before accepting duplicate work.",
    ),
    (
        "Explain calibration drift in deployed systems.",
        "Calibration data is an approximation of future usage. Changes in language, task mix or input length can make that approximation less representative. Monitor important segments independently rather than relying on one global average. Keep a stable evaluation set and add new holdout cases without mixing them into optimization. Changes should be tested against both old and new distributions before becoming the default.",
    ),
    (
        "Describe reliable experiment provenance.",
        "Record source checksums, code identity, data identity and the exact runtime settings alongside each result. Distinguish metadata checks from hashing complete weight files. Keep failed attempts visible so a later success does not erase the original error. A report should identify skipped tests explicitly. Do not claim two experiments are equivalent merely because they share a friendly model name or nominal bit width.",
    ),
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_hash(tensor):
    import torch

    return hashlib.sha256(
        tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def atomic_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def execute(args, report):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeDecoderLayer,
        Qwen3_5MoeTopKRouter,
    )

    from gptqmodel.quantization.refinement import (
        PackedLinear,
        RefinementConfig,
        reconstruction_error,
        refine_block,
        swiglu_forward,
        validate_splits,
    )
    from gptqmodel.utils.quality_coverage import audit_quant_log, recount_calibration

    torch.set_num_threads(2)
    device = torch.device("cpu" if args.device == "cpu" else "cuda:0")
    if device.type == "cuda":
        if not os.environ.get("CUDA_VISIBLE_DEVICES") or torch.cuda.device_count() != 1:
            raise RuntimeError(
                "CUDA mode requires one gpuq-assigned device; do not override its assignment"
            )
        if torch.cuda.mem_get_info()[0] < 3 * 2**30:
            raise RuntimeError("need 3 GiB free admitted VRAM")
    elif torch.cuda.is_available():
        raise RuntimeError("CPU probe must run with CUDA_VISIBLE_DEVICES empty")
    # Local process only; never change server settings.
    torch.backends.cuda.matmul.allow_tf32 = False
    config = AutoConfig.from_pretrained(
        args.source, local_files_only=True, trust_remote_code=False
    )
    if config.model_type != "qwen3_5_moe":
        raise ValueError("probe requires original Qwen3.5-MoE/Qwen3.6 architecture")
    cfg = config.text_config
    if (
        cfg.layer_types[0] != "linear_attention"
        or cfg.hidden_size != 2048
        or cfg.moe_intermediate_size != 512
    ):
        raise ValueError(
            "probe bound requires first linear-attention block, hidden=2048, expert=512"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.source, local_files_only=True, trust_remote_code=False
    )
    source_index = json.loads(
        (args.source / "model.safetensors.index.json").read_text()
    )["weight_map"]
    quant_index = json.loads(
        (args.checkpoint / "model.safetensors.index.json").read_text()
    )["weight_map"]
    quant_config = json.loads((args.checkpoint / "quantize_config.json").read_text())
    checkpoint_format = quant_config.get(
        "checkpoint_format", quant_config.get("format")
    )
    if checkpoint_format not in ("gptq", "gptq_v2"):
        raise ValueError(
            "checkpoint must explicitly declare GPTQ v1 or v2 zero-point storage"
        )
    zero_format = 1 if checkpoint_format == "gptq" else 2
    report["input_checkpoint_format"] = checkpoint_format
    report["candidate_fixture_format"] = "gptq_v2; canonical zero-point eight"

    def read(root, index, name, selection=None):
        path = (root / index[name]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("shard index escapes checkpoint directory")
        with safe_open(path, framework="pt", device="cpu") as handle:
            return (
                handle.get_tensor(name)
                if selection is None
                else handle.get_slice(name)[selection]
            )

    coverage = audit_quant_log(args.quant_log, legacy_solver="gptq")
    # Large per-projection ledger is kept locally, not expanded into the final report.
    atomic_json(args.output_dir / "coverage.json", coverage)
    report["coverage"] = {k: v for k, v in coverage.items() if k != "entries"}
    report["calibration"] = recount_calibration(args.calibration)
    report["identity"] = {
        "source_config_sha256": sha(args.source / "config.json"),
        "source_index_sha256": sha(args.source / "model.safetensors.index.json"),
        "checkpoint_index_sha256": sha(
            args.checkpoint / "model.safetensors.index.json"
        ),
        "probe_sha256": sha(__file__),
        "source_weight_hash_scope": "selected tensors only, not whole checkpoint",
    }
    cfg._attn_implementation = "eager"
    with torch.device("meta"):
        block = Qwen3_5MoeDecoderLayer(cfg, 0)
        router = Qwen3_5MoeTopKRouter(cfg)

    class Capture(torch.nn.Module):
        value = None

        def forward(self, hidden):
            self.value = hidden.detach().clone()
            return torch.zeros_like(hidden)

    capture = Capture()
    block.mlp = capture
    prefix = "model.language_model.layers.0."
    state = {
        name: read(args.source, source_index, prefix + name).to(device)
        for name in block.state_dict()
    }
    block.load_state_dict(state, assign=True, strict=True)
    router.load_state_dict(
        {
            "weight": read(args.source, source_index, prefix + "mlp.gate.weight").to(
                device
            )
        },
        assign=True,
    )
    block.eval()
    router.eval()
    embedding = read(
        args.source, source_index, "model.language_model.embed_tokens.weight"
    )
    if embedding.dtype != torch.bfloat16:
        raise ValueError("source embedding must be BF16")
    corpus = [json.loads(line) for line in args.calibration.open()]
    train_rows = corpus[: args.train_examples]
    encoded = {"train": [row["input_ids"][: args.max_tokens] for row in train_rows]}
    # IDs hash whole documents rather than token rows. Include existing source IDs.
    ids = {
        "train": [
            str(
                row.get(
                    "source_id",
                    hashlib.sha256(json.dumps(row["input_ids"]).encode()).hexdigest(),
                )
            )
            for row in train_rows
        ]
    }
    for split, examples in (("validation", VALIDATION), ("test", TEST)):
        encoded[split], ids[split] = [], []
        for question, answer in examples:
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(tokens) > args.max_tokens:
                raise ValueError(
                    "heldout document exceeds bound; increase max_tokens rather than silently truncate"
                )
            encoded[split].append(tokens)
            ids[split].append(
                hashlib.sha256(
                    json.dumps(messages, sort_keys=True).encode()
                ).hexdigest()
            )
    validate_splits(ids["train"], ids["validation"], ids["test"])
    # Also catch identical token sequences even when external document IDs differ.
    token_ids = {
        split: [hashlib.sha256(json.dumps(row).encode()).hexdigest() for row in rows]
        for split, rows in encoded.items()
    }
    validate_splits(token_ids["train"], token_ids["validation"], token_ids["test"])
    records, counts = {}, {}
    with torch.inference_mode():
        for split, documents in encoded.items():
            values, routes = [], []
            for index, tokens in enumerate(documents):
                token_tensor = torch.tensor([tokens], dtype=torch.long)
                hidden = torch.nn.functional.embedding(token_tensor, embedding).to(
                    device
                )
                block(
                    hidden,
                    position_embeddings=None,
                    attention_mask=torch.ones_like(token_tensor, device=device),
                    past_key_values=None,
                )
                value = capture.value.reshape(-1, cfg.hidden_size)
                if value.dtype != torch.bfloat16 or not bool(
                    torch.isfinite(value).all()
                ):
                    raise FloatingPointError(
                        "source attention capture must remain finite BF16"
                    )
                _, _, selected = router(value)
                values.append(value.cpu())
                routes.append(selected.cpu())
                if (index + 1) % 8 == 0:
                    print(
                        json.dumps(
                            {"phase": "capture", "split": split, "examples": index + 1}
                        ),
                        flush=True,
                    )
            records[split] = (torch.cat(values), torch.cat(routes))
            counts[split] = torch.bincount(
                records[split][1].reshape(-1), minlength=cfg.num_experts
            )
    del embedding, block, router, state, capture
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    chosen = sorted(
        range(cfg.num_experts), key=lambda i: (-int(counts["train"][i]), i)
    )[: args.experts]
    report["data"] = {
        "scope": "small first-block pilot; real BF16 source activations; not frozen task benchmark",
        "template_sha256": sha(args.source / "chat_template.jinja")
        if (args.source / "chat_template.jinja").is_file()
        else None,
        "ids": ids,
        "token_sha256": token_ids,
        "tokens": {k: sum(map(len, v)) for k, v in encoded.items()},
        "selection": "experts ranked by training routing count only, before inspecting errors",
        "selected_experts": chosen,
        "routing_counts": {k: v.tolist() for k, v in counts.items()},
    }
    report["experts"] = []
    atomic_json(args.output_dir / "report.json", report)
    for expert in chosen:
        tensors = {}
        for split, (values, routes) in records.items():
            selected = (routes == expert).any(-1)
            # Evenly sample bounded rows; use separate whole-document split IDs.
            x = values[selected]
            if len(x) > args.max_rows:
                x = x[torch.linspace(0, len(x) - 1, args.max_rows).long()]
            tensors[split] = x.contiguous()
        expert_report = {
            "layer": 0,
            "expert": expert,
            "rows": {k: len(v) for k, v in tensors.items()},
            "arms": {},
        }
        if (
            min(expert_report["rows"].values()) < 32
            or expert_report["rows"]["train"] < 128
        ):
            expert_report["status"] = "blocked_undercoverage"
            report["experts"].append(expert_report)
            continue
        gate_up = read(
            args.source,
            source_index,
            prefix + "mlp.experts.gate_up_proj",
            selection=expert,
        )
        down = read(
            args.source,
            source_index,
            prefix + "mlp.experts.down_proj",
            selection=expert,
        )
        if gate_up.shape != (1024, 2048) or down.shape != (2048, 512):
            raise ValueError("unexpected fused expert source orientation")
        source = {
            "gate_proj": gate_up[:512].contiguous().to(device),
            "up_proj": gate_up[512:].contiguous().to(device),
            "down_proj": down.to(device),
        }
        expert_report["source_sha256"] = {k: tensor_hash(v) for k, v in source.items()}
        baseline = {}
        existing_weights = {}
        expert_report["existing_zero_scale_groups"] = {}
        for projection, source_weight in source.items():
            name = prefix + f"mlp.experts.{expert}.{projection}."
            packed = {
                key: read(args.checkpoint, quant_index, name + key).to(device)
                for key in ("qweight", "scales", "qzeros", "g_idx")
            }
            group = source_weight.shape[1] // packed["scales"].shape[0]
            from gptqmodel.utils.gptq_pro_contract import dequantize_v3

            existing_weights[projection] = dequantize_v3(
                **packed, group_size=group, qzero_format=zero_format
            ).T.contiguous()
            expert_report["existing_zero_scale_groups"][projection] = int(
                (packed["scales"] == 0).sum()
            )
            if args.baseline == "checkpoint":
                baseline[projection] = PackedLinear.from_packed(
                    **packed, group_size=group, qzero_format=zero_format
                )
            else:
                from gptqmodel.quantization.config import QuantizeConfig
                from gptqmodel.quantization.gptq import GPTQ

                clean_input = tensors["train"].to(device)
                with torch.no_grad():
                    if projection == "down_proj":
                        clean_input = torch.nn.functional.silu(
                            torch.nn.functional.linear(clean_input, source["gate_proj"])
                        ) * torch.nn.functional.linear(clean_input, source["up_proj"])
                    dense = torch.nn.Linear(
                        source_weight.shape[1],
                        source_weight.shape[0],
                        bias=False,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    dense.weight.copy_(source_weight)
                qcfg = QuantizeConfig.gptq_pro(group_size=group, offload_to_disk=False)
                solver = GPTQ(dense, qcfg)
                solver.quantizer.configure(perchannel=True)
                solver.add_batch(
                    clean_input.unsqueeze(0), torch.empty(0, device=device)
                )
                quant, scale, zero, idx, *_ = solver.quantize()
                baseline[projection] = PackedLinear.from_solver(
                    quant, scale, zero, idx, group
                )
                expert_report.setdefault("fresh_recipes", {})[projection] = (
                    qcfg.to_dict()
                )
                del solver, dense, quant, scale, zero, idx, clean_input
                gc.collect()
                print(
                    json.dumps(
                        {
                            "phase": "fresh_baseline",
                            "expert": expert,
                            "projection": projection,
                        }
                    ),
                    flush=True,
                )
        # Independently score the unchanged original artifact, including its zero
        # groups, without admitting those groups into the positive-scale tuner.
        with torch.no_grad():
            error, energy = 0.0, 0.0
            for x in tensors["test"].to(device).split(128):
                ref = swiglu_forward(
                    x.float(), {k: w.float() for k, w in source.items()}
                )
                old = swiglu_forward(
                    x.float(), {k: w.float() for k, w in existing_weights.items()}
                )
                if not bool(torch.isfinite(old).all() & torch.isfinite(ref).all()):
                    raise FloatingPointError(
                        "nonfinite original artifact reconstruction"
                    )
                error += (old.double() - ref.double()).square().sum().item()
                energy += ref.double().square().sum().item()
            expert_report["existing_checkpoint_test_nmse"] = error / energy
            expert_report["baseline_kind"] = args.baseline
        save_file(
            {
                **{k: v.cpu() for k, v in tensors.items()},
                **{"source." + k: v.cpu() for k, v in source.items()},
                **{
                    f"baseline.{name}.{key}": value.cpu()
                    for name, q in baseline.items()
                    for key, value in q.pack().items()
                },
            },
            str(args.output_dir / f"expert-{expert}-capture.safetensors"),
        )
        inputs = {k: v.to(device) for k, v in tensors.items()}
        base_test = reconstruction_error(
            source, baseline, inputs["test"], swiglu_forward
        )
        expert_report["baseline_test_nmse"] = base_test
        expert_report["baseline_validation_nmse"] = reconstruction_error(
            source, baseline, inputs["validation"], swiglu_forward
        )
        schedules = [
            ("scale-50", RefinementConfig(steps=50, tune_rounding=False)),
            ("joint-50", RefinementConfig(steps=50)),
            ("joint-200", RefinementConfig(steps=args.long_steps)),
        ]
        # All arms and steps are declared before any heldout errors are observed.
        for label, options in schedules:
            started = time.monotonic()
            best, details = refine_block(
                source,
                baseline,
                inputs["train"],
                inputs["validation"],
                train_ids=ids["train"],
                validation_ids=ids["validation"],
                config=options,
                forward=swiglu_forward,
            )
            test_error = reconstruction_error(
                source, best, inputs["test"], swiglu_forward
            )
            details.update(
                test_nmse=test_error,
                baseline_test_nmse=base_test,
                test_relative_improvement=1 - test_error / base_test,
                seconds=time.monotonic() - started,
                test_used_for_selection=False,
            )
            candidate_tensors = {
                f"{name}.{key}": value.cpu()
                for name, q in best.items()
                for key, value in q.pack().items()
            }
            save_file(
                candidate_tensors,
                str(args.output_dir / f"expert-{expert}-{label}.safetensors"),
            )
            expert_report["arms"][label] = details
            print(
                json.dumps(
                    {
                        "phase": "refined",
                        "expert": expert,
                        "arm": label,
                        "test_nmse": test_error,
                        "relative_improvement": details["test_relative_improvement"],
                    }
                ),
                flush=True,
            )
        expert_report["status"] = "measured"
        report["experts"].append(expert_report)
        atomic_json(args.output_dir / "report.json", report)
    report["status"] = (
        "measured"
        if any(e["status"] == "measured" for e in report["experts"])
        else "blocked_undercoverage"
    )
    report["full_model_quality_tested"] = False
    report["cuda_kernel_tested"] = False
    report["production_weights_modified"] = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "checkpoint", "calibration", "quant-log", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--baseline", choices=("checkpoint", "fresh"), default="checkpoint"
    )
    parser.add_argument("--train-examples", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-rows", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=2)
    parser.add_argument("--long-steps", type=int, default=200)
    args = parser.parse_args()
    if not (
        1 <= args.train_examples <= 64
        and 128 <= args.max_tokens <= 512
        and 128 <= args.max_rows <= 4096
        and 1 <= args.experts <= 4
        and 1 <= args.long_steps <= 200
    ):
        parser.error("resource bounds exceeded")
    for root in (args.source, args.checkpoint):
        if args.output_dir.resolve().is_relative_to(root.resolve()):
            parser.error("output must not be inside source/checkpoint")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(args.output_dir).free < 2 * 2**30:
        raise RuntimeError("need 2 GiB spare disk for bounded pilot")
    report = {
        "schema": "quality-refinement-real-pilot-v1",
        "status": "running",
        "device": args.device,
        "started_unix": time.time(),
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
    }
    try:
        execute(args, report)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["completed_unix"] = time.time()
        atomic_json(args.output_dir / "report.json", report)
    return 0 if report["status"] == "measured" else 3


if __name__ == "__main__":
    raise SystemExit(main())
