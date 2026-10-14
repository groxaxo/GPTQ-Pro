# SPDX-License-Identifier: Apache-2.0
"""Replay historical campaign defects, without loading a model or using CUDA.

This executes trusted, hash-pinned campaign code; it is NOT a security sandbox.
Exit 0 confirms undesirable behaviours, not release readiness. See README.md.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import runpy
import sys
import tempfile
import traceback
import types
from pathlib import Path


def verify_source(job: Path, manifest: dict) -> None:
    """Reject a missing or changed source before executing any campaign code."""
    if not job.is_dir():
        raise ValueError("campaign-root must be an existing trusted directory")
    for name in manifest["replay_required_files"]:
        path = (job / name).resolve()
        if not path.is_relative_to(job):
            raise ValueError(f"Campaign source escapes root: {name}")
        if not path.is_file():
            raise ValueError(f"Missing required audited source: {name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest["files"][name]:
            raise ValueError(
                f"Audited source hash mismatch: {name}; this replay does not target modified code"
            )


def replay(job: Path, workspace: Path) -> list[dict]:
    """Run the eight original reproductions with CPU fixtures and a fake provider."""
    JOB = job
    ROOT = workspace
    results = []

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def record(name, reproduced, **details):
        results.append({"case": name, "reproduced": bool(reproduced), **details})

    # Execute the exact outer entrypoint AST, with a harmless mocked main().
    source = (JOB / "verify_after.py").read_text()
    tree = ast.parse(source)
    outer = tree.body[-1]
    assert isinstance(outer, ast.If)
    code = compile(
        ast.Module(body=[outer], type_ignores=[]), str(JOB / "verify_after.py"), "exec"
    )
    for return_code, original in [
        (0, "passed"),
        (4, "not_ready"),
        (5, "blocked_runtime_resources"),
    ]:
        writes = []

        def fake_main():
            writes.append(("status.json", {"status": original}))
            return return_code

        namespace = {
            "__name__": "__main__",
            "main": fake_main,
            "sys": sys,
            "traceback": traceback,
            "save": lambda name, payload: writes.append((name, payload)),
        }
        observed = None
        try:
            exec(code, namespace)
        except SystemExit as exc:
            observed = exc.code
        record(
            "validator_exit_" + str(return_code),
            writes[-1][1]["status"] == "validation_failed",
            exit_code=observed,
            original_status=original,
            final_status=writes[-1][1]["status"],
            reported_error=writes[-1][1].get("error"),
        )

    # Execute the full runtime-probe script with an in-memory fake vLLM provider.
    original_vllm = sys.modules.get("vllm")
    original_argv = sys.argv[:]
    for responses, label in [
        (["version 4", "5"], "false_positive_wrong_arithmetic"),
        (["ready", "5"], "quality_exit_code_reclassified"),
    ]:
        stub = types.ModuleType("vllm")

        class FakeLLM:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def generate(self, prompts, params):
                return [
                    types.SimpleNamespace(
                        prompt=prompt,
                        outputs=[types.SimpleNamespace(text=text, token_ids=[1])],
                    )
                    for prompt, text in zip(prompts, responses)
                ]

        stub.LLM = FakeLLM
        stub.SamplingParams = lambda **kwargs: kwargs
        sys.modules["vllm"] = stub
        sys.argv = [str(JOB / "vllm_probe.py"), str(ROOT / "synthetic-not-a-model")]
        capture = io.StringIO()
        exit_code = 0
        try:
            with contextlib.redirect_stdout(capture):
                runpy.run_path(str(JOB / "vllm_probe.py"), run_name="__main__")
        except SystemExit as exc:
            exit_code = exc.code
        text = capture.getvalue()
        record(
            label,
            ("runtime_smoke_pass" in text and exit_code == 0)
            if label.startswith("false")
            else exit_code == 2,
            responses=responses,
            exit_code=exit_code,
            stdout=text,
        )
    sys.argv = original_argv
    if original_vllm is None:
        sys.modules.pop("vllm", None)
    else:
        sys.modules["vllm"] = original_vllm

    # A deliberately malformed tiny checkpoint must NOT be structurally certified.
    # Call only the verifier main: CUDA is hidden, so runtime resource validation stops it.
    import torch
    from safetensors.torch import save_file

    verifier = load("audit_verify_after", JOB / "verify_after.py")
    with tempfile.TemporaryDirectory(prefix="invalid-checkpoint-", dir=ROOT) as tmp:
        fixture = Path(tmp)
        output = fixture / "output"
        output.mkdir()
        full = fixture / "full"
        full.mkdir()
        (full / "phase.txt").write_text("exported_unverified")
        (full / "artifact.json").write_text("{}")
        (output / "config.json").write_text("{}")
        (output / "quantize_config.json").write_text(
            json.dumps(
                {
                    "bits": 8,
                    "group_size": 999,
                    "dynamic": {"not-a-policy": {"group_size": 32}},
                }
            )
        )
        tensors = {f"mtp.fake.{i}": torch.zeros(1) for i in range(19)}
        tensors.update({f"model.visual.fake.{i}": torch.zeros(1) for i in range(200)})
        tensors["model.fake.qweight"] = torch.zeros(
            1
        )  # wrong dtype, shape, count, and no scales/qzeros/g_idx
        save_file(tensors, str(output / "model.safetensors"))
        verifier.ROOT = fixture
        verifier.OUTPUT = output
        verifier.RESULT = fixture / "validation"
        return_code = verifier.main()
        structure = json.loads((verifier.RESULT / "structural.json").read_text())
        status = json.loads((verifier.RESULT / "status.json").read_text())
        record(
            "malformed_checkpoint_structure_accepted",
            structure["status"] == "structure_checked"
            and structure["tensors"]["qweights"] == 1,
            return_code=return_code,
            structure=structure,
            runtime_status=status,
            scope="structural gate only; no GPU runtime was attempted",
        )

    # Assertions used as source-integrity checks disappear with Python -O.
    func = next(
        n
        for n in ast.parse((JOB / "run_quant.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "source_check"
    )
    with tempfile.TemporaryDirectory(prefix="invalid-source-", dir=ROOT) as tmp:
        p = Path(tmp)
        (p / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "NOT_QWEN",
                    "text_config": {
                        "num_hidden_layers": 1,
                        "num_experts": 1,
                        "num_experts_per_tok": 1,
                        "mtp_num_hidden_layers": 0,
                    },
                }
            )
        )
        (p / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"fake": "missing.safetensors"}})
        )
        namespace = {"SOURCE": p, "json": json}
        exec(
            compile(
                ast.Module(body=[func], type_ignores=[]),
                str(JOB / "run_quant.py"),
                "exec",
                optimize=1,
            ),
            namespace,
        )
        stats = namespace["source_check"]()
        record(
            "optimized_python_removes_source_gates",
            stats["mtp"] == 0 and stats["vision"] == 0,
            result=stats,
        )

    # Dense-style FIFO numbering is incompatible with skipped global batch indices.
    spill = load(
        "audit_activation_spill", JOB / "engine/gptqmodel/utils/activation_spill.py"
    )
    with tempfile.TemporaryDirectory(prefix="spill-", dir=ROOT) as tmp:
        queue = spill.DiskTensorQueue(tmp, "example.sparse.expert")
        try:
            queue.append(torch.ones(1, 4), batch_index=2)
            message = "accepted"
        except Exception as exc:
            message = str(exc)
        record(
            "native_spill_rejects_sparse_global_batch_indices",
            "Native input order mismatch" in message,
            error=message,
            scope="conditional: routed experts with skipped batches; current quality_4bit job has GPTAQ disabled",
        )

    return results


def sanitize_results(results: list[dict], job: Path, workspace: Path) -> list[dict]:
    """Omit raw traces and private paths while retaining diagnostic observations."""
    clean = json.loads(json.dumps(results))
    for case in clean:
        stdout = case.pop("stdout", None)
        if stdout:
            case["events"] = [
                json.loads(line)["event"]
                for line in stdout.splitlines()
                if line.startswith("{")
            ]
        if "structure" in case:
            case["structure"]["checkpoint"] = "<temporary-invalid-checkpoint>/output"
    text = (
        json.dumps(clean)
        .replace(str(job), "<campaign-root>")
        .replace(str(workspace), "<temporary-workspace>")
    )
    return json.loads(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root",
        required=True,
        type=Path,
        help="Trusted, unmodified campaign matching source-manifest.json",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New JSON file outside the campaign; refuses overwrite",
    )
    args = parser.parse_args()
    job = args.campaign_root.resolve()
    output = args.output.resolve()
    try:
        if output.is_relative_to(job):
            raise ValueError("Output must be outside the campaign root")
        if output.exists():
            raise ValueError("Output already exists; refusing overwrite")
        if not output.parent.is_dir():
            raise ValueError("Output parent directory must already exist")
        manifest = json.loads(
            Path(__file__).with_name("source-manifest.json").read_text()
        )
        verify_source(job, manifest)
        # Set before any Torch/campaign import. Do not touch production processes.
        os.environ.update(
            CUDA_VISIBLE_DEVICES="",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            OMP_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
        )
        sys.dont_write_bytecode = True
        with tempfile.TemporaryDirectory(prefix="gptq-pro-audit-") as tmp:
            workspace = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                results = replay(job, workspace)
            verify_source(job, manifest)
            results = sanitize_results(results, job, workspace)
        # Exclusive creation protects previous evidence, including concurrent writes.
        with output.open("x", encoding="utf-8") as target:
            json.dump(results, target, indent=2, allow_nan=False)
            target.write("\n")
        confirmed = sum(bool(case["reproduced"]) for case in results)
        print(
            f"Historical undesirable behaviours reproduced: {confirmed}/{len(results)}."
        )
        print("This is not a model-quality test or a passing release gate.")
        return 0 if len(results) == 8 and confirmed == 8 else 1
    except Exception as exc:
        message = str(exc).replace(str(job), "<campaign-root>")
        print(f"Replay invalid or incomplete: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
