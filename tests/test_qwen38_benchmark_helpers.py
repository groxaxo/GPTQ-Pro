from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_qwen38_gptqpro",
    ROOT / "scripts" / "benchmark_qwen38_gptqpro.py",
)
assert SPEC and SPEC.loader
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class ByteTokenizer:
    bos_token_id = 256

    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        return {"input_ids": list(text.encode("utf-8"))}


def contains_subsequence(values: list[int], needle: list[int]) -> bool:
    width = len(needle)
    return any(values[index:index + width] == needle for index in range(len(values) - width + 1))


def test_context_builder_is_exact_and_contains_needle() -> None:
    tokenizer = ByteTokenizer()
    ids, needle, fraction = bench.build_context_token_ids(tokenizer, 2048, 0)
    assert len(ids) == 2048
    assert 0.0 < fraction < 1.0
    assert contains_subsequence(ids, list(needle.encode("utf-8")))


def test_artifact_metadata_gate(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    report = {
        "definition": "Qwen3_5QModel",
        "expected_quantized_modules": 400,
        "packed_modules": 400,
        "quantized_layers": 64,
        "total_layers": 64,
        "group_size": 64,
        "preset": "max_quality",
        "symmetric": True,
        "desc_act": False,
        "assembled_from_resumable_chunks": True,
        "calibration_sha256": "abc",
    }
    manifest = {
        "schema": "qwen38-gptq-pro-resume-assembled/v2",
        "expected_layers": 64,
        "chunk_layers": 4,
        "packed_qweight_tensors": 400,
        "calibration_sha256": "abc",
        "chunks": [{} for _ in range(16)],
        "recipe": {
            "group_size": 64,
            "preset": "max_quality",
            "symmetric": True,
            "desc_act": False,
        },
    }
    (model / "qwen3_8_27b_preflight.json").write_text(json.dumps(report), encoding="utf-8")
    (model / "qwen38_resumable_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = bench.validate_artifact(
        model,
        require_manifest=True,
        expected_packed_modules=400,
        expected_group_size=64,
        expected_preset="max_quality",
    )
    assert result["passed"], result["errors"]


def test_structural_quality_validators() -> None:
    python_ok, _ = bench.validate_python_response(
        "```python\ndef parse_jsonl(lines):\n    return [line for line in lines]\n```"
    )
    json_ok, _ = bench.validate_json_response(
        '{"action":"deploy","risk":"low","rollback":"revert"}'
    )
    spanish_ok, _ = bench.validate_spanish_response(
        "El prefill procesa el contexto de entrada, mientras que el decode genera los tokens de salida con la caché KV."
    )
    diagnostic_ok, _ = bench.validate_diagnostic_response(
        "Profile prefill and attention, inspect KV cache growth, and record VRAM memory use."
    )
    assert python_ok
    assert json_ok
    assert spanish_ok
    assert diagnostic_ok
