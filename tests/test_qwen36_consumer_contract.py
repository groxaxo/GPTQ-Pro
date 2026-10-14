"""CPU contract regressions; these do not qualify an inference backend."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "qwen36_heretic"
sys.path.insert(0, str(SCRIPTS))
import consumer_contract as consumer
import policy


def module(layer, projection):
    return f"model.language_model.layers.{layer}.mlp.experts.0.{projection}"


def runtime():
    return {
        "backend": "vllm",
        "version": consumer.AUDITED_VLLM_VERSION,
        "loader_sources": {"auto_gptq.py": consumer.AUDITED_AUTO_GPTQ_SHA256},
        "loader_contract": consumer.FUSED_SINGLE_GROUP,
    }


def evidence():
    data = consumer.qualification_evidence_requirements("artifact", runtime(), "source")
    data["representative_layers"] = [0, 10, 39]
    data["checks"] = dict.fromkeys(consumer.REQUIRED_CHECKS, True)
    data["external_report"] = {"path": "/fabricated/report.json", "sha256": "0" * 64}
    return data


def test_exact_recipe_counts_and_identity():
    report = policy.verify()
    assert report["counts"] == {
        "INT4-g64": 19968,
        "INT4-g32": 10496,
        "BF16-preserve": 256,
    }
    assert report["recipe_id"] == policy.RECIPE_ID
    assert len(report["fingerprint"]) == 64
    assert policy.classify(module(0, "gate_proj")) == "INT4-g64"
    assert policy.classify(module(0, "up_proj")) == "INT4-g64"
    assert policy.classify(module(0, "down_proj")) == "INT4-g32"
    assert policy.classify(module(10, "gate_proj")) == "INT4-g32"
    assert policy.classify(module(10, "up_proj")) == "INT4-g32"
    assert policy.classify(module(10, "down_proj")) == "BF16-preserve"


@pytest.mark.parametrize("value", [[], [-1], [40], ["0"], [0.1], [True]])
def test_invalid_pilot_layers_are_explicit_errors(value):
    with pytest.raises(policy.PolicyError):
        policy.dynamic_rules(value)


@pytest.mark.parametrize(
    "name",
    [
        "anything.up_proj",
        "model.language_model.layers.0.mlp.experts.0.generic_proj",
        "model.language_model.layers.40.mlp.experts.0.up_proj",
        "model.language_model.layers.0.mlp.experts.256.up_proj",
        "model.language_model.layers.0.mlp.shared_expert_typo.up_proj",
    ],
)
def test_unknown_and_out_of_range_projections_fail_closed(name):
    with pytest.raises(policy.PolicyError):
        policy.classify(name)


def test_pilot_excludes_other_layers_and_protects_router():
    rules = policy.dynamic_rules([0, 10, 39])
    assert policy.classify(module(1, "gate_proj"), rules) == "BF16-preserve"
    assert policy.classify(module(10, "gate_proj"), rules) == "INT4-g32"
    for name in ("mlp.gate", "mlp.shared_expert_gate", "mlp.shared_expert.up_proj"):
        assert policy.classify("model.language_model.layers.0." + name) == "BF16-preserve"
    # The loader normalizes negative rules before positives on config reload.
    # A plain -:.* catch-all would therefore silently preserve ALL projections.
    reordered = {
        **{k: v for k, v in rules.items() if k.startswith("-:")},
        **{k: v for k, v in rules.items() if not k.startswith("-:")},
    }
    assert policy.classify(module(0, "gate_proj"), reordered) == "INT4-g64"
    assert policy.classify(module(0, "down_proj"), reordered) == "INT4-g32"
    assert policy.classify(module(10, "gate_proj"), reordered) == "INT4-g32"
    assert policy.classify(module(10, "down_proj"), reordered) == "BF16-preserve"


def test_policy_validation_survives_python_optimization():
    code = f"""
import sys
sys.path.insert(0, {str(SCRIPTS)!r})
import policy
for fn in (lambda: policy.dynamic_rules([40]), lambda: policy.classify("generic_proj")):
    try: fn()
    except policy.PolicyError: pass
    else: raise RuntimeError("optimized Python bypassed required validation")
policy.verify()
"""
    result = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_real_quantize_config_save_reload_retains_policy_and_pcre_matching(tmp_path):
    from gptqmodel.quantization.config import QuantizeConfig, dynamic_get

    config = QuantizeConfig.quality_4bit(
        group_size=64,
        dynamic=policy.dynamic_rules(),
        offload_to_disk_path=str(tmp_path / "offload"),
    )
    config.save_pretrained(str(tmp_path))
    restored = QuantizeConfig.from_pretrained(str(tmp_path))
    assert list(restored.dynamic.items()) == list(policy.dynamic_rules().items())
    for layer in (0, 10, 39):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            name = module(layer, projection)
            precision = policy.classify(name)
            result = dynamic_get(restored.dynamic, name, "group_size", 64)
            expected = False if precision == "BF16-preserve" else int(precision.split("g")[1])
            assert result == expected
    assert dynamic_get(restored.dynamic, "generic.new_projection") is False
    assert dynamic_get(restored.dynamic, "model.language_model.layers.0.mlp.gate") is False


@pytest.mark.parametrize("pilot", [None, [0], [0, 10, 39], [10]])
def test_real_transformers_sorted_model_config_preserves_precision(tmp_path, pilot):
    from transformers import PretrainedConfig

    from gptqmodel.quantization.config import QuantizeConfig, dynamic_get

    rules = policy.dynamic_rules(pilot)
    quantizer = QuantizeConfig.quality_4bit(
        group_size=64,
        dynamic=rules,
        offload_to_disk_path=str(tmp_path / "offload"),
    )
    model_config = PretrainedConfig(quantization_config=quantizer.to_dict())
    model_config.save_pretrained(str(tmp_path))
    saved = json.loads((tmp_path / "config.json").read_text())["quantization_config"]
    assert list(saved["dynamic"]) == sorted(saved["dynamic"])
    assert saved["dynamic"] == rules  # Exact mapping, despite JSON key order.
    hydrated = QuantizeConfig.from_quant_config(saved)
    for layer in range(policy.LAYER_COUNT):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            name = module(layer, projection)
            precision = policy.classify(name, rules)
            expected = False if precision == "BF16-preserve" else int(precision.split("g")[1])
            assert policy.classify(name, saved["dynamic"]) == precision
            # Even the raw, recursively sorted config is order-independent.
            assert dynamic_get(saved["dynamic"], name, "group_size", 64) == expected
            assert dynamic_get(hydrated.dynamic, name, "group_size", 64) == expected
    assert dynamic_get(saved["dynamic"], "generic.new_projection") is False
    assert dynamic_get(hydrated.dynamic, "generic.new_projection") is False


def test_mixed_groups_and_partial_fused_bf16_are_explicitly_rejected():
    result = consumer.consumer_preflight("artifact", runtime(), "source")
    assert result["supported"] is False
    assert result["status"] == "unsupported"
    assert any("one group_size" in reason for reason in result["reasons"])
    assert any("Layer 10" in reason for reason in result["reasons"])
    assert "no exact-backend GPU reload" in result["qualification_scope"]


def test_fabricated_matching_evidence_cannot_qualify_a_backend():
    result = consumer.consumer_preflight("artifact", runtime(), "source", evidence())
    assert not result["supported"]
    assert any("Evidence claims cannot add backend support" in reason for reason in result["reasons"])


@pytest.mark.parametrize(
    "field",
    [
        "artifact_identity",
        "source_identity",
        "runtime_identity",
        "recipe_id",
        "policy_fingerprint",
        "contract_fingerprint",
    ],
)
def test_evidence_must_bind_every_identity(field):
    data = evidence()
    data[field] = "mismatch"
    result = consumer.consumer_preflight("artifact", runtime(), "source", data)
    assert not result["supported"]
    assert any(field + " does not match" in reason for reason in result["reasons"])


def test_unknown_backend_is_unqualified_even_with_pass_claims():
    data = {"backend": "native-gptq", "version": "future", "loader_contract": "unfused"}
    result = consumer.consumer_preflight("artifact", data, "source", evidence())
    assert not result["supported"]
    assert any("No reviewed consumer implementation" in reason for reason in result["reasons"])


def test_validation_wrapper_does_not_accept_a_forged_json_file(tmp_path):
    data = evidence()
    path = tmp_path / "consumer.json"
    path.write_text(json.dumps(data))
    with pytest.raises(consumer.ConsumerCompatibilityError, match="Evidence claims"):
        consumer.validate_consumer_evidence(
            path,
            data["contract_fingerprint"],
            runtime_identity=runtime(),
            source_identity="source",
            artifact_identity="artifact",
        )


def test_runtime_identity_inspects_requested_python_without_gpu_import(monkeypatch, tmp_path):
    interpreter = tmp_path / "python"
    interpreter.touch()
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(runtime()))

    monkeypatch.setattr(consumer.subprocess, "run", fake_run)
    actual = consumer.installed_vllm_identity(interpreter)
    assert observed["command"][:3] == [str(interpreter), "-I", "-c"]
    assert "importlib.metadata" in observed["command"][3]
    assert "import torch" not in observed["command"][3]
    assert "import vllm" not in observed["command"][3]
    assert actual["loader_contract"] == consumer.FUSED_SINGLE_GROUP
    assert len(actual["source_fingerprint"]) == 64


def test_cpu_projection_fixture_save_reload_preserves_exact_boundaries(tmp_path):
    # This is a tensor packing/serialization reference, NOT GPTQModel.load or a
    # vLLM fused-kernel test. It cannot generate consumer approval.
    import torch
    from safetensors.torch import load_file, save_file

    tensors = {}
    expected = {}
    for layer in (0, 10, 39):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            name = module(layer, projection)
            in_features, out_features = (64, 128) if projection == "down_proj" else (128, 64)
            precision = policy.classify(name)
            if precision == "BF16-preserve":
                weight = torch.arange(in_features * out_features).reshape(out_features, in_features)
                tensors[name + ".weight"] = (weight / 8192).to(torch.bfloat16)
                expected[name] = tensors[name + ".weight"].float()
                continue
            group_size = int(precision.split("g")[1])
            q = torch.arange(in_features * out_features, dtype=torch.int64).reshape(in_features, out_features) % 16
            scales = torch.full((in_features // group_size, out_features), 0.125, dtype=torch.bfloat16)
            packed = torch.zeros((in_features // 8, out_features), dtype=torch.int64)
            for offset in range(8):
                packed |= q[offset::8] << (4 * offset)
            tensors[name + ".qweight"] = packed.to(torch.int32)
            tensors[name + ".scales"] = scales
            tensors[name + ".g_idx"] = torch.arange(in_features, dtype=torch.int32) // group_size
            expected[name] = ((q.float() - 8) * scales.float().repeat_interleave(group_size, 0)).T
    path = tmp_path / "cpu-projection-fixture.safetensors"
    save_file(tensors, path, metadata={"scope": "CPU reference only; no backend qualification"})
    restored = load_file(path)
    assert restored.keys() == tensors.keys()
    for key in tensors:
        assert torch.equal(restored[key], tensors[key])
    assert restored[module(0, "gate_proj") + ".scales"].shape == (2, 64)
    assert restored[module(0, "down_proj") + ".scales"].shape == (2, 128)
    assert restored[module(10, "gate_proj") + ".scales"].shape == (4, 64)
    assert restored[module(10, "down_proj") + ".weight"].dtype == torch.bfloat16
    assert module(10, "down_proj") + ".qweight" not in restored
    for name, expected_weight in expected.items():
        if name + ".weight" in restored:
            actual_weight = restored[name + ".weight"].float()
        else:
            packed = restored[name + ".qweight"].to(torch.int64)
            unpacked = torch.stack([(packed >> (4 * offset)) & 15 for offset in range(8)], dim=1)
            unpacked = unpacked.reshape(-1, packed.shape[1])
            actual_weight = (
                (unpacked.float() - 8) * restored[name + ".scales"].float()[restored[name + ".g_idx"].long()]
            ).T
        x = torch.arange(expected_weight.shape[1], dtype=torch.float32) / 128
        assert torch.equal(actual_weight @ x, expected_weight @ x)
    assert not consumer.consumer_preflight("cpu-projection-fixture", runtime(), "source", evidence())["supported"]
