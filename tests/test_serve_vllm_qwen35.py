import importlib.util
import sys
import types
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "serve_vllm_qwen35.py"
MODULE_SPEC = importlib.util.spec_from_file_location("serve_vllm_qwen35", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
serve_vllm_qwen35 = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(serve_vllm_qwen35)


def test_qwen35_detection_reads_local_config(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "qwen3_5_text"}', encoding="utf-8")

    assert serve_vllm_qwen35._is_qwen35_text_checkpoint(str(tmp_path))


def test_qwen35_detection_uses_autoconfig_for_repo_ids(monkeypatch):
    calls = []

    class DummyConfig:
        def to_dict(self):
            return {"model_type": "qwen3_5_text"}

    class DummyAutoConfig:
        @staticmethod
        def from_pretrained(model_id, trust_remote_code):
            calls.append((model_id, trust_remote_code))
            return DummyConfig()

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoConfig=DummyAutoConfig))

    assert serve_vllm_qwen35._is_qwen35_text_checkpoint("groxaxo/qwen35-gptq-pro")
    assert calls == [("groxaxo/qwen35-gptq-pro", True)]
