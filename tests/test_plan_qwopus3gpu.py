"""Dependency-free regression tests for the three-GPU read-only admission gate."""
import importlib.util
import sys
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "scripts" / "plan_qwopus3gpu.py"
spec = importlib.util.spec_from_file_location("qwopus_plan", path)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

def sample_gpus(free_mib=23000):
    text = "\n".join(
        f"{i}, NVIDIA GeForce RTX 3090, 24576, {free_mib}, 0" for i in range(3))
    return mod.read_gpu_csv(text)

class CapacityTests(unittest.TestCase):
    def test_three_cards_csv(self):
        self.assertEqual([x["index"] for x in sample_gpus()], [0, 1, 2])
        self.assertEqual(sample_gpus(512)[0]["free_gib"], 0.5)

    def test_reject_bad_csv(self):
        for text in ("0, 3090, 24576, 500",
                     "0, 3090, 24576, 25000, 0",
                     "0, 3090, 24576, 500, 0\n0, 3090, 24576, 500, 0"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                mod.read_gpu_csv(text)

    def test_busy_host_blocks(self):
        result = mod.assess(sample_gpus(510), [0, 1, 2], 60, 13)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(len(result["blockers"]), 5)

    def test_capacity_never_claims_qualification(self):
        result = mod.assess(sample_gpus(), [0, 1, 2], 180, 95)
        self.assertEqual(result["status"], "RESOURCE_CANDIDATE_NOT_QUALIFIED")
        self.assertFalse(result["blockers"])
        self.assertIn("concurrent three-GPU sequential GPTQ scheduling", result["unqualified"])

    def test_missing_gpu_blocks(self):
        result = mod.assess(sample_gpus()[:2], [0, 1, 2], 180, 95)
        self.assertIn("GPU 2 not found", result["blockers"])

    def test_mem_available(self):
        self.assertEqual(mod.read_mem_available("MemAvailable: 1048576 kB\n"), 1.0)
        with self.assertRaises(ValueError):
            mod.read_mem_available("MemFree: 1048576 kB\n")

    def test_source_architecture(self):
        d = {"model_type": "qwen3_5",
             "architectures": ["Qwen3_5ForConditionalGeneration"],
             "quantization_config": None,
             "text_config": {"model_type": "qwen3_5_text", "num_hidden_layers": 64,
                             "hidden_size": 5120, "intermediate_size": 17408,
                             "layer_types": ["linear_attention", "linear_attention",
                                             "linear_attention", "full_attention"] * 16}}
        self.assertEqual(mod.validate_source_config(d), [])
        d["quantization_config"] = {"method": "gptq"}
        self.assertIn("source is prequantized", mod.validate_source_config(d))

if __name__ == "__main__":
    unittest.main()
