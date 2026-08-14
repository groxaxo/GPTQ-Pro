from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "plan_qwen38_gptqpro.py"
SPEC = importlib.util.spec_from_file_location("plan_qwen38_gptqpro", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
planner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


def test_parameter_parser() -> None:
    assert planner.parse_param_count("27B") == 27_000_000_000
    assert planner.parse_param_count("2.4T") == 2_400_000_000_000


def test_qwen38_is_not_qwen3_8b() -> None:
    assert planner.detect_qwen_name("Qwen/Qwen3-8B") == (False, True)
    assert planner.detect_qwen_name("Qwen/Qwen3.8-27B") == (True, False)


def test_group_64_costs_more_space_than_group_128() -> None:
    params = 27_000_000_000
    assert planner.projected_int4_bytes(
        params, 64
    ) > planner.projected_int4_bytes(params, 128)


def test_27b_uses_quality_demon_profile_and_fits_lab() -> None:
    hardware = planner.Hardware(3, 24, 128, 2_000)
    plan = planner.build_plan(
        model="Qwen/Qwen3.8-27B",
        total_params=27_000_000_000,
        source_dtype="bf16",
        hardware=hardware,
        trust_remote_code=False,
        assume_qwen38_max=False,
    )
    assert plan.recipe.preset == "max_quality"
    assert plan.recipe.group_size == 64
    assert plan.capacity.quantization_feasible is True
    assert plan.status == "unverified"


def test_announced_2_4t_model_is_blocked_on_lab() -> None:
    hardware = planner.Hardware(3, 24, 128, 2_000)
    plan = planner.build_plan(
        model="",
        total_params=None,
        source_dtype="bf16",
        hardware=hardware,
        trust_remote_code=False,
        assume_qwen38_max=True,
    )
    assert plan.model.total_params == 2_400_000_000_000
    assert plan.capacity.quantization_feasible is False
    assert plan.capacity.storage_feasible is False
    assert plan.status == "blocked"
