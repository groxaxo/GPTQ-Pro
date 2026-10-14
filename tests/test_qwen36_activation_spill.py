"""CPU-only regressions for qualified layer spill and rejected native FIFO spill."""

import gc
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file

from gptqmodel.looper.forward_executor import ForwardExecutor
from gptqmodel.looper.loop_processor import LoopProcessor
from gptqmodel.looper.native_processor import NativeProcessor
from gptqmodel.looper.stage_inputs_capture import StageInputsCapture
from gptqmodel.nn_modules.hooked_linear import STOP_FORWARD_EXCEPTION
from gptqmodel.utils import activation_spill as spill


@pytest.fixture(autouse=True)
def clean_spill_environment(monkeypatch):
    monkeypatch.delenv("GPTQ_HQ_LAYER_SPILL", raising=False)
    monkeypatch.delenv("GPTQ_HQ_NATIVE_SPILL", raising=False)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16, torch.int64, torch.bool])
def test_spill_roundtrip_snapshot_and_repeatability(tmp_path, dtype):
    cache = spill.DiskLayerList(tmp_path)
    source = torch.arange(12).reshape(3, 4).to(dtype).t()
    expected = source.clone()
    cache.append([source, torch.tensor(3).to(dtype)])
    source.fill_(0)
    first = cache[0]
    assert first[0].device.type == "cpu"
    assert first[0].dtype == dtype and tuple(first[0].shape) == (4, 3)
    assert torch.equal(first[0], expected)
    first[0].fill_(0)
    assert torch.equal(cache[0][0], expected)
    assert torch.equal(cache[0][1], torch.tensor(3).to(dtype))
    cache.close()


def test_spill_preserves_empty_batches_order_slices_and_replays(tmp_path):
    cache = spill.DiskLayerList(tmp_path)
    cache.append([])
    cache.append([torch.tensor([8.0])])
    cache.append([])
    cache.append([torch.tensor([2.0]), torch.empty(0, 3)])
    for _ in range(3):
        batches = list(cache)
        assert len(batches) == 4 and batches[0] == [] and batches[2] == []
        assert batches[1][0].item() == 8 and batches[3][0].item() == 2
        assert batches[3][1].shape == (0, 3)
    assert cache[-1][0].item() == 2
    assert cache[::-2][0][0].item() == 2
    with pytest.raises(IndexError):
        cache[4]
    cache.close()


@pytest.mark.parametrize(
    "replacement",
    [
        {"0": torch.tensor([9.0, 2.0])},
        {"0": torch.tensor([[1.0, 2.0]])},
        {"0": torch.tensor([1.0, 2.0], dtype=torch.float64)},
        {"0": torch.tensor([1.0, 2.0]), "extra": torch.ones(1)},
        {},
    ],
)
def test_spill_rejects_content_shape_dtype_inventory_corruption(tmp_path, replacement):
    cache = spill.DiskLayerList(tmp_path)
    cache.append([torch.tensor([1.0, 2.0])])
    save_file(replacement, str(cache.entries[0][0]))
    with pytest.raises(ValueError, match="Layer spill integrity failure"):
        cache[0]
    cache.close()


def test_spill_rejects_truncated_and_missing_file(tmp_path):
    cache = spill.DiskLayerList(tmp_path)
    cache.append([torch.ones(3)])
    path = cache.entries[0][0]
    path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="Layer spill integrity failure"):
        cache[0]
    path.unlink()
    with pytest.raises(ValueError, match="Layer spill integrity failure"):
        cache[0]
    cache.close()


def test_spill_rejects_nonfinite_and_non_tensor_batches(tmp_path):
    cache = spill.DiskLayerList(tmp_path)
    for batch in ([torch.tensor([float("nan")])], [torch.tensor([float("inf")])]):
        with pytest.raises(ValueError, match="Nonfinite"):
            cache.append(batch)
    for batch in (torch.ones(2), [None], {"0": torch.ones(1)}):
        with pytest.raises(TypeError):
            cache.append(batch)
    assert len(cache) == 0 and list(cache.path.iterdir()) == []
    cache.close()


def test_spill_failed_serialization_leaves_no_partial_entry(tmp_path, monkeypatch):
    cache = spill.DiskLayerList(tmp_path)

    def fail_save(_values, path):
        from pathlib import Path

        Path(path).write_bytes(b"partial")
        raise OSError("injected disk full")

    monkeypatch.setattr(spill, "save_file", fail_save)
    with pytest.raises(OSError, match="disk full"):
        cache.append([torch.ones(2)])
    assert len(cache) == 0 and list(cache.path.iterdir()) == []
    cache.close()


def test_clear_reuse_close_and_finalizer_release_all_files(tmp_path):
    before = spill.DiskLayerList.snapshot()["live_bytes"]
    cache = spill.DiskLayerList(tmp_path)
    cache.append([torch.ones(3)])
    assert spill.DiskLayerList.snapshot()["live_bytes"] > before
    cache.clear()
    cache.clear()
    assert spill.DiskLayerList.snapshot()["live_bytes"] == before
    cache.append([])
    cache.close()
    cache.close()
    assert not cache.path.exists()
    with pytest.raises(ValueError, match="closed"):
        cache.append([])
    for populated in (False, True):
        other = spill.DiskLayerList(tmp_path)
        path = other.path
        if populated:
            other.append([torch.ones(3)])
        del other
        gc.collect()
        assert not path.exists()
    assert spill.DiskLayerList.snapshot()["live_bytes"] == before


def test_factory_default_and_enabled(tmp_path, monkeypatch):
    assert spill.layer_cache_list() == []
    monkeypatch.setenv("GPTQ_HQ_LAYER_SPILL", str(tmp_path))
    cache = spill.layer_cache_list()
    assert isinstance(cache, spill.DiskLayerList)
    cache.close()


@pytest.mark.parametrize("placement", ["balanced", "cuda", torch.device("cuda:1")])
def test_layer_spill_rejects_non_cpu_cache_placement(tmp_path, monkeypatch, placement):
    monkeypatch.setenv("GPTQ_HQ_LAYER_SPILL", str(tmp_path))
    with pytest.raises(ValueError, match="CPU calibration_data_device"):
        spill.validate_layer_spill_placement(placement)


@pytest.mark.parametrize("placement", [None, "cpu", torch.device("cpu")])
def test_layer_spill_supports_cpu_cache_placement(tmp_path, monkeypatch, placement):
    monkeypatch.setenv("GPTQ_HQ_LAYER_SPILL", str(tmp_path))
    spill.validate_layer_spill_placement(placement)


def test_native_spill_rejected_before_any_capture_or_initialization(tmp_path, monkeypatch):
    monkeypatch.setenv("GPTQ_HQ_NATIVE_SPILL", str(tmp_path))

    def unexpected_initialization(*_args, **_kwargs):
        pytest.fail("Unsupported native spill reached capture initialization")

    monkeypatch.setattr(LoopProcessor, "__init__", unexpected_initialization)
    # Reject before observing any routing, including skipped experts, no captures,
    # and repeated captures in one global batch. No FIFO ordering claim is made.
    with pytest.raises(ValueError, match="per-expert capture/replay alignment"):
        NativeProcessor(None, None, None, None, None, None, 1)
    with pytest.raises(ValueError, match="per-expert capture/replay alignment"):
        spill.layer_cache_list()
    assert list(tmp_path.iterdir()) == []


class Layer(torch.nn.Module):
    def forward(self, hidden_states, **_kwargs):
        return hidden_states + 1


class CaptureModel:
    ATTENTION_MASKS_REQUIRED_FOR_INPUT = False

    def __init__(self):
        self.layer = Layer()
        self.model = self.layer
        self.quantize_config = SimpleNamespace(device=torch.device("cpu"), calibration_data_device=None)
        self.finished = False
        self.cache_devices = []

    def shell_module_materialize(self, target_submodule, device):
        return target_submodule.to(device)

    def get_base_modules(self, _model):
        return []

    def pre_quantize_generate_hook_start(self):
        pass

    def pre_quantize_generate_hook_end(self):
        self.finished = True

    def move_input_capture_example(self, example, _device):
        return example

    def run_input_capture(self, example, **_kwargs):
        self.layer(example["hidden_states"])

    def capture_first_layer_positional_inputs(self, args, kwargs, batch_device):
        self.cache_devices.append(batch_device)
        return [value.to(batch_device) for value in args]

    def capture_first_layer_input_kwargs(self, **kwargs):
        return kwargs["layer_input_kwargs"]


def test_capture_stage_spills_real_inputs_and_clears_sentinel_traceback(tmp_path, monkeypatch):
    monkeypatch.setenv("GPTQ_HQ_LAYER_SPILL", str(tmp_path))
    model = CaptureModel()
    looper = SimpleNamespace(gptq_model=model, _batch_row_count=lambda batch: len(batch[0]))
    stage = StageInputsCapture(looper, logger=MagicMock())
    cache = stage.cache_inputs(
        [model.layer],
        [{"hidden_states": torch.full((1, 2, 3), float(i))} for i in range(3)],
        False,
    )
    assert isinstance(cache.layer_inputs, spill.DiskLayerList)
    assert [batch[0][0, 0, 0].item() for batch in cache.layer_inputs] == [0, 1, 2]
    assert all(device == torch.device("cpu") for device in model.cache_devices)
    assert STOP_FORWARD_EXCEPTION.__traceback__ is None
    assert model.finished and len(model.layer._forward_pre_hooks) == 0
    cache.layer_inputs.close()


def make_executor():
    model = SimpleNamespace(
        quantize_config=SimpleNamespace(calibration_data_device=None),
        prepare_layer_replay_kwargs=lambda **kwargs: kwargs["additional_inputs"],
    )
    looper = SimpleNamespace(
        gptq_model=model,
        support_batch_quantize=False,
        moe_routing_override=None,
        _resolve_batch_total=lambda _num_batches, inputs: len(inputs),
        _collect_row_counts=lambda inputs: [len(batch[0]) for batch in inputs],
        _batch_row_count=lambda batch: len(batch[0]),
        _set_processor_mask=lambda *_args: None,
    )
    processor = SimpleNamespace(num_batches=None, _set_current_batch_index=lambda _index: None)
    return ForwardExecutor(looper), processor


def executor_kwargs(processor, inputs):
    return {
        "module": Layer(),
        "processor": processor,
        "layer_inputs": inputs,
        "layer_input_kwargs": [{} for _ in inputs],
        "position_ids": [],
        "attention_masks": [None for _ in inputs],
        "cur_layer_device": torch.device("cpu"),
        "is_lm_head_module": False,
        "shared_kv_cache_dict": {},
        "layer_index": 0,
        "need_outputs": True,
        "reuse_kv": False,
    }


def test_serial_forward_spill_matches_in_memory_across_replays(tmp_path, monkeypatch):
    executor, processor = make_executor()
    inputs = [[torch.full((1, 2, 3), float(i))] for i in range(3)]
    expected = executor.run_single(**executor_kwargs(processor, inputs))
    monkeypatch.setenv("GPTQ_HQ_LAYER_SPILL", str(tmp_path))
    disk_inputs = spill.layer_cache_list()
    for batch in inputs:
        disk_inputs.append(batch)
    for _ in range(2):
        result = executor.run_single(**executor_kwargs(processor, disk_inputs))
        assert isinstance(result, spill.DiskLayerList)
        assert len(result) == len(expected)
        for actual, reference in zip(result, expected):
            assert torch.equal(actual[0], reference[0])
        result.close()
    disk_inputs.close()


def test_parallel_forward_rejected_before_replicas_or_reads(tmp_path, monkeypatch):
    executor, processor = make_executor()
    monkeypatch.setenv("GPTQ_HQ_LAYER_SPILL", str(tmp_path))
    with pytest.raises(ValueError, match="auto_forward_data_parallel=False"):
        executor.run_parallel(
            **executor_kwargs(processor, [[torch.ones(1, 2, 3)]]),
            devices=[torch.device("cuda:0"), torch.device("cuda:1")],
        )
    assert list(tmp_path.iterdir()) == []


def test_capture_cpu_cache_placement_survives_forward_device_change(tmp_path, monkeypatch):
    from contextlib import nullcontext

    import gptqmodel.looper.stage_inputs_capture as capture_impl

    monkeypatch.setenv("GPTQ_HQ_LAYER_SPILL", str(tmp_path))
    model = CaptureModel()
    model.ATTENTION_MASKS_REQUIRED_FOR_INPUT = True
    model.quantize_config.device = torch.device("meta")
    model.quantize_config.calibration_data_device = torch.device("cpu")
    monkeypatch.setattr(capture_impl, "DEVICE_THREAD_POOL", SimpleNamespace(read_lock=lambda _device: nullcontext()))
    monkeypatch.setattr(capture_impl, "device_ctx", lambda _device: nullcontext())
    looper = SimpleNamespace(gptq_model=model, _batch_row_count=lambda batch: len(batch[0]))
    cache = StageInputsCapture(looper, logger=MagicMock()).cache_inputs(
        [model.layer],
        [{"hidden_states": torch.ones(1, 2, 3)}],
        False,
    )
    assert model.cache_devices == [torch.device("cpu")]
    assert cache.layer_inputs[0][0].device.type == "cpu"
    cache.layer_inputs.close()


def test_capture_failure_removes_hook_and_finishes_capture(tmp_path, monkeypatch):
    monkeypatch.setenv("GPTQ_HQ_LAYER_SPILL", str(tmp_path))
    model = CaptureModel()

    def fail_capture(*_args, **_kwargs):
        raise RuntimeError("injected capture failure")

    model.run_input_capture = fail_capture
    looper = SimpleNamespace(gptq_model=model, _batch_row_count=lambda batch: len(batch[0]))
    with pytest.raises(RuntimeError, match="injected capture failure"):
        StageInputsCapture(looper, logger=MagicMock()).cache_inputs(
            [model.layer],
            [{"hidden_states": torch.ones(1, 2, 3)}],
            False,
        )
    assert model.finished and len(model.layer._forward_pre_hooks) == 0
