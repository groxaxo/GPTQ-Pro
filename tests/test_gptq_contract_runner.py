"""The live-kernel entrypoint must never turn missing hardware/builds into a pass."""
import importlib.util
from pathlib import Path

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('_contract_runner',ROOT/'scripts/quality_gptq_contract.py')
RUNNER=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
CASE={'K':32,'N':32,'group_size':16,'M':1}


def test_no_cases_is_not_a_kernel_pass():
    report={}
    with pytest.raises(ValueError,match='empty CUDA'):
        RUNNER.cuda_audit([],report,lambda:None)
    assert report.get('live_kernel_shape_parity') is not True


def test_no_gpu_admission_is_not_a_kernel_pass(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    report={}
    with pytest.raises(RuntimeError,match='gpuq-admitted'):
        RUNNER.cuda_audit([CASE],report,lambda:None)
    assert report.get('live_kernel_shape_parity') is not True


def test_missing_prebuilt_extension_fails_without_jit_or_gpu_work(monkeypatch):
    from gptqmodel.utils import _extension_loader
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','test-admitted-placeholder')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda index:(8,6))
    calls=[]
    def absent(name):
        calls.append(name)
        raise ImportError('prebuilt deliberately absent')
    monkeypatch.setattr(_extension_loader,'load_extension_module',absent)
    report={}
    with pytest.raises(ImportError,match='prebuilt deliberately absent'):
        RUNNER.cuda_audit([CASE],report,lambda:None)
    assert calls==['gptqmodel_gptq_pro_kernels_v3']
    assert report.get('live_kernel_shape_parity') is not True
