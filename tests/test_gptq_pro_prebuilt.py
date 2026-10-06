"""Reject wrong binaries before native initialization; CPU tests never load fake code."""
import hashlib
import sys
from types import SimpleNamespace

import pytest

from gptqmodel.utils import gptq_pro_prebuilt as loader
from gptqmodel.utils import _extension_loader


@pytest.fixture
def binary(tmp_path, monkeypatch):
    path = tmp_path / (loader.MODULE_NAME + ".so")
    path.write_bytes(b"not native code; the initializer is mocked in tests")
    monkeypatch.delitem(sys.modules, loader.MODULE_NAME, raising=False)
    def forbidden(*args, **kwargs):
        pytest.fail("A rejected binary reached native initialization")
    monkeypatch.setattr(loader.importlib.util, "module_from_spec", forbidden)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("digest", [None, "", "0"*63, "0"*65, "g"*64, 0])
def test_invalid_digest_is_rejected(binary, digest):
    with pytest.raises(ValueError, match="hexadecimal"):
        loader.load_pinned_v3(binary[0], digest)


def test_relative_path_is_rejected(binary):
    with pytest.raises(ValueError, match="absolute"):
        loader.load_pinned_v3(binary[0].name, binary[1])


def test_missing_path_is_rejected(binary):
    with pytest.raises(FileNotFoundError):
        loader.load_pinned_v3(binary[0].parent / "missing.so", binary[1])


def test_directory_is_rejected(binary):
    with pytest.raises(ValueError, match="regular file"):
        loader.load_pinned_v3(binary[0].parent, binary[1])


def test_wrong_name_is_rejected(binary):
    path = binary[0].with_name("another.so")
    path.write_bytes(binary[0].read_bytes())
    with pytest.raises(ValueError, match="module name"):
        loader.load_pinned_v3(path, binary[1])


def test_hash_mismatch_is_rejected(binary):
    with pytest.raises(ValueError, match="mismatch before import"):
        loader.load_pinned_v3(binary[0], "0"*64)


def test_cached_module_is_not_silently_reused(binary, monkeypatch):
    monkeypatch.setitem(sys.modules, loader.MODULE_NAME, SimpleNamespace())
    with pytest.raises(RuntimeError, match="already imported"):
        loader.load_pinned_v3(*binary)


def mock_native_import(monkeypatch, path, callback=lambda: None, has_gemm=True):
    monkeypatch.setattr(_extension_loader, "_ensure_torch_shared_libraries_loaded", lambda: None)
    spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: callback()))
    monkeypatch.setattr(loader.importlib.util, "spec_from_file_location", lambda name,p: spec)
    module = SimpleNamespace(__file__=str(path))
    if has_gemm:
        module.gptq_pro_gemm = lambda *args: None
    monkeypatch.setattr(loader.importlib.util, "module_from_spec", lambda found: module)
    return module


def test_explicit_file_and_uppercase_digest_are_verified(binary, monkeypatch):
    module = mock_native_import(monkeypatch, binary[0])
    old_path = list(sys.path)
    loaded, identity = loader.load_pinned_v3(binary[0], binary[1].upper())
    assert loaded is module
    assert identity["sha256_pin_verified"] is True
    assert identity["source_build_identity_proven"] is False
    assert identity["sha256"] == binary[1]
    assert sys.path == old_path


def test_changed_binary_after_import_fails(binary, monkeypatch):
    mock_native_import(monkeypatch, binary[0], lambda: binary[0].write_bytes(b"changed"))
    with pytest.raises(RuntimeError, match="bytes changed"):
        loader.load_pinned_v3(*binary)


def test_missing_gemm_fails(binary, monkeypatch):
    mock_native_import(monkeypatch, binary[0], has_gemm=False)
    with pytest.raises(ImportError, match="callable"):
        loader.load_pinned_v3(*binary)


def test_wrong_loaded_path_fails(binary, monkeypatch, tmp_path):
    another = tmp_path / "wrong.so"
    another.write_bytes(b"placeholder")
    mock_native_import(monkeypatch, another)
    with pytest.raises(ImportError, match="different binary path"):
        loader.load_pinned_v3(*binary)
