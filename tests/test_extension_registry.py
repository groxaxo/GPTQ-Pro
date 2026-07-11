import pytest

from gptqmodel import extension


def test_extension_registry_only_lists_live_cpu_helpers():
    assert set(extension.available_extensions()) == {"pack_block_cpu", "floatx_cpu"}


def test_removed_marlin_alias_fails_cleanly():
    with pytest.raises(ValueError, match="Unknown extension `marlin`"):
        extension.load("marlin")
