from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_package_metadata_does_not_advertise_removed_backend_extras():
    pyproject = _read("pyproject.toml")

    for optional_group in (
        "vllm",
        "sglang",
        "bitblas",
        "bitsandbytes",
        "marlin-cuda12",
        "marlin-cuda",
        "mlx",
    ):
        assert f"{optional_group} = [" not in pyproject

    assert 'Homepage = "https://github.com/groxaxo/GPTQ-Pro"' in pyproject
    assert 'Upstream = "https://github.com/ModelCloud/GPTQModel"' in pyproject


def test_source_manifest_only_packages_live_extension_sources():
    manifest = _read("MANIFEST.in")

    assert "gptqmodel_ext/gptq_pro" in manifest
    assert "gptqmodel_ext/pack_block_cpu.cpp" in manifest
    assert "gptqmodel_ext/floatx_cpu.cpp" in manifest

    for removed_tree in (
        "gptqmodel_ext/awq",
        "gptqmodel_ext/exllama",
        "gptqmodel_ext/marlin",
        "gptqmodel_ext/machete",
        "gptqmodel_ext/qqq",
    ):
        assert removed_tree not in manifest


def test_container_configuration_uses_single_backend_environment():
    environment = _read("environment.yml")
    dockerfile = _read("Dockerfile")

    assert environment.startswith("name: gptq-pro\n")
    assert "ARG ENV_NAME=gptq-pro" in dockerfile
    assert ' -e ".[vllm' not in dockerfile
    assert ' -e ".[sglang' not in dockerfile
    assert "gptq-pro-vllm" not in dockerfile


def test_readme_does_not_offer_removed_runtime_commands():
    readme = _read("README.md")

    assert "BACKEND.MARLIN" not in readme
    assert "BACKEND.MACHETE" not in readme
    assert "pip install -e .[vllm" not in readme
    assert "pip install -e .[sglang" not in readme
    assert "max_quality_4bit()` does **not** enable FOEM or rotation" in readme
