"""Compatibility imports for the standalone Inkling Mini recipe scripts."""

from gptqmodel.models.inkling2_mini import (  # noqa: F401
    CpuStagedCudaTransfers,
    LAYER_PREFIX,
    MODEL_TYPE,
    REVISION,
    SOURCE,
    apply,
    load,
    native_linear,
    replace_linears,
    sha256,
    target_modules,
)
