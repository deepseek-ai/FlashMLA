__version__ = "1.0.0"

from pathlib import Path

import torch

try:
    from flash_mla._build_info import BACKEND, EXTENSION
except ImportError as e:
    raise ImportError(
        "flash_mla has not been built yet: flash_mla/_build_info.py is missing. "
        "Build it in place with `python setup.py build_ext --inplace`, or reinstall flash_mla."
    ) from e

# Loading the extension runs its static STABLE_TORCH_LIBRARY registrations.
if BACKEND == "npu":
    # Initialize PrivateUse1 before loading an Ascend extension.
    try:
        import torch_npu  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "flash_mla was built for Ascend and requires torch_npu. Please install torch_npu"
        ) from e

_extension_path = Path(__file__).parent / EXTENSION
if not _extension_path.is_file():
    raise ImportError(
        f"FlashMLA extension not found: {_extension_path}. "
        "Reinstall flash_mla, or build it in place with "
        "`python setup.py build_ext --inplace`."
    )
torch.ops.load_library(_extension_path)

from flash_mla.flash_mla_interface import (
    get_mla_metadata,
    flash_mla_with_kvcache,
    flash_attn_varlen_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_varlen_kvpacked_func,
    flash_mla_sparse_fwd
)

from . import fused_norm_rope_attn_rope_cast

__all__ = [
    "get_mla_metadata",
    "flash_mla_with_kvcache",
    "flash_attn_varlen_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_varlen_kvpacked_func",
    "flash_mla_sparse_fwd",
    "fused_norm_rope_attn_rope_cast"
]
