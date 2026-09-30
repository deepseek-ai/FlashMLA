__version__ = "1.0.0"

from pathlib import Path

import torch

# Loading the extension runs its static TORCH_LIBRARY registrations. The glob
# accepts either platform build and both CPython-specific and abi3 suffixes.
_so_files = [
    *Path(__file__).parent.glob("cuda*.so"),
    *Path(__file__).parent.glob("npu*.so"),
]
assert len(_so_files) == 1, f"Expected one FlashMLA extension, found {_so_files}"
# Initialize PrivateUse1 before loading an Ascend extension. Keying off the
# artifact is robust in containers where /dev/davinci_manager is hidden.
if _so_files[0].name.startswith("npu"):
    import torch_npu  # noqa: F401
torch.ops.load_library(_so_files[0])

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
