__version__ = "1.0.0"

import torch
from pathlib import Path

# Load the compiled extension so its static TORCH_LIBRARY registrations run.
# The operators are then available as torch.ops.flash_mla.<op>. The glob matches
# both the Python-version-specific name (cuda.cpython-*.so) and the abi3 name
# (cuda.abi3.so) produced when Py_LIMITED_API is enabled.
_so_files = list(Path(__file__).parent.glob("cuda*.so"))
assert len(_so_files) == 1, f"Expected one cuda*.so file, found {_so_files}"
torch.ops.load_library(_so_files[0])

from flash_mla.flash_mla_interface import (
    get_mla_metadata,
    flash_mla_with_kvcache,
    flash_attn_varlen_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_varlen_kvpacked_func,
    flash_mla_sparse_fwd
)

__all__ = [
    "get_mla_metadata",
    "flash_mla_with_kvcache",
    "flash_attn_varlen_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_varlen_kvpacked_func",
    "flash_mla_sparse_fwd"
]
