__version__ = "1.0.0"

from .flash_mla_interface import (
    get_mla_metadata,
    flash_mla_with_kvcache,
    get_mla_workspace_bytes,
    alloc_mla_workspace,
    sm86_config,
)
