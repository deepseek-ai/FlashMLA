"""FlashMLA: An efficient MLA decoding kernel for Hopper GPUs."""

from flash_mla.flash_mla_interface import (
    get_mla_metadata,
    flash_mla_with_kvcache,
)


__all__ = [
    "get_mla_metadata",
    "flash_mla_with_kvcache",
]


__version__ = "1.0.0"
