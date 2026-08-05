"""Lazy wrappers for the unified persistent sparse-attention extension."""

from __future__ import annotations

from types import ModuleType
from typing import Any


def _extension() -> ModuleType:
    try:
        from flash_mla import persistent_sparse_attention_cuda
    except ImportError as error:
        raise RuntimeError(
            "FlashMLA persistent pipeline kernels are unavailable; rebuild "
            "with FLASH_MLA_BUILD_PERSISTENT_SPARSE_ATTENTION_SUITE=1"
        ) from error
    return persistent_sparse_attention_cuda


def _call(name: str, *args: Any, **kwargs: Any) -> Any:
    return getattr(_extension(), name)(*args, **kwargs)


def csa_root_fusion(*args: Any, **kwargs: Any) -> Any:
    return _call("csa_root_fusion", *args, **kwargs)


def state_save(*args: Any, **kwargs: Any) -> Any:
    return _call("state_save", *args, **kwargs)


def main_compress(*args: Any, **kwargs: Any) -> Any:
    return _call("main_compress", *args, **kwargs)


def swa_cache_insert(*args: Any, **kwargs: Any) -> Any:
    return _call("swa_cache_insert", *args, **kwargs)


def index_cache_gather(*args: Any, **kwargs: Any) -> Any:
    return _call("index_cache_gather", *args, **kwargs)


def learned_topk(*args: Any, **kwargs: Any) -> Any:
    return _call("learned_topk", *args, **kwargs)


def combine_indices(*args: Any, **kwargs: Any) -> Any:
    return _call("combine_indices", *args, **kwargs)


def attention_output_quant(*args: Any, **kwargs: Any) -> Any:
    return _call("attention_output_quant", *args, **kwargs)


def output_quant(*args: Any, **kwargs: Any) -> Any:
    return _call("output_quant", *args, **kwargs)


def reset_ready(*args: Any, **kwargs: Any) -> Any:
    return _call("reset_ready", *args, **kwargs)


def index_query_quant(*args: Any, **kwargs: Any) -> Any:
    return _call("index_query_quant", *args, **kwargs)


def index_weight_scale(*args: Any, **kwargs: Any) -> Any:
    return _call("index_weight_scale", *args, **kwargs)


def index_cache_pack(*args: Any, **kwargs: Any) -> Any:
    return _call("index_cache_pack", *args, **kwargs)


def kv_cache_gather_persistent(*args: Any, **kwargs: Any) -> Any:
    return _call("kv_cache_gather_persistent", *args, **kwargs)


def _sparse_attention_lane(
    lane: str, *args: Any, **kwargs: Any
) -> Any:
    from flash_mla.persistent_sparse_attention_interface import (
        flash_mla_persistent_sparse_attention_fwd,
    )

    if "psa" in kwargs or "csa_lane" in kwargs:
        raise TypeError("CSA sparse-attention wrappers select their own lane")
    return flash_mla_persistent_sparse_attention_fwd(
        *args, psa=True, csa_lane=lane, **kwargs
    )


def sparse_attention(*args: Any, **kwargs: Any) -> Any:
    return _sparse_attention_lane("generic", *args, **kwargs)


def sparse_attention_prefix(*args: Any, **kwargs: Any) -> Any:
    return _sparse_attention_lane("prefix", *args, **kwargs)


def sparse_attention_tail(*args: Any, **kwargs: Any) -> Any:
    return _sparse_attention_lane("tail", *args, **kwargs)


__all__ = [
    "attention_output_quant",
    "csa_root_fusion",
    "combine_indices",
    "index_cache_gather",
    "index_cache_pack",
    "index_query_quant",
    "index_weight_scale",
    "kv_cache_gather_persistent",
    "learned_topk",
    "main_compress",
    "output_quant",
    "reset_ready",
    "sparse_attention",
    "sparse_attention_prefix",
    "sparse_attention_tail",
    "state_save",
    "swa_cache_insert",
]
