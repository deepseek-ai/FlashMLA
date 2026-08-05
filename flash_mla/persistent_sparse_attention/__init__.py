"""Persistent sparse-attention and fused pipeline kernels for SM100."""

from .kernels import index_query_prepare, kv_cache_gather
from .aot_adapters import (
    index_logits_out,
    persistent_attention_frontend,
    wqb_ready_out,
    wqb_ready_packed_out,
)
from .ops import (
    attention_output_quant,
    csa_root_fusion,
    combine_indices,
    index_cache_gather,
    index_cache_pack,
    index_query_quant,
    index_weight_scale,
    kv_cache_gather_persistent,
    learned_topk,
    main_compress,
    output_quant,
    reset_ready,
    sparse_attention,
    sparse_attention_prefix,
    sparse_attention_tail,
    state_save,
    swa_cache_insert,
)
from .sm_config import ProductionSMConfig, resolve_sm_config

__all__ = [
    "ProductionSMConfig",
    "attention_output_quant",
    "combine_indices",
    "csa_root_fusion",
    "index_cache_gather",
    "index_cache_pack",
    "index_logits_out",
    "index_query_prepare",
    "index_query_quant",
    "index_weight_scale",
    "kv_cache_gather",
    "kv_cache_gather_persistent",
    "learned_topk",
    "main_compress",
    "output_quant",
    "persistent_attention_frontend",
    "reset_ready",
    "resolve_sm_config",
    "sparse_attention",
    "sparse_attention_prefix",
    "sparse_attention_tail",
    "state_save",
    "swa_cache_insert",
    "wqb_ready_out",
    "wqb_ready_packed_out",
]
