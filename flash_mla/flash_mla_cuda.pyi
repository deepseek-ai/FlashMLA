import torch

def get_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def fwd_kvcache_mla(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor | None,
    head_dim_v: int,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]: ...
