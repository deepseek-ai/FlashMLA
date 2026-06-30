#include <torch/library.h>

TORCH_LIBRARY(flash_mla, m) {
    m.def("sparse_decode_fwd(Tensor q, Tensor kv, Tensor indices, Tensor? topk_length, Tensor? attn_sink, Tensor(a)? tile_scheduler_metadata, Tensor(b)? num_splits, Tensor? extra_kv, Tensor? extra_indices, Tensor? extra_topk_length, int d_v, float sm_scale) -> (Tensor, Tensor, Tensor(a)?, Tensor(b)?)");
    m.def("dense_decode_fwd(Tensor q, Tensor kcache, int head_size_v, Tensor seqlens_k, Tensor block_table, float softmax_scale, bool is_causal, Tensor(a)? tile_scheduler_metadata, Tensor(b)? num_splits) -> (Tensor, Tensor, Tensor(a)?, Tensor(b)?)");
    m.def("sparse_prefill_fwd(Tensor q, Tensor kv, Tensor indices, float sm_scale, int d_v, Tensor? attn_sink, Tensor? topk_length) -> Tensor[]");
    m.def("dense_prefill_fwd(Tensor workspace_buffer, Tensor q, Tensor k, Tensor v, Tensor cumulative_seqlen_q, Tensor cumulative_seqlen_kv, Tensor(a!) o, Tensor(b!) lse, int mask_mode_code, float softmax_scale, int max_seqlen_q, int max_seqlen_kv, bool is_varlen) -> ()");
    m.def("dense_prefill_bwd(Tensor(a!) workspace_buffer, Tensor d_o, Tensor q, Tensor k, Tensor v, Tensor o, Tensor lse, Tensor cumulative_seqlen_q, Tensor cumulative_seqlen_kv, Tensor(b!) dq, Tensor(c!) dk, Tensor(d!) dv, int mask_mode_code, float softmax_scale, int max_seqlen_q, int max_seqlen_kv, bool is_varlen) -> ()");
    m.def("fused_norm_rope_attn_rope_cast_fwd(Tensor q, Tensor kv, Tensor indices, float sm_scale, int d_v, Tensor? attn_sink, Tensor? topk_length, bool enable_q_norm, float rms_norm_eps, Tensor token_positions, bool is_rope_neox_style, int rope_dim, Tensor cos_sin_cache, int n_wv_group, int num_per_channels, bool use_tma_aligned_col_major_sf, bool round_sf, bool use_packed_ue8m0) -> Tensor[]");
    m.def("fused_norm_rope_attn_rope_cast_decode(Tensor q, Tensor kv, Tensor indices, float sm_scale, int d_v, Tensor? attn_sink, Tensor? topk_length, Tensor? extra_kv, Tensor? extra_indices, Tensor? extra_topk_length, bool enable_q_norm, float rms_norm_eps, Tensor token_positions, bool is_rope_neox_style, int rope_dim, Tensor cos_sin_cache, int n_wv_group, int num_per_channels, bool use_tma_aligned_col_major_sf, bool round_sf, bool use_packed_ue8m0) -> Tensor[]");
    m.def("permute_q_b_proj(Tensor q_b_proj, Tensor scale_factors, int h_q, int d_q) -> Tensor[]");
    m.def("permute_wv_proj(Tensor wv_proj, Tensor scale_factors, int wv_group_size, int d_o) -> Tensor[]");
}
