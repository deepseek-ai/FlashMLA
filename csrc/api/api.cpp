#include <torch/csrc/stable/library.h>

#include "sparse_fwd.h"
#include "sparse_decode.h"
#include "dense_decode.h"
#include "dense_fwd.h"

STABLE_TORCH_LIBRARY(flash_mla, m) {
    m.def("sparse_decode_fwd(Tensor q, Tensor kv, Tensor indices, Tensor? topk_length, Tensor? attn_sink, Tensor(a)? tile_scheduler_metadata, Tensor(b)? num_splits, Tensor? extra_kv, Tensor? extra_indices, Tensor? extra_topk_length, int d_v, float sm_scale) -> (Tensor, Tensor, Tensor(a)?, Tensor(b)?)");
    m.def("dense_decode_fwd(Tensor q, Tensor kcache, int head_size_v, Tensor seqlens_k, Tensor block_table, float softmax_scale, bool is_causal, Tensor(a)? tile_scheduler_metadata, Tensor(b)? num_splits) -> (Tensor, Tensor, Tensor(a)?, Tensor(b)?)");
    m.def("sparse_prefill_fwd(Tensor q, Tensor kv, Tensor indices, float sm_scale, int d_v, Tensor? attn_sink, Tensor? topk_length) -> Tensor[]");
    m.def("dense_prefill_fwd(Tensor workspace_buffer, Tensor q, Tensor k, Tensor v, Tensor cumulative_seqlen_q, Tensor cumulative_seqlen_kv, Tensor(a!) o, Tensor(b!) lse, int mask_mode_code, float softmax_scale, int max_seqlen_q, int max_seqlen_kv, bool is_varlen) -> ()");
    m.def("dense_prefill_bwd(Tensor(a!) workspace_buffer, Tensor d_o, Tensor q, Tensor k, Tensor v, Tensor o, Tensor lse, Tensor cumulative_seqlen_q, Tensor cumulative_seqlen_kv, Tensor(b!) dq, Tensor(c!) dk, Tensor(d!) dv, int mask_mode_code, float softmax_scale, int max_seqlen_q, int max_seqlen_kv, bool is_varlen) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(flash_mla, CUDA, m) {
    m.impl("sparse_decode_fwd", TORCH_BOX(&sparse_attn_decode_interface));
    m.impl("dense_decode_fwd", TORCH_BOX(&dense_attn_decode_interface));
    m.impl("sparse_prefill_fwd", TORCH_BOX(&sparse_attn_prefill_interface));
    m.impl("dense_prefill_fwd", TORCH_BOX(&FMHACutlassSM100FwdRun));
    m.impl("dense_prefill_bwd", TORCH_BOX(&FMHACutlassSM100BwdRun));
}
