#include <Python.h>

#include <optional>
#include <vector>

#include <torch/all.h>
#include <torch/nn/functional.h>

extern
std::vector<at::Tensor>
fwd_kvcache_mla_fp8(
    at::Tensor &q,                               // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor &kcache,                    // num_blocks x page_block_size x num_heads_k x head_size (when is_fp8 is False) or num_blocks x num_heads_k x (page_block_size*656) (when is_fp8 is True)
    const int64_t head_size_v,
    const at::Tensor &seqlens_k,                 // batch_size
    const at::Tensor &block_table,               // batch_size x max_num_blocks_per_seq
    const double softmax_scale,
    bool is_causal,
    const at::Tensor &tile_scheduler_metadata,   // num_sm_parts x TileSchedulerMetaDataSize
    const at::Tensor &num_splits,                // batch_size + 1
    const std::optional<at::Tensor> &descale_q,  // None or batch_size
    const std::optional<at::Tensor> &descale_k   // None or batch_size
);

extern
std::vector<at::Tensor>
get_mla_decoding_metadata_dense_fp8(
    at::Tensor &seqlens_k,
    const int64_t num_heads_per_head_k,
    const int64_t num_heads_k
);
