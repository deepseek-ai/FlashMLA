#pragma once

#include "params.h"

namespace nv_smallbatch::sm100 {

struct DecodingParamsCompat {
    using index_t = int64_t;

    int b;
    int s_q;
    int q_seq_per_hk;
    int d, d_v;
    int h_q, h_k;
    int num_blocks;
    int q_head_per_hk;
    bool is_causal;
    float scale_softmax, scale_softmax_log2;
    int topk;

    void* __restrict__ q_ptr;
    void* __restrict__ k_ptr;
    void* __restrict__ o_ptr;
    void* __restrict__ softmax_lse_ptr;
    int* __restrict__ indices_ptr;

    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t o_batch_stride;
    index_t q_seq_stride;
    index_t o_seq_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t o_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t o_head_stride;
    index_t indices_batch_stride;
    index_t indices_row_stride;

    int* __restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;
    int tma_coords_step_per_block; // k_batch_stride / k_row_stride; accounts for block padding in main's KV layout
    int* __restrict__ seqlens_k_ptr;

    DecodingSchedMeta* __restrict__ tile_scheduler_metadata_ptr;
    int num_sm_parts;
    int* __restrict__ num_splits_ptr;

    int total_num_splits;
    void* __restrict__ softmax_lseaccum_ptr;
    void* __restrict__ oaccum_ptr;

    cudaStream_t stream;
};

inline DecodingParamsCompat make_decoding_params_compat(const SparseAttnDecodeParams& params) {
    DecodingParamsCompat compat = {};
    compat.b = params.b;
    compat.s_q = params.s_q;
    compat.q_seq_per_hk = params.s_q * (params.h_q / params.h_kv);
    compat.d = params.d_qk;
    compat.d_v = params.d_v;
    compat.h_q = params.h_q;
    compat.h_k = params.h_kv;
    compat.num_blocks = params.num_blocks;
    compat.q_head_per_hk = params.h_q / params.h_kv;
    compat.is_causal = false;
    compat.scale_softmax = params.sm_scale;
    compat.scale_softmax_log2 = params.sm_scale_div_log2;
    compat.topk = params.topk;

    compat.q_ptr = params.q;
    compat.k_ptr = params.kv;
    compat.o_ptr = params.out;
    compat.softmax_lse_ptr = params.lse;
    compat.indices_ptr = params.indices;

    compat.q_batch_stride = params.stride_q_b;
    compat.k_batch_stride = params.stride_kv_block;
    compat.o_batch_stride = params.stride_o_b;
    compat.q_seq_stride = params.stride_q_s_q;
    compat.o_seq_stride = params.stride_o_s_q;
    compat.q_row_stride = params.stride_q_h_q;
    compat.k_row_stride = params.stride_kv_row;
    compat.o_row_stride = params.stride_o_h_q;
    compat.q_head_stride = params.stride_q_h_q;
    compat.k_head_stride = 0;
    compat.o_head_stride = params.stride_o_h_q;
    compat.indices_batch_stride = params.stride_indices_b;
    compat.indices_row_stride = params.stride_indices_s_q;

    compat.block_table = nullptr;
    compat.block_table_batch_stride = 0;
    compat.page_block_size = params.page_block_size;
    // step per block in row-stride units — handles main's kv block padding (block_size+1 internal padding)
    compat.tma_coords_step_per_block = (int)(params.stride_kv_block / params.stride_kv_row);
    compat.seqlens_k_ptr = nullptr;

    compat.tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr;
    compat.num_sm_parts = params.num_sm_parts;
    compat.num_splits_ptr = params.num_splits_ptr;

    compat.total_num_splits = params.b + params.num_sm_parts;
    compat.softmax_lseaccum_ptr = params.lse_accum;
    compat.oaccum_ptr = params.o_accum;
    compat.stream = params.stream;
    return compat;
}

}  // namespace nv_smallbatch::sm100
