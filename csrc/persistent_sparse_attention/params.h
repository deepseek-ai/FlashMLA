#pragma once

#include <cstdint>
#include <type_traits>

#include "cutlass/bfloat16.h"

namespace persistent_sparse_attention {

enum class ModelType {
    V32,
    MODEL1
};

struct __align__(4*8) DecodingSchedMeta {
    int begin_req_idx, end_req_idx;     // Both inclusive
    int begin_block_idx, end_block_idx; // Inclusive, exclusive
    int begin_split_idx;
    int is_first_req_splitted, is_last_req_splitted;
    int _pad[1];
};
static constexpr int DecodingSchedMetaSize = sizeof(DecodingSchedMeta);

struct DenseAttnDecodeParams { // TODO Change name to DenseAttnDecodeParams
    using index_t = int64_t;

    int b;              // batch size
    int s_q;
    int q_seq_per_hk;   // The number of q(s) per KV head, = h_q / h_k * s_q
    int d, d_v;         // K/V dimension
    int h_q, h_k;       // The number of Q/K heads
    int num_blocks;     // Number of blocks in total
    int q_head_per_hk;  // The number of q_head(s) per KV head, = h_q / h_k
    bool is_causal;
    float scale_softmax, scale_softmax_log2;

    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ o_ptr;
    float *__restrict__ softmax_lse_ptr;

    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t o_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t o_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t o_head_stride;

    int *__restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;
    int *__restrict__ seqlens_k_ptr;

    DecodingSchedMeta *__restrict__ tile_scheduler_metadata_ptr;
    int num_sm_parts;
    int *__restrict__ num_splits_ptr;

    int total_num_splits;
    float *__restrict__ softmax_lseaccum_ptr;
    float *__restrict__ oaccum_ptr;

    cudaStream_t stream;
};

struct SparseAttnDecodeParams {
    int b, s_q;
    int h_q, h_kv;
    int d_qk, d_v;
    float sm_scale, sm_scale_div_log2;
    int num_blocks, page_block_size, topk;
    ModelType model_type;

    cutlass::bfloat16_t* __restrict__ q;   // [b, s_q, h_q, d_qk]
    cutlass::bfloat16_t* __restrict__ kv;  // [num_blocks, page_block_size, d_qk]
    int* __restrict__ indices;   // [b, s_q, topk]
    int* __restrict__ topk_length;  // [b], may be nullptr
    float* __restrict__ attn_sink;  // [h_q], may be nullptr

    float* __restrict__ lse;    // [b, s_q, h_q]
    cutlass::bfloat16_t* __restrict__ out;   // [b, s_q, h_q, d_v]

    int extra_num_blocks, extra_page_block_size, extra_topk;
    cutlass::bfloat16_t* __restrict__ extra_kv;  // [extra_num_blocks, extra_page_block_size, d_qk]
    int* __restrict__ extra_indices;   // [b, s_q, extra_topk]
    int* __restrict__ extra_topk_length;  // [b], may be nullptr

    int stride_q_b, stride_q_s_q, stride_q_h_q;
    int stride_kv_block, stride_kv_row;
    int stride_indices_b, stride_indices_s_q;
    int stride_lse_b, stride_lse_s_q;
    int stride_o_b, stride_o_s_q, stride_o_h_q;
    int stride_extra_kv_block, stride_extra_kv_row;
    int stride_extra_indices_b, stride_extra_indices_s_q;

    cudaStream_t stream;

    // SplitKV-related parameters
    float* __restrict__ lse_accum;  // [num_splits, s_q, h_q]
    float* __restrict__ o_accum;    // [num_splits, s_q, h_q, d_v]
    int stride_lse_accum_split, stride_lse_accum_s_q;
    int stride_o_accum_split, stride_o_accum_s_q, stride_o_accum_h_q;
    DecodingSchedMeta* __restrict__ tile_scheduler_metadata_ptr; // [num_sm_parts, ], contiguous
    int* __restrict__ num_splits_ptr; // [batch_size+1, ], contiguous
    int num_sm_parts;

    // Optional producer/consumer hand-off for CTA-specialized persistent sparse attention.
    // When non-null, the SM100 head128 kernel waits for row_ready[batch_idx]
    // and assigns complete requests round-robin across scheduler partitions.
    int* __restrict__ row_ready; // [b], int32, zero -> not ready
    int initial_ready_rows;      // dependency-protected prefix, no polling

    // Optional DeepSeek-V4 raw-Q preprocessing.  When both pointers are
    // non-null, the SM100 head128 kernel performs per-head RMSNorm and GPT-J
    // RoPE in the shared-memory Q-load prologue before copying Q to TMEM.
    // A negative position is an internal sentinel for a row that the
    // producer already transformed and that must skip this prologue.
    int64_t* __restrict__ q_positions;   // [b]
    float* __restrict__ q_cos_sin_cache; // [max_position, 64] = cos || sin
    float q_norm_eps;

    // Optional ready-decode MA-O0 epilogue. Each completed attention row is
    // inverse-RoPE transformed and block-quantized directly from TMEM into the
    // group-major packed-UE8M0 layout consumed by DeepGEMM WoA.
    uint8_t* __restrict__ fused_o_fp8;       // logical [b, 16, 4096]
    int* __restrict__ fused_o_scale;         // logical [b, 16, 8]
    int64_t* __restrict__ fused_o_positions; // [b]
    int stride_fused_o_fp8_b, stride_fused_o_fp8_g;
    int stride_fused_o_fp8_k;
    int stride_fused_o_scale_b, stride_fused_o_scale_g;
    int stride_fused_o_scale_k;
    bool fused_o_skip_bf16;
};

struct CombineParams {
    int b, s_q, h_q, d_v;

    float* __restrict__ lse;    // [b, s_q, h_q]
    void* __restrict__ out;   // [b, s_q, h_q, d_v]
    int stride_lse_b, stride_lse_s_q;
    int stride_o_b, stride_o_s_q, stride_o_h_q;

    float* __restrict__ lse_accum;  // [num_splits, s_q, h_q]
    float* __restrict__ o_accum;    // [num_splits, s_q, h_q, d_v]
    int stride_lse_accum_split, stride_lse_accum_s_q;
    int stride_o_accum_split, stride_o_accum_s_q, stride_o_accum_h_q;

    DecodingSchedMeta* __restrict__ tile_scheduler_metadata_ptr; // [num_sm_parts, ], contiguous
    int* __restrict__ num_splits_ptr; // [batch_size+1, ], contiguous
    int num_sm_parts;

    float* attn_sink;  // [h_q], may be nullptr

    cudaStream_t stream;
};

struct GetDecodeSchedMetaParams {
    int b;  // batch size
    int s_q;
    int block_size_n;
    int fixed_overhead_num_blocks;

    int topk, extra_topk;   // -1 if sparse attention (or extra topk) is disabled
    int *__restrict__ topk_length, *__restrict__ extra_topk_length;

    int *__restrict__ seqlens_k_ptr;    // Only necessary for dense attention

    DecodingSchedMeta *__restrict__ tile_scheduler_metadata_ptr;
    int *__restrict__ num_splits_ptr;
    int num_sm_parts;

    cudaStream_t stream;
};

struct SparseAttnFwdParams {
    int s_q, s_kv, h_q, h_kv, d_qk, d_v, topk;
    float sm_scale, sm_scale_div_log2;

    // Input tensors
    cutlass::bfloat16_t* __restrict__ q;    // [s_q, h_q, d_qk]
    cutlass::bfloat16_t* __restrict__ kv;   // [s_kv, h_kv, d_qk]
    int* __restrict__ indices;   // [s_q, h_kv, topk]
    float* __restrict__ attn_sink;   // [h_q], may be nullptr
    int* __restrict__ topk_length;    // [s_q], may be nullptr

    // Optional second BF16 KV pool. DeepSeek-V4 HCA uses the primary pool
    // for the 128-token SWA window and this pool for compressed history.
    int s_extra_kv, extra_topk;
    cutlass::bfloat16_t* __restrict__ extra_kv;
    int* __restrict__ extra_indices;
    int* __restrict__ extra_topk_length;

    // Strides
    int stride_q_s_q; int stride_q_h_q;
    int stride_kv_s_kv; int stride_kv_h_kv;
    int stride_indices_s_q; int stride_indices_h_kv;
    int stride_extra_kv_s_kv; int stride_extra_kv_h_kv;
    int stride_extra_indices_s_q; int stride_extra_indices_h_kv;

    // Output tensors
    cutlass::bfloat16_t* __restrict__ out;   // [s_q, h_q, d_v]
    float* __restrict__ max_logits; // [s_q, h_q]
    float* __restrict__ lse; // [s_q, h_q]

    // Optional producer/consumer hand-off for persistent sparse Prefill.
    // q_ready publishes groups of consecutive Q rows.  The producer must use
    // release ordering after all rows in a group have been stored.
    int* __restrict__ q_ready; // [ceil(s_q / q_ready_chunk_size)], int32
    int q_ready_chunk_size;

    // Optional two-CTA output completion hand-off. Each query owns two
    // consecutive flags, one per CTA in the cluster. A CTA release-publishes
    // its flag only after its async TMA output store has completed.
    int* __restrict__ o_ready; // [2 * s_q], int32

    // Optional DeepSeek-V4 raw-Q preprocessing.  When both pointers are
    // non-null, the SM100 head128 small-topk kernel performs per-head RMSNorm
    // and GPT-J RoPE in shared memory before copying Q to TMEM. A negative
    // position is an internal sentinel for a producer-preprocessed row.
    int64_t* __restrict__ q_positions;   // [s_q]
    float* __restrict__ q_cos_sin_cache; // [max_position, 64] = cos || sin
    float q_norm_eps;

    // Optional DeepSeek-V4 MA-O0 epilogue.  The SM100 head128 kernel can
    // inverse-RoPE and block-quantize each completed attention head directly
    // from TMEM into the group-major layout consumed by DeepGEMM WoA.
    uint8_t* __restrict__ fused_o_fp8;       // logical [s_q, 16, 4096]
    int* __restrict__ fused_o_scale;         // logical [s_q, 16, 8]
    int64_t* __restrict__ fused_o_positions; // [s_q]
    int stride_fused_o_fp8_b, stride_fused_o_fp8_g;
    int stride_fused_o_fp8_k;
    int stride_fused_o_scale_b, stride_fused_o_scale_g;
    int stride_fused_o_scale_k;
    bool fused_o_skip_bf16;

    int num_sm;
    // Prefill only. When set, the grid is capped at `num_sm` cluster-aligned
    // CTAs and each resident cluster walks a closed-form sequence of query
    // rows instead of pulling them from Cluster Launch Control. Off keeps the
    // stock problem-sized [2*s_q] CLC grid byte-for-byte.
    bool persistent_grid;
    cudaStream_t stream;
};

// We have some kernels that implement both prefill and decode modes in a single kernel (with different template instantiations). The following enum helps to distinguish the modes.
enum class SparseAttnFwdMode {
    Prefill,            // Normal prefill mode
    DecodeWithSplitKV,  // To trigger decoding mode for kernels that support both prefill and decode
    DecodeReady,        // Whole-row round-robin decode gated by row_ready
};

template<SparseAttnFwdMode FWD_MODE>
inline constexpr bool is_decode_v = std::bool_constant<FWD_MODE != SparseAttnFwdMode::Prefill>::value;

template<SparseAttnFwdMode FWD_MODE>
inline constexpr bool is_ready_decode_v = std::bool_constant<FWD_MODE == SparseAttnFwdMode::DecodeReady>::value;

template<SparseAttnFwdMode FWD_MODE>
using SparseFwdArgT = std::conditional_t<is_decode_v<FWD_MODE>, SparseAttnDecodeParams, SparseAttnFwdParams>;

}  // namespace persistent_sparse_attention
