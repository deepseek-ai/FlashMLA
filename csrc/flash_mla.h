#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_mla_params {
    using index_t = int64_t;

    int b, seqlen_q, d, d_v;
    int h, h_h_k_ratio, ngroups;
    bool is_causal;
    float scale_softmax, scale_softmax_log2;
    int *__restrict__ cu_seqlens_k;

    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;
    void *__restrict__ o_ptr;
    void *__restrict__ softmax_lse_ptr;

    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t o_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t o_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;
    index_t o_head_stride;

    int *__restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;

    int *__restrict__ tile_scheduler_metadata_ptr;
    int num_sm_parts;
    int *__restrict__ num_splits_ptr;

    void *__restrict__ softmax_lseaccum_ptr;
    void *__restrict__ oaccum_ptr;

    // ---- Interfaces for the Ampere (sm86) kernels; unused by the legacy sm80/sm90 kernels. ----
    // Caller-owned persistent scratch (fp32 partial O / LSE for split-KV). nullptr => allocate per call.
    void *__restrict__ workspace = nullptr;
    // Per-(batch, m-block) arrival counters for the fused split-KV combine. Must be all-zero on entry;
    // the last-arriving CTA resets its counter to 0 (stays valid across CUDA Graph replays). nullptr => separate combine kernel.
    int *__restrict__ combine_counters = nullptr;
    // Per-token fp32 dequant scale, row-major [num_blocks, page_block_size]; nullptr when kv_dtype == kKVNative.
    // See docs/kv_quant_format.md.
    const float *__restrict__ kv_scale_ptr = nullptr;
    // One of the kKV* values below.
    int kv_dtype = 0;
};

// Values for Flash_fwd_mla_params::kv_dtype.
static constexpr int kKVNative = 0;     // bf16 / fp16 KV cache (D = 576 elements per token)
static constexpr int kKVFp8E4m3 = 1;    // 512 latent dims fp8 e4m3 + fp32 per-token scale, 64 rope dims bf16
static constexpr int kKVInt8 = 2;       // 512 latent dims int8     + fp32 per-token scale, 64 rope dims bf16

static constexpr int TileSchedulerMetaDataSize = 8;
// [begin_idx, begin_seqlen, end_idx, end_seqlen, begin_n_split_idx, _, _, _]

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int Headdim, bool is_sm90>
struct mha_fwd_splitkv_mla {
    static void run(Flash_fwd_mla_params &params, cudaStream_t stream);
};

template<typename T, int Headdim>
struct mha_fwd_splitkv_mla_ws {
    static void run(Flash_fwd_mla_params &params, cudaStream_t stream);
};

struct Mla_metadata_params {
    int *__restrict__ seqlens_k_ptr;
    int *__restrict__ tile_scheduler_metadata_ptr;
    int *__restrict__ num_splits_ptr;
    int batch_size;
    int block_size_n;
    int fixed_overhead_num_blocks;
    int num_sm_parts;
};

void get_mla_metadata_func(Mla_metadata_params &params, cudaStream_t stream);
