#pragma once

#include <cutlass/half.h>
#include <cutlass/fast_math.h>

#include "common.h"
#include "params.h"

#include "sm90/decode/dense/splitkv_mla.h"
#include "smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.h"
#include "smxx/decode/combine/combine.h"

static std::tuple<Tensor, Tensor, std::optional<Tensor>, std::optional<Tensor>>
dense_attn_decode_interface(
    Tensor q,                                    // batch_size x seqlen_q x num_heads x head_size
    const Tensor &kcache,                        // num_blocks x page_block_size x num_heads_k x head_size (when is_fp8 is False) or num_blocks x num_heads_k x (page_block_size*656) (when is_fp8 is True)
    const int64_t head_size_v,
    const Tensor &seqlens_k,                     // batch_size
    const Tensor &block_table,                   // batch_size x max_num_blocks_per_seq
    const double softmax_scale,
    bool is_causal,
    std::optional<Tensor> tile_scheduler_metadata,    // num_sm_parts x (DecodingSchedMetaSize/4)
    std::optional<Tensor> num_splits                  // batch_size + 1
) {
    // Check arch
    Arch arch = Arch();
    if (!arch.is_sm90a()) {
        STD_TORCH_CHECK(false, "Dense decode MLA is only supported on SM90a architecture");
    }

    // Check data types
    auto q_dtype = q.scalar_type();
    STD_TORCH_CHECK(q_dtype == ScalarType::BFloat16 || q_dtype == ScalarType::Half);

    STD_TORCH_CHECK(kcache.scalar_type() == q_dtype, "query and key must have the same dtype");
    STD_TORCH_CHECK(seqlens_k.scalar_type() == ScalarType::Int, "seqlens_k must have dtype int32");
    STD_TORCH_CHECK(block_table.scalar_type() == ScalarType::Int, "block_table must have dtype torch.int32");

    // Check device
    KU_CHECK_DEVICE(q);
    KU_CHECK_DEVICE(kcache);
    KU_CHECK_DEVICE(seqlens_k);
    KU_CHECK_DEVICE(block_table);
    KU_CHECK_DEVICE(tile_scheduler_metadata);
    KU_CHECK_DEVICE(num_splits);

    // Check layout
    STD_TORCH_CHECK(q.stride(-1) == 1, "q must have contiguous last dimension");
    STD_TORCH_CHECK(kcache.stride(-1) == 1, "kcache must have contiguous last dimension");
    KU_CHECK_CONTIGUOUS(seqlens_k);
    STD_TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
    KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
    KU_CHECK_CONTIGUOUS(num_splits);

    const int batch_size = q.size(0);
    const int seqlen_q_ori = q.size(1);
    const int num_heads_q = q.size(2);
    const int head_size_k = q.size(3);
    STD_TORCH_CHECK(head_size_k == 576 || head_size_k == 512, "Only head_size_k == 576 or 512 is supported");
    STD_TORCH_CHECK(head_size_v == 512, "Only head_size_v == 512 is supported");

    const int max_num_blocks_per_seq = block_table.size(1);
    const int num_blocks = kcache.size(0);
    const int page_block_size = kcache.size(1);
    const int num_heads_k = kcache.size(2);
    STD_TORCH_CHECK(page_block_size == 64, "Currently page_block_size must be 64");
    STD_TORCH_CHECK(batch_size > 0, "batch size must be positive");
    STD_TORCH_CHECK(num_heads_q % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    
    if (seqlen_q_ori == 1) { is_causal = false; }
    
    const int num_q_heads_per_hk = num_heads_q / num_heads_k;
    const int q_seq_per_hk = seqlen_q_ori * num_q_heads_per_hk;
    const int num_heads = num_heads_k;
    q = torch::stable::reshape(
            torch::stable::transpose(
                torch::stable::view(q, {batch_size, seqlen_q_ori, num_heads_k, num_q_heads_per_hk, head_size_k}),
                2, 3),
            {batch_size, q_seq_per_hk, num_heads, head_size_k});
    int num_sm_parts = std::max(arch.num_sms / num_heads_k / cutlass::ceil_div(seqlen_q_ori*num_heads_q/num_heads_k, 64), 1);

    KU_CHECK_SHAPE(q, batch_size, q_seq_per_hk, num_heads, head_size_k);
    KU_CHECK_SHAPE(kcache, num_blocks, page_block_size, num_heads_k, head_size_k);
    KU_CHECK_SHAPE(seqlens_k, batch_size);
    KU_CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
    KU_CHECK_SHAPE(tile_scheduler_metadata, num_sm_parts, DecodingSchedMetaSize/sizeof(int));
    KU_CHECK_SHAPE(num_splits, batch_size+1);

    torch::stable::accelerator::DeviceGuard device_guard(q.get_device_index());

    Tensor out = torch::stable::new_empty(q, {batch_size, num_heads, q_seq_per_hk, head_size_v});
    Tensor lse = torch::stable::new_empty(q, {batch_size, num_heads, q_seq_per_hk}, ScalarType::Float);
    KU_CHECK_CONTIGUOUS(out);
    KU_CHECK_CONTIGUOUS(lse);

    if (!tile_scheduler_metadata.has_value()) {
        tile_scheduler_metadata = torch::stable::new_empty(q, {num_sm_parts, sizeof(DecodingSchedMeta)/4}, ScalarType::Int);
        num_splits = torch::stable::new_empty(q, {batch_size+1}, ScalarType::Int);
        KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
        KU_CHECK_CONTIGUOUS(num_splits);
    
        GetDecodeSchedMetaParams get_sched_meta_params = {
            batch_size, seqlen_q_ori,
            64,
            5,
            -1, -1,
            nullptr, nullptr,
            const_cast<int*>(seqlens_k.const_data_ptr<int>()),
            (DecodingSchedMeta*)tile_scheduler_metadata->data_ptr(),
            num_splits->mutable_data_ptr<int>(),
            num_sm_parts,
            get_current_cuda_stream(q)
        };
        smxx::decode::run_get_decoding_sched_meta_kernel(get_sched_meta_params);
    } else {
        KU_CHECK_DTYPE(tile_scheduler_metadata, ScalarType::Int);
        KU_CHECK_DTYPE(num_splits, ScalarType::Int);
        KU_CHECK_DEVICE(tile_scheduler_metadata);
        KU_CHECK_DEVICE(num_splits);
        KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
        KU_CHECK_CONTIGUOUS(num_splits);
        KU_CHECK_SHAPE(tile_scheduler_metadata, num_sm_parts, sizeof(DecodingSchedMeta)/sizeof(int));
        KU_CHECK_SHAPE(num_splits, batch_size+1);
    }

    // Set the sizes
    DenseAttnDecodeParams params;
    params.b = batch_size;
    params.s_q = seqlen_q_ori;
    params.q_seq_per_hk = q_seq_per_hk;
    params.seqlens_k_ptr = const_cast<int*>(seqlens_k.const_data_ptr<int>());
    params.h_q = num_heads_q;
    params.h_k = num_heads_k;
    params.num_blocks = num_blocks;
    params.q_head_per_hk = num_q_heads_per_hk;
    params.is_causal = is_causal;
    params.d = head_size_k;
    params.d_v = head_size_v;
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = float(softmax_scale * M_LOG2E);
    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = kcache.data_ptr();
    params.o_ptr = out.data_ptr();
    params.softmax_lse_ptr = lse.mutable_data_ptr<float>();
    // All stride are in elements, not bytes.
    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = kcache.stride(0);
    params.o_batch_stride = out.stride(0);
    params.q_row_stride = q.stride(1);
    params.k_row_stride = kcache.stride(1);
    params.o_row_stride = out.stride(2);
    params.q_head_stride = q.stride(2);
    params.k_head_stride = kcache.stride(2);
    params.o_head_stride = out.stride(1);

    params.block_table = const_cast<int*>(block_table.const_data_ptr<int>());
    params.block_table_batch_stride = block_table.stride(0);
    params.page_block_size = page_block_size;
    
    params.tile_scheduler_metadata_ptr = (DecodingSchedMeta*)tile_scheduler_metadata->data_ptr();
    params.num_sm_parts = num_sm_parts;
    params.num_splits_ptr = num_splits->mutable_data_ptr<int>();

    const int total_num_splits = batch_size + params.num_sm_parts;
    Tensor lse_accum = torch::stable::new_empty(q, {total_num_splits, num_heads, q_seq_per_hk}, ScalarType::Float);
    Tensor out_accum = torch::stable::new_empty(q, {total_num_splits, num_heads, q_seq_per_hk, head_size_v}, ScalarType::Float);
    KU_CHECK_CONTIGUOUS(lse_accum);
    KU_CHECK_CONTIGUOUS(out_accum);
    params.total_num_splits = total_num_splits;
    params.softmax_lseaccum_ptr = lse_accum.mutable_data_ptr<float>();
    params.oaccum_ptr = out_accum.mutable_data_ptr<float>();

    params.stream = get_current_cuda_stream(q);

    if (q_dtype == ScalarType::BFloat16) {
        sm90::run_flash_splitkv_mla_kernel<cutlass::bfloat16_t>(params);
    } else if (q_dtype == ScalarType::Half) {
#ifdef FLASH_MLA_DISABLE_FP16
        STD_TORCH_CHECK(false, "FlashMLA is compiled with -DFLASH_MLA_DISABLE_FP16. Please remove this flag from your environment and re-compile FlashMLA.");
#else
        sm90::run_flash_splitkv_mla_kernel<cutlass::half_t>(params);
#endif
    } else {
        STD_TORCH_CHECK(false, "Unsupported dtype for dense MLA on SM90");
    }

    CombineParams combine_params = {
        batch_size, seqlen_q_ori,
        num_heads_q, head_size_v,

        params.softmax_lse_ptr,
        params.o_ptr,
        num_heads*q_seq_per_hk, num_heads_q,
        num_heads_q*seqlen_q_ori*head_size_v, num_heads_q*head_size_v, head_size_v,

        params.softmax_lseaccum_ptr,
        params.oaccum_ptr,
        num_heads*q_seq_per_hk, num_heads_q,
        num_heads_q*seqlen_q_ori*head_size_v, num_heads_q*head_size_v, head_size_v,

        params.tile_scheduler_metadata_ptr,
        params.num_splits_ptr,
        params.num_sm_parts,

        nullptr,
        get_current_cuda_stream(q)
    };

    if (q_dtype == ScalarType::BFloat16) {
        smxx::decode::run_flash_mla_combine_kernel<cutlass::bfloat16_t>(combine_params);
    } else if (q_dtype == ScalarType::Half) {
#ifndef FLASH_MLA_DISABLE_FP16
        smxx::decode::run_flash_mla_combine_kernel<cutlass::half_t>(combine_params);
#endif
    } else {
        STD_TORCH_CHECK(false, "Unsupported tensor dtype for query");
    }

    out = torch::stable::reshape(
            torch::stable::transpose(
                torch::stable::view(out, {batch_size, num_heads_k, seqlen_q_ori, num_q_heads_per_hk, head_size_v}),
                1, 2),
            {batch_size, seqlen_q_ori, num_heads_q, head_size_v});
    lse = torch::stable::reshape(
            torch::stable::transpose(
                torch::stable::view(lse, {batch_size, num_heads_k, seqlen_q_ori, num_q_heads_per_hk}),
                2, 3),
            {batch_size, num_heads_q, seqlen_q_ori});

    return {out, lse, tile_scheduler_metadata, num_splits};
}
