#pragma once

#include "common.h"
#include "sparse_decode.h"

#include "sm90/decode/sparse_fp8/pack_selected_kv.h"

static constexpr int PACKED_SPARSE_RECORD_BYTES = 656;
static constexpr int PACKED_SPARSE_TILE_SIZE = 64;

static at::Tensor pack_selected_kv_interface(
    const at::Tensor &kv,
    const at::Tensor &indices,
    const std::optional<at::Tensor> &topk_length,
    const std::optional<at::Tensor> &out_
) {
    Arch arch;
    TORCH_CHECK(
        arch.is_sm90a(),
        "pack_selected_kv is available only on SM90"
    );

    KU_CHECK_NDIM(kv, 4);
    KU_CHECK_NDIM(indices, 3);
    const int num_blocks = kv.size(0);
    const int page_block_size = kv.size(1);
    const int h_kv = kv.size(2);
    const int record_bytes = kv.size(3);
    const int b = indices.size(0);
    const int s_q = indices.size(1);
    const int topk = indices.size(2);

    TORCH_CHECK(b > 0 && s_q > 0 && topk > 0);
    TORCH_CHECK(
        topk % PACKED_SPARSE_TILE_SIZE == 0,
        "packed sparse topk must be divisible by 64"
    );
    TORCH_CHECK(h_kv == 1, "packed sparse decode requires MQA KV");
    TORCH_CHECK(
        record_bytes == PACKED_SPARSE_RECORD_BYTES,
        "pack_selected_kv requires the 656-byte V3.2 FP8 KV format"
    );

    KU_CHECK_DEVICE(kv);
    KU_CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(topk_length);
    TORCH_CHECK(
        kv.device() == indices.device(),
        "KV and indices must be on the same device"
    );
    if (topk_length.has_value()) {
        TORCH_CHECK(
            kv.device() == topk_length->device(),
            "KV and topk_length must be on the same device"
        );
    }
    TORCH_CHECK(
        kv.dtype() == torch::kFloat8_e4m3fn
            || kv.dtype() == torch::kInt8
            || kv.dtype() == torch::kUInt8,
        "KV must have dtype float8_e4m3fn, int8, or uint8"
    );
    KU_CHECK_DTYPE(indices, torch::kInt32);
    KU_CHECK_DTYPE(topk_length, torch::kInt32);
    KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
    TORCH_CHECK(
        kv.stride(1) == PACKED_SPARSE_RECORD_BYTES,
        "each source KV page must be contiguous"
    );
    TORCH_CHECK(
        kv.stride(0) % 16 == 0,
        "source KV blocks must preserve 16-byte record alignment"
    );
    KU_CHECK_CONTIGUOUS(indices);
    KU_CHECK_CONTIGUOUS(topk_length);
    KU_CHECK_SHAPE(topk_length, b);

    const int packed_pages =
        b * s_q * (topk / PACKED_SPARSE_TILE_SIZE);
    const std::vector<int64_t> packed_shape = {
        packed_pages,
        PACKED_SPARSE_TILE_SIZE,
        1,
        PACKED_SPARSE_RECORD_BYTES
    };

    at::Tensor out;
    if (out_.has_value()) {
        at::Tensor storage = *out_;
        KU_CHECK_DEVICE(storage);
        KU_CHECK_DTYPE(storage, torch::kUInt8);
        KU_CHECK_CONTIGUOUS(storage);
        TORCH_CHECK(
            storage.device() == kv.device(),
            "packed output and source KV must be on the same device"
        );
        TORCH_CHECK(
            storage.numel()
                == int64_t(packed_pages)
                    * PACKED_SPARSE_TILE_SIZE
                    * PACKED_SPARSE_RECORD_BYTES,
            "packed output storage has the wrong size"
        );
        out = storage.view(packed_shape);
    } else {
        out = torch::empty(
            packed_shape,
            kv.options().dtype(torch::kUInt8)
        );
    }

    at::cuda::CUDAGuard device_guard{(char)kv.get_device()};
    sm90::decode::sparse_fp8::PackSelectedKvParams params = {
        reinterpret_cast<const uint8_t*>(kv.data_ptr()),
        indices.data_ptr<int>(),
        ku::get_optional_tensor_ptr<int>(topk_length),
        out.data_ptr<uint8_t>(),
        num_blocks * page_block_size,
        page_block_size,
        kv.stride(0),
        kv.stride(1),
        b,
        s_q,
        topk,
        at::cuda::getCurrentCUDAStream().stream()
    };
    sm90::decode::sparse_fp8::run_pack_selected_kv_kernel(params);
    return out;
}

static std::tuple<
    at::Tensor,
    at::Tensor,
    std::optional<at::Tensor>,
    std::optional<at::Tensor>
>
packed_sparse_attn_decode_interface(
    const at::Tensor &q,
    const at::Tensor &packed_kv,
    const std::optional<at::Tensor> &topk_length,
    const std::optional<at::Tensor> &attn_sink,
    std::optional<at::Tensor> tile_scheduler_metadata,
    std::optional<at::Tensor> num_splits,
    int d_v,
    float sm_scale
) {
    using bf16 = cutlass::bfloat16_t;

    Arch arch;
    TORCH_CHECK(
        arch.is_sm90a(),
        "packed sparse decode is available only on SM90"
    );

    KU_CHECK_NDIM(q, 4);
    KU_CHECK_NDIM(packed_kv, 4);
    const int b = q.size(0);
    const int s_q = q.size(1);
    const int h_q = q.size(2);
    const int d_qk = q.size(3);
    const int num_blocks = packed_kv.size(0);
    const int page_block_size = packed_kv.size(1);
    const int h_kv = packed_kv.size(2);
    const int record_bytes = packed_kv.size(3);

    TORCH_CHECK(b > 0 && s_q > 0);
    TORCH_CHECK(
        h_q == 64 || h_q == 128,
        "packed sparse decode supports 64 or 128 query heads"
    );
    TORCH_CHECK(
        d_qk == 576 && d_v == 512,
        "packed sparse decode supports only V3.2 [d_qk=576,d_v=512]"
    );
    TORCH_CHECK(
        page_block_size == PACKED_SPARSE_TILE_SIZE
            && h_kv == 1
            && record_bytes == PACKED_SPARSE_RECORD_BYTES,
        "packed KV must have opaque shape [N,64,1,656]"
    );
    TORCH_CHECK(
        num_blocks % (b * s_q) == 0,
        "packed KV page count must be divisible by B*S_q"
    );
    const int topk =
        (num_blocks / (b * s_q)) * PACKED_SPARSE_TILE_SIZE;
    TORCH_CHECK(topk > 0);

    KU_CHECK_DEVICE(q);
    KU_CHECK_DEVICE(packed_kv);
    KU_CHECK_DEVICE(topk_length);
    KU_CHECK_DEVICE(attn_sink);
    KU_CHECK_DEVICE(tile_scheduler_metadata);
    KU_CHECK_DEVICE(num_splits);
    TORCH_CHECK(
        q.device() == packed_kv.device(),
        "q and packed KV must be on the same device"
    );
    if (topk_length.has_value()) {
        TORCH_CHECK(
            q.device() == topk_length->device(),
            "q and topk_length must be on the same device"
        );
    }
    if (attn_sink.has_value()) {
        TORCH_CHECK(
            q.device() == attn_sink->device(),
            "q and attn_sink must be on the same device"
        );
    }
    if (tile_scheduler_metadata.has_value()) {
        TORCH_CHECK(
            q.device() == tile_scheduler_metadata->device(),
            "q and scheduler metadata must be on the same device"
        );
    }
    if (num_splits.has_value()) {
        TORCH_CHECK(
            q.device() == num_splits->device(),
            "q and num_splits must be on the same device"
        );
    }
    KU_CHECK_DTYPE(q, torch::kBFloat16);
    KU_CHECK_DTYPE(packed_kv, torch::kUInt8);
    KU_CHECK_DTYPE(topk_length, torch::kInt32);
    KU_CHECK_DTYPE(attn_sink, torch::kFloat32);
    KU_CHECK_DTYPE(tile_scheduler_metadata, torch::kInt32);
    KU_CHECK_DTYPE(num_splits, torch::kInt32);
    KU_CHECK_LAST_DIM_CONTIGUOUS(q);
    KU_CHECK_CONTIGUOUS(packed_kv);
    KU_CHECK_CONTIGUOUS(topk_length);
    KU_CHECK_CONTIGUOUS(attn_sink);
    KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
    KU_CHECK_CONTIGUOUS(num_splits);
    KU_CHECK_SHAPE(q, b, s_q, h_q, d_qk);
    KU_CHECK_SHAPE(
        packed_kv,
        num_blocks,
        PACKED_SPARSE_TILE_SIZE,
        1,
        PACKED_SPARSE_RECORD_BYTES
    );
    KU_CHECK_SHAPE(topk_length, b);
    KU_CHECK_SHAPE(attn_sink, h_q);
    TORCH_CHECK(
        tile_scheduler_metadata.has_value()
            == num_splits.has_value(),
        "tile_scheduler_metadata and num_splits must be supplied together"
    );

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto opts = q.options();
    at::Tensor out = torch::empty({b, s_q, h_q, d_v}, opts);
    at::Tensor lse =
        torch::empty({b, s_q, h_q}, opts.dtype(at::kFloat));

    DecodeImplMeta impl_meta = {
        std::max(arch.num_sms / s_q / (h_q / 64), 1),
        5,
        PACKED_SPARSE_TILE_SIZE
    };
    if (tile_scheduler_metadata.has_value()) {
        TORCH_CHECK(
            tile_scheduler_metadata->size(0) > 0,
            "external packed scheduler must contain at least one row"
        );
        impl_meta.num_sm_parts = tile_scheduler_metadata->size(0);
    }
    TORCH_CHECK(
        impl_meta.num_sm_parts <= 160,
        "packed sparse decode supports at most 160 scheduler rows"
    );

    SparseAttnDecodeParams params = {
        b, s_q, h_q, h_kv, d_qk, d_v,
        sm_scale, sm_scale * LOG_2_E,
        num_blocks, page_block_size, topk,
        ModelType::V32,

        (bf16*)q.data_ptr(),
        (bf16*)packed_kv.data_ptr(),
        nullptr,
        ku::get_optional_tensor_ptr<int>(topk_length),
        ku::get_optional_tensor_ptr<float>(attn_sink),
        (float*)lse.data_ptr(),
        (bf16*)out.data_ptr(),

        0, 0, 0,
        nullptr, nullptr, nullptr,

        int64_stride_to_int(q.stride(0)),
        int64_stride_to_int(q.stride(1)),
        int64_stride_to_int(q.stride(2)),
        int64_stride_to_int(packed_kv.stride(0)),
        int64_stride_to_int(packed_kv.stride(1)),
        0, 0,
        int64_stride_to_int(lse.stride(0)),
        int64_stride_to_int(lse.stride(1)),
        int64_stride_to_int(out.stride(0)),
        int64_stride_to_int(out.stride(1)),
        int64_stride_to_int(out.stride(2)),

        0, 0, 0, 0,
        at::cuda::getCurrentCUDAStream().stream()
    };

    if (!tile_scheduler_metadata.has_value()) {
        tile_scheduler_metadata = torch::empty(
            {
                impl_meta.num_sm_parts,
                sizeof(DecodingSchedMeta) / sizeof(int)
            },
            opts.dtype(torch::kInt32)
        );
        num_splits =
            torch::empty({b + 1}, opts.dtype(torch::kInt32));
        GetDecodeSchedMetaParams get_sched_meta_params = {
            b, s_q,
            impl_meta.block_size_topk,
            impl_meta.fixed_overhead_num_blocks,
            topk,
            0,
            ku::get_optional_tensor_ptr<int>(topk_length),
            nullptr,
            nullptr,
            (DecodingSchedMeta*)tile_scheduler_metadata->data_ptr(),
            num_splits->data_ptr<int>(),
            impl_meta.num_sm_parts,
            at::cuda::getCurrentCUDAStream().stream()
        };
        smxx::decode::run_get_decoding_sched_meta_kernel(
            get_sched_meta_params
        );
    }

    KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
    KU_CHECK_CONTIGUOUS(num_splits);
    KU_CHECK_SHAPE(
        tile_scheduler_metadata,
        impl_meta.num_sm_parts,
        sizeof(DecodingSchedMeta) / sizeof(int)
    );
    KU_CHECK_SHAPE(num_splits, b + 1);
    params.tile_scheduler_metadata_ptr =
        (DecodingSchedMeta*)tile_scheduler_metadata->data_ptr();
    params.num_splits_ptr = num_splits->data_ptr<int>();
    params.num_sm_parts = impl_meta.num_sm_parts;

    const int total_num_splits = b + impl_meta.num_sm_parts;
    at::Tensor lse_accum = torch::empty(
        {total_num_splits, s_q, h_q},
        opts.dtype(at::kFloat)
    );
    at::Tensor o_accum = torch::empty(
        {total_num_splits, s_q, h_q, d_v},
        opts.dtype(at::kFloat)
    );
    params.lse_accum = lse_accum.data_ptr<float>();
    params.o_accum = o_accum.data_ptr<float>();
    params.stride_lse_accum_split =
        int64_stride_to_int(lse_accum.stride(0));
    params.stride_lse_accum_s_q =
        int64_stride_to_int(lse_accum.stride(1));
    params.stride_o_accum_split =
        int64_stride_to_int(o_accum.stride(0));
    params.stride_o_accum_s_q =
        int64_stride_to_int(o_accum.stride(1));
    params.stride_o_accum_h_q =
        int64_stride_to_int(o_accum.stride(2));

    DISPATCH_NUM_HEADS(h_q, NUM_HEADS, [&]() {
        sm90::decode::sparse_fp8::
            run_flash_splitkv_mla_fp8_packed_multi_query_kernel<
                ModelType::V32,
                NUM_HEADS
            >(params);
    });

    CombineParams combine_params = {
        b, s_q, h_q, d_v,

        params.lse,
        params.out,
        params.stride_lse_b,
        params.stride_lse_s_q,
        params.stride_o_b,
        params.stride_o_s_q,
        params.stride_o_h_q,

        params.lse_accum,
        params.o_accum,
        params.stride_lse_accum_split,
        params.stride_lse_accum_s_q,
        params.stride_o_accum_split,
        params.stride_o_accum_s_q,
        params.stride_o_accum_h_q,

        params.tile_scheduler_metadata_ptr,
        params.num_splits_ptr,
        params.num_sm_parts,

        ku::get_optional_tensor_ptr<float>(attn_sink),
        at::cuda::getCurrentCUDAStream().stream()
    };
    smxx::decode::run_flash_mla_combine_kernel<bf16>(
        combine_params
    );

    return {
        out,
        lse.transpose(1, 2),
        tile_scheduler_metadata,
        num_splits
    };
}
