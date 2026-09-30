#include "common.h"
#include "params.h"

#include <algorithm>

template<bool ENABLE_SPLIT_KV>
static constexpr SparseAttnFwdMode get_decode_fwd_mode() {
    if constexpr (ENABLE_SPLIT_KV) {
        return SparseAttnFwdMode::DecodeWithSplitKV;
    } else {
        return SparseAttnFwdMode::Decode;
    }
}

enum class SparseDecodeFeatures : int {
    HEAD_64,
    HEAD_128,

    ATTN_SINK,
    TOPK_LENGTH,
    EXTRA_KVCACHE,
    EXTRA_TOPK_LENGTH,

    BATCH_INVARIANT
};

struct SparseDecodeImplMeta {
    int num_sm_parts;
    int fixed_overhead_num_blocks;
    int block_size_topk;
};


class SparseDecodeImplBase : public ImplBase<
    SparseAttnDecodeParams,
    SparseDecodeFeatures
> {
public:
    virtual SparseDecodeImplMeta get_meta(int h_q, int s_q) = 0;
};


#ifdef FLASH_MLA_IS_BUILD_ON_CUDA

#include "cuda_kernels/sm100/decode/sparse/head64/kernel.h"
#include "cuda_kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/phase1.h"
#include "cuda_kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.h"
#include "cuda_kernels/smxx/decode/combine/combine.h"

class SparseDecode_Sm100_Head64_Impl : public SparseDecodeImplBase {
    DECLARE_SUPPORTED_FEATURES(
        SparseDecodeFeatures::HEAD_64,
        SparseDecodeFeatures::ATTN_SINK,
        SparseDecodeFeatures::TOPK_LENGTH,
        SparseDecodeFeatures::EXTRA_KVCACHE,
        SparseDecodeFeatures::EXTRA_TOPK_LENGTH,
        SparseDecodeFeatures::BATCH_INVARIANT
    )
    using SupportedKVFormats = KVFormatPairs<KVFormatPair<ModelType::V41>,
                                             KVFormatPair<ModelType::V41, ModelType::V41_FP4>>;

public:
    SparseDecodeImplMeta get_meta(int h_q, int s_q) override {
        Arch arch = Arch();
        return {
            std::max(arch.num_sms / s_q, 1),
            5,
            64
        };
    }

protected:
    void run_(const SparseAttnDecodeParams &params, const std::vector<FeatureT> &required_features) override {
        dispatch_kv_formats(SupportedKVFormats{}, params.model_type, params.extra_model_type, [&]<ModelType MODEL_TYPE, ModelType EXTRA_MODEL_TYPE>() {
            DISPATCH_BOOLEAN_FLAG(params.enable_split_kv, ENABLE_SPLIT_KV, ([&]() {
                STD_TORCH_CHECK(params.h_q == 64, "Unsupported h_q: ", params.h_q);
                using sm100::decode::sparse::head64::Config;
                sm100::decode::sparse::head64::run_flash_splitkv_mla_fp8_sparse_kernel<Config{MODEL_TYPE, EXTRA_MODEL_TYPE, ENABLE_SPLIT_KV}>(params);
            }));
        });
    }
};


// An implementation for d_qk = 512 head128 sparse decoding
class SparseDecode_Sm100_Head128_Impl : public SparseDecodeImplBase {
    DECLARE_SUPPORTED_FEATURES(
        SparseDecodeFeatures::HEAD_128,
        SparseDecodeFeatures::ATTN_SINK,
        SparseDecodeFeatures::TOPK_LENGTH,
        SparseDecodeFeatures::EXTRA_KVCACHE,
        SparseDecodeFeatures::EXTRA_TOPK_LENGTH,
        SparseDecodeFeatures::BATCH_INVARIANT
    )
    using SupportedKVFormats = KVFormatPairs<KVFormatPair<ModelType::V41>,
                                             KVFormatPair<ModelType::V41, ModelType::V41_FP4>>;

public:
    SparseDecodeImplMeta get_meta(int h_q, int s_q) override {
        Arch arch = Arch();
        return {
            std::max(arch.num_sms / s_q / 2, 1),
            3,  // TODO Tune
            64
        };
    }

protected:
    void run_(const SparseAttnDecodeParams &params, const std::vector<FeatureT> &required_features) override {
        SparseAttnDecodeParams hotfixed_params = params;
        if (params.s_q == 1 && params.b > 1) {
            // For this kernel, we require `params.stride_q_b % params.stride_q_s_q == 0`, since we "squeeze" the batch size dimention and the sequence length q dimension together during tensormap creation
            hotfixed_params.stride_q_s_q = hotfixed_params.stride_q_b;
        }
        dispatch_kv_formats(SupportedKVFormats{}, params.model_type, params.extra_model_type, [&]<ModelType MODEL_TYPE, ModelType EXTRA_MODEL_TYPE>() {
            DISPATCH_BOOLEAN_FLAG(params.enable_split_kv, ENABLE_SPLIT_KV, ([&]() {
                sm100::prefill::sparse_fwd_for_small_topk::head128::run_sparse_fwd_for_small_topk_phase1_kernel<get_decode_fwd_mode<ENABLE_SPLIT_KV>(), MODEL_TYPE, EXTRA_MODEL_TYPE>(hotfixed_params);
            }));
        });
    }
};

#endif  // FLASH_MLA_IS_BUILD_ON_CUDA

#ifdef FLASH_MLA_IS_BUILD_ON_ASCEND

#include "ascend_kernels/prefill/sparse/kernel.h"

class SparseDecodeImpl : public SparseDecodeImplBase {
    DECLARE_SUPPORTED_FEATURES(
        SparseDecodeFeatures::HEAD_64,
        SparseDecodeFeatures::ATTN_SINK,
        SparseDecodeFeatures::TOPK_LENGTH,
        SparseDecodeFeatures::EXTRA_KVCACHE,
        SparseDecodeFeatures::EXTRA_TOPK_LENGTH,
        SparseDecodeFeatures::BATCH_INVARIANT
    )
    using SupportedKVFormats = KVFormatPairs<
        KVFormatPair<ModelType::V41>, KVFormatPair<ModelType::V41, ModelType::V41_FP4>>;

public:
    SparseDecodeImplMeta get_meta(int h_q, int s_q) override {
        return {    // Return whatever we like since this impl doesn't support splitKV
            0,
            0,
            0
        };
    }

protected:
    void run_(const SparseAttnDecodeParams &params, const std::vector<FeatureT> &required_features) override {
        dispatch_kv_formats(SupportedKVFormats{}, params.model_type, params.extra_model_type, [&]<ModelType MODEL_TYPE, ModelType EXTRA_MODEL_TYPE>() {
            DISPATCH_BOOLEAN_FLAG(params.attn_sink != nullptr, HAVE_ATTN_SINK, ([&]() {
                static constexpr ascend::prefill::sparse_fwd::Config CONFIG = {
                    SparseAttnFwdMode::Decode,
                    64,
                    640,
                    HAVE_ATTN_SINK,
                    MODEL_TYPE,
                    EXTRA_MODEL_TYPE
                };
                ascend::prefill::sparse_fwd::run_sparse_fwd_kernel<CONFIG>(params);
            }));
        });
    }
};

#endif  // FLASH_MLA_IS_BUILD_ON_ASCEND


static std::tuple<Tensor, Tensor, std::optional<Tensor>, std::optional<Tensor>>
sparse_decode_fwd(
    const Tensor &q,   // [b, s_q, h_q, d_qk]
    const Tensor &kv,   // [num_blocks, page_block_size, h_k, bytes_per_token]
    const Tensor &indices,    // [b, s_q, topk]
    const std::optional<Tensor> &topk_length,   // [b]
    const std::optional<Tensor> &attn_sink, // [h_q]
    std::optional<Tensor> tile_scheduler_metadata,   // num_sm_parts x (DecodingSchedMetaSize/4)
    std::optional<Tensor> num_splits,                // batch_size + 1
    const std::optional<Tensor> &extra_kv,
    const std::optional<Tensor> &extra_indices,
    const std::optional<Tensor> &extra_topk_length,
    int64_t d_v,
    double sm_scale,
    bool enable_batch_invariant
) {
#ifdef FLASH_MLA_IS_BUILD_ON_CUDA
    // Check the architecture
    Arch arch = Arch();
    STD_TORCH_CHECK(arch.is_sm100f(), "Sparse Attention Decode Kernel (sparse_decode_fwd) is only supported on SM100f architectures.");
#endif
    KU_CHECK_NDIM(q, 4);
    KU_CHECK_NDIM(kv, 4);
    KU_CHECK_NDIM(indices, 3);

    int b = q.size(0);
    int s_q = q.size(1);
    int h_q = q.size(2);
    int d_qk = q.size(3);
    int num_blocks = kv.size(0);
    int page_block_size = kv.size(1);
    int h_kv = kv.size(2);
    int topk = indices.size(2);

    bool have_topk_length = topk_length.has_value();
    bool have_extra_kcache = extra_kv.has_value();
    bool have_extra_topk_length = extra_topk_length.has_value();
    bool have_attn_sink = attn_sink.has_value();

    int extra_num_blocks = 0, extra_page_block_size = 0, extra_topk = 0;
    if (have_extra_kcache) {
        extra_num_blocks = extra_kv->size(0);
        extra_page_block_size = extra_kv->size(1);
    }
    if (extra_indices.has_value()) {
        extra_topk = extra_indices->size(-1);
    }

    // Split-KV only pays off when a request has enough work
#ifdef FLASH_MLA_IS_BUILD_ON_ASCEND
    bool enable_split_kv = false;   // Ascend does not support split-KV
#else
    bool enable_split_kv = !enable_batch_invariant && !(topk + extra_topk <= 640);
#endif

    // metadata sanity check
    STD_TORCH_CHECK(b > 0);
    STD_TORCH_CHECK(s_q > 0);
    STD_TORCH_CHECK(h_q > 0);
    STD_TORCH_CHECK(h_kv == 1, "Currently only MQA (i.e. h_kv == 1) is supported for sparse decoding");
    STD_TORCH_CHECK(d_qk == 512, "Only head_size_k == 512 is supported for sparse decoding");
    STD_TORCH_CHECK(d_v == 512, "Only head_size_v == 512 is supported for sparse decoding");
    STD_TORCH_CHECK(topk > 0);

    if (have_extra_kcache) {
        STD_TORCH_CHECK(extra_indices.has_value(), "extra_indices_in_kvcache must be provided when extra_kcache is provided for sparse attention");
    } else {
        STD_TORCH_CHECK(!extra_indices.has_value(), "extra_indices_in_kvcache must not be provided when extra_k_cache is not provided");
        STD_TORCH_CHECK(!extra_topk_length.has_value(), "extra_topk_length must not be provided when extra_k_cache is not provided");
    }

    // Check device
    KU_CHECK_DEVICE(q);
    KU_CHECK_DEVICE(kv);
    KU_CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(topk_length);
    KU_CHECK_DEVICE(attn_sink);
    KU_CHECK_DEVICE(tile_scheduler_metadata);
    KU_CHECK_DEVICE(num_splits);
    KU_CHECK_DEVICE(extra_kv);
    KU_CHECK_DEVICE(extra_indices);
    KU_CHECK_DEVICE(extra_topk_length);

    // Check data type
    KU_CHECK_DTYPE(q, ScalarType::BFloat16);
    STD_TORCH_CHECK(kv.scalar_type() == ScalarType::Float8_e4m3fn || kv.scalar_type() == ScalarType::Char || kv.scalar_type() == ScalarType::Byte, "key must have dtype fp8_e4m3fn, int8 or uint8");
    if (extra_kv.has_value()) {
        STD_TORCH_CHECK(extra_kv->scalar_type() == ScalarType::Float8_e4m3fn || extra_kv->scalar_type() == ScalarType::Char || extra_kv->scalar_type() == ScalarType::Byte, "extra k cache must have dtype fp8_e4m3fn, int8 or uint8");
    }
    KU_CHECK_DTYPE(indices, ScalarType::Int);
    KU_CHECK_DTYPE(topk_length, ScalarType::Int);
    KU_CHECK_DTYPE(attn_sink, ScalarType::Float);
    KU_CHECK_DTYPE(tile_scheduler_metadata, ScalarType::Int);
    KU_CHECK_DTYPE(num_splits, ScalarType::Int);
    KU_CHECK_DTYPE(extra_indices, ScalarType::Int);
    KU_CHECK_DTYPE(extra_topk_length, ScalarType::Int);

    // Check layout
    KU_CHECK_LAST_DIM_CONTIGUOUS(q);
    KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    KU_CHECK_CONTIGUOUS(topk_length);
    KU_CHECK_CONTIGUOUS(attn_sink);

    KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
    KU_CHECK_CONTIGUOUS(num_splits);

    KU_CHECK_LAST_DIM_CONTIGUOUS(extra_kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(extra_indices);
    KU_CHECK_CONTIGUOUS(extra_topk_length);

    // Check shape
    KU_CHECK_SHAPE(q, b, s_q, h_q, d_qk);
    // The formats of `kv` and `extra_kv`
    ModelType model_type = detect_kv_cache_format_for_headdim_512(kv.size(3));
    ModelType extra_model_type = have_extra_kcache ? detect_kv_cache_format_for_headdim_512(extra_kv->size(3)) : model_type;
    STD_TORCH_CHECK(model_type != ModelType::V41_FP4, "The fp4 KV cache is only supported as extra_kv");
    STD_TORCH_CHECK(is_valid_kv_format_pair(model_type, extra_model_type), "invalid kv format pair, ", get_dynamic_enum_name(model_type), " and ", get_dynamic_enum_name(extra_model_type));
    KU_CHECK_SHAPE(kv, num_blocks, page_block_size, h_kv, kv_cache_bytes_per_token(model_type));
    KU_CHECK_SHAPE(extra_kv, extra_num_blocks, extra_page_block_size, h_kv, kv_cache_bytes_per_token(extra_model_type));
    STD_TORCH_CHECK(kv.stride(1) == kv_cache_bytes_per_token(model_type), "The whole block must be contiguous when is_fp8_cache is True for kv cache");
    if (have_extra_kcache) {
        STD_TORCH_CHECK(extra_kv->stride(1) == kv_cache_bytes_per_token(extra_model_type), "The whole block must be contiguous when is_fp8_cache is True for extra kv cache");
    }
    KU_CHECK_SHAPE(indices, b, s_q, topk);
    KU_CHECK_SHAPE(topk_length, b);
    KU_CHECK_SHAPE(attn_sink, h_q);
    KU_CHECK_SHAPE(extra_indices, b, s_q, extra_topk);
    KU_CHECK_SHAPE(extra_topk_length, b);

    torch::stable::accelerator::DeviceGuard device_guard(q.get_device_index());
    Tensor out = torch::stable::new_empty(q, {b, s_q, h_q, d_v});
    Tensor lse = torch::stable::new_empty(q, {b, s_q, h_q}, ScalarType::Float);

    std::vector<SparseDecodeFeatures> features;
    if (h_q == 64) {
        features.push_back(SparseDecodeFeatures::HEAD_64);
    } else if (h_q == 128) {
        features.push_back(SparseDecodeFeatures::HEAD_128);
    } else {
        STD_TORCH_CHECK(false, "Unsupported h_q: ", h_q);
    }
    if (have_attn_sink) {
        features.push_back(SparseDecodeFeatures::ATTN_SINK);
    }
    if (have_topk_length) {
        features.push_back(SparseDecodeFeatures::TOPK_LENGTH);
    }
    if (have_extra_kcache) {
        features.push_back(SparseDecodeFeatures::EXTRA_KVCACHE);
    }
    if (have_extra_topk_length) {
        features.push_back(SparseDecodeFeatures::EXTRA_TOPK_LENGTH);
    }
    if (enable_batch_invariant) {
        features.push_back(SparseDecodeFeatures::BATCH_INVARIANT);
    }

    SparseDecodeImplBase* impl;
#ifdef FLASH_MLA_IS_BUILD_ON_CUDA
    if (h_q == 64) {
        impl = new SparseDecode_Sm100_Head64_Impl();
    } else {
        impl = new SparseDecode_Sm100_Head128_Impl();
    }

    SparseDecodeImplMeta impl_meta = impl->get_meta(h_q, s_q);
#endif
#ifdef FLASH_MLA_IS_BUILD_ON_ASCEND
    impl = new SparseDecodeImpl();
#endif

    SparseAttnDecodeParams params = {
        b, s_q, h_q, h_kv, d_qk, static_cast<int>(d_v),
        static_cast<float>(sm_scale), static_cast<float>(sm_scale * LOG_2_E),
        num_blocks, page_block_size, topk,
        model_type, extra_model_type,

        (bf16*)q.data_ptr(),
        (bf16*)kv.data_ptr(),
        (int*)indices.data_ptr(),
        ku::get_optional_tensor_ptr<int>(topk_length),
        ku::get_optional_tensor_ptr<float>(attn_sink),
        (float*)lse.data_ptr(),
        (bf16*)out.data_ptr(),

        extra_num_blocks, extra_page_block_size, extra_topk,
        ku::get_optional_tensor_ptr<bf16>(extra_kv),
        ku::get_optional_tensor_ptr<int>(extra_indices),
        ku::get_optional_tensor_ptr<int>(extra_topk_length),

        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)), int64_stride_to_int(q.stride(2)),
        int64_stride_to_int(kv.stride(0)), int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),
        int64_stride_to_int(lse.stride(0)), int64_stride_to_int(lse.stride(1)),
        int64_stride_to_int(out.stride(0)), int64_stride_to_int(out.stride(1)), int64_stride_to_int(out.stride(2)),

        have_extra_kcache ? int64_stride_to_int(extra_kv->stride(0)) : 0,
        have_extra_kcache ? int64_stride_to_int(extra_kv->stride(1)) : 0,
        have_extra_kcache ? int64_stride_to_int(extra_indices->stride(0)) : 0,
        have_extra_kcache ? int64_stride_to_int(extra_indices->stride(1)) : 0,
        get_num_sms(),
        kerutils::get_current_stream<stream_t>(q),

        enable_split_kv,
    };

#ifdef FLASH_MLA_IS_BUILD_ON_CUDA
    Tensor o_accum, lse_accum;
    if (enable_split_kv) {
        // Get MLA metadata if necessary
        if (!tile_scheduler_metadata.has_value()) {
            tile_scheduler_metadata = torch::stable::new_empty(q, {impl_meta.num_sm_parts, sizeof(DecodingSchedMeta)/4}, ScalarType::Int);
            num_splits = torch::stable::new_empty(q, {b+1}, ScalarType::Int);
            KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
            KU_CHECK_CONTIGUOUS(num_splits);

            GetDecodeSchedMetaParams get_sched_meta_params = {
                b, s_q,
                impl_meta.block_size_topk,
                impl_meta.fixed_overhead_num_blocks,
                impl_meta.num_sm_parts,
                topk,
                extra_topk,
                ku::get_optional_tensor_ptr<int>(topk_length),
                ku::get_optional_tensor_ptr<int>(extra_topk_length),
                (DecodingSchedMeta*)tile_scheduler_metadata->data_ptr(),
                num_splits->mutable_data_ptr<int>(),
                kerutils::get_current_stream<stream_t>(q)
            };
            smxx::decode::run_get_decoding_sched_meta_kernel(get_sched_meta_params);
        }
        KU_CHECK_DEVICE(tile_scheduler_metadata);
        KU_CHECK_DEVICE(num_splits);
        KU_CHECK_DTYPE(tile_scheduler_metadata, ScalarType::Int);
        KU_CHECK_DTYPE(num_splits, ScalarType::Int);
        KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
        KU_CHECK_CONTIGUOUS(num_splits);
        KU_CHECK_SHAPE(tile_scheduler_metadata, impl_meta.num_sm_parts, sizeof(DecodingSchedMeta)/4);
        KU_CHECK_SHAPE(num_splits, b+1);
        // Stick the metadata pointers to `params`
        params.tile_scheduler_metadata_ptr = (DecodingSchedMeta*)tile_scheduler_metadata->data_ptr();
        params.num_splits_ptr = num_splits->mutable_data_ptr<int>();
        params.num_sm_parts = impl_meta.num_sm_parts;
        // Allocate intermediate buffers for split-KV
        const int total_num_splits = b + params.num_sm_parts;
        lse_accum = torch::stable::new_empty(q, {total_num_splits, s_q, h_q}, ScalarType::Float);
        o_accum = torch::stable::new_empty(q, {total_num_splits, s_q, h_q, d_v}, ScalarType::Float);
        KU_CHECK_CONTIGUOUS(lse_accum);
        KU_CHECK_CONTIGUOUS(o_accum);
        params.lse_accum = lse_accum.mutable_data_ptr<float>();
        params.o_accum = o_accum.mutable_data_ptr<float>();
        params.stride_lse_accum_split = int64_stride_to_int(lse_accum.stride(0));
        params.stride_lse_accum_s_q = int64_stride_to_int(lse_accum.stride(1));
        params.stride_o_accum_split = int64_stride_to_int(o_accum.stride(0));
        params.stride_o_accum_s_q = int64_stride_to_int(o_accum.stride(1));
        params.stride_o_accum_h_q = int64_stride_to_int(o_accum.stride(2));
    }
#endif

    impl->run(params, features);
    if (enable_split_kv) {
#ifdef FLASH_MLA_IS_BUILD_ON_CUDA
        smxx::decode::run_flash_mla_combine_kernel(params);
#endif
    }

    delete impl;

    Tensor transposed_lse = torch::stable::transpose(lse, 1, 2);
    return {out, transposed_lse, tile_scheduler_metadata, num_splits};
}

#ifdef FLASH_MLA_IS_BUILD_ON_CUDA
STABLE_TORCH_LIBRARY_IMPL(flash_mla, CUDA, m) {
    m.impl("sparse_decode_fwd", TORCH_BOX(&sparse_decode_fwd));
}
#endif

#ifdef FLASH_MLA_IS_BUILD_ON_ASCEND
STABLE_TORCH_LIBRARY_IMPL(flash_mla, PrivateUse1, m) {
    m.impl("sparse_decode_fwd", TORCH_BOX(&sparse_decode_fwd));
}
#endif
