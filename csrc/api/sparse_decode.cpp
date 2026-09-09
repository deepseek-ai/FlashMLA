#include "common.h"

#include "kernels/params.h"

#include "kernels/sm90/decode/sparse/splitkv_mla.h"
#include "kernels/sm100/decode/sparse/head64/kernel.h"
#include "kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/phase1.h"
#include "kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.h"
#include "kernels/smxx/decode/combine/combine.h"

template<bool ENABLE_SPLIT_KV>
static constexpr SparseAttnFwdMode get_decode_fwd_mode() {
    if constexpr (ENABLE_SPLIT_KV) {
        return SparseAttnFwdMode::DecodeWithSplitKV;
    } else {
        return SparseAttnFwdMode::Decode;
    }
}

// Feature set of sparse decoding kernels
enum class DecodeFeatures : int {
    HEAD_64,
    HEAD_128,

    HEAD_DIM_576,
    HEAD_DIM_512,

    V32_KVCACHE_FORMAT,
    V4_KVCACHE_FORMAT,
    V41_KVCACHE_FORMAT,
    V41_FP4_KVCACHE_FORMAT,

    ATTN_SINK,
    TOPK_LENGTH,
    EXTRA_KVCACHE,
    EXTRA_TOPK_LENGTH
};

struct DecodeImplMeta {
    int num_sm_parts;
    int fixed_overhead_num_blocks;
    int block_size_topk;
};

class DecodeImplBase : public ImplBase<
    SparseAttnDecodeParams,
    DecodeFeatures
> {
public:
    virtual DecodeImplMeta get_meta(int h_q, int s_q) = 0;
};

class Decode_Sm90_Impl : public DecodeImplBase {
    DECLARE_SUPPORTED_FEATURES(
        DecodeFeatures::HEAD_64,
        DecodeFeatures::HEAD_128,
        DecodeFeatures::HEAD_DIM_512,
        DecodeFeatures::HEAD_DIM_576,
        DecodeFeatures::V32_KVCACHE_FORMAT,
        DecodeFeatures::V4_KVCACHE_FORMAT,
        DecodeFeatures::ATTN_SINK,
        DecodeFeatures::TOPK_LENGTH,
        DecodeFeatures::EXTRA_KVCACHE,
        DecodeFeatures::EXTRA_TOPK_LENGTH
    )

public:
    DecodeImplMeta get_meta(int h_q, int s_q) override {
        Arch arch = Arch();
        return {
            std::max(arch.num_sms / s_q / (h_q/64), 1),
            5,
            64
        };
    }

protected:
    void run_(const SparseAttnDecodeParams &params, const std::vector<FeatureT> &required_features) override {
        DISPATCH_MODEL_TYPE(params.model_type, MODEL_TYPE, [&]() {
            DISPATCH_NUM_HEADS(params.h_q, NUM_HEADS, [&]() {
                sm90::decode::sparse::run_flash_splitkv_mla_fp8_sparse_kernel<MODEL_TYPE, NUM_HEADS>(params);
            });
        });
    }
};

class Decode_Sm100_Head64_Impl : public DecodeImplBase {
    DECLARE_SUPPORTED_FEATURES(
        DecodeFeatures::HEAD_64,
        DecodeFeatures::HEAD_DIM_512,
        DecodeFeatures::HEAD_DIM_576,
        DecodeFeatures::V32_KVCACHE_FORMAT,
        DecodeFeatures::V4_KVCACHE_FORMAT,
        DecodeFeatures::V41_KVCACHE_FORMAT,
        DecodeFeatures::V41_FP4_KVCACHE_FORMAT,
        DecodeFeatures::ATTN_SINK,
        DecodeFeatures::TOPK_LENGTH,
        DecodeFeatures::EXTRA_KVCACHE,
        DecodeFeatures::EXTRA_TOPK_LENGTH
    )
    using SupportedKVFormats = KVFormatPairs<KVFormatPair<ModelType::V32>, KVFormatPair<ModelType::V4>, KVFormatPair<ModelType::V41>,
                                             KVFormatPair<ModelType::V41, ModelType::V41_FP4>>;

public:
    DecodeImplMeta get_meta(int h_q, int s_q) override {
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
                TORCH_CHECK(params.h_q == 64, "Unsupported h_q: ", params.h_q);
                using sm100::decode::sparse::head64::Config;
                sm100::decode::sparse::head64::run_flash_splitkv_mla_fp8_sparse_kernel<Config{MODEL_TYPE, EXTRA_MODEL_TYPE, ENABLE_SPLIT_KV}>(params);
            }));
        });
    }
};


// An implementation that calls the head64 kernel twice to process head128
// Necessary for running V3.2 shape (i.e. h = 128, d_qk = 576) on SM100f
class Decode_Sm100_Head64x2_Impl : public DecodeImplBase {
    DECLARE_SUPPORTED_FEATURES(
        DecodeFeatures::HEAD_128,
        DecodeFeatures::HEAD_DIM_512,
        DecodeFeatures::HEAD_DIM_576,
        DecodeFeatures::V32_KVCACHE_FORMAT,
        DecodeFeatures::V4_KVCACHE_FORMAT,
        DecodeFeatures::ATTN_SINK,
        DecodeFeatures::TOPK_LENGTH,
        DecodeFeatures::EXTRA_KVCACHE,
        DecodeFeatures::EXTRA_TOPK_LENGTH
    )
    using SupportedKVFormats = KVFormatPairs<KVFormatPair<ModelType::V32>, KVFormatPair<ModelType::V4>>;

public:
    DecodeImplMeta get_meta(int h_q, int s_q) override {
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
                for (int start_head_idx = 0; start_head_idx < 128; start_head_idx += 64) {
                    SparseAttnDecodeParams cur_params = params;
                    cur_params.q += start_head_idx * params.stride_q_h_q;
                    if (cur_params.attn_sink) {
                        cur_params.attn_sink += start_head_idx;
                    }
                    cur_params.lse += start_head_idx;
                    cur_params.out += start_head_idx * params.stride_o_h_q;
                    if (cur_params.enable_split_kv) {
                        cur_params.lse_accum += start_head_idx;
                        cur_params.o_accum += start_head_idx * params.stride_o_accum_h_q;
                    }
                    cur_params.h_q = 64;
                    using sm100::decode::sparse::head64::Config;
                    sm100::decode::sparse::head64::run_flash_splitkv_mla_fp8_sparse_kernel<Config{MODEL_TYPE, EXTRA_MODEL_TYPE, ENABLE_SPLIT_KV}>(cur_params);
                }
            }));
        });
    }
};


class Decode_Sm100_Head128_Impl : public DecodeImplBase {
    DECLARE_SUPPORTED_FEATURES(
        DecodeFeatures::HEAD_128,
        DecodeFeatures::HEAD_DIM_512,
        DecodeFeatures::V4_KVCACHE_FORMAT,
        DecodeFeatures::V41_KVCACHE_FORMAT,
        DecodeFeatures::V41_FP4_KVCACHE_FORMAT,
        DecodeFeatures::ATTN_SINK,
        DecodeFeatures::TOPK_LENGTH,
        DecodeFeatures::EXTRA_KVCACHE,
        DecodeFeatures::EXTRA_TOPK_LENGTH
    )
    using SupportedKVFormats = KVFormatPairs<KVFormatPair<ModelType::V4>, KVFormatPair<ModelType::V41>,
                                             KVFormatPair<ModelType::V41, ModelType::V41_FP4>>;

public:
    DecodeImplMeta get_meta(int h_q, int s_q) override {
        Arch arch = Arch();
        return {
            std::max(arch.num_sms / s_q / 2, 1),
            3,
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
                sm100::prefill::sparse_fwd_for_small_topk::head128::run_sparse_fwd_for_small_topk_phase1_kernel<get_decode_fwd_mode<ENABLE_SPLIT_KV>(), 512, MODEL_TYPE, EXTRA_MODEL_TYPE>(hotfixed_params);
            }));
        });
    }
};


static std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>>
sparse_attn_decode_interface(
    const at::Tensor &q,   // [b, s_q, h_q, d_qk]
    const at::Tensor &kv,   // [num_blocks, page_block_size, h_k, d_qk]
    const at::Tensor &indices,    // [b, s_q, topk]
    const std::optional<at::Tensor> &topk_length,   // [b, s_q]
    const std::optional<at::Tensor> &attn_sink, // [h_q]
    std::optional<at::Tensor> &tile_scheduler_metadata,   // num_sm_parts x (DecodingSchedMetaSize/4)
    std::optional<at::Tensor> &num_splits,                // batch_size + 1
    const std::optional<at::Tensor> &extra_kv,
    const std::optional<at::Tensor> &extra_indices,
    const std::optional<at::Tensor> &extra_topk_length,
    int d_v,
    float sm_scale
) {
    using bf16 = cutlass::bfloat16_t;

    // Check the architecture
    Arch arch = Arch();

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

    // Split-KV only pays off when a request has enough work. The sm90 kernel always splits.
    bool enable_split_kv = arch.is_sm90a() || !(topk + extra_topk <= 640);

    // metadata sanity check
    TORCH_CHECK(b > 0);
    TORCH_CHECK(s_q > 0);
    TORCH_CHECK(h_q > 0);
    TORCH_CHECK(h_kv == 1, "Currently only MQA (i.e. h_kv == 1) is supported for sparse decoding");
    TORCH_CHECK(d_qk == 576 || d_qk == 512, "Only head_size_k == 576 or 512 is supported for sparse decoding");
    TORCH_CHECK(d_v == 512, "Only head_size_v == 512 is supported for sparse decoding");
    TORCH_CHECK(topk > 0);

    if (have_extra_kcache) {
        TORCH_CHECK(extra_indices.has_value(), "extra_indices_in_kvcache must be provided when extra_kcache is provided for sparse attention");
    } else {
        TORCH_CHECK(!extra_indices.has_value(), "extra_indices_in_kvcache must not be provided when extra_k_cache is not provided");
        TORCH_CHECK(!extra_topk_length.has_value(), "extra_topk_length must not be provided when extra_k_cache is not provided");
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
    KU_CHECK_DTYPE(q, torch::kBFloat16);
    TORCH_CHECK(kv.dtype() == torch::kFloat8_e4m3fn || kv.dtype() == torch::kInt8 || kv.dtype() == torch::kUInt8, "key must have dtype fp8_e4m3fn, int8 or uint8");
    if (extra_kv.has_value()) {
        TORCH_CHECK(extra_kv->dtype() == torch::kFloat8_e4m3fn || extra_kv->dtype() == torch::kInt8 || extra_kv->dtype() == torch::kUInt8, "extra k cache must have dtype fp8_e4m3fn, int8 or uint8");
    }
    KU_CHECK_DTYPE(indices, torch::kInt32);
    KU_CHECK_DTYPE(topk_length, torch::kInt32);
    KU_CHECK_DTYPE(attn_sink, torch::kFloat32);
    KU_CHECK_DTYPE(tile_scheduler_metadata, torch::kInt32);
    KU_CHECK_DTYPE(num_splits, torch::kInt32);
    KU_CHECK_DTYPE(extra_indices, torch::kInt32);
    KU_CHECK_DTYPE(extra_topk_length, torch::kInt32);
    
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
    ModelType model_type, extra_model_type;
    if (d_qk == 576 && d_v == 512) {
        model_type = extra_model_type = ModelType::V32;
    } else if (d_qk == 512 && d_v == 512) {
        model_type = detect_kv_cache_format_for_headdim_512(kv.size(3));
        extra_model_type = have_extra_kcache ? detect_kv_cache_format_for_headdim_512(extra_kv->size(3)) : model_type;
    } else {
        TORCH_CHECK(false, "Unsupported head sizes for is_fp8_kvcache == True");
    }
    TORCH_CHECK(model_type != ModelType::V41_FP4, "The fp4 KV cache is only supported as extra_kv");
    TORCH_CHECK(is_valid_kv_format_pair(model_type, extra_model_type), "invalid kv format pair, ", get_dynamic_enum_name(model_type), " and ", get_dynamic_enum_name(extra_model_type));
    KU_CHECK_SHAPE(kv, num_blocks, page_block_size, h_kv, kv_cache_bytes_per_token(model_type));
    KU_CHECK_SHAPE(extra_kv, extra_num_blocks, extra_page_block_size, h_kv, kv_cache_bytes_per_token(extra_model_type));
    TORCH_CHECK(kv.stride(1) == kv_cache_bytes_per_token(model_type), "The whole block must be contiguous when is_fp8_cache is True for kv cache");
    if (have_extra_kcache) {
        TORCH_CHECK(extra_kv->stride(1) == kv_cache_bytes_per_token(extra_model_type), "The whole block must be contiguous when is_fp8_cache is True for extra kv cache");
    }
    KU_CHECK_SHAPE(indices, b, s_q, topk);
    KU_CHECK_SHAPE(topk_length, b);
    KU_CHECK_SHAPE(attn_sink, h_q);
    KU_CHECK_SHAPE(extra_indices, b, s_q, extra_topk);
    KU_CHECK_SHAPE(extra_topk_length, b);

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto opts = q.options();

    at::Tensor out = torch::empty({b, s_q, h_q, d_v}, opts);
    at::Tensor lse = torch::empty({b, s_q, h_q}, opts.dtype(at::kFloat));

    std::vector<DecodeFeatures> features;
    if (h_q == 64) {
        features.push_back(DecodeFeatures::HEAD_64);
    } else if (h_q == 128) {
        features.push_back(DecodeFeatures::HEAD_128);
    } else {
        TORCH_CHECK(false, "Unsupported h_q: ", h_q);
    }
    if (d_qk == 576) {
        features.push_back(DecodeFeatures::HEAD_DIM_576);
    } else if (d_qk == 512) {
        features.push_back(DecodeFeatures::HEAD_DIM_512);
    } else {
        TORCH_CHECK(false, "Unsupported d_qk: ", d_qk);
    }
    if (have_attn_sink) {
        features.push_back(DecodeFeatures::ATTN_SINK);
    }
    if (have_topk_length) {
        features.push_back(DecodeFeatures::TOPK_LENGTH);
    }
    if (have_extra_kcache) {
        features.push_back(DecodeFeatures::EXTRA_KVCACHE);
    }
    if (have_extra_topk_length) {
        features.push_back(DecodeFeatures::EXTRA_TOPK_LENGTH);
    }
    for (ModelType mt : {model_type, extra_model_type}) {
        if (mt == ModelType::V32) {
            features.push_back(DecodeFeatures::V32_KVCACHE_FORMAT);
        } else if (mt == ModelType::V4) {
            features.push_back(DecodeFeatures::V4_KVCACHE_FORMAT);
        } else if (mt == ModelType::V41) {
            features.push_back(DecodeFeatures::V41_KVCACHE_FORMAT);
        } else if (mt == ModelType::V41_FP4) {
            features.push_back(DecodeFeatures::V41_FP4_KVCACHE_FORMAT);
        } else {
            TORCH_CHECK(false, "Unsupported model type: ", (int)mt);
        }
    }

    DecodeImplBase* impl;
    if (arch.is_sm100f()) {
        if (h_q == 64) {
            impl = new Decode_Sm100_Head64_Impl();
        } else if (h_q == 128) {
            if (d_qk == 576) {
                impl = new Decode_Sm100_Head64x2_Impl();
            } else if (d_qk == 512) {
                impl = new Decode_Sm100_Head128_Impl();
            } else {
                TORCH_CHECK(false, "Unsupported d_qk: ", d_qk);
            }
        } else {
            TORCH_CHECK(false, "Unsupported h_q: ", h_q);
        }
    } else if (arch.is_sm90a()) {
        impl = new Decode_Sm90_Impl();
    } else {
        TORCH_CHECK(false, "Unsupported architecture for sparse decode fwd");
    }

    DecodeImplMeta impl_meta = impl->get_meta(h_q, s_q);

    SparseAttnDecodeParams params = {
        b, s_q, h_q, h_kv, d_qk, d_v,
        sm_scale, sm_scale * LOG_2_E,
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
        at::cuda::getCurrentCUDAStream().stream(),

        enable_split_kv,
    };

    at::Tensor o_accum, lse_accum;
    if (enable_split_kv) {
        // Get MLA metadata if necessary
        if (!tile_scheduler_metadata.has_value()) {
            tile_scheduler_metadata = torch::empty({impl_meta.num_sm_parts, sizeof(DecodingSchedMeta)/4}, opts.dtype(torch::kInt32));
            num_splits = torch::empty({b+1}, opts.dtype(torch::kInt32));
            KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
            KU_CHECK_CONTIGUOUS(num_splits);

            GetDecodeSchedMetaParams get_sched_meta_params = {
                b, s_q,
                impl_meta.block_size_topk,
                impl_meta.fixed_overhead_num_blocks,
                topk,
                extra_topk,
                ku::get_optional_tensor_ptr<int>(topk_length),
                ku::get_optional_tensor_ptr<int>(extra_topk_length),
                nullptr,
                (DecodingSchedMeta*)tile_scheduler_metadata->data_ptr(),
                num_splits->data_ptr<int>(),
                impl_meta.num_sm_parts,
                at::cuda::getCurrentCUDAStream().stream()
            };
            smxx::decode::run_get_decoding_sched_meta_kernel(get_sched_meta_params);
        }
        KU_CHECK_DEVICE(tile_scheduler_metadata);
        KU_CHECK_DEVICE(num_splits);
        KU_CHECK_DTYPE(tile_scheduler_metadata, torch::kInt32);
        KU_CHECK_DTYPE(num_splits, torch::kInt32);
        KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
        KU_CHECK_CONTIGUOUS(num_splits);
        KU_CHECK_SHAPE(tile_scheduler_metadata, impl_meta.num_sm_parts, sizeof(DecodingSchedMeta)/4);
        KU_CHECK_SHAPE(num_splits, b+1);
        // Stick the metadata pointers to `params`
        params.tile_scheduler_metadata_ptr = (DecodingSchedMeta*)tile_scheduler_metadata->data_ptr();
        params.num_splits_ptr = num_splits->data_ptr<int>();
        params.num_sm_parts = impl_meta.num_sm_parts;
        // Allocate intermediate buffers for split-KV
        const int total_num_splits = b + params.num_sm_parts;
        lse_accum = torch::empty({total_num_splits, s_q, h_q}, opts.dtype(at::kFloat));
        o_accum = torch::empty({total_num_splits, s_q, h_q, d_v}, opts.dtype(at::kFloat));
        KU_CHECK_CONTIGUOUS(lse_accum);
        KU_CHECK_CONTIGUOUS(o_accum);
        params.lse_accum = lse_accum.data_ptr<float>();
        params.o_accum = o_accum.data_ptr<float>();
        params.stride_lse_accum_split = int64_stride_to_int(lse_accum.stride(0));
        params.stride_lse_accum_s_q = int64_stride_to_int(lse_accum.stride(1));
        params.stride_o_accum_split = int64_stride_to_int(o_accum.stride(0));
        params.stride_o_accum_s_q = int64_stride_to_int(o_accum.stride(1));
        params.stride_o_accum_h_q = int64_stride_to_int(o_accum.stride(2));
    }

    impl->run(params, features);
    if (enable_split_kv) {
        CombineParams combine_params = {
            b, s_q, h_q, d_v,

            params.lse,
            params.out,
            params.stride_lse_b, params.stride_lse_s_q,
            params.stride_o_b, params.stride_o_s_q, params.stride_o_h_q,

            params.lse_accum,
            params.o_accum,
            params.stride_lse_accum_split, params.stride_lse_accum_s_q,
            params.stride_o_accum_split, params.stride_o_accum_s_q, params.stride_o_accum_h_q,

            params.tile_scheduler_metadata_ptr,
            params.num_splits_ptr,
            params.num_sm_parts,

            ku::get_optional_tensor_ptr<float>(attn_sink),
            at::cuda::getCurrentCUDAStream().stream()
        };
        smxx::decode::run_flash_mla_combine_kernel<bf16>(combine_params);
    }

    delete impl;

    return {out, lse.transpose(1, 2), tile_scheduler_metadata, num_splits};
}

void register_sparse_decode(pybind11::module_& m) {
    m.def("sparse_decode_fwd",
        &sparse_attn_decode_interface,
        "Run Sparse Attention Decode Forward");
}
