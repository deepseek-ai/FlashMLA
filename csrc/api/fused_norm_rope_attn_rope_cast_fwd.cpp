#include "common.h"

#include "kernels/params.h"

#include "kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/kernel.h"
#include "kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/permute_q_b_proj/kernel.h"
#include "kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/permute_wv_proj/kernel.h"

// Local aliases: `kernels/defines.h` declares this type as `fp8`, and the fused kernel headers declare
// it inside their own namespace, so this translation unit needs the explicit name at file scope.
using bf16 = cutlass::bfloat16_t;
using fp8_e4m3 = cutlass::float_e4m3_t;

using Params = sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn::ParamT<SparseAttnFwdMode::Prefill>;
using DecodeParams = sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn::ParamT<SparseAttnFwdMode::Decode>;
using Config = sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn::Config;

static at::Tensor allocate_scale_factor(uint32_t batch_size, uint32_t hidden_dim, uint32_t scale_gran, const at::TensorOptions &opts, std::optional<uint32_t> extra_dim = std::nullopt) {
    // Allocate a scale factor tensor, which should have shape ([extra_dim], batch_size, hidden_dim / (scale_gran*4))
    // Meet DeepGEMM's SF requirement under use_tma_aligned_col_major_sf == True, round_sf == True, and use_packed_ue8m0 == True
    TORCH_CHECK(hidden_dim % (scale_gran*4) == 0);
    uint32_t sf_align_requirement = 16u / sizeof(int32_t);
    uint32_t aligned_batch_size_for_sf = (batch_size + sf_align_requirement - 1) / sf_align_requirement * sf_align_requirement;
    uint32_t leading_dim = extra_dim.value_or(1);
    at::Tensor sf = torch::empty({leading_dim, hidden_dim / (scale_gran * 4), aligned_batch_size_for_sf}, opts.dtype(torch::kInt32));
    KU_CHECK_CONTIGUOUS(sf);
    sf = sf.transpose(1, 2).slice(1, 0, batch_size);   // [leading_dim, batch_size, hidden_dim / (scale_gran*4)], int32
    if (!extra_dim.has_value())
        sf = sf.squeeze(0);
    return sf;
}

static std::vector<at::Tensor> fused_norm_rope_attn_rope_cast_fwd(
    const at::Tensor &q,
    const at::Tensor &kv,
    const at::Tensor &indices,
    float sm_scale,
    int d_v,    // TODO change int to uint32_t
    const std::optional<at::Tensor> &attn_sink,
    const std::optional<at::Tensor> &topk_length,
    bool enable_q_norm,
    float rms_norm_eps,
    const at::Tensor &token_positions,
    bool is_rope_neox_style,
    uint32_t rope_dim,
    const at::Tensor &cos_sin_cache,

    uint32_t n_wv_group,
    uint32_t num_per_channels,
    bool use_tma_aligned_col_major_sf,
    bool round_sf,
    bool use_packed_ue8m0
) {
    Arch arch = Arch();
    bool is_sm100f = arch.is_sm100f();
    TORCH_CHECK(is_sm100f, "Fused Norm + RoPE + Core Attn + RoPE + Cast (fused_norm_rope_attn_rope_cast_fwd) is only supported on SM100f architectures.");

    KU_CHECK_NDIM(q, 3);
    KU_CHECK_NDIM(kv, 3);
    KU_CHECK_NDIM(indices, 3);
    KU_CHECK_NDIM(attn_sink, 1);
    KU_CHECK_NDIM(topk_length, 1);
    KU_CHECK_NDIM(token_positions, 1);
    KU_CHECK_NDIM(cos_sin_cache, 2);

    int s_q = q.size(0);
    int s_kv = kv.size(0);
    int h_q = q.size(1);
    int h_kv = kv.size(1);
    int d_qk = q.size(2);
    int topk = indices.size(2);
    uint32_t wv_group_size = h_q / n_wv_group;

    TORCH_CHECK(h_q % n_wv_group == 0, "h_q %% n_wv_group != 0");
    TORCH_CHECK(is_rope_neox_style == false, "Only `is_rope_neox_style == False` is supported");
    TORCH_CHECK(use_tma_aligned_col_major_sf == true, "`use_tma_aligned_col_major_sf` must be True");
    TORCH_CHECK(round_sf == true, "`round_sf` must be True");
    TORCH_CHECK(use_packed_ue8m0 == true, "`use_packed_ue8m0` must be True");

    KU_CHECK_DEVICE(q);
    KU_CHECK_DEVICE(kv);
    KU_CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(attn_sink);
    KU_CHECK_DEVICE(topk_length);
    KU_CHECK_DEVICE(token_positions);
    KU_CHECK_DEVICE(cos_sin_cache);
    
    KU_CHECK_DTYPE(q, torch::kBFloat16);
    KU_CHECK_DTYPE(kv, torch::kBFloat16);
    KU_CHECK_DTYPE(indices, torch::kInt32);
    KU_CHECK_DTYPE(attn_sink, torch::kFloat32);
    KU_CHECK_DTYPE(topk_length, torch::kInt32);
    KU_CHECK_DTYPE(token_positions, torch::kInt32);
    KU_CHECK_DTYPE(cos_sin_cache, torch::kFloat32);
    
    KU_CHECK_SHAPE(q, s_q, h_q, d_qk);
    KU_CHECK_SHAPE(kv, s_kv, h_kv, d_qk);
    KU_CHECK_SHAPE(indices, s_q, h_kv, topk);
    KU_CHECK_SHAPE(attn_sink, h_q);
    KU_CHECK_SHAPE(topk_length, s_q);
    KU_CHECK_SHAPE(token_positions, s_q);
    KU_CHECK_SHAPE(cos_sin_cache, cos_sin_cache.size(0), rope_dim);
    
    KU_CHECK_LAST_DIM_CONTIGUOUS(q);
    // q is in the permuted layout (see permute_q_b_proj), so the kernel assumes that the h_q*d_qk elements of one token are contiguous (only q.stride(0) is used by the kernel)
    TORCH_CHECK(q.stride(1) == d_qk, "q must be contiguous within each token (i.e. q.stride(1) == d_qk), since q is in the permuted layout, got q.stride(1) = ", q.stride(1));
    KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    KU_CHECK_LAST_DIM_CONTIGUOUS(attn_sink);
    KU_CHECK_CONTIGUOUS(topk_length);
    KU_CHECK_CONTIGUOUS(token_positions);
    KU_CHECK_CONTIGUOUS(cos_sin_cache);

    TORCH_CHECK(num_per_channels == 32, "num_per_channels must be 32, got ", num_per_channels);

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto opts = q.options();
    
    TORCH_CHECK(d_v % (num_per_channels * 4) == 0); // 4 is the number of uint8 in uint32, since `use_packed_ue8m0` is `True`
    at::Tensor out_fp8 = torch::empty({s_q, n_wv_group, wv_group_size * d_v}, opts.dtype(torch::kFloat8_e4m3fn));
    uint32_t out_sf_scale_gran = 32; // Since the weight is per-32 scaled and deep_gemm.einsum requires A and B to have the same scale granularity, the output sf is always stored in a per-32 scaled format, although it will be actually per-128 scaled when num_per_channels is 128
    at::Tensor out_sf = allocate_scale_factor(s_q, wv_group_size * d_v, out_sf_scale_gran, opts, n_wv_group).transpose(0, 1);   // [s_q, n_wv_group, wv_group_size * d_v / (out_sf_scale_gran*4)]
    at::Tensor max_logits = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat));
    at::Tensor lse = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat));
    KU_CHECK_CONTIGUOUS(out_fp8);
    TORCH_CHECK(out_sf.stride(0) == 1);
    KU_CHECK_CONTIGUOUS(max_logits);
    KU_CHECK_CONTIGUOUS(lse);

    Params params = {
        s_q, s_kv, h_q, h_kv, d_qk, d_v, topk,
        sm_scale, sm_scale * LOG_2_E,

        (bf16*)q.data_ptr(),
        (bf16*)kv.data_ptr(),
        (int*)indices.data_ptr(),
        ku::get_optional_tensor_ptr<float>(attn_sink),
        ku::get_optional_tensor_ptr<int>(topk_length),

        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)),
        int64_stride_to_int(kv.stride(0)), int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),

        nullptr,
        (float*)max_logits.data_ptr(),
        (float*)lse.data_ptr(),

        arch.num_sms,
        at::cuda::getCurrentCUDAStream().stream(),

        enable_q_norm,
        rms_norm_eps,
        (uint32_t*)token_positions.data_ptr(),
        is_rope_neox_style,
        rope_dim,
        (float*)cos_sin_cache.data_ptr(),

        n_wv_group,
        wv_group_size,
        num_per_channels,
        use_tma_aligned_col_major_sf,
        round_sf,
        use_packed_ue8m0,

        (fp8_e4m3*)out_fp8.data_ptr(),
        (uint32_t*)out_sf.data_ptr(),
        (uint32_t)int64_stride_to_int(out_sf.stride(1)),
        (uint32_t)int64_stride_to_int(out_sf.stride(2))
    };

    TORCH_CHECK(h_q == 64 || h_q == 128, "Only h_q == 64 or 128 is supported for fused_norm_rope_attn_rope_cast_fwd, got ", h_q);
    DISPATCH_NUM_HEADS(h_q, H_Q, ([&]() {
        DISPATCH_BOOLEAN_FLAG(enable_q_norm, ENABLE_Q_NORM, ([&]() {
            sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn::run_fused_norm_rope_attn_rope_cast_fwd_kernel<Config{SparseAttnFwdMode::Prefill, ModelType::V4, ModelType::V4, H_Q, ENABLE_Q_NORM}>(params);
        }));
    }));

    return {out_fp8, out_sf, max_logits, lse};
}


static std::vector<at::Tensor> fused_norm_rope_attn_rope_cast_decode(
    const at::Tensor &q,        // [s_q, h_q, d_qk]
    const at::Tensor &kv,       // [num_blocks, page_block_size, h_kv, bytes_per_token], paged quantized KV cache
    const at::Tensor &indices,  // [s_q, topk]
    float sm_scale,
    int d_v,
    const std::optional<at::Tensor> &attn_sink,         // [h_q]
    const std::optional<at::Tensor> &topk_length,       // [s_q]
    const std::optional<at::Tensor> &extra_kv,          // [extra_num_blocks, extra_page_block_size, h_kv, bytes_per_token]
    const std::optional<at::Tensor> &extra_indices,     // [s_q, extra_topk]
    const std::optional<at::Tensor> &extra_topk_length, // [s_q]
    bool enable_q_norm,
    float rms_norm_eps,
    const at::Tensor &token_positions,  // [s_q]
    bool is_rope_neox_style,
    uint32_t rope_dim,
    const at::Tensor &cos_sin_cache,    // [*, rope_dim]

    uint32_t n_wv_group,
    uint32_t num_per_channels,
    bool use_tma_aligned_col_major_sf,
    bool round_sf,
    bool use_packed_ue8m0
) {
    Arch arch = Arch();
    TORCH_CHECK(arch.is_sm100f(), "Fused Norm + RoPE + Core Attn + RoPE + Cast (fused_norm_rope_attn_rope_cast_decode) is only supported on SM100f architectures.");

    KU_CHECK_NDIM(q, 3);
    KU_CHECK_NDIM(kv, 4);
    KU_CHECK_NDIM(indices, 2);
    KU_CHECK_NDIM(attn_sink, 1);
    KU_CHECK_NDIM(topk_length, 1);
    KU_CHECK_NDIM(extra_kv, 4);
    KU_CHECK_NDIM(extra_indices, 2);
    KU_CHECK_NDIM(extra_topk_length, 1);
    KU_CHECK_NDIM(token_positions, 1);
    KU_CHECK_NDIM(cos_sin_cache, 2);

    int s_q = q.size(0);
    int h_q = q.size(1);
    int d_qk = q.size(2);
    int num_blocks = kv.size(0);
    int page_block_size = kv.size(1);
    int h_kv = kv.size(2);
    int topk = indices.size(1);

    bool have_extra_kvcache = extra_kv.has_value();

    int extra_num_blocks = 0, extra_page_block_size = 0, extra_topk = 0;
    if (have_extra_kvcache) {
        extra_num_blocks = extra_kv->size(0);
        extra_page_block_size = extra_kv->size(1);
    }
    if (extra_indices.has_value()) {
        extra_topk = extra_indices->size(-1);
    }

    // Metadata sanity check
    TORCH_CHECK(s_q > 0);
    TORCH_CHECK(h_q == 64 || h_q == 128, "Only h_q == 64 or 128 is supported for fused_norm_rope_attn_rope_cast_decode, got ", h_q);
    TORCH_CHECK(h_kv == 1, "Currently only MQA (i.e. h_kv == 1) is supported");
    TORCH_CHECK(d_qk == 512, "Only head_size_k == 512 (V4 / V4.1) is supported");
    TORCH_CHECK(d_v == 512, "Only head_size_v == 512 is supported");
    TORCH_CHECK(topk > 0);
    TORCH_CHECK(h_q % n_wv_group == 0, "h_q %% n_wv_group != 0");
    uint32_t wv_group_size = h_q / n_wv_group;

    if (have_extra_kvcache) {
        TORCH_CHECK(extra_indices.has_value(), "extra_indices must be provided when extra_kv is provided");
    } else {
        TORCH_CHECK(!extra_indices.has_value(), "extra_indices must not be provided when extra_kv is not provided");
        TORCH_CHECK(!extra_topk_length.has_value(), "extra_topk_length must not be provided when extra_kv is not provided");
    }

    TORCH_CHECK(is_rope_neox_style == false, "Only `is_rope_neox_style == False` is supported");
    TORCH_CHECK(use_tma_aligned_col_major_sf == true, "`use_tma_aligned_col_major_sf` must be True");
    TORCH_CHECK(round_sf == true, "`round_sf` must be True");
    TORCH_CHECK(use_packed_ue8m0 == true, "`use_packed_ue8m0` must be True");

    // Check device
    KU_CHECK_DEVICE(q);
    KU_CHECK_DEVICE(kv);
    KU_CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(attn_sink);
    KU_CHECK_DEVICE(topk_length);
    KU_CHECK_DEVICE(extra_kv);
    KU_CHECK_DEVICE(extra_indices);
    KU_CHECK_DEVICE(extra_topk_length);
    KU_CHECK_DEVICE(token_positions);
    KU_CHECK_DEVICE(cos_sin_cache);

    // Check data type
    KU_CHECK_DTYPE(q, torch::kBFloat16);
    TORCH_CHECK(kv.dtype() == torch::kFloat8_e4m3fn || kv.dtype() == torch::kInt8 || kv.dtype() == torch::kUInt8, "kv must have dtype fp8_e4m3fn, int8 or uint8");
    if (have_extra_kvcache) {
        TORCH_CHECK(extra_kv->dtype() == torch::kFloat8_e4m3fn || extra_kv->dtype() == torch::kInt8 || extra_kv->dtype() == torch::kUInt8, "extra_kv must have dtype fp8_e4m3fn, int8 or uint8");
    }
    KU_CHECK_DTYPE(indices, torch::kInt32);
    KU_CHECK_DTYPE(attn_sink, torch::kFloat32);
    KU_CHECK_DTYPE(topk_length, torch::kInt32);
    KU_CHECK_DTYPE(extra_indices, torch::kInt32);
    KU_CHECK_DTYPE(extra_topk_length, torch::kInt32);
    KU_CHECK_DTYPE(token_positions, torch::kInt32);
    KU_CHECK_DTYPE(cos_sin_cache, torch::kFloat32);

    // Check layout
    KU_CHECK_LAST_DIM_CONTIGUOUS(q);
    // q is in the permuted layout (see permute_q_b_proj), so the kernel assumes that the h_q*d_qk elements of one token are contiguous (only q.stride(0) is used by the kernel)
    TORCH_CHECK(q.stride(1) == d_qk, "q must be contiguous within each token (i.e. q.stride(1) == d_qk), since q is in the permuted layout, got q.stride(1) = ", q.stride(1));
    KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    KU_CHECK_CONTIGUOUS(attn_sink);
    KU_CHECK_CONTIGUOUS(topk_length);
    KU_CHECK_LAST_DIM_CONTIGUOUS(extra_kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(extra_indices);
    KU_CHECK_CONTIGUOUS(extra_topk_length);
    KU_CHECK_CONTIGUOUS(token_positions);
    KU_CHECK_CONTIGUOUS(cos_sin_cache);

    // The formats of `kv` and `extra_kv` (V4 / V4.1 / V4.1 fp4, see KVCacheFormat), detected by bytes_per_token
    ModelType model_type = detect_kv_cache_format_for_headdim_512(kv.size(3));
    ModelType extra_model_type = have_extra_kvcache ? detect_kv_cache_format_for_headdim_512(extra_kv->size(3)) : model_type;
    TORCH_CHECK(model_type != ModelType::V41_FP4, "The fp4 KV cache is only supported as extra_kv");
    TORCH_CHECK(is_valid_kv_format_pair(model_type, extra_model_type), "extra_kv must have the format of kv, or the V4.1 fp4 format when kv has the V4.1 format, got ",
        get_dynamic_enum_name(model_type), " and ", get_dynamic_enum_name(extra_model_type));
    KU_CHECK_SHAPE(kv, num_blocks, page_block_size, h_kv, kv_cache_bytes_per_token(model_type));
    KU_CHECK_SHAPE(extra_kv, extra_num_blocks, extra_page_block_size, h_kv, kv_cache_bytes_per_token(extra_model_type));
    TORCH_CHECK(kv.stride(1) == kv_cache_bytes_per_token(model_type), "The whole block must be contiguous for the paged KV cache");
    if (have_extra_kvcache) {
        TORCH_CHECK(extra_kv->stride(1) == kv_cache_bytes_per_token(extra_model_type), "The whole block must be contiguous for the paged extra KV cache");
    }
    TORCH_CHECK(num_per_channels == 32, "num_per_channels must be 32, got ", num_per_channels);

    // Check shape
    KU_CHECK_SHAPE(q, s_q, h_q, d_qk);
    KU_CHECK_SHAPE(indices, s_q, topk);
    KU_CHECK_SHAPE(attn_sink, h_q);
    KU_CHECK_SHAPE(topk_length, s_q);
    KU_CHECK_SHAPE(extra_indices, s_q, extra_topk);
    KU_CHECK_SHAPE(extra_topk_length, s_q);
    KU_CHECK_SHAPE(token_positions, s_q);
    KU_CHECK_SHAPE(cos_sin_cache, cos_sin_cache.size(0), rope_dim);

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto opts = q.options();

    TORCH_CHECK(d_v % (num_per_channels * 4) == 0); // 4 is the number of uint8 in uint32, since `use_packed_ue8m0` is `True`
    at::Tensor out_fp8 = torch::empty({s_q, n_wv_group, wv_group_size * d_v}, opts.dtype(torch::kFloat8_e4m3fn));
    uint32_t out_sf_scale_gran = 32; // Since the weight is per-32 scaled and deep_gemm.einsum requires A and B to have the same scale granularity, the output sf is always stored in a per-32 scaled format, although it will be actually per-128 scaled when num_per_channels is 128
    at::Tensor out_sf = allocate_scale_factor(s_q, wv_group_size * d_v, out_sf_scale_gran, opts, n_wv_group).transpose(0, 1);   // [s_q, n_wv_group, wv_group_size * d_v / (out_sf_scale_gran*4)]
    at::Tensor lse = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat));
    KU_CHECK_CONTIGUOUS(out_fp8);
    TORCH_CHECK(out_sf.stride(0) == 1);
    KU_CHECK_CONTIGUOUS(lse);

    SparseAttnDecodeParams base_params = {
        1 /* b */, s_q, h_q, h_kv, d_qk, d_v,
        sm_scale, sm_scale * LOG_2_E,
        num_blocks, page_block_size, topk,
        model_type, extra_model_type,

        (bf16*)q.data_ptr(),
        (bf16*)kv.data_ptr(),
        (int*)indices.data_ptr(),
        ku::get_optional_tensor_ptr<int>(topk_length),
        ku::get_optional_tensor_ptr<float>(attn_sink),
        (float*)lse.data_ptr(),
        nullptr,    // `out` (bf16) is unused; the FP8 output goes to `out_fp8` + `out_sf`

        extra_num_blocks, extra_page_block_size, extra_topk,
        ku::get_optional_tensor_ptr<bf16>(extra_kv),
        ku::get_optional_tensor_ptr<int>(extra_indices),
        ku::get_optional_tensor_ptr<int>(extra_topk_length),

        0, int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)),  // stride_q_b is unused since b == 1
        int64_stride_to_int(kv.stride(0)), int64_stride_to_int(kv.stride(1)),
        0, int64_stride_to_int(indices.stride(0)),      // stride_indices_b is unused since b == 1
        0, int64_stride_to_int(lse.stride(0)),          // stride_lse_b is unused since b == 1
        0, 0, 0,    // stride_o_b, stride_o_s_q, stride_o_h_q: unused since `out` is unused

        have_extra_kvcache ? int64_stride_to_int(extra_kv->stride(0)) : 0,
        have_extra_kvcache ? int64_stride_to_int(extra_kv->stride(1)) : 0,
        0,                                              // stride_extra_indices_b is unused since b == 1
        have_extra_kvcache ? int64_stride_to_int(extra_indices->stride(0)) : 0,
        at::cuda::getCurrentCUDAStream().stream(),

        false,      // enable_split_kv: split-KV is not supported by this kernel
        // The remaining split-KV related fields are zero-initialized
    };

    DecodeParams params = {
        base_params,

        enable_q_norm,
        rms_norm_eps,
        (uint32_t*)token_positions.data_ptr(),
        is_rope_neox_style,
        rope_dim,
        (float*)cos_sin_cache.data_ptr(),

        n_wv_group,
        wv_group_size,
        num_per_channels,
        use_tma_aligned_col_major_sf,
        round_sf,
        use_packed_ue8m0,

        (fp8_e4m3*)out_fp8.data_ptr(),
        (uint32_t*)out_sf.data_ptr(),
        (uint32_t)int64_stride_to_int(out_sf.stride(1)),
        (uint32_t)int64_stride_to_int(out_sf.stride(2))
    };

    DISPATCH_NUM_HEADS(h_q, H_Q, ([&]() {
        DISPATCH_BOOLEAN_FLAG(enable_q_norm, ENABLE_Q_NORM, ([&]() {
            if (extra_model_type == ModelType::V41_FP4) {
                sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn::run_fused_norm_rope_attn_rope_cast_fwd_kernel<Config{SparseAttnFwdMode::Decode, ModelType::V41, ModelType::V41_FP4, H_Q, ENABLE_Q_NORM}>(params);
            } else if (model_type == ModelType::V4) {
                sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn::run_fused_norm_rope_attn_rope_cast_fwd_kernel<Config{SparseAttnFwdMode::Decode, ModelType::V4, ModelType::V4, H_Q, ENABLE_Q_NORM}>(params);
            } else if (model_type == ModelType::V41) {
                sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn::run_fused_norm_rope_attn_rope_cast_fwd_kernel<Config{SparseAttnFwdMode::Decode, ModelType::V41, ModelType::V41, H_Q, ENABLE_Q_NORM}>(params);
            } else {
                TORCH_CHECK(false, "Unsupported model_type: ", get_dynamic_enum_name(model_type));
            }
        }));
    }));

    return {out_fp8, out_sf, lse};
}


static std::vector<at::Tensor> permute_q_b_proj(
    const at::Tensor &q_b_proj,
    const at::Tensor &scale_factors,
    int h_q,
    int d_q
) {
    KU_CHECK_NDIM(q_b_proj, 2);
    KU_CHECK_NDIM(scale_factors, 2);

    int h_q_d_q = h_q * d_q;
    int q_lora_rank = q_b_proj.size(1);

    int gran = q_lora_rank / (4 * scale_factors.size(1));
    TORCH_CHECK(gran == 32 || gran == 128, "gran must be 32 or 128, got ", gran);
    TORCH_CHECK(q_lora_rank % (gran * 4) == 0, "q_lora_rank must be divisible by gran * 4");

    KU_CHECK_DEVICE(q_b_proj);
    KU_CHECK_DEVICE(scale_factors);

    KU_CHECK_DTYPE(q_b_proj, torch::kFloat8_e4m3fn);
    KU_CHECK_DTYPE(scale_factors, torch::kInt32);

    KU_CHECK_SHAPE(q_b_proj, h_q_d_q, q_lora_rank);
    KU_CHECK_SHAPE(scale_factors, h_q_d_q, q_lora_rank / gran / 4);

    KU_CHECK_LAST_DIM_CONTIGUOUS(q_b_proj);
    TORCH_CHECK(scale_factors.stride(0) == 1, "scale_factors must be contiguous on the first dimension");

    at::cuda::CUDAGuard device_guard{(char)q_b_proj.get_device()};

    at::Tensor q_b_proj_permuted = torch::empty_like(q_b_proj);
    at::Tensor scale_factors_permuted = allocate_scale_factor(h_q_d_q, q_lora_rank, gran, q_b_proj.options());
    KU_CHECK_CONTIGUOUS(q_b_proj_permuted);

    sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::permute_q_b_proj::Params params = {
        (uint32_t)h_q,
        (uint32_t)d_q,
        (uint32_t)q_lora_rank,
        (uint32_t)gran,

        (fp8_e4m3*)q_b_proj.data_ptr(),
        (uint64_t)q_b_proj.stride(0),
        (int32_t*)scale_factors.data_ptr(),
        (uint64_t)scale_factors.stride(1),

        (fp8_e4m3*)q_b_proj_permuted.data_ptr(),
        (uint64_t)q_b_proj_permuted.stride(0),
        (int32_t*)scale_factors_permuted.data_ptr(),
        (uint64_t)scale_factors_permuted.stride(1),
        
        at::cuda::getCurrentCUDAStream().stream(),
    };

    sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::permute_q_b_proj::run_permute_q_b_proj_kernel(params);

    return {q_b_proj_permuted, scale_factors_permuted};
}


static std::vector<at::Tensor> permute_wv_proj(
    const at::Tensor &wv_proj,
    const at::Tensor &scale_factors,
    int wv_group_size,
    int d_o
) {
    KU_CHECK_NDIM(wv_proj, 3);
    KU_CHECK_NDIM(scale_factors, 3);

    KU_CHECK_DEVICE(wv_proj);
    KU_CHECK_DEVICE(scale_factors);
    
    KU_CHECK_DTYPE(wv_proj, torch::kFloat8_e4m3fn);
    KU_CHECK_DTYPE(scale_factors, torch::kInt32);
    
    int n_wv_group = wv_proj.size(0);
    int d_proj_out = wv_proj.size(1);
    KU_CHECK_SHAPE(wv_proj, n_wv_group, d_proj_out, wv_group_size * d_o);

    int input_gran = wv_group_size * d_o / (4 * scale_factors.size(2));
    int output_gran = 32;   // Fixed to 32, otherwise permution between chunk (which has 32 elements) will be impossible
    TORCH_CHECK(input_gran == 32, "input scale granularity must be 32, got ", input_gran);
    TORCH_CHECK((wv_group_size * d_o) % (input_gran * 4) == 0, "q_lora_rank must be divisible by gran * 4");
    KU_CHECK_SHAPE(scale_factors, n_wv_group, d_proj_out, (wv_group_size * d_o) / input_gran / 4);

    KU_CHECK_LAST_DIM_CONTIGUOUS(wv_proj);
    TORCH_CHECK(scale_factors.stride(1) == 1, "scale_factors must be contiguous on the second dimension");

    at::cuda::CUDAGuard device_guard{(char)wv_proj.get_device()};

    at::Tensor wv_proj_permuted = torch::empty_like(wv_proj);
    at::Tensor scale_factors_permuted = allocate_scale_factor(d_proj_out, wv_group_size * d_o, output_gran, wv_proj.options(), n_wv_group);
    KU_CHECK_CONTIGUOUS(wv_proj_permuted);

    sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::permute_wv_proj::Params params = {
        (uint32_t)d_o,
        (uint32_t)input_gran,
        (uint32_t)wv_group_size,
        (uint32_t)n_wv_group,
        (uint32_t)d_proj_out,

        (fp8_e4m3*)wv_proj.data_ptr(),
        (uint64_t)wv_proj.stride(0),
        (uint64_t)wv_proj.stride(1),
        (int32_t*)scale_factors.data_ptr(),
        (uint64_t)scale_factors.stride(0),
        (uint64_t)scale_factors.stride(2),

        (fp8_e4m3*)wv_proj_permuted.data_ptr(),
        (uint64_t)wv_proj_permuted.stride(0),
        (uint64_t)wv_proj_permuted.stride(1),
        (int32_t*)scale_factors_permuted.data_ptr(),
        (uint64_t)scale_factors_permuted.stride(0),
        (uint64_t)scale_factors_permuted.stride(2),
        
        at::cuda::getCurrentCUDAStream().stream(),
    };

    sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::permute_wv_proj::run_permute_wv_proj_kernel(params);

    return {wv_proj_permuted, scale_factors_permuted};
}


void register_fused_norm_rope_attn_rope_cast_fwd(pybind11::module_& m) {
    m.def("fused_norm_rope_attn_rope_cast_fwd", 
        &fused_norm_rope_attn_rope_cast_fwd, 
        "Run Fused Norm + RoPE + Core Attn + RoPE + Cast");
    m.def("fused_norm_rope_attn_rope_cast_decode",
        &fused_norm_rope_attn_rope_cast_decode,
        "Run Fused Norm + RoPE + Core Attn + RoPE + Cast (Decoding, with paged quantized KV cache)");
    m.def("permute_q_b_proj", &permute_q_b_proj,
        "Permute q_b_proj weight layout for fused norm rope attn rope cast fwd");
    m.def("permute_wv_proj", &permute_wv_proj,
        "Permute wv_proj weight layout for fused norm rope attn rope cast fwd");
}
