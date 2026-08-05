#pragma once

#include <optional>
#include <tuple>
#include <vector>

#include "api/common.h"
#include "persistent_sparse_attention/params.h"
#include "persistent_sparse_attention/fwd_for_small_topk/head128/phase1.h"
#include "persistent_sparse_attention/psa_fwd/sparse_attention.h"

namespace persistent_sparse_attention {

inline std::vector<at::Tensor> sparse_prefill_fwd(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    float sm_scale,
    int64_t d_v,
    const std::optional<at::Tensor>& attn_sink,
    const std::optional<at::Tensor>& topk_length,
    const std::optional<at::Tensor>& out_,
    const std::optional<int64_t>& num_sms_override,
    const std::optional<at::Tensor>& q_ready,
    int64_t q_ready_chunk_size,
    const std::optional<at::Tensor>& q_positions,
    const std::optional<at::Tensor>& q_cos_sin_cache,
    double q_norm_eps,
    const std::optional<at::Tensor>& o_ready,
    const std::optional<at::Tensor>& fused_o_fp8,
    const std::optional<at::Tensor>& fused_o_scale,
    const std::optional<at::Tensor>& fused_o_positions,
    bool fused_o_skip_bf16,
    const std::optional<at::Tensor>& extra_kv,
    const std::optional<at::Tensor>& extra_indices,
    const std::optional<at::Tensor>& extra_topk_length,
    bool psa,
    int64_t csa_lane
) {
    using bf16 = cutlass::bfloat16_t;

    Arch arch;
    TORCH_CHECK(
        arch.is_sm100f(),
        "persistent sparse-attention prefill requires SM100F, got SM ",
        arch.major, ".", arch.minor);

    KU_CHECK_NDIM(q, 3);
    KU_CHECK_NDIM(kv, 3);
    KU_CHECK_NDIM(indices, 3);
    KU_CHECK_NDIM(attn_sink, 1);
    KU_CHECK_NDIM(topk_length, 1);
    KU_CHECK_NDIM(q_ready, 1);
    KU_CHECK_NDIM(q_positions, 1);
    KU_CHECK_NDIM(q_cos_sin_cache, 2);
    KU_CHECK_NDIM(o_ready, 1);
    KU_CHECK_NDIM(fused_o_fp8, 3);
    KU_CHECK_NDIM(fused_o_scale, 3);
    KU_CHECK_NDIM(fused_o_positions, 1);
    KU_CHECK_NDIM(extra_kv, 3);
    KU_CHECK_NDIM(extra_indices, 3);
    KU_CHECK_NDIM(extra_topk_length, 1);

    const int s_q = q.size(0);
    const int s_kv = kv.size(0);
    const int h_q = q.size(1);
    const int h_kv = kv.size(1);
    const int d_qk = q.size(2);
    const int topk = indices.size(2);

    const bool have_extra_kv = extra_kv.has_value();
    const bool have_extra_indices = extra_indices.has_value();
    const bool have_extra = have_extra_kv && have_extra_indices;
    const bool have_extra_topk_length = extra_topk_length.has_value();
    const int s_extra_kv = have_extra ? extra_kv->size(0) : 0;
    const int extra_topk = have_extra ? extra_indices->size(2) : 0;
    const bool have_fused_o =
        fused_o_fp8.has_value() &&
        fused_o_scale.has_value() &&
        fused_o_positions.has_value();

    TORCH_CHECK(h_q == 128, "persistent sparse-attention requires h_q == 128");
    TORCH_CHECK(h_kv == 1, "persistent sparse-attention requires h_kv == 1");
    TORCH_CHECK(d_qk == 512 && d_v == 512,
                "persistent sparse-attention requires d_qk == d_v == 512");
    TORCH_CHECK(topk > 0 && topk % 64 == 0 && topk <= 1280,
                "topk must be a positive multiple of 64 not exceeding 1280");
    TORCH_CHECK(have_extra_kv == have_extra_indices,
                "extra_kv and extra_indices must be provided together");
    TORCH_CHECK(!have_extra_topk_length || have_extra,
                "extra_topk_length requires extra_kv and extra_indices");
    TORCH_CHECK(!have_extra || topk + extra_topk <= 1280,
                "combined primary and extra topk must not exceed 1280");
    TORCH_CHECK(!have_extra || extra_topk % 64 == 0,
                "extra topk must be a multiple of 64");
    TORCH_CHECK(!q_positions.has_value() || q_cos_sin_cache.has_value(),
                "q_positions requires q_cos_sin_cache");
    TORCH_CHECK(!q_ready.has_value() || q_ready_chunk_size > 0,
                "q_ready_chunk_size must be positive with q_ready");
    TORCH_CHECK(
        fused_o_fp8.has_value() == fused_o_scale.has_value() &&
        fused_o_scale.has_value() == fused_o_positions.has_value(),
        "fused_o_fp8, fused_o_scale, and fused_o_positions must be provided together");
    TORCH_CHECK(!fused_o_skip_bf16 || have_fused_o,
                "fused_o_skip_bf16 requires fused output tensors");
    TORCH_CHECK(!have_fused_o || q_cos_sin_cache.has_value(),
                "fused-O requires q_cos_sin_cache");
    TORCH_CHECK(!(psa && have_fused_o),
                "persistent topology and fused-O are mutually exclusive");
    TORCH_CHECK(csa_lane >= 0 && csa_lane <= 2,
                "csa_lane must be 0 (generic), 1 (prefix), or 2 (tail)");
    TORCH_CHECK(csa_lane == 0 || psa,
                "CSA prefix/tail lanes require psa=True");
    TORCH_CHECK(q_norm_eps > 0.0, "q_norm_eps must be positive");

    int num_sms = arch.num_sms;
    if (num_sms_override.has_value()) {
        TORCH_CHECK(*num_sms_override > 0 && *num_sms_override <= arch.num_sms,
                    "num_sms must be in [1, ", arch.num_sms, "]");
        num_sms = static_cast<int>(*num_sms_override);
    }
    if (psa) {
        TORCH_CHECK(num_sms_override.has_value(),
                    "persistent topology requires an explicit num_sms");
        TORCH_CHECK(num_sms % 2 == 0,
                    "persistent topology requires an even num_sms");
    }
    if (csa_lane == 1) {
        TORCH_CHECK(s_q == 6784,
                    "CSA prefix lane requires exactly 6784 query rows");
        TORCH_CHECK(q_ready.has_value() && q_ready_chunk_size == 128,
                    "CSA prefix lane requires q_ready chunks of 128 rows");
        TORCH_CHECK(q_positions.has_value() && q_cos_sin_cache.has_value(),
                    "CSA prefix lane requires raw-Q positions and RoPE cache");
    } else if (csa_lane == 2) {
        TORCH_CHECK(s_q == 1408,
                    "CSA tail lane requires exactly 1408 query rows");
        TORCH_CHECK(q_ready.has_value() && q_ready_chunk_size == 128,
                    "CSA tail lane requires q_ready chunks of 128 rows");
        TORCH_CHECK(q_positions.has_value() && q_cos_sin_cache.has_value(),
                    "CSA tail lane requires raw-Q positions and RoPE cache");
    }

    KU_CHECK_DEVICE(q);
    KU_CHECK_DEVICE(kv);
    KU_CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(attn_sink);
    KU_CHECK_DEVICE(topk_length);
    KU_CHECK_DEVICE(q_ready);
    KU_CHECK_DEVICE(q_positions);
    KU_CHECK_DEVICE(q_cos_sin_cache);
    KU_CHECK_DEVICE(o_ready);
    KU_CHECK_DEVICE(fused_o_fp8);
    KU_CHECK_DEVICE(fused_o_scale);
    KU_CHECK_DEVICE(fused_o_positions);
    KU_CHECK_DEVICE(extra_kv);
    KU_CHECK_DEVICE(extra_indices);
    KU_CHECK_DEVICE(extra_topk_length);

    KU_CHECK_DTYPE(q, torch::kBFloat16);
    KU_CHECK_DTYPE(kv, torch::kBFloat16);
    KU_CHECK_DTYPE(indices, torch::kInt32);
    KU_CHECK_DTYPE(attn_sink, torch::kFloat32);
    KU_CHECK_DTYPE(topk_length, torch::kInt32);
    KU_CHECK_DTYPE(q_ready, torch::kInt32);
    KU_CHECK_DTYPE(q_positions, torch::kInt64);
    KU_CHECK_DTYPE(q_cos_sin_cache, torch::kFloat32);
    KU_CHECK_DTYPE(o_ready, torch::kInt32);
    KU_CHECK_DTYPE(fused_o_fp8, torch::kFloat8_e4m3fn);
    KU_CHECK_DTYPE(fused_o_scale, torch::kInt32);
    KU_CHECK_DTYPE(fused_o_positions, torch::kInt64);
    KU_CHECK_DTYPE(extra_kv, torch::kBFloat16);
    KU_CHECK_DTYPE(extra_indices, torch::kInt32);
    KU_CHECK_DTYPE(extra_topk_length, torch::kInt32);

    KU_CHECK_SHAPE(q, s_q, h_q, d_qk);
    KU_CHECK_SHAPE(kv, s_kv, h_kv, d_qk);
    KU_CHECK_SHAPE(indices, s_q, h_kv, topk);
    KU_CHECK_SHAPE(attn_sink, h_q);
    KU_CHECK_SHAPE(topk_length, s_q);
    KU_CHECK_SHAPE(q_positions, s_q);
    KU_CHECK_SHAPE(fused_o_fp8, s_q, 16, 4096);
    KU_CHECK_SHAPE(fused_o_scale, s_q, 16, 8);
    KU_CHECK_SHAPE(fused_o_positions, s_q);
    if (have_extra) {
        KU_CHECK_SHAPE(extra_kv, s_extra_kv, h_kv, d_qk);
        KU_CHECK_SHAPE(extra_indices, s_q, h_kv, extra_topk);
        KU_CHECK_SHAPE(extra_topk_length, s_q);
    }
    if (q_ready.has_value()) {
        const int64_t required =
            (s_q + q_ready_chunk_size - 1) / q_ready_chunk_size;
        TORCH_CHECK(q_ready->numel() >= required,
                    "q_ready has too few entries");
    }
    if (o_ready.has_value()) {
        TORCH_CHECK(o_ready->numel() >= 2LL * s_q,
                    "o_ready must have at least 2*s_q entries");
    }
    if (q_cos_sin_cache.has_value()) {
        TORCH_CHECK(q_cos_sin_cache->size(1) == 64,
                    "q_cos_sin_cache must have shape [max_position, 64]");
    }

    KU_CHECK_LAST_DIM_CONTIGUOUS(q);
    KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    KU_CHECK_LAST_DIM_CONTIGUOUS(attn_sink);
    KU_CHECK_LAST_DIM_CONTIGUOUS(topk_length);
    KU_CHECK_LAST_DIM_CONTIGUOUS(q_ready);
    KU_CHECK_LAST_DIM_CONTIGUOUS(q_positions);
    KU_CHECK_LAST_DIM_CONTIGUOUS(q_cos_sin_cache);
    KU_CHECK_LAST_DIM_CONTIGUOUS(o_ready);
    KU_CHECK_LAST_DIM_CONTIGUOUS(fused_o_positions);
    KU_CHECK_LAST_DIM_CONTIGUOUS(extra_kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(extra_indices);
    KU_CHECK_LAST_DIM_CONTIGUOUS(extra_topk_length);

    at::cuda::CUDAGuard device_guard{static_cast<char>(q.get_device())};
    auto opts = q.options();
    at::Tensor out = out_.has_value()
        ? *out_
        : torch::empty({s_q, h_q, d_v}, opts);
    at::Tensor max_logits =
        torch::empty({s_q, h_q}, opts.dtype(torch::kFloat32));
    at::Tensor lse =
        torch::empty({s_q, h_q}, opts.dtype(torch::kFloat32));

    KU_CHECK_DEVICE(out);
    KU_CHECK_DTYPE(out, torch::kBFloat16);
    KU_CHECK_SHAPE(out, s_q, h_q, d_v);
    KU_CHECK_CONTIGUOUS(out);
    KU_CHECK_CONTIGUOUS(max_logits);
    KU_CHECK_CONTIGUOUS(lse);

    if (have_fused_o) {
        TORCH_CHECK(
            fused_o_fp8->stride(0) == 4096 &&
            fused_o_fp8->stride(1) >= static_cast<int64_t>(s_q) * 4096 &&
            fused_o_fp8->stride(1) % 4096 == 0 &&
            fused_o_fp8->stride(2) == 1,
            "fused_o_fp8 must use physical [16, S_Q, 4096] group-major layout");
        TORCH_CHECK(
            fused_o_scale->stride(0) == 1 &&
            fused_o_scale->stride(1) == 8 * fused_o_scale->stride(2) &&
            fused_o_scale->stride(2) >= s_q &&
            fused_o_scale->stride(2) % 4 == 0,
            "fused_o_scale must use packed DeepGEMM MN-major layout");
    }

    SparseAttnFwdParams params{};
    params.s_q = s_q;
    params.s_kv = s_kv;
    params.h_q = h_q;
    params.h_kv = h_kv;
    params.d_qk = d_qk;
    params.d_v = static_cast<int>(d_v);
    params.topk = topk;
    params.sm_scale = sm_scale;
    params.sm_scale_div_log2 = sm_scale * LOG_2_E;
    params.q = reinterpret_cast<bf16*>(q.data_ptr());
    params.kv = reinterpret_cast<bf16*>(kv.data_ptr());
    params.indices = reinterpret_cast<int*>(indices.data_ptr());
    params.attn_sink = ku::get_optional_tensor_ptr<float>(attn_sink);
    params.topk_length = ku::get_optional_tensor_ptr<int>(topk_length);
    params.s_extra_kv = s_extra_kv;
    params.extra_topk = extra_topk;
    params.extra_kv = ku::get_optional_tensor_ptr<bf16>(extra_kv);
    params.extra_indices = ku::get_optional_tensor_ptr<int>(extra_indices);
    params.extra_topk_length =
        ku::get_optional_tensor_ptr<int>(extra_topk_length);
    params.stride_q_s_q = int64_stride_to_int(q.stride(0));
    params.stride_q_h_q = int64_stride_to_int(q.stride(1));
    params.stride_kv_s_kv = int64_stride_to_int(kv.stride(0));
    params.stride_kv_h_kv = int64_stride_to_int(kv.stride(1));
    params.stride_indices_s_q = int64_stride_to_int(indices.stride(0));
    params.stride_indices_h_kv = int64_stride_to_int(indices.stride(1));
    params.stride_extra_kv_s_kv =
        have_extra ? int64_stride_to_int(extra_kv->stride(0)) : 0;
    params.stride_extra_kv_h_kv =
        have_extra ? int64_stride_to_int(extra_kv->stride(1)) : 0;
    params.stride_extra_indices_s_q =
        have_extra ? int64_stride_to_int(extra_indices->stride(0)) : 0;
    params.stride_extra_indices_h_kv =
        have_extra ? int64_stride_to_int(extra_indices->stride(1)) : 0;
    params.out = reinterpret_cast<bf16*>(out.data_ptr());
    params.max_logits = max_logits.data_ptr<float>();
    params.lse = lse.data_ptr<float>();
    params.q_ready = ku::get_optional_tensor_ptr<int>(q_ready);
    params.q_ready_chunk_size = static_cast<int>(q_ready_chunk_size);
    params.o_ready = ku::get_optional_tensor_ptr<int>(o_ready);
    params.q_positions = ku::get_optional_tensor_ptr<int64_t>(q_positions);
    params.q_cos_sin_cache =
        ku::get_optional_tensor_ptr<float>(q_cos_sin_cache);
    params.q_norm_eps = static_cast<float>(q_norm_eps);
    params.fused_o_fp8 =
        ku::get_optional_tensor_ptr<uint8_t>(fused_o_fp8);
    params.fused_o_scale =
        ku::get_optional_tensor_ptr<int>(fused_o_scale);
    params.fused_o_positions =
        ku::get_optional_tensor_ptr<int64_t>(fused_o_positions);
    params.stride_fused_o_fp8_b =
        have_fused_o ? int64_stride_to_int(fused_o_fp8->stride(0)) : 0;
    params.stride_fused_o_fp8_g =
        have_fused_o ? int64_stride_to_int(fused_o_fp8->stride(1)) : 0;
    params.stride_fused_o_fp8_k =
        have_fused_o ? int64_stride_to_int(fused_o_fp8->stride(2)) : 0;
    params.stride_fused_o_scale_b =
        have_fused_o ? int64_stride_to_int(fused_o_scale->stride(0)) : 0;
    params.stride_fused_o_scale_g =
        have_fused_o ? int64_stride_to_int(fused_o_scale->stride(1)) : 0;
    params.stride_fused_o_scale_k =
        have_fused_o ? int64_stride_to_int(fused_o_scale->stride(2)) : 0;
    params.fused_o_skip_bf16 = fused_o_skip_bf16;
    params.num_sm = num_sms;
    params.persistent_grid = num_sms_override.has_value();
    params.stream = at::cuda::getCurrentCUDAStream().stream();

    if (csa_lane == 1) {
        sparse_attention::launch_sparse_attention_prefix(params);
    } else if (csa_lane == 2) {
        sparse_attention::launch_sparse_attention_tail(params);
    } else if (psa) {
        sparse_attention::launch_sparse_attention(params);
    } else {
        fwd_for_small_topk::head128::run_fwd_for_small_topk_phase1_kernel<
            SparseAttnFwdMode::Prefill, 512>(params);
    }
    return {out, max_logits, lse};
}

inline std::vector<at::Tensor> sparse_decode_ready_fwd(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    const std::optional<at::Tensor>& topk_length,
    const std::optional<at::Tensor>& attn_sink,
    const std::optional<at::Tensor>& extra_kv,
    const std::optional<at::Tensor>& extra_indices,
    const std::optional<at::Tensor>& extra_topk_length,
    int64_t d_v,
    double sm_scale,
    const std::optional<at::Tensor>& out_,
    int64_t num_sm_parts,
    const at::Tensor& row_ready,
    int64_t initial_ready_rows,
    const std::optional<at::Tensor>& q_positions,
    const std::optional<at::Tensor>& q_cos_sin_cache,
    double q_norm_eps,
    const std::optional<at::Tensor>& fused_o_fp8,
    const std::optional<at::Tensor>& fused_o_scale,
    const std::optional<at::Tensor>& fused_o_positions,
    bool fused_o_skip_bf16
) {
    using bf16 = cutlass::bfloat16_t;

    Arch arch;
    TORCH_CHECK(arch.is_sm100f(),
                "persistent sparse-attention ready decode requires SM100F");
    KU_CHECK_NDIM(q, 4);
    KU_CHECK_NDIM(kv, 4);
    KU_CHECK_NDIM(indices, 3);
    KU_CHECK_NDIM(topk_length, 1);
    KU_CHECK_NDIM(attn_sink, 1);
    KU_CHECK_NDIM(extra_kv, 4);
    KU_CHECK_NDIM(extra_indices, 3);
    KU_CHECK_NDIM(extra_topk_length, 1);
    KU_CHECK_NDIM(row_ready, 1);
    KU_CHECK_NDIM(q_positions, 1);
    KU_CHECK_NDIM(q_cos_sin_cache, 2);
    KU_CHECK_NDIM(fused_o_fp8, 3);
    KU_CHECK_NDIM(fused_o_scale, 3);
    KU_CHECK_NDIM(fused_o_positions, 1);

    const int b = q.size(0);
    const int s_q = q.size(1);
    const int h_q = q.size(2);
    const int d_qk = q.size(3);
    const int num_blocks = kv.size(0);
    const int page_block_size = kv.size(1);
    const int h_kv = kv.size(2);
    const int topk = indices.size(2);
    const bool have_extra = extra_kv.has_value();
    const int extra_num_blocks = have_extra ? extra_kv->size(0) : 0;
    const int extra_page_block_size = have_extra ? extra_kv->size(1) : 0;
    const int extra_topk = extra_indices.has_value()
        ? extra_indices->size(2) : 0;
    const bool have_fused_o =
        fused_o_fp8.has_value() &&
        fused_o_scale.has_value() &&
        fused_o_positions.has_value();

    TORCH_CHECK(s_q == 1 && h_q == 128 && h_kv == 1,
                "ready decode requires s_q=1, h_q=128, h_kv=1");
    TORCH_CHECK(d_qk == 512 && d_v == 512,
                "ready decode requires d_qk=d_v=512");
    TORCH_CHECK(topk > 0 && topk % 64 == 0,
                "ready decode topk must be a positive multiple of 64");
    TORCH_CHECK(!have_extra || extra_topk % 64 == 0,
                "ready decode extra topk must be a multiple of 64");
    TORCH_CHECK(num_sm_parts > 0 && num_sm_parts <= arch.num_sms / 2,
                "num_sm_parts must be in [1, num_sms/2]");
    TORCH_CHECK(initial_ready_rows >= 0 && initial_ready_rows <= b,
                "initial_ready_rows must be in [0, b]");
    TORCH_CHECK(extra_kv.has_value() == extra_indices.has_value(),
                "extra_kv and extra_indices must be provided together");
    TORCH_CHECK(!extra_topk_length.has_value() || have_extra,
                "extra_topk_length requires extra_kv");
    TORCH_CHECK(
        q_positions.has_value() == q_cos_sin_cache.has_value(),
        "q_positions and q_cos_sin_cache must be provided together");
    TORCH_CHECK(
        fused_o_fp8.has_value() == fused_o_scale.has_value() &&
        fused_o_scale.has_value() == fused_o_positions.has_value(),
        "all fused-O tensors must be provided together");
    TORCH_CHECK(!fused_o_skip_bf16 || have_fused_o,
                "fused_o_skip_bf16 requires fused-O");
    TORCH_CHECK(!have_fused_o || q_cos_sin_cache.has_value(),
                "fused-O requires q_cos_sin_cache");
    TORCH_CHECK(q_norm_eps > 0.0, "q_norm_eps must be positive");

    const int bytes_per_token = 448 + 64 * 2 + (448 / 64) + 1;
    TORCH_CHECK(kv.size(3) == bytes_per_token,
                "MODEL1 ready-decode KV token must have ",
                bytes_per_token, " packed bytes");

    KU_CHECK_DEVICE(q);
    KU_CHECK_DEVICE(kv);
    KU_CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(row_ready);
    KU_CHECK_DEVICE(topk_length);
    KU_CHECK_DEVICE(attn_sink);
    KU_CHECK_DEVICE(extra_kv);
    KU_CHECK_DEVICE(extra_indices);
    KU_CHECK_DEVICE(extra_topk_length);
    KU_CHECK_DEVICE(q_positions);
    KU_CHECK_DEVICE(q_cos_sin_cache);
    KU_CHECK_DEVICE(fused_o_fp8);
    KU_CHECK_DEVICE(fused_o_scale);
    KU_CHECK_DEVICE(fused_o_positions);

    KU_CHECK_DTYPE(q, torch::kBFloat16);
    TORCH_CHECK(
        kv.scalar_type() == at::ScalarType::Float8_e4m3fn ||
        kv.scalar_type() == at::ScalarType::Char ||
        kv.scalar_type() == at::ScalarType::Byte,
        "kv must contain packed FP8 bytes");
    if (have_extra) {
        TORCH_CHECK(
            extra_kv->scalar_type() == at::ScalarType::Float8_e4m3fn ||
            extra_kv->scalar_type() == at::ScalarType::Char ||
            extra_kv->scalar_type() == at::ScalarType::Byte,
            "extra_kv must contain packed FP8 bytes");
    }
    KU_CHECK_DTYPE(indices, torch::kInt32);
    KU_CHECK_DTYPE(row_ready, torch::kInt32);
    KU_CHECK_DTYPE(topk_length, torch::kInt32);
    KU_CHECK_DTYPE(attn_sink, torch::kFloat32);
    KU_CHECK_DTYPE(extra_indices, torch::kInt32);
    KU_CHECK_DTYPE(extra_topk_length, torch::kInt32);
    KU_CHECK_DTYPE(q_positions, torch::kInt64);
    KU_CHECK_DTYPE(q_cos_sin_cache, torch::kFloat32);
    KU_CHECK_DTYPE(fused_o_fp8, torch::kFloat8_e4m3fn);
    KU_CHECK_DTYPE(fused_o_scale, torch::kInt32);
    KU_CHECK_DTYPE(fused_o_positions, torch::kInt64);

    KU_CHECK_SHAPE(q, b, s_q, h_q, d_qk);
    KU_CHECK_SHAPE(kv, num_blocks, page_block_size, h_kv, bytes_per_token);
    KU_CHECK_SHAPE(indices, b, s_q, topk);
    KU_CHECK_SHAPE(row_ready, b);
    KU_CHECK_SHAPE(topk_length, b);
    KU_CHECK_SHAPE(attn_sink, h_q);
    KU_CHECK_SHAPE(q_positions, b);
    KU_CHECK_SHAPE(fused_o_fp8, b, 16, 4096);
    KU_CHECK_SHAPE(fused_o_scale, b, 16, 8);
    KU_CHECK_SHAPE(fused_o_positions, b);
    if (have_extra) {
        KU_CHECK_SHAPE(extra_kv, extra_num_blocks, extra_page_block_size,
                       h_kv, bytes_per_token);
        KU_CHECK_SHAPE(extra_indices, b, s_q, extra_topk);
        KU_CHECK_SHAPE(extra_topk_length, b);
    }
    TORCH_CHECK(kv.stride(1) == bytes_per_token,
                "each packed KV block must be contiguous");
    if (have_extra) {
        TORCH_CHECK(extra_kv->stride(1) == bytes_per_token,
                    "each packed extra-KV block must be contiguous");
    }
    if (q_cos_sin_cache.has_value()) {
        TORCH_CHECK(q_cos_sin_cache->size(1) == 64,
                    "q_cos_sin_cache must have shape [max_position, 64]");
    }

    KU_CHECK_LAST_DIM_CONTIGUOUS(q);
    KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    KU_CHECK_CONTIGUOUS(topk_length);
    KU_CHECK_CONTIGUOUS(attn_sink);
    KU_CHECK_LAST_DIM_CONTIGUOUS(extra_kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(extra_indices);
    KU_CHECK_CONTIGUOUS(extra_topk_length);
    KU_CHECK_CONTIGUOUS(row_ready);
    KU_CHECK_CONTIGUOUS(q_positions);
    KU_CHECK_CONTIGUOUS(q_cos_sin_cache);
    KU_CHECK_CONTIGUOUS(fused_o_positions);

    at::cuda::CUDAGuard device_guard{static_cast<char>(q.get_device())};
    auto opts = q.options();
    at::Tensor out = out_.has_value()
        ? *out_
        : torch::empty({b, s_q, h_q, d_v}, opts);
    at::Tensor lse =
        torch::empty({b, s_q, h_q}, opts.dtype(torch::kFloat32));

    KU_CHECK_DEVICE(out);
    KU_CHECK_DTYPE(out, torch::kBFloat16);
    KU_CHECK_SHAPE(out, b, s_q, h_q, d_v);
    KU_CHECK_CONTIGUOUS(out);
    KU_CHECK_CONTIGUOUS(lse);

    if (have_fused_o) {
        TORCH_CHECK(
            fused_o_fp8->stride(0) == 4096 &&
            fused_o_fp8->stride(1) >= static_cast<int64_t>(b) * 4096 &&
            fused_o_fp8->stride(1) % 4096 == 0 &&
            fused_o_fp8->stride(2) == 1,
            "fused_o_fp8 must use physical [16, B, 4096] group-major layout");
        TORCH_CHECK(
            fused_o_scale->stride(0) == 1 &&
            fused_o_scale->stride(1) == 8 * fused_o_scale->stride(2) &&
            fused_o_scale->stride(2) >= b &&
            fused_o_scale->stride(2) % 4 == 0,
            "fused_o_scale must use packed DeepGEMM MN-major layout");
    }

    SparseAttnDecodeParams params{};
    params.b = b;
    params.s_q = s_q;
    params.h_q = h_q;
    params.h_kv = h_kv;
    params.d_qk = d_qk;
    params.d_v = static_cast<int>(d_v);
    params.sm_scale = static_cast<float>(sm_scale);
    params.sm_scale_div_log2 = static_cast<float>(sm_scale) * LOG_2_E;
    params.num_blocks = num_blocks;
    params.page_block_size = page_block_size;
    params.topk = topk;
    params.model_type = ModelType::MODEL1;
    params.q = reinterpret_cast<bf16*>(q.data_ptr());
    params.kv = reinterpret_cast<bf16*>(kv.data_ptr());
    params.indices = reinterpret_cast<int*>(indices.data_ptr());
    params.topk_length = ku::get_optional_tensor_ptr<int>(topk_length);
    params.attn_sink = ku::get_optional_tensor_ptr<float>(attn_sink);
    params.lse = lse.data_ptr<float>();
    params.out = reinterpret_cast<bf16*>(out.data_ptr());
    params.extra_num_blocks = extra_num_blocks;
    params.extra_page_block_size = extra_page_block_size;
    params.extra_topk = extra_topk;
    params.extra_kv = ku::get_optional_tensor_ptr<bf16>(extra_kv);
    params.extra_indices = ku::get_optional_tensor_ptr<int>(extra_indices);
    params.extra_topk_length =
        ku::get_optional_tensor_ptr<int>(extra_topk_length);
    params.stride_q_b = int64_stride_to_int(q.stride(0));
    params.stride_q_s_q = int64_stride_to_int(q.stride(1));
    params.stride_q_h_q = int64_stride_to_int(q.stride(2));
    params.stride_kv_block = int64_stride_to_int(kv.stride(0));
    params.stride_kv_row = int64_stride_to_int(kv.stride(1));
    params.stride_indices_b = int64_stride_to_int(indices.stride(0));
    params.stride_indices_s_q = int64_stride_to_int(indices.stride(1));
    params.stride_lse_b = int64_stride_to_int(lse.stride(0));
    params.stride_lse_s_q = int64_stride_to_int(lse.stride(1));
    params.stride_o_b = int64_stride_to_int(out.stride(0));
    params.stride_o_s_q = int64_stride_to_int(out.stride(1));
    params.stride_o_h_q = int64_stride_to_int(out.stride(2));
    params.stride_extra_kv_block =
        have_extra ? int64_stride_to_int(extra_kv->stride(0)) : 0;
    params.stride_extra_kv_row =
        have_extra ? int64_stride_to_int(extra_kv->stride(1)) : 0;
    params.stride_extra_indices_b =
        have_extra ? int64_stride_to_int(extra_indices->stride(0)) : 0;
    params.stride_extra_indices_s_q =
        have_extra ? int64_stride_to_int(extra_indices->stride(1)) : 0;
    params.stream = at::cuda::getCurrentCUDAStream().stream();
    params.tile_scheduler_metadata_ptr = nullptr;
    params.num_splits_ptr = nullptr;
    params.num_sm_parts = static_cast<int>(num_sm_parts);
    params.row_ready = row_ready.data_ptr<int>();
    params.initial_ready_rows = static_cast<int>(initial_ready_rows);
    params.q_positions = ku::get_optional_tensor_ptr<int64_t>(q_positions);
    params.q_cos_sin_cache =
        ku::get_optional_tensor_ptr<float>(q_cos_sin_cache);
    params.q_norm_eps = static_cast<float>(q_norm_eps);
    params.fused_o_fp8 =
        ku::get_optional_tensor_ptr<uint8_t>(fused_o_fp8);
    params.fused_o_scale =
        ku::get_optional_tensor_ptr<int>(fused_o_scale);
    params.fused_o_positions =
        ku::get_optional_tensor_ptr<int64_t>(fused_o_positions);
    params.stride_fused_o_fp8_b =
        have_fused_o ? int64_stride_to_int(fused_o_fp8->stride(0)) : 0;
    params.stride_fused_o_fp8_g =
        have_fused_o ? int64_stride_to_int(fused_o_fp8->stride(1)) : 0;
    params.stride_fused_o_fp8_k =
        have_fused_o ? int64_stride_to_int(fused_o_fp8->stride(2)) : 0;
    params.stride_fused_o_scale_b =
        have_fused_o ? int64_stride_to_int(fused_o_scale->stride(0)) : 0;
    params.stride_fused_o_scale_g =
        have_fused_o ? int64_stride_to_int(fused_o_scale->stride(1)) : 0;
    params.stride_fused_o_scale_k =
        have_fused_o ? int64_stride_to_int(fused_o_scale->stride(2)) : 0;
    params.fused_o_skip_bf16 = fused_o_skip_bf16;

    fwd_for_small_topk::head128::run_fwd_for_small_topk_phase1_kernel<
        SparseAttnFwdMode::DecodeReady, 512>(params);
    return {out, lse.transpose(1, 2)};
}

}  // namespace persistent_sparse_attention
