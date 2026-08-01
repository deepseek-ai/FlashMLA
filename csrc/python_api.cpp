#include <algorithm>
#include <optional>
#include <tuple>
#include <vector>

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/fast_math.h>

#include "api/common.h"
#include "api/dense_decode.h"
#include "api/sparse_decode.h"
#include "api/sparse_fwd.h"

std::vector<at::Tensor> get_mla_decoding_metadata(
    at::Tensor& seqlens_k,
    const int64_t num_q_tokens_per_head_k,
    const int64_t h_k,
    const std::optional<int64_t> h_q,
    const bool is_fp8_kvcache,
    const std::optional<int64_t> topk) {
    TORCH_CHECK(seqlens_k.is_cuda(), "seqlens_k must be on CUDA device");
    TORCH_CHECK(seqlens_k.dtype() == torch::kInt32, "seqlens_k must have dtype int32");
    TORCH_CHECK(seqlens_k.is_contiguous(), "seqlens_k must be contiguous");

    const int batch_size = seqlens_k.size(0);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");

    const int num_q_tokens_per_head_k_int = static_cast<int>(num_q_tokens_per_head_k);
    const int h_k_int = static_cast<int>(h_k);
    const std::optional<int> h_q_int = h_q.has_value() ? std::make_optional(static_cast<int>(*h_q)) : std::nullopt;
    const std::optional<int> topk_int = topk.has_value() ? std::make_optional(static_cast<int>(*topk)) : std::nullopt;

    TORCH_CHECK(h_k_int > 0, "num_heads_k must be positive");
    if (topk_int.has_value()) {
        TORCH_CHECK(h_q_int.has_value(), "num_heads_q must be provided when topk is provided");
        TORCH_CHECK(is_fp8_kvcache, "Sparse decoding requires is_fp8_kvcache=true");
    }

    // Keep dense FP8 metadata on the dedicated API path for compatibility.
    TORCH_CHECK(!(is_fp8_kvcache && !topk_int.has_value()),
                "Use get_mla_decoding_metadata_dense_fp8 for dense fp8 metadata");

    const int num_heads_q = h_q_int.value_or(h_k_int * num_q_tokens_per_head_k_int);
    TORCH_CHECK(num_heads_q > 0, "num_heads_q must be positive");

    const int heads_ratio = std::max(1, num_heads_q / h_k_int);
    const int s_q = std::max(1, num_q_tokens_per_head_k_int / heads_ratio);

    Arch arch;
    int num_sm_parts;
    if (topk_int.has_value()) {
        if (arch.is_sm100f()) {
            // sm100 sparse kernels use a larger split count envelope.
            num_sm_parts = std::max(arch.num_sms / s_q, 1);
        } else {
            const int heads_per_64 = std::max(1, num_heads_q / 64);
            num_sm_parts = std::max(arch.num_sms / s_q / heads_per_64, 1);
        }
    } else {
        num_sm_parts = std::max(
            arch.num_sms / h_k_int / cutlass::ceil_div(s_q * num_heads_q / h_k_int, 64),
            1);
    }

    at::cuda::CUDAGuard device_guard{static_cast<char>(seqlens_k.get_device())};
    auto opts = seqlens_k.options().dtype(torch::kInt32);

    at::Tensor tile_scheduler_metadata =
        torch::empty({num_sm_parts, DecodingSchedMetaSize / static_cast<int>(sizeof(int))}, opts);
    at::Tensor num_splits = torch::empty({batch_size + 1}, opts);

    GetDecodeSchedMetaParams params = {
        batch_size,
        s_q,
        64,
        5,
        topk_int.value_or(-1),
        -1,
        nullptr,
        nullptr,
        seqlens_k.data_ptr<int>(),
        reinterpret_cast<DecodingSchedMeta*>(tile_scheduler_metadata.data_ptr<int>()),
        num_splits.data_ptr<int>(),
        num_sm_parts,
        at::cuda::getCurrentCUDAStream().stream(),
    };

    smxx::decode::run_get_decoding_sched_meta_kernel(params);
    return {tile_scheduler_metadata, num_splits};
}

std::vector<at::Tensor> fwd_kvcache_mla(
    at::Tensor& q,
    const at::Tensor& kcache,
    const int64_t head_size_v,
    const at::Tensor& seqlens_k,
    const at::Tensor& block_table,
    const double softmax_scale,
    bool is_causal,
    const at::Tensor& tile_scheduler_metadata,
    const at::Tensor& num_splits,
    const bool& is_fp8,
    const std::optional<at::Tensor>& indices) {
    const int head_size_v_int = static_cast<int>(head_size_v);
    const float softmax_scale_float = static_cast<float>(softmax_scale);

    std::optional<at::Tensor> tile_scheduler_metadata_opt = tile_scheduler_metadata;
    std::optional<at::Tensor> num_splits_opt = num_splits;

    if (indices.has_value()) {
        TORCH_CHECK(is_fp8, "Sparse decode path requires is_fp8=true");
        auto result = sparse_attn_decode_interface(
            q,
            kcache,
            indices.value(),
            std::nullopt,
            std::nullopt,
            tile_scheduler_metadata_opt,
            num_splits_opt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            head_size_v_int,
            softmax_scale_float);
        return {std::get<0>(result), std::get<1>(result)};
    }

    TORCH_CHECK(!is_fp8,
                "Dense FP8 decode is exposed via fwd_kvcache_mla_fp8, not fwd_kvcache_mla");
    auto result = dense_attn_decode_interface(
        q,
        kcache,
        head_size_v_int,
        seqlens_k,
        block_table,
        softmax_scale_float,
        is_causal,
        tile_scheduler_metadata_opt,
        num_splits_opt);
    return {std::get<0>(result), std::get<1>(result)};
}

std::vector<at::Tensor> sparse_prefill_fwd(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    double sm_scale,
    int64_t d_v) {
    auto result = sparse_attn_prefill_interface(
        q,
        kv,
        indices,
        static_cast<float>(sm_scale),
        static_cast<int>(d_v),
        std::nullopt,
        std::nullopt);
    // Keep SGL compatibility: this API historically returns max_logits/lse in log2 space.
    result[1].mul_(LOG_2_E);
    result[2].mul_(LOG_2_E);
    return result;
}
