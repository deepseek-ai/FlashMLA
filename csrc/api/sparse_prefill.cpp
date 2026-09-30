#include "common.h"
#include "params.h"

enum class SparseFwdFeatures : int {
    HEAD_64,
    HEAD_128,

    ATTN_SINK,
    TOPK_LENGTH
};

class SparseFwdImplBase : public ImplBase<
    SparseAttnFwdParams,
    SparseFwdFeatures
> {};

#ifdef FLASH_MLA_IS_BUILD_ON_CUDA

#include "cuda_kernels/sm100/prefill/sparse/fwd/head64/phase1.h"
#include "cuda_kernels/sm100/prefill/sparse/fwd/head128/phase1.h"
#include "cuda_kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/phase1.h"

class SparseFwd_Sm100_Head64_Impl : public SparseFwdImplBase {
    DECLARE_SUPPORTED_FEATURES(
        SparseFwdFeatures::HEAD_64,
        SparseFwdFeatures::ATTN_SINK,
        SparseFwdFeatures::TOPK_LENGTH
    )

protected:
    void run_(const SparseAttnFwdParams &params, const std::vector<FeatureT> &required_features) override {
        sm100::prefill::sparse_fwd::head64::run_sparse_fwd_phase1_kernel(params);
    }
};

class SparseFwd_Sm100_Head128_Impl : public SparseFwdImplBase {
    DECLARE_SUPPORTED_FEATURES(
        SparseFwdFeatures::HEAD_128,
        SparseFwdFeatures::ATTN_SINK,
        SparseFwdFeatures::TOPK_LENGTH
    )

protected:
    void run_(const SparseAttnFwdParams &params, const std::vector<FeatureT> &required_features) override {
        sm100::prefill::sparse_fwd::head128::run_sparse_fwd_phase1_kernel(params);
    }
};

class SparseFwd_Sm100_Head128_Small_TopK_Impl : public SparseFwdImplBase {
    DECLARE_SUPPORTED_FEATURES(
        SparseFwdFeatures::HEAD_128,
        SparseFwdFeatures::ATTN_SINK,
        SparseFwdFeatures::TOPK_LENGTH
    )

protected:
    void run_(const SparseAttnFwdParams &params, const std::vector<FeatureT> &required_features) override {
        sm100::prefill::sparse_fwd_for_small_topk::head128::run_sparse_fwd_for_small_topk_phase1_kernel<SparseAttnFwdMode::Prefill, ModelType::V41, ModelType::V41>(params);
    }
};

#endif  // FLASH_MLA_IS_BUILD_ON_CUDA

#ifdef FLASH_MLA_IS_BUILD_ON_ASCEND

#include "ascend_kernels/prefill/sparse/kernel.h"

class SparseFwdImpl : public SparseFwdImplBase {
    DECLARE_SUPPORTED_FEATURES(
        SparseFwdFeatures::HEAD_64,
        SparseFwdFeatures::ATTN_SINK,
        SparseFwdFeatures::TOPK_LENGTH
    )

protected:
    void run_(const SparseAttnFwdParams &params, const std::vector<FeatureT> &required_features) override {
        DISPATCH_BOOLEAN_FLAG(params.attn_sink != nullptr, HAVE_ATTN_SINK, ([&]() {
            static constexpr ascend::prefill::sparse_fwd::Config CONFIG = {
                SparseAttnFwdMode::Prefill,
                64,
                640,
                HAVE_ATTN_SINK,
                ModelType::V41,
                ModelType::V41
            };
            ascend::prefill::sparse_fwd::run_sparse_fwd_kernel<CONFIG>(params);
        }));
    }
};

#endif  // FLASH_MLA_IS_BUILD_ON_ASCEND

static std::vector<at::Tensor> sparse_prefill_fwd(
    const at::Tensor &q,
    const at::Tensor &kv,
    const at::Tensor &indices,
    float sm_scale,
    int d_v,
    const std::optional<at::Tensor> &attn_sink,
    const std::optional<at::Tensor> &topk_length
) {
#ifdef FLASH_MLA_IS_BUILD_ON_CUDA
    Arch arch = Arch();
    TORCH_CHECK(arch.is_sm100f(), "Sparse Attention Prefill Kernel (sparse_prefill_fwd) is only supported on SM100f architectures.");
#endif
    KU_CHECK_NDIM(q, 3);
    KU_CHECK_NDIM(kv, 3);
    KU_CHECK_NDIM(indices, 3);
    KU_CHECK_NDIM(attn_sink, 1);
    KU_CHECK_NDIM(topk_length, 1);

    int s_q = q.size(0);
    int s_kv = kv.size(0);
    int h_q = q.size(1);
    int h_kv = kv.size(1);
    int d_qk = q.size(2);
    int topk = indices.size(2);
    bool have_topk_length = topk_length.has_value();

    TORCH_CHECK(d_qk == 512, "Invalid d_qk: ", d_qk);
    TORCH_CHECK(d_v == 512, "Invalid d_v", d_v);

    KU_CHECK_DEVICE(q);
    KU_CHECK_DEVICE(kv);
    KU_CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(attn_sink);
    KU_CHECK_DEVICE(topk_length);

    KU_CHECK_DTYPE(q, torch::kBFloat16);
    KU_CHECK_DTYPE(kv, torch::kBFloat16);
    KU_CHECK_DTYPE(indices, torch::kInt32);
    KU_CHECK_DTYPE(attn_sink, torch::kFloat32);
    KU_CHECK_DTYPE(topk_length, torch::kInt32);

    KU_CHECK_SHAPE(q, s_q, h_q, d_qk);
    KU_CHECK_SHAPE(kv, s_kv, h_kv, d_qk);
    KU_CHECK_SHAPE(indices, s_q, h_kv, topk);
    KU_CHECK_SHAPE(attn_sink, h_q);
    KU_CHECK_SHAPE(topk_length, s_q);

    KU_CHECK_LAST_DIM_CONTIGUOUS(q);
    KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    KU_CHECK_LAST_DIM_CONTIGUOUS(attn_sink);
    KU_CHECK_LAST_DIM_CONTIGUOUS(topk_length);

    // Allocate results
#ifdef FLASH_MLA_IS_BUILD_ON_CUDA
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
#endif
    auto opts = q.options();

    at::Tensor out = torch::empty({s_q, h_q, d_v}, opts);
    at::Tensor max_logits = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat));
    at::Tensor lse = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat));
    KU_CHECK_CONTIGUOUS(out);
    KU_CHECK_CONTIGUOUS(max_logits);
    KU_CHECK_CONTIGUOUS(lse);

    SparseAttnFwdParams params = {
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

        (bf16*)out.data_ptr(),
        (float*)max_logits.data_ptr(),
        (float*)lse.data_ptr(),

        SparseAttnFwdMode::Prefill,
        get_num_sms(),
        get_current_stream()
    };

    std::vector<SparseFwdFeatures> required_features;
    if (h_q == 64) {
        required_features.push_back(SparseFwdFeatures::HEAD_64);
    } else if (h_q == 128) {
        required_features.push_back(SparseFwdFeatures::HEAD_128);
    } else {
        TORCH_CHECK(false, "Unsupported h_q: ", h_q);
    }
    if (attn_sink.has_value()) {
        required_features.push_back(SparseFwdFeatures::ATTN_SINK);
    }
    if (have_topk_length) {
        required_features.push_back(SparseFwdFeatures::TOPK_LENGTH);
    }

#ifdef FLASH_MLA_IS_BUILD_ON_CUDA
    if (h_q == 64) {
        SparseFwd_Sm100_Head64_Impl fwd_impl;
        fwd_impl.run(params, required_features);
    } else {
        SparseFwd_Sm100_Head128_Small_TopK_Impl small_topk_impl;
        SparseFwd_Sm100_Head128_Impl regular_impl;
        bool use_small_topk_impl = false;
        if (
            (topk <= 1280 && small_topk_impl.check_if_all_features_are_supported(required_features)) ||
            !regular_impl.check_if_all_features_are_supported(required_features)
        ) {
            use_small_topk_impl = true;
        }
        if (use_small_topk_impl) {
            small_topk_impl.run(params, required_features);
        } else {
            regular_impl.run(params, required_features);
        }
    }
#endif  // FLASH_MLA_IS_BUILD_ON_CUDA

#ifdef FLASH_MLA_IS_BUILD_ON_ASCEND
    SparseFwdImpl fwd_impl;
    fwd_impl.run(params, required_features);
#endif  // FLASH_MLA_IS_BUILD_ON_ASCEND

    return {out, max_logits, lse};
}

void register_sparse_prefill(pybind11::module_& m) {
    m.def("sparse_prefill_fwd",
        &sparse_prefill_fwd,
        "Run Sparse Attention Prefill Forward");
}
