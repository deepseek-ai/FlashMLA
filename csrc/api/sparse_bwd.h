#pragma once

#include "common.h"
#include "../sm100/prefill/sparse/bwd/params.h"

// Forward declaration of the SM100 sparse backward launcher (host wrapper lives
// in csrc/sm100/prefill/sparse/bwd/instantiations/sparse_bwd_k576.cu).
namespace sm100::sparse_bwd {
    void launch(SparseAttnBwdParams& params);
}

// Sparse MLA backward -- host dispatch wrapper. SM100-only (SM90 port pending).

enum class SparseBwdFeatures : int {
    HEAD_64,
    HEAD_128,
    HEAD_DIM_576,
    HEAD_DIM_512,
    FUSED_REDUCESUM,
};

class SparseBwdImplBase : public ImplBase<
    SparseAttnBwdParams,
    SparseBwdFeatures
> {};

class SparseBwd_Sm100_Impl : public SparseBwdImplBase {
    // CUTLASS-3 SM100 warp-specialized sparse MLA backward. Built with B_H=64
    // (per-CTA token batch) -- matches DSA production h_q=64.
    DECLARE_SUPPORTED_FEATURES(
        SparseBwdFeatures::HEAD_64,
        SparseBwdFeatures::HEAD_DIM_576,
        SparseBwdFeatures::FUSED_REDUCESUM
    )

protected:
    void run_(const SparseAttnBwdParams& params,
              const std::vector<FeatureT>& required_features) override {
        // DSA only uses d_qk=576; the 512 case would have D_QK_PARTIAL=170
        // (=512/3) which violates SM100 UMMA BLK_K0 multiple-of-8 constraint.
        TORCH_CHECK(params.d_qk == 576,
            "FlashMLA sparse bwd currently only supports d_qk=576 (DSA absorbed-MLA);"
            " got d_qk=", params.d_qk);
        sm100::sparse_bwd::launch(const_cast<SparseAttnBwdParams&>(params));
    }
};

// Host-side torch op. Inputs and outputs follow the contract in
// the docstring of flash_mla_sparse_bwd.
//
// Returns:
//   { dq, dkv, kl_target }   if fuse_reducesum is true
//   { dq, dkv }              otherwise (kl_target is an empty tensor)
static std::vector<at::Tensor> sparse_attn_prefill_bwd_interface(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& o,
    const at::Tensor& do_grad,
    const at::Tensor& indices,
    const at::Tensor& lse,
    const at::Tensor& dq_acc_workspace,   // [s_q*h_q*d_qk] fp32
    const at::Tensor& dkv_acc_workspace,  // [s_kv*d_qk]    fp32
    float sm_scale,
    int d_v,
    bool fuse_reducesum
) {
    using bf16 = cutlass::bfloat16_t;

    Arch arch = Arch();
    bool is_sm100f = arch.is_sm100f();
    TORCH_CHECK(is_sm100f,
        "Sparse Attention Backward is currently only supported on SM100f (B200/B300). "
        "SM90 port is not yet available.");

    KU_CHECK_NDIM(q, 3);
    KU_CHECK_NDIM(kv, 3);
    KU_CHECK_NDIM(o, 3);
    KU_CHECK_NDIM(do_grad, 3);
    KU_CHECK_NDIM(indices, 3);
    KU_CHECK_NDIM(lse, 2);

    int s_q  = q.size(0);
    int s_kv = kv.size(0);
    int h_q  = q.size(1);
    int h_kv = kv.size(1);
    int d_qk = q.size(2);
    int topk = indices.size(2);

    TORCH_CHECK(d_qk == 576 || d_qk == 512, "Invalid d_qk: ", d_qk);
    TORCH_CHECK(d_v == 512,                "Invalid d_v: ", d_v);
    TORCH_CHECK(h_kv == 1,                  "MLA expects h_kv == 1, got ", h_kv);

    KU_CHECK_DEVICE(q); KU_CHECK_DEVICE(kv); KU_CHECK_DEVICE(o);
    KU_CHECK_DEVICE(do_grad); KU_CHECK_DEVICE(indices); KU_CHECK_DEVICE(lse);

    KU_CHECK_DTYPE(q, torch::kBFloat16);
    KU_CHECK_DTYPE(kv, torch::kBFloat16);
    KU_CHECK_DTYPE(o, torch::kBFloat16);
    KU_CHECK_DTYPE(do_grad, torch::kBFloat16);
    KU_CHECK_DTYPE(indices, torch::kInt32);
    KU_CHECK_DTYPE(lse, torch::kFloat32);

    KU_CHECK_SHAPE(q,        s_q, h_q, d_qk);
    KU_CHECK_SHAPE(kv,       s_kv, h_kv, d_qk);
    KU_CHECK_SHAPE(o,        s_q, h_q, d_v);
    KU_CHECK_SHAPE(do_grad,  s_q, h_q, d_v);
    KU_CHECK_SHAPE(indices,  s_q, h_kv, topk);
    KU_CHECK_SHAPE(lse,      s_q, h_q);

    KU_CHECK_LAST_DIM_CONTIGUOUS(q);
    KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(o);
    KU_CHECK_LAST_DIM_CONTIGUOUS(do_grad);
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    KU_CHECK_LAST_DIM_CONTIGUOUS(lse);

    // workspace shape/dtype validation.
    KU_CHECK_DEVICE(dq_acc_workspace);
    KU_CHECK_DEVICE(dkv_acc_workspace);
    KU_CHECK_DTYPE(dq_acc_workspace,  torch::kFloat32);
    KU_CHECK_DTYPE(dkv_acc_workspace, torch::kFloat32);
    TORCH_CHECK(dq_acc_workspace.numel()  >= (int64_t)s_q * h_q * d_qk,
        "dq_acc_workspace too small: need ", (int64_t)s_q * h_q * d_qk,
        " floats, got ", dq_acc_workspace.numel());
    TORCH_CHECK(dkv_acc_workspace.numel() >= (int64_t)s_kv * d_qk,
        "dkv_acc_workspace too small: need ", (int64_t)s_kv * d_qk,
        " floats, got ", dkv_acc_workspace.numel());

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto opts_bf16  = q.options();
    auto opts_float = q.options().dtype(torch::kFloat);

    at::Tensor dq        = torch::zeros({s_q,  h_q,  d_qk}, opts_bf16);
    at::Tensor dkv       = torch::zeros({s_kv, h_kv, d_qk}, opts_bf16);
    at::Tensor d_row     = torch::empty({s_q,  h_q},         opts_float);
    at::Tensor kl_target = fuse_reducesum
        ? torch::zeros({s_q, topk}, opts_float)
        : at::Tensor();

    SparseAttnBwdParams params = {
        s_q, s_kv, h_q, h_kv, d_qk, d_v, topk,
        sm_scale, sm_scale * LOG_2_E,

        (bf16*)q.data_ptr(),
        (bf16*)kv.data_ptr(),
        (bf16*)o.data_ptr(),
        (bf16*)do_grad.data_ptr(),
        (int*)indices.data_ptr(),
        (float*)lse.data_ptr(),
        (float*)d_row.data_ptr(),

        // caller-supplied fp32 workspaces (replace internal alloc).
        (float*)dq_acc_workspace.data_ptr(),
        (float*)dkv_acc_workspace.data_ptr(),

        int64_stride_to_int(q.stride(0)),       int64_stride_to_int(q.stride(1)),
        int64_stride_to_int(kv.stride(0)),      int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(o.stride(0)),       int64_stride_to_int(o.stride(1)),
        int64_stride_to_int(do_grad.stride(0)), int64_stride_to_int(do_grad.stride(1)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),

        (bf16*)dq.data_ptr(),
        (bf16*)dkv.data_ptr(),
        fuse_reducesum ? (float*)kl_target.data_ptr() : nullptr,

        int64_stride_to_int(dq.stride(0)),  int64_stride_to_int(dq.stride(1)),
        int64_stride_to_int(dkv.stride(0)), int64_stride_to_int(dkv.stride(1)),

        arch.num_sms,
        at::cuda::getCurrentCUDAStream().stream(),
        {}   // tensor_map_kv -- built inside run_bwd_kernel (.cu compilation
             // unit, where CUTLASS_CUDA_DRIVER_WRAPPER_CALL resolves).
    };

    std::vector<SparseBwdFeatures> required_features;
    if (h_q == 128)      required_features.push_back(SparseBwdFeatures::HEAD_128);
    else if (h_q == 64)  required_features.push_back(SparseBwdFeatures::HEAD_64);
    else TORCH_CHECK(false, "Unsupported h_q: ", h_q);
    if (d_qk == 576)     required_features.push_back(SparseBwdFeatures::HEAD_DIM_576);
    else if (d_qk == 512)required_features.push_back(SparseBwdFeatures::HEAD_DIM_512);
    if (fuse_reducesum)  required_features.push_back(SparseBwdFeatures::FUSED_REDUCESUM);

    // Kernel uses B_H=64 internally (per-CTA token batch), so dispatch h_q==64.
    // h_q==128 case would need a separate kernel variant (follow-up).
    if (h_q == 64) {
        SparseBwd_Sm100_Impl bwd_impl;
        bwd_impl.run(params, required_features);
    } else {
        TORCH_CHECK(false, "h_q=", h_q, " not yet wired; kernel currently built for h_q=64.");
    }

    return fuse_reducesum
        ? std::vector<at::Tensor>{dq, dkv, kl_target}
        : std::vector<at::Tensor>{dq, dkv};
}
