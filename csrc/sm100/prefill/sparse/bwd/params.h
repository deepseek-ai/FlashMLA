#pragma once

#include "cutlass/bfloat16.h"
#include <cuda.h>           // CUtensorMap

// Sparse MLA backward parameters. Mirrors SparseAttnFwdParams in
// FlashMLA's top-level params.h plus the gradient-side tensors.
//
// Layout contract :
//   q   [s_q, h_q, d_qk]                bf16
//   kv  [s_kv, h_kv=1, d_qk]            bf16   (V == kv[:, :, :d_v])
//   o   [s_q, h_q, d_v]                 bf16   (forward output, saved)
//   do  [s_q, h_q, d_v]                 bf16   (upstream grad)
//   indices [s_q, h_kv=1, topk]         int32  (causal-respecting top-k, -1 = pad)
//   lse [s_q, h_q]                      float  (BASE-2; multiply external base-e LSE by log2(e))
//   d_row [s_q, h_q]                    float  (precomputed sum_d(O*dO), see sum_OdO.cuh)
//
//   dq  [s_q, h_q, d_qk]                bf16   (out)
//   dkv [s_kv, h_kv=1, d_qk]            bf16   (out; dV occupies dkv[:, :, :d_v])
//   kl_target [s_q, topk]               float  (out, only when fused reducesum is on)
//
// All tensors are last-dim contiguous; strides supplied for the leading 2.
struct SparseAttnBwdParams {
    int s_q, s_kv;
    int h_q, h_kv;          // h_kv == 1 for MLA
    int d_qk, d_v;          // 576 / 512 for DSA absorbed MLA
    int topk;               // 2048 in production
    float sm_scale, sm_scale_div_log2;

    // Inputs (all bf16 unless noted)
    cutlass::bfloat16_t* __restrict__ q;
    cutlass::bfloat16_t* __restrict__ kv;
    cutlass::bfloat16_t* __restrict__ o;
    cutlass::bfloat16_t* __restrict__ do_grad;   // 'do' is a C++ keyword
    int*                 __restrict__ indices;
    float*               __restrict__ lse;       // base-2
    float*               __restrict__ d_row;     // sum_OdO precompute

    // caller-supplied fp32 workspaces (mirrors dense FMHA BWD pattern).
    //   dq_acc_workspace  : [s_q * h_q * d_qk] fp32, TMA_DQ REDUCE_ADD target
    //   dkv_acc_workspace : [s_kv * d_qk]       fp32, dKV scatter atomic target
    // Caller (Python wrapper) allocates persistent
    // buffers and passes them; the kernel no longer allocates internally.
    float*               __restrict__ dq_acc_workspace;
    float*               __restrict__ dkv_acc_workspace;

    // Strides -- same convention as fwd, in element units
    int stride_q_s_q,      stride_q_h;
    int stride_kv_s_kv,    stride_kv_h;
    int stride_o_s_q,      stride_o_h;
    int stride_do_s_q,     stride_do_h;
    int stride_indices_s_q,stride_indices_h;

    // Outputs
    cutlass::bfloat16_t* __restrict__ dq;       // [s_q, h_q, d_qk]
    cutlass::bfloat16_t* __restrict__ dkv;      // [s_kv, h_kv=1, d_qk]; cleared to 0 by host
    float*               __restrict__ kl_target;// [s_q, topk]; nullptr disables fusion

    int stride_dq_s_q,  stride_dq_h;
    int stride_dkv_s_kv,stride_dkv_h;

    int num_sms;
    cudaStream_t stream;

    // Sparse gather TMA descriptor for K and V (both come from `kv`). Built
    // host-side via cuTensorMapEncodeTiled -- identical layout to sparse fwd's
    // `tensor_map_kv` (same source tensor `kv`, same swizzle, same box size).
    // Consumed in the Load warp via `ku::tma_gather4_cta_group_2`.
    CUtensorMap tensor_map_kv;

    // TMA descriptors for output stores (built in run_bwd_kernel).
    //   tensor_map_dq        -- per-token regular store, 2-rank (D_QK, s_q).
    //   tensor_map_dkv_scatter -- scatter via indices, used for dV/dK to dkv.
    //   tensor_map_kl_target -- per-token kl_target store, 2-rank (topk, s_q).
    // For now tensor_map_dq + tensor_map_kl_target are built; dkv_scatter
    // comes with the Reduce warp's scatter store (follow-up).
    CUtensorMap tensor_map_dq;
    CUtensorMap tensor_map_kl_target;     // [s_q, topk] float, fused reducesum output

    // Input-side TMA descriptors used by the Load warp.
    //   tensor_map_q   -- Q loaded per-Q-token (full D_QK=576), 2-rank.
    //   tensor_map_do  -- dO loaded per-Q-token (full D_V=512), 2-rank.
    CUtensorMap tensor_map_q;
    CUtensorMap tensor_map_do;

    // Scatter destination descriptor for dV/dK -> dkv at indices[k_pos]
    // (per-row SM90_TMA_REDUCE_ADD_2D issues; standard scatter4 not in atom set).
    CUtensorMap tensor_map_dkv;

    // KV-outer block-sparse (Phase 2b, 576/512): k2q CSR (built host-side from the
    // per-q-token block selection). At the END of the struct so the existing
    // positional aggregate-init in sparse_bwd.h leaves them value-initialized
    // (nullptr/0 => the original per-token Q-outer + atomicAdd-scatter path).
    // When k2q_row_ptr != nullptr the 576 KV-outer kernel runs (CTA = KV-block,
    // iterate attending q-tokens via the CSR; contiguous block, accumulate-store).
    const int* k2q_row_ptr = nullptr;     // [num_kv_blocks + 1]
    const int* k2q_q_indices = nullptr;   // [s_q * topk_blocks] attending q-token ids per KV-block
    int num_kv_blocks = 0;
};
