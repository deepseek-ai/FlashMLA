#pragma once
//
// Block-sparse MLA backward parameters (MSA-inspired KV-outer design; see
// docs/MSA_BLOCK_SPARSE_KV_OUTER_BWD_DESIGN.md). Distinct from the per-token
// SparseAttnBwdParams (bwd/params.h): sparsity is per-(query -> KV BLOCK) and
// the kernel runs KV-outer (one CTA per KV block) using the k2q CSR, so dK/dV
// accumulate-then-store with no atomicAdd scatter.
//
// Layout contract (single KV head, MLA: h_kv == 1):
//   q   [s_q, h_q, d_qk]                 bf16
//   kv  [s_kv, 1, d_qk]                  bf16   (V == kv[:, :, :d_v])
//   o   [s_q, h_q, d_v]                  bf16   (fwd output, saved)
//   do  [s_q, h_q, d_v]                  bf16   (upstream grad)
//   lse [s_q, h_q]                       float  (BASE-2)
//   d_row [s_q, h_q]                     float  (sum_d(O*dO) precompute)
//
//   q2k_indices  [s_q, topk_blocks]      int32  (KV-block ids per query, -1=pad)
//   k2q_row_ptr  [num_kv_blocks + 1]     int32  (built by build_k2q_csr)
//   k2q_q_indices[s_q * topk_blocks]     int32  (attending q per block, CSR)
//
//   dq  [s_q, h_q, d_qk]                 bf16   (out; via FP32 REDUCE_ADD acc)
//   dkv [s_kv, 1, d_qk]                  bf16   (out; accumulate-then-store per block)
//
#include "cutlass/bfloat16.h"
#include <cuda.h>

struct BlockSparseAttnBwdParams {
    int s_q, s_kv;
    int h_q, h_kv;              // h_kv == 1 for MLA
    int d_qk, d_v;              // 576 / 512 for K2 DSA
    int kv_block_size;          // KV tokens per sparse block (e.g. 128)
    int num_kv_blocks;          // ceil(s_kv / kv_block_size)
    int topk_blocks;            // selected KV blocks per query (q2k_indices dim 1)
    float sm_scale, sm_scale_div_log2;

    // Inputs (bf16 unless noted)
    cutlass::bfloat16_t* __restrict__ q;
    cutlass::bfloat16_t* __restrict__ kv;
    cutlass::bfloat16_t* __restrict__ o;
    cutlass::bfloat16_t* __restrict__ do_grad;
    float*               __restrict__ lse;        // base-2
    float*               __restrict__ d_row;      // sum_OdO precompute

    // Block-sparse selection + reverse CSR (k2q built host-side via build_k2q_csr).
    int* __restrict__ q2k_indices;     // [s_q, topk_blocks]
    int* __restrict__ k2q_row_ptr;     // [num_kv_blocks + 1]
    int* __restrict__ k2q_q_indices;   // [s_q * topk_blocks]

    // Caller-supplied FP32 workspaces (mirror the dense FMHA BWD pattern).
    //   dq_acc_workspace  : [s_q * h_q * d_qk] fp32, TMA_DQ REDUCE_ADD target.
    //   dkv_acc_workspace : [s_kv * d_qk]       fp32, dKV accumulate target
    //                       (KV-outer => written once per block, not scattered).
    float* __restrict__ dq_acc_workspace;
    float* __restrict__ dkv_acc_workspace;

    // Strides (element units), leading 2 dims (last dim contiguous).
    int stride_q_s_q,   stride_q_h;
    int stride_kv_s_kv, stride_kv_h;
    int stride_o_s_q,   stride_o_h;
    int stride_do_s_q,  stride_do_h;

    // Outputs
    cutlass::bfloat16_t* __restrict__ dq;    // [s_q, h_q, d_qk]
    cutlass::bfloat16_t* __restrict__ dkv;   // [s_kv, 1, d_qk] (dV in [:, :, :d_v])
    int stride_dq_s_q,  stride_dq_h;
    int stride_dkv_s_kv, stride_dkv_h;

    int num_sms;
    cudaStream_t stream;

    // TMA descriptors (built host-side, like the per-token sparse bwd). Block
    // gather is contiguous per KV block (no gather4-per-key). Filled in the host
    // wrapper (Phase 2).
    CUtensorMap tensor_map_kv;
    CUtensorMap tensor_map_q;
    CUtensorMap tensor_map_do;
    CUtensorMap tensor_map_dq;
    CUtensorMap tensor_map_dkv;
};
