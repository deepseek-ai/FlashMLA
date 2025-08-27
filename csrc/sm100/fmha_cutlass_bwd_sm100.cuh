/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/


#pragma once

#include <iostream>
#include <random>
#include <regex>

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/kernel_hardware_info.h>

#include <cutlass/util/command_line.h>
#include <cutlass/util/distribution.h>
#include <cutlass/util/reference/device/tensor_fill.h>

#include "common/utils.hpp"
#include "collective/fmha_fusion.hpp"
#include "device/fmha_device_bwd.hpp"

using namespace cute;
using namespace cutlass::fmha::kernel;
using namespace cutlass::fmha::collective;
using namespace cutlass::fmha;
using namespace cutlass;


template<
  class DType,
  bool kIsVarlen,
  bool kIsMla,
  class TileShape,
  class ActiveMask
>
struct BwdRunner {

  using Element = DType;
  using ElementAccumulator = float;

  // Q K D D_VO (H B)
  using ProblemShape = std::conditional_t<
    kIsVarlen,
    cute::tuple<VariableLength, VariableLength, int, int, cute::tuple<int, int>>,
    cute::tuple<int, int, int, int, cute::tuple<int, int>>
  >;

  using Operation = cutlass::fmha::device::Sm100FmhaBwd<ProblemShape, Element, ElementAccumulator, TileShape, kIsMla, ActiveMask>;
  
  using TensorStride = Stride<int, _1, Stride<int, int>>; 
  using StrideQ = TensorStride;                               // Seq DQK (H B)
  using StrideK = TensorStride;                               // Seq DQK (H B)
  using StrideV = TensorStride;                               // Seq DVO (H B)
  using StrideO = TensorStride;                               // Seq DVO (H B)
  using StrideLSE = Stride<_1, Stride<int, int>>;             // Seq (H B)

  // Backwards specific
  using StrideDQ = TensorStride;
  using StrideDK = TensorStride;                              // Seq DQK (H B)
  using StrideDV = TensorStride;                              // Seq DVO (H B)
  using StrideDO = TensorStride;

  static void run(at::Tensor workspace_buffer, at::Tensor d_o, at::Tensor q, at::Tensor k,
                  at::Tensor v, at::Tensor o, at::Tensor lse,
                  at::Tensor cumulative_seqlen_q, at::Tensor cumulative_seqlen_kv,
                  at::Tensor dq, at::Tensor dk, at::Tensor dv,
                  float softmax_scale, int max_seqlen_q, int max_seqlen_kv) {
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
    ProblemShape problem_shape;
    cute::tuple<int, int, int, int, cute::tuple<int, int>> tensor_shape;


    int d = q.size(-1);
    int d_vo = v.size(-1);
    int batch_size = cumulative_seqlen_q.size(0) - 1;
    int num_qo_heads = q.size(1);
    int total_seqlen_q = q.size(0);
    int total_seqlen_kv = k.size(0);
    
    //varlen: q: [Q, H, D]
    //fixedlen: q: [B, H, Q, D] 
    if constexpr (kIsVarlen) {
      problem_shape = cute::make_tuple(
        VariableLength{max_seqlen_q, static_cast<int*>(cumulative_seqlen_q.data_ptr()), total_seqlen_q},
        VariableLength{max_seqlen_kv, static_cast<int*>(cumulative_seqlen_kv.data_ptr()), total_seqlen_kv},
        d, d_vo, cute::make_tuple(num_qo_heads, batch_size));
      tensor_shape = make_shape(total_seqlen_q, total_seqlen_kv, d, d_vo, make_shape(num_qo_heads, 1));
    } else {
      int q_len = total_seqlen_q / batch_size;
      int kv_len = total_seqlen_kv / batch_size;
      problem_shape = cute::make_tuple(q_len, kv_len, d, d_vo, cute::make_tuple(num_qo_heads, batch_size));
      tensor_shape = problem_shape;
    }

    auto [Q, K, D, D_VO, HB] = tensor_shape;
    auto [H, B] = HB;

    int q_stride0 = q.stride(0), q_stride1 = q.stride(1), q_stride2 = q.stride(2);
    int k_stride0 = k.stride(0), k_stride1 = k.stride(1), k_stride2 = k.stride(2);
    int v_stride0 = v.stride(0), v_stride1 = v.stride(1), v_stride2 = v.stride(2);
    int o_stride0 = o.stride(0), o_stride1 = o.stride(1), o_stride2 = o.stride(2);
    int lse_stride0 = lse.stride(0), lse_stride1 = lse.stride(1);
    int dq_stride0 = dq.stride(0), dq_stride1 = dq.stride(1), dq_stride2 = dq.stride(2);
    int dk_stride0 = dk.stride(0), dk_stride1 = dk.stride(1), dk_stride2 = dk.stride(2);
    int dv_stride0 = dv.stride(0), dv_stride1 = dv.stride(1), dv_stride2 = dv.stride(2);
    int do_stride0 = d_o.stride(0), do_stride1 = d_o.stride(1), do_stride2 = d_o.stride(2);
    TORCH_CHECK(q_stride2 == 1);
    TORCH_CHECK(k_stride2 == 1);
    TORCH_CHECK(v_stride2 == 1);
    TORCH_CHECK(o_stride2 == 1);
    TORCH_CHECK(lse_stride0 == 1);
    TORCH_CHECK(dq_stride2 == 1);
    TORCH_CHECK(dk_stride2 == 1);
    TORCH_CHECK(dv_stride2 == 1);
    TORCH_CHECK(do_stride2 == 1);

    StrideQ stride_Q = make_stride(q_stride0, _1{}, make_stride(q_stride1, B == 1 ? 0 : q_stride0*Q));
    StrideK stride_K = make_stride(k_stride0, _1{}, make_stride(k_stride1, B == 1 ? 0 : k_stride0*K));
    StrideV stride_V = make_stride(v_stride0, _1{}, make_stride(v_stride1, B == 1 ? 0 : v_stride0*K));
    StrideO stride_O = make_stride(o_stride0, _1{}, make_stride(o_stride1, B == 1 ? 0 : o_stride0*Q));
    StrideLSE stride_LSE = make_stride(_1{}, make_stride(lse_stride1, B == 1 ? 0 : Q));

    StrideDQ stride_dQ = make_stride(dq_stride0, _1{}, make_stride(dq_stride1, B == 1 ? 0 : dq_stride0*Q));
    StrideDK stride_dK = make_stride(dk_stride0, _1{}, make_stride(dk_stride1, B == 1 ? 0 : dk_stride0*K));
    StrideDV stride_dV = make_stride(dv_stride0, _1{}, make_stride(dv_stride1, B == 1 ? 0 : dv_stride0*K));
    StrideDO stride_dO = make_stride(do_stride0, _1{}, make_stride(do_stride1, B == 1 ? 0 : do_stride0*Q));

    typename Operation::Arguments arguments{
      problem_shape,
      (static_cast<Element*>(q.data_ptr())), stride_Q,
      (static_cast<Element*>(k.data_ptr())), stride_K,
      (static_cast<Element*>(v.data_ptr())), stride_V,
      (static_cast<Element*>(o.data_ptr())), stride_O,
      (static_cast<ElementAccumulator*>(lse.data_ptr())), stride_LSE,
      (static_cast<Element*>(d_o.data_ptr())), stride_dO,
      (static_cast<Element*>(dq.data_ptr())), stride_dQ,
      (static_cast<Element*>(dk.data_ptr())), stride_dK,
      (static_cast<Element*>(dv.data_ptr())), stride_dV,
      static_cast<ElementAccumulator>(softmax_scale),
      hw_info
    };

    Operation op;

    uint8_t* workspace_ptr = static_cast<uint8_t*>(workspace_buffer.data_ptr());

    CUTLASS_CHECK(op.can_implement(arguments));
    CUTLASS_CHECK(op.initialize(arguments, workspace_ptr));
    CUTLASS_CHECK(op.run(at::cuda::getCurrentCUDAStream()));
  }

};


template <typename DType, bool kIsVarlen, bool kIsMla, typename TileShape, typename Mask>
void run_fmha_bwd(at::Tensor workspace_buffer, at::Tensor d_o, at::Tensor q, at::Tensor k,
                  at::Tensor v, at::Tensor o, at::Tensor lse,
                  at::Tensor cumulative_seqlen_q, at::Tensor cumulative_seqlen_kv,
                  at::Tensor dq, at::Tensor dk, at::Tensor dv,
                  float softmax_scale, int max_seqlen_q, int total_seqlen_kv) {
  BwdRunner<DType, kIsVarlen, kIsMla, TileShape, Mask>::run(workspace_buffer, d_o, q, k, v, o, lse,
                                                     cumulative_seqlen_q, cumulative_seqlen_kv,
                                                     dq, dk, dv,
                                                     softmax_scale, max_seqlen_q, total_seqlen_kv);
}
