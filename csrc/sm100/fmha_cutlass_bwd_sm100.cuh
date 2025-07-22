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
    
    //varlen: q: [Q, H, D]
    //fixedlen: q: [B, H, Q, D] 
    if constexpr (kIsVarlen) {
      int d = q.size(-1);
      int d_vo = v.size(-1);
      int batch_size = cumulative_seqlen_q.size(0) - 1;
      int num_qo_heads = q.size(1);
      int total_seqlen_q = q.size(0);
      int total_seqlen_kv = k.size(0);

      problem_shape = cute::make_tuple(
        VariableLength{max_seqlen_q, static_cast<int*>(cumulative_seqlen_q.data_ptr()), total_seqlen_q},
        VariableLength{max_seqlen_kv, static_cast<int*>(cumulative_seqlen_kv.data_ptr()), total_seqlen_kv},
        d, d_vo, cute::make_tuple(num_qo_heads, batch_size));
      tensor_shape = make_shape(total_seqlen_q, total_seqlen_kv, d, d_vo, make_shape(num_qo_heads, 1));
    } else {
      int q_len = q.size(1);
      int kv_len = k.size(1);
      int d = q.size(-1);
      int d_vo = v.size(-1);
      int batch_size = q.size(0);
      int num_qo_heads = q.size(2);

      problem_shape = cute::make_tuple(q_len, kv_len, d, d_vo, cute::make_tuple(num_qo_heads, batch_size));
      tensor_shape = problem_shape;
    }

    auto [Q, K, D, D_VO, HB] = tensor_shape;
    auto [H, B] = HB;

    StrideQ stride_Q = make_stride(H*D, _1{}, make_stride(D, B == 1 ? 0 : D*Q*H));
    StrideK stride_K = make_stride(H*D, _1{}, make_stride(D, B == 1 ? 0 : D*K*H));
    StrideV stride_V = make_stride(H*D_VO, _1{}, make_stride(D_VO, B == 1 ? 0 : D_VO*K*H));
    StrideO stride_O = make_stride(H*D_VO, _1{}, make_stride(D_VO, B == 1 ? 0 : D_VO*Q*H));
    StrideLSE stride_LSE = make_stride(_1{}, make_stride(Q, B == 1 ? 0 : Q*H));

    StrideDQ stride_dQ = stride_Q;
    StrideDK stride_dK = stride_K;
    StrideDV stride_dV = stride_V;
    StrideDO stride_dO = stride_O;

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

    size_t workspace_size = 0;
    workspace_size = Operation::get_workspace_size(arguments);
    DeviceAllocation<uint8_t> workspace(workspace_size);
    uint8_t* workspace_ptr = workspace.get();

    CUTLASS_CHECK(op.can_implement(arguments));
    CUTLASS_CHECK(op.initialize(arguments, workspace.get()));
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
