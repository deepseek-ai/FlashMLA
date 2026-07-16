/***************************************************************************************************
 * Copyright (c) 2025  - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cute/arch/simd_sm100.hpp"

#include "cutlass/arch/arch.h"
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include <kerutils/kerutils.cuh> // for  KERUTILS_ENABLE_SM100A

// include dense MLA bwd's collective deps via absolute path. The original
// `#include "../collective/fmha_common.hpp"` resolved against the dense kernel dir;
// our v2 lives under `sparse/bwd/v2/` so we go up three levels to reach
// `prefill/dense/collective/`.
#include "../../dense/collective/fmha_common.hpp"
#include "../../dense/collective/fmha_fusion.hpp"

// kerutils sparse-gather intrinsic (the sparse variant will swap in for K/V TMA load).
#include <kerutils/device/sm100/intrinsics.cuh>

#include <cmath>

// keep the inner struct under cutlass::fmha::kernel for type compatibility
// with the dense MLA bwd's collective/util headers, but rename the struct itself so
// it can coexist with `Sm100SparseBwdMlaKernelTmaWarpSpecialized` (dense path) in the
// same translation unit.
namespace cutlass::fmha::kernel {

using namespace cutlass::fmha::collective;

using namespace cute;

template<
    class ProblemShape,
    class Element,
    class ElementAcc,
    class TileShape,
    class Mask
>
struct Sm100SparseBwdMlaKernelTmaWarpSpecialized {

  using TileShapeQ = decltype(get<0>(TileShape{}));
  using TileShapeK = decltype(get<1>(TileShape{}));
  using TileShapeDQK = decltype(get<2>(TileShape{}));
  using TileShapeDVO = decltype(get<3>(TileShape{}));

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  struct TmemAllocation {
    static constexpr uint32_t kDK = 0;                     // TileShapeK x TileShapeDQK x acc
    static constexpr uint32_t kDV = kDK + TileShapeDQK{};  // TileShapeK x TileShapeDVO x acc
    static constexpr uint32_t kDQ = kDV + TileShapeDVO{};  // TileShapeQ x TileShapeDQK x acc
    static constexpr uint32_t kDP = kDQ;                   // TileShapeK x TileShapeQ   x inp
    static constexpr uint32_t kS = kDQ + 65536 * 16;
    static constexpr uint32_t kP = kS;
    static constexpr uint32_t kTotal = kDQ + TileShapeDQK{};
  };

  static_assert(
      static_cast<int>(TmemAllocation::kTotal) <= TmemAllocator::Sm100TmemCapacityColumns,
      "using too much tmem"
  );

  enum class WarpRole {
    Empty = 0x0, Load = 0x1, Mma = 0x2, Compute = 0x3, Reduce = 0x4
  };

  static constexpr unsigned long long kWarpAssignment = 0x12'3333'3333'4444ull;
  static constexpr int kNumComputeWarps = 8;
  static constexpr int kNumReduceWarps = 4;

  static constexpr int kLoadPerThread = TileShapeQ{} / NumThreadsPerWarp;
  static_assert(TileShapeQ{} % NumThreadsPerWarp == 0, "TileShapeQ must be divisible by NumThreadsPerWarp");
  CUTLASS_DEVICE WarpRole warp_idx_to_role(int warp_idx) {
    return static_cast<WarpRole>((kWarpAssignment >> (4 * warp_idx)) & 0xF);
  }

  struct RegisterAllocation {
    static constexpr int kWarpgroup0 = 160-8;
    static constexpr int kWarpgroup1 = 128;
    static constexpr int kWarpgroup2 = 96;
    static constexpr int kReduce = kWarpgroup0;
    static constexpr int kCompute = kWarpgroup1;
    static constexpr int kMma = kWarpgroup2;
    static constexpr int kEmpty = kWarpgroup2;
    static constexpr int kLoad = kWarpgroup2;

    static_assert(kWarpgroup0 + 2 * kWarpgroup1 + kWarpgroup2 <= 512);
  };

  using ArchTag = cutlass::arch::Sm100;

  // cluster<3,1,1> (cross-CTA partial-S reduction) hit "unspecified launch
  // failure" at runtime and was reverted; the D_QK split is handled by giving
  // each of 3 CTAs its own 192-col slice instead (see grid layout).
  using ClusterShape = Shape<_1, _1, _1>;
  using Schedule = cutlass::gemm::KernelTmaWarpSpecialized1SmSm100;

  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int kNumWarps = kNumComputeWarps + kNumReduceWarps + 4;
  static constexpr int MaxThreadsPerBlock = NumThreadsPerWarp * kNumWarps;

  static constexpr int Alignment = 128 / sizeof_bits_v<Element>;
  static constexpr int kStages = 2;

  using TensorStrideContiguousK = Stride<int, _1, Stride<int, int>>;
  using TensorStrideContiguousMN = Stride<_1, int, Stride<int, int>>;

  // compute S
  using CollectiveMmaQK = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      Element, TensorStrideContiguousK, Alignment,
      Element, TensorStrideContiguousK, Alignment,
      ElementAcc,
      Shape<TileShapeQ, TileShapeK, TileShapeDQK>,
      ClusterShape, cutlass::gemm::collective::StageCount<kStages>,
      Schedule>::CollectiveOp;
  using TileShapeQK = typename CollectiveMmaQK::TileShape;
  using TiledMmaQK = typename CollectiveMmaQK::TiledMma;

  // compute dP
  using CollectiveMmaDOV = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      Element, TensorStrideContiguousK, Alignment,
      Element, TensorStrideContiguousK, Alignment,
      ElementAcc,
      Shape<TileShapeQ, TileShapeK, TileShapeDVO>,
      ClusterShape, cutlass::gemm::collective::StageCount<kStages>,
      Schedule>::CollectiveOp;
  using TileShapeDOV = typename CollectiveMmaDOV::TileShape;
  using TiledMmaDOV = typename CollectiveMmaDOV::TiledMma;

  // compute dV
  using CollectiveMmaPDO = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      // needs to match ordering of S calculation
      Element, TensorStrideContiguousK, Alignment,
      Element, TensorStrideContiguousMN, Alignment,
      ElementAcc,
      Shape<TileShapeK, TileShapeDVO, TileShapeQ>,
      ClusterShape, cutlass::gemm::collective::StageCount<kStages>,
      Schedule>::CollectiveOp;
  using TileShapePDO = typename CollectiveMmaPDO::TileShape;
  using TiledMmaPDO = typename CollectiveMmaPDO::TiledMma;

  // compute dK
  using CollectiveMmaDSQ = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      // somewhat arbitrary since we dump to smem, need to agree with the next one
      Element, TensorStrideContiguousK , Alignment,
      Element, TensorStrideContiguousMN, Alignment,
      ElementAcc,
      Shape<TileShapeK, TileShapeDQK, TileShapeQ>,
      ClusterShape, cutlass::gemm::collective::StageCount<kStages>,
      Schedule>::CollectiveOp;
  using TileShapeDSQ = typename CollectiveMmaDSQ::TileShape;
  using TiledMmaDSQ = typename CollectiveMmaDSQ::TiledMma;

  // compute dQ
  using CollectiveMmaDSK = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      // somewhat arbitrary since we dump to smem, need to agree with the previous one
      Element, TensorStrideContiguousMN, Alignment,
      Element, TensorStrideContiguousMN, Alignment,
      ElementAcc,
      Shape<TileShapeQ, TileShapeDQK, TileShapeK>,
      ClusterShape, cutlass::gemm::collective::StageCount<kStages>,
      Schedule>::CollectiveOp;
  using TileShapeDSK = typename CollectiveMmaDSK::TileShape;
  using TiledMmaDSK = typename CollectiveMmaDSK::TiledMma;

  // pipelines are named Pipeline<Producer><Consumer><Resource>
  static constexpr int kStagesComputeSmem = 1;
  // D path: pipeline kStages 2 -> 1 forces Load+MMA strict serial (Load cannot
  // race-ahead within iter). Fixes K SMEM race where Load's phase N+1 K write
  // overwrites slot 0 before MMA chunk N reads it. SMEM unchanged.
  using PipelineLoadMmaQ = PipelineTmaUmmaAsync<1, ClusterShape>;
  using PipelineLoadMmaDO = PipelineTmaUmmaAsync<1, ClusterShape>;
  using PipelineLoadComputeLSE = PipelineAsync<1>;
  using PipelineLoadComputeSumOdO = PipelineAsync<1>;
  using PipelineMmaComputeS = PipelineUmmaAsync<1>;
  using PipelineMmaComputeDP = PipelineUmmaAsync<1>;
  using PipelineMmaReduceDQ = PipelineUmmaAsync<1>;
  using PipelineComputeMmaP = PipelineUmmaConsumerAsync<1>;
  using PipelineComputeMmaDS = PipelineUmmaConsumerAsync<kStagesComputeSmem>;
  // Note: kStages=2 + compute restructure. kStages=1 hung at 879;
  // kStages=8 left race window (882 dV cos 0.07-0.27 for chunks 0..2). kStages=2
  // tighter buffer: MMA can have 1 outstanding ahead of consumer release, but
  // chunk c+2 blocks at acquire until consumer released chunk c. TMEM race
  // narrowed to 1 chunk window (still possible but very small).
  using PipelineMmaComputeDKDV = PipelineUmmaAsync<2>;
  static constexpr int kStagesReduceTmaStore = 2;
  using PipelineReduceTmaStore = PipelineTmaStore<kStagesReduceTmaStore>;

  struct PipelineStorage {
    alignas(16) typename PipelineLoadMmaQ::SharedStorage load_mma_q;
    alignas(16) typename PipelineLoadMmaDO::SharedStorage load_mma_do;
    alignas(16) typename PipelineLoadComputeLSE::SharedStorage load_compute_lse;
    alignas(16) typename PipelineLoadComputeSumOdO::SharedStorage load_compute_sum_odo;
    alignas(16) typename PipelineMmaComputeS::SharedStorage mma_compute_s;
    alignas(16) typename PipelineMmaComputeDP::SharedStorage mma_compute_dp;
    alignas(16) typename PipelineMmaReduceDQ::SharedStorage mma_reduce_dq;
    alignas(16) typename PipelineComputeMmaP::SharedStorage compute_mma_p;
    alignas(16) typename PipelineComputeMmaDS::SharedStorage compute_mma_ds;
    alignas(16) typename PipelineMmaComputeDKDV::SharedStorage mma_compute_dkdv;
  };

  template<class Layout, class Stages = _1>
  static CUTE_DEVICE constexpr auto restage(Layout const& layout, Stages stages = {}) {
    return composition(layout, make_tuple(_, _, _, make_layout(stages)));
  }

  using SmemLayoutK = decltype(restage(typename CollectiveMmaQK::SmemLayoutB{}));
  using SmemLayoutV = decltype(restage(typename CollectiveMmaDOV::SmemLayoutB{}));
  using SmemLayoutQ = decltype(restage(typename CollectiveMmaQK::SmemLayoutA{}, _2{}));
  using SmemLayoutDO = decltype(restage(typename CollectiveMmaDOV::SmemLayoutA{}, _1{}));
  using SmemLayoutDS = decltype(restage(typename CollectiveMmaDSK::SmemLayoutA{}, Int<kStagesComputeSmem>{}));
  using SmemLayoutLSE = Layout<Shape<TileShapeQ, _1>>;
  using SmemLayoutSumOdO = Layout<Shape<TileShapeQ, _1>>;

  using SmemLayoutQT = decltype(restage(typename CollectiveMmaDSQ::SmemLayoutB{}, _2{}));
  using SmemLayoutKT = decltype(restage(typename CollectiveMmaDSK::SmemLayoutB{}));
  using SmemLayoutDST = decltype(restage(typename CollectiveMmaDSQ::SmemLayoutA{}, Int<kStagesComputeSmem>{}));
  using SmemLayoutDOT = decltype(restage(typename CollectiveMmaPDO::SmemLayoutB{}, _1{}));
  using SmemLayoutP = decltype(restage(typename CollectiveMmaPDO::SmemLayoutA{}, _1{}));
  using SmemLayoutPT = decltype(restage(typename CollectiveMmaDSK::SmemLayoutA{}, _1{}));

  using TileShapeDQ = _32;
  using SmemAtomDQ = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      cute::UMMA::Major::K, ElementAcc, TileShapeQ, TileShapeDQ
  >());
  using SmemShapeDQ = Shape<TileShapeQ, TileShapeDQ, Int<kStagesReduceTmaStore>>;
  using SmemLayoutDQ = decltype(tile_to_shape(SmemAtomDQ{}, SmemShapeDQ{}, Step<_2, _1, _3>{}));

  struct TensorStorage {
    union {
      alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutK>> smem_k;
      alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutKT>> smem_k_t;
    };
    alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutV>> smem_v;
    union {
      alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
      alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutQT>> smem_q_t;
    };
    union {
      alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutDO>> smem_do;
      alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutDOT>> smem_do_t;
    };
    union {
      alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutDS>> smem_ds;
      alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutDST>> smem_ds_t;
    };
    union{
      alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutP>> smem_p;
      alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutPT>> smem_p_t;
    };
    alignas(1024) cute::array<ElementAcc, cute::cosize_v<SmemLayoutDQ>> smem_dq;
    alignas(16) cute::array<ElementAcc, cute::cosize_v<SmemLayoutLSE>> smem_lse;
    alignas(16) cute::array<ElementAcc, cute::cosize_v<SmemLayoutSumOdO>> smem_sum_odo;
    // custom transac_bar_t for K gather4, mirroring sparse FWD's
    // pattern. We use this instead of the pipeline's mbar to verify whether
    // pipeline mbar + raw gather4 PTX is the incompatibility source.
    alignas(16) kerutils::transac_bar_t sparse_k_gather_bar;
  };

  static constexpr int kTransactionsBytesLoadQ = cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutQ{})) * cute::sizeof_bits_v<Element>);
  static constexpr int kTransactionsBytesLoadDO = cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutDO{})) * cute::sizeof_bits_v<Element>);

  static constexpr int kTransactionsBytesLoadK = cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutK{})) * cute::sizeof_bits_v<Element>);
  static constexpr int kTransactionsBytesLoadV = cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutV{})) * cute::sizeof_bits_v<Element>);

  struct SharedStorage {
    TensorStorage tensors;
    PipelineStorage pipelines;
    uint32_t tmem_base_ptr;
  };

  // this is tight enough that it won't work with sizeof due to padding for alignment
  static constexpr int SharedStorageSize = offsetof(SharedStorage, tmem_base_ptr) + sizeof(uint32_t);
  static_assert(SharedStorageSize <= cutlass::arch::sm100_smem_capacity_bytes, "using too much smem");

  using TensorStride = TensorStrideContiguousK;  // S D (H B)
  using RowTensorStride = Stride<_1, Stride<int, int>>;    // S (H B)

  // Index stride for the sparse indices tensor (per-q-token int32 [topk] view).
  using IndicesStride = Stride<int, _1, Stride<int, int>>;   // Seq Topk (H B)

  struct MainloopArguments {
    const Element* ptr_q;
    TensorStride stride_q;
    // Sparse K/V: one buffer (DSA absorbed-MLA); V occupies kv[:, :, :D_V].
    const Element* ptr_kv;
    TensorStride stride_kv;
    // Kept for type signature compatibility with helpers ported from dense MLA bwd.
    // The actual load fires through `tensor_map_kv` + ku::tma_gather4_cta_group_1.
    const Element* ptr_k;   // alias for ptr_kv
    TensorStride stride_k;  // alias for stride_kv
    const Element* ptr_v;   // alias for ptr_kv
    TensorStride stride_v;  // alias for stride_kv
    const Element* ptr_do;
    TensorStride stride_do;

    // Sparse indices: int32 [s_q, h_kv=1, topk].
    const int* ptr_indices;
    IndicesStride stride_indices;
    int topk;

    const ElementAcc* ptr_lse;
    RowTensorStride stride_lse;

    const ElementAcc* ptr_sum_odo;
    RowTensorStride stride_sum_odo;

    ElementAcc* ptr_dq_acc;
    TensorStride stride_dq_acc;

    ElementAcc softmax_scale = 1.0f / sqrtf(TileShapeDQK{});
  };

  using TMA_K = typename CollectiveMmaQK::Params::TMA_B;
  using TMA_V = typename CollectiveMmaDOV::Params::TMA_B;
  using TMA_Q = typename CollectiveMmaQK::Params::TMA_A;
  using TMA_DO = typename CollectiveMmaDOV::Params::TMA_A;

  using TMA_DQ = decltype(make_tma_copy(SM90_TMA_REDUCE_ADD{},
      make_tensor((const ElementAcc*)nullptr, make_shape(1, 1, make_shape(1, 1)), TensorStride{}),
      SmemLayoutDQ{}(_, _, _0{})
  ));

  struct MainloopParams {
    // Inherited from dense MLA bwd. tma_load_k/v are kept as compile-time
    // placeholders (some helpers reference their layout types); the actual
    // load fires through `tensor_map_kv` + ku::tma_gather4_cta_group_1.
    TMA_K tma_load_k;
    TMA_V tma_load_v;
    TMA_Q tma_load_q;
    TMA_DO tma_load_do;
    TMA_DQ tma_red_dq;

    // Sparse gather TMA descriptor for K and V (single buffer; V = kv[:, :, :D_V]).
    // Built host-side via cuTensorMapEncodeTiled in the run wrapper.
    CUtensorMap tensor_map_kv;

    // dKV scatter descriptor (per-row SM90_TMA_REDUCE_ADD_2D issues at row
    // indices[k_pos]; standard scatter4 not in atom set).
    CUtensorMap tensor_map_dkv;

    // kl_target [s_q, topk] FP32 store. FP32 + 64-elem-bf16-style box=64
    // hits CUDA_ERROR_INVALID_VALUE on cuTensorMapEncodeTiled; use SWIZZLE_NONE.
    CUtensorMap tensor_map_kl_target;
  };

  struct EpilogueArguments {
    Element* ptr_dk;
    TensorStride stride_dk;
    Element* ptr_dv;
    TensorStride stride_dv;
    // FP32 dKV accumulator workspace (deterministic atomicAdd target).
    // Shape [s_kv, h_kv=1, d_qk] FP32. Host allocates + zeros + post-casts to
    // bf16 dKV after kernel. Replaces non-deterministic bf16 atomicAdd ordering.
    ElementAcc* ptr_dkv_acc = nullptr;
    // Fused reducesum: kl_target = sum_h(P) per (q, k_pos). FP32.
    // Shape [s_q, topk]; nullptr disables the kl_target path.
    ElementAcc* ptr_kl_target = nullptr;
  };

  struct Arguments {
    ProblemShape problem_shape;
    MainloopArguments mainloop;
    EpilogueArguments epilogue;
    KernelHardwareInfo hw_info;
  };

  struct Params {
    ProblemShape problem_shape;
    MainloopArguments mainloop;
    MainloopParams mainloop_params;
    EpilogueArguments epilogue;
    KernelHardwareInfo hw_info;
  };


  static bool can_implement(Arguments const& args) {
    auto [Q, K, D, D_VO, HB] = args.problem_shape;
    auto [H, B] = HB;
    if (Q <= 0 || K <= 0 || D <= 0 || H <= 0 || B <= 0 || D_VO <= 0) {
      return false;
    }
    if (D % Alignment != 0 || D_VO % Alignment != 0) {
      return false;
    }
    return true;
  }


  static Status initialize_workspace(Arguments const&, void*, cudaStream_t) {
    return Status::kSuccess;
  }


  static Params to_underlying_arguments(Arguments const& args, void*) {
    auto [Q_, K_, D, D_VO, HB] = args.problem_shape;
    int Q = Q_;
    int K = K_;

    if constexpr (is_variable_length_v<decltype(Q_)>) {
      Q = Q_.total_length;
    }
    if constexpr (is_variable_length_v<decltype(K_)>) {
      K = K_.total_length;
    }

    auto params_kq = CollectiveMmaQK::to_underlying_arguments(
      make_shape(Q, K, D, HB),
      typename CollectiveMmaQK::Arguments {
        args.mainloop.ptr_q, args.mainloop.stride_q,
        args.mainloop.ptr_k, args.mainloop.stride_k,
      }, /*workspace=*/nullptr);

    auto params_vdo = CollectiveMmaDOV::to_underlying_arguments(
      make_shape(Q, K, D_VO, HB),
      typename CollectiveMmaDOV::Arguments {
        args.mainloop.ptr_do, args.mainloop.stride_do,
        args.mainloop.ptr_v, args.mainloop.stride_v,
      }, /*workspace=*/nullptr);

    TMA_DQ tma_red_dq = make_tma_copy(
        SM90_TMA_REDUCE_ADD{},
        make_tensor(args.mainloop.ptr_dq_acc, make_shape(Q_, D, HB), args.mainloop.stride_dq_acc),
        SmemLayoutDQ{}(_, _, _0{})
    );

    // Sparse CUtensorMap descriptors (gather K/V, scatter dKV, store kl_target).
    // Built via the host-side helper in sparse_bwd_host.cuh -- can't be done
    // here because to_underlying_arguments must be device-callable in some
    // CUTLASS paths and CUTLASS_CUDA_DRIVER_WRAPPER_CALL needs the driver shim
    // resolved in a .cu compilation unit. So we zero-initialize them here and
    // expect the host wrapper to fill them in before launch.
    CUtensorMap zero_map{};
    return Params{
      args.problem_shape,
      args.mainloop,
      MainloopParams{
        params_kq.tma_load_b,
        params_vdo.tma_load_b,
        params_kq.tma_load_a,
        params_vdo.tma_load_a,
        tma_red_dq,
        zero_map,   // tensor_map_kv     -- filled host-side
        zero_map,   // tensor_map_dkv    -- filled host-side
        zero_map,   // tensor_map_kl_target -- filled host-side
      },
      args.epilogue,
      args.hw_info
    };
  }


  template<class T>
  static CUTLASS_DEVICE auto quantize(T const& input) {
    constexpr int AlignmentS = 4;
    auto output = make_tensor<Element>(shape(input));
    auto input_vec = recast<Array<ElementAcc, AlignmentS>>(input);
    auto output_vec = recast<Array<Element, AlignmentS>>(output);

    cutlass::NumericArrayConverter<Element, ElementAcc, AlignmentS> epilogue_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(input_vec); i++) {
      output_vec(i) = epilogue_op(input_vec(i));
    }

    return output;
  }


  template<class BlkCoord, class BlkOffset, class ProblemShape_>
  CUTLASS_DEVICE void load(
      BlkCoord const& blk_coord,
      BlkOffset const& blk_offset,
      ProblemShape_ const& problem_shape,
      // in sparse semantics iter_index is K-tile-index (was Q-block in
      // dense). Q/dO/LSE/sum_OdO load use `sq_idx` (= get<1>(blk_coord))
      // instead, since the Q-token is fixed across all K-tile iterations.
      int iter_index,
      int iter_count,
      MainloopArguments const& mainloop_args,
      MainloopParams const& mainloop_params,
      TensorStorage& shared_tensors,
      PipelineLoadMmaQ& pipeline_load_mma_q,
      typename PipelineLoadMmaQ::PipelineState& pipeline_load_mma_q_producer_state,
      PipelineLoadMmaDO& pipeline_load_mma_do,
      typename PipelineLoadMmaDO::PipelineState& pipeline_load_mma_do_producer_state,
      PipelineLoadComputeLSE& pipeline_load_compute_lse,
      typename PipelineLoadComputeLSE::PipelineState& pipeline_load_compute_lse_producer_state,
      PipelineLoadComputeSumOdO& pipeline_load_compute_sum_odo,
      typename PipelineLoadComputeSumOdO::PipelineState& pipeline_load_compute_sum_odo_producer_state) {

    auto [Q, K, D, D_VO, HB] = problem_shape;

    using X = Underscore;

    uint16_t mcast_mask = 0;

    auto mK_in = mainloop_params.tma_load_k.get_tma_tensor(make_shape(K, D, HB));
    auto mV_in = mainloop_params.tma_load_v.get_tma_tensor(make_shape(K, D_VO, HB));
    auto mQ_in = mainloop_params.tma_load_q.get_tma_tensor(make_shape(Q, D, HB));
    auto mDO_in = mainloop_params.tma_load_do.get_tma_tensor(make_shape(Q, D_VO, HB));

    auto mK = domain_offset(select<1,2,4>(blk_offset), mK_in);
    auto mV = domain_offset(select<1,3,4>(blk_offset), mV_in);
    auto mQ = domain_offset(select<0,2,4>(blk_offset), mQ_in);
    auto mDO = domain_offset(select<0,3,4>(blk_offset), mDO_in);

    auto gK = local_tile(mK, TileShapeQK{}, make_coord(_,_,_), Step<X, _1, _1>{});
    auto gQ = local_tile(mQ, TileShapeQK{}, make_coord(_,_,_), Step<_1, X, _1>{});
    auto gV = local_tile(mV, TileShapeDOV{}, make_coord(_,_,_), Step<X, _1, _1>{});
    auto gDO = local_tile(mDO, TileShapeDOV{}, make_coord(_,_,_), Step<_1, X, _1>{});

    ThrMMA cta_mma_kq = TiledMmaQK{}.get_slice(_0{});
    ThrMMA cta_mma_vdo = TiledMmaDOV{}.get_slice(_0{});

    auto tSTgK = cta_mma_kq.partition_B(gK);
    auto tSTgQ = cta_mma_kq.partition_A(gQ);
    auto tDPTgV = cta_mma_vdo.partition_B(gV);
    auto tDPTgDO = cta_mma_vdo.partition_A(gDO);

    auto sQ = make_tensor(make_smem_ptr(shared_tensors.smem_q.begin()), SmemLayoutQ{});
    auto sK = make_tensor(make_smem_ptr(shared_tensors.smem_k.begin()), SmemLayoutK{});
    auto sV = make_tensor(make_smem_ptr(shared_tensors.smem_v.begin()), SmemLayoutV{});
    auto sDO = make_tensor(make_smem_ptr(shared_tensors.smem_do.begin()), SmemLayoutDO{});

    auto [tKgK_mkl, tKsK] = tma_partition(
        mainloop_params.tma_load_k, _0{}, make_layout(_1{}),
        group_modes<0,3>(sK), group_modes<0,3>(tSTgK));
    auto [tQgQ_mkl, tQsQ] = tma_partition(
        mainloop_params.tma_load_q, _0{}, make_layout(_1{}),
        group_modes<0,3>(sQ), group_modes<0,3>(tSTgQ));
    auto [tVgV_mkl, tVsV] = tma_partition(
        mainloop_params.tma_load_v, _0{}, make_layout(_1{}),
        group_modes<0,3>(sV), group_modes<0,3>(tDPTgV));
    auto [tDOgDO_mkl, tDOsDO] = tma_partition(
        mainloop_params.tma_load_do, _0{}, make_layout(_1{}),
        group_modes<0,3>(sDO), group_modes<0,3>(tDPTgDO));

    // set up lse and sum_odo

    auto [blk_coord_q, blk_coord_k, blk_coord_d, blk_coord_dv, blk_coord_batch] = blk_coord;
    // in sparse semantics get<1>(blk_coord) IS the Q-token index.
    // Use this for Q/dO/LSE/sum_OdO TMA partitions instead of iter_index.
    const int sq_idx = blk_coord_k;  // Q-token index (renamed for clarity)

    // D path: K gather + Q load are wrapped in a 3-chunk loop. Each chunk c
    // delivers Q_c (cols [c*192,(c+1)*192)) and K_c (same cols, gathered) into
    // the pipeline's stage. The consumer (MMA) drains 3 stages per K-iter for
    // S accumulate, then 3 more for dK/dQ. Pipeline kStages=2 cycles slots.
    static constexpr int kSparseKGatherBytes =
        4 * TileShapeDQK::value * sizeof(Element) * (TileShapeK::value / 4);
    auto tma_barrier = pipeline_load_mma_q.producer_get_barrier(pipeline_load_mma_q_producer_state);

    // ===== Setup phase: S accumulate across 3 d_qk chunks =====
    CUTLASS_PRAGMA_NO_UNROLL
    for (int chunk_idx = 0; chunk_idx < 3; ++chunk_idx) {
      pipeline_load_mma_q.producer_acquire(pipeline_load_mma_q_producer_state);
      tma_barrier = pipeline_load_mma_q.producer_get_barrier(pipeline_load_mma_q_producer_state);
      pipeline_load_mma_q.producer_expect_transaction(pipeline_load_mma_q_producer_state, kSparseKGatherBytes);

      // K gather chunk_idx (cols [chunk_idx*D_QK_PARTIAL, (chunk_idx+1)*D_QK_PARTIAL))
      if (cute::elect_one_sync()) {
        using bf16 = Element;
        static constexpr int B_TOPK_         = decltype(get<1>(TileShape{}))::value;  // 64
        static constexpr int D_QK_PARTIAL_   = decltype(get<2>(TileShape{}))::value;  // 192
        static constexpr int kNumRowChunks   = B_TOPK_ / 4;                            // 16
        const int* gIndices = mainloop_args.ptr_indices
                            + sq_idx * mainloop_args.topk;
        bf16* sK_dst_base = reinterpret_cast<bf16*>(shared_tensors.smem_k.begin());
        // D path: K is 2-staged. Write to slot = producer_state.index().
        const int sK_slot_setup = pipeline_load_mma_q_producer_state.index();
        bf16* sK_dst = sK_dst_base + sK_slot_setup * (B_TOPK_ * D_QK_PARTIAL_);
        static constexpr int kCols  = 64;
        static constexpr int kNumColIters = D_QK_PARTIAL_ / kCols;  // 3
        CUTLASS_PRAGMA_UNROLL
        for (int rc = 0; rc < kNumRowChunks; ++rc) {
          int4 row_idxs = *reinterpret_cast<const int4*>(
              gIndices + iter_index * B_TOPK_ + rc * 4);
          // clamp invalid (negative sentinel OR out-of-range) indices.
          // Real-data causal masking can emit indices >= K; missing >=K check
          // here causes TMA gather illegal access at scale (job 869 root cause).
          if (row_idxs.x < 0 || row_idxs.x >= K) row_idxs.x = 0;
          if (row_idxs.y < 0 || row_idxs.y >= K) row_idxs.y = 0;
          if (row_idxs.z < 0 || row_idxs.z >= K) row_idxs.z = 0;
          if (row_idxs.w < 0 || row_idxs.w >= K) row_idxs.w = 0;
          CUTLASS_PRAGMA_UNROLL
          for (int cc = 0; cc < kNumColIters; ++cc) {
            ku::tma_gather4_cta_group_1_pipe(
                &mainloop_params.tensor_map_kv,
                tma_barrier,
                sK_dst + cc * B_TOPK_ * kCols + rc * 4 * kCols,
                /*col_idx=*/chunk_idx * D_QK_PARTIAL_ + cc * kCols,
                row_idxs,
                static_cast<int64_t>(cute::TMA::CacheHintSm90::EVICT_LAST));
          }
        }
      }

      // load Q chunk_idx (d_qk tile = chunk_idx)
      if (cute::elect_one_sync()) {
        cute::copy(
            mainloop_params.tma_load_q.with(*tma_barrier, mcast_mask),
            tQgQ_mkl(_, sq_idx, chunk_idx, blk_coord_batch),
            tQsQ(_, pipeline_load_mma_q_producer_state.index())
        );
      }

      ++pipeline_load_mma_q_producer_state;
    }

    pipeline_load_compute_lse.producer_acquire(pipeline_load_compute_lse_producer_state);

    // load LSE
    // 32 threads loading kLoadPerThread * 32 values of 32b each

    int thread_idx = threadIdx.x % NumThreadsPerWarp;
    int smem_idx = TileShapeQ{} * pipeline_load_compute_lse_producer_state.index() + thread_idx * kLoadPerThread;
    // sparse: gmem_idx uses sq_idx (fixed Q-token) for LSE addressing.
    int gmem_idx = TileShapeQ{} * sq_idx + thread_idx * kLoadPerThread;
    auto mLSE = make_tensor(mainloop_args.ptr_lse, make_shape(Q, HB), mainloop_args.stride_lse);
    for (int i = 0; i < kLoadPerThread; i++) {
      cutlass::arch::cp_async_zfill<4>(
          shared_tensors.smem_lse.begin() + smem_idx + i,
          &mLSE(gmem_idx + i, blk_coord_batch),
          gmem_idx + i < Q
      );
    }

    pipeline_load_compute_lse.producer_commit(pipeline_load_compute_lse_producer_state, cutlass::arch::cpasync_barrier_arrive);
    ++pipeline_load_compute_lse_producer_state;


    pipeline_load_mma_do.producer_acquire(pipeline_load_mma_do_producer_state);
    tma_barrier = pipeline_load_mma_do.producer_get_barrier(pipeline_load_mma_do_producer_state);

    // Option A: V gather4 delivers bytes to pipeline_load_mma_do's mbar
    // (same as dO TMA). Mma's consumer_wait(load_mma_do) gets dO+V both.
    // V gather covers D_V_CHUNK cols (= 128 = 2 col-iters of 64 each) per row-chunk,
    // for B_TOPK rows.
    static constexpr int kSparseVGatherBytes =
        4 * TileShapeDVO::value * sizeof(Element) * (TileShapeK::value / 4);
    pipeline_load_mma_do.producer_expect_transaction(pipeline_load_mma_do_producer_state, kSparseVGatherBytes);

    // load V -- SPARSE gather4 path.
    // Same row indices as K, but col_idx=0 (V = kv[:, :, :D_V]).
    // V load uses pipeline_load_mma_do's tma_barrier (different pipeline from K).
    if (cute::elect_one_sync()) {
      using bf16 = Element;
      static constexpr int B_TOPK_       = decltype(get<1>(TileShape{}))::value;  // 64
      static constexpr int D_V_CHUNK_    = decltype(get<3>(TileShape{}))::value;  // 128
      static constexpr int kNumRowChunks = B_TOPK_ / 4;                            // 16
      // sparse: use sq_idx (Q-token index = get<1>(blk_coord)), not blk_coord_q
      // which is always _0 in our sparse setup -> all CTAs would gather Q-token 0.
      const int* gIndices = mainloop_args.ptr_indices
                          + sq_idx * mainloop_args.topk;
      bf16* sV_dst = reinterpret_cast<bf16*>(shared_tensors.smem_v.begin());
      // V col loop with col-tile-major SMEM offset (matching SmemLayoutV).
      // D_V_CHUNK=128, box=64 cols -> 2 col-iters.
      static constexpr int kCols          = 64;
      static constexpr int kNumColItersV  = D_V_CHUNK_ / kCols;  // 2
      CUTLASS_PRAGMA_UNROLL
      for (int rc = 0; rc < kNumRowChunks; ++rc) {
        int4 row_idxs = *reinterpret_cast<const int4*>(
            gIndices + iter_index * B_TOPK_ + rc * 4);
        // clamp invalid (negative sentinel OR out-of-range) indices.
        // Real-data causal masking can emit indices >= K; missing >=K check
        // here causes TMA gather illegal access at scale (job 869 root cause).
        if (row_idxs.x < 0 || row_idxs.x >= K) row_idxs.x = 0;
        if (row_idxs.y < 0 || row_idxs.y >= K) row_idxs.y = 0;
        if (row_idxs.z < 0 || row_idxs.z >= K) row_idxs.z = 0;
        if (row_idxs.w < 0 || row_idxs.w >= K) row_idxs.w = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int cc = 0; cc < kNumColItersV; ++cc) {
          ku::tma_gather4_cta_group_1_pipe(
              &mainloop_params.tensor_map_kv,
              tma_barrier,
              sV_dst + cc * B_TOPK_ * kCols + rc * 4 * kCols,
              /*col_idx=*/cc * kCols,
              row_idxs,
              static_cast<int64_t>(cute::TMA::CacheHintSm90::EVICT_LAST));
        }
      }
    }

    // load dO chunk 0 -- sparse: sq_idx, d_v tile 0.
    if (cute::elect_one_sync()) {
      cute::copy(
          mainloop_params.tma_load_do.with(*tma_barrier, mcast_mask),
          tDOgDO_mkl(_, sq_idx, _0{}, blk_coord_batch),
          tDOsDO(_, pipeline_load_mma_do_producer_state.index())
      );
    }

    ++pipeline_load_mma_do_producer_state;

    // hang-fix: load sum_OdO BEFORE dO chunks 1..3 so Compute
    // warp can produce dS (which needs sum_OdO) and start consuming dkdv
    // pipeline. Otherwise: MMA blocks at chunk 2 dkdv_acq (kStages=2 limit),
    // Load blocks at chunk 3 acquire (waiting MMA release), Compute blocks
    // at OdO wait (Load hasn't committed sum_OdO yet) -- cyclic deadlock.
    pipeline_load_compute_sum_odo.producer_acquire(pipeline_load_compute_sum_odo_producer_state);

    // load sum_OdO -- sparse: sq_idx.
    smem_idx = TileShapeQ{} * pipeline_load_compute_sum_odo_producer_state.index() + thread_idx * kLoadPerThread;
    gmem_idx = TileShapeQ{} * sq_idx + thread_idx * kLoadPerThread;
    auto mSumOdO = make_tensor(mainloop_args.ptr_sum_odo, make_shape(Q, HB), mainloop_args.stride_sum_odo);
    for (int i = 0; i < kLoadPerThread; i++) {
      cutlass::arch::cp_async_zfill<4>(
          shared_tensors.smem_sum_odo.begin() + smem_idx + i,
          &mSumOdO(gmem_idx + i, blk_coord_batch),
          gmem_idx + i < Q
      );
    }

    pipeline_load_compute_sum_odo.producer_commit(pipeline_load_compute_sum_odo_producer_state, cutlass::arch::cpasync_barrier_arrive);
    ++pipeline_load_compute_sum_odo_producer_state;

    // Note: load V + dO chunks 1..3 (D_V multi-chunk).
    // Each chunk loads V[indices, d_v[chunk*128 : (chunk+1)*128]] + dO[chunk].
    // V chunks 1..3 enable dP multi-chunk accumulate (dQ correctness).
    {
      static constexpr int D_V_NUM_CHUNKS = /*D_V=512 / D_V_CHUNK=128 = */ 4;
      // NO_UNROLL: prevents PTX size explosion from 4x duplicated TMA gather4
      // inline asm (each gather is 16 row * 2 col = 32 inline asms; 3 chunks
      // unrolled = 96 + 96 = 192 inline asms in this block alone).
      CUTLASS_PRAGMA_NO_UNROLL
      for (int chunk = 1; chunk < D_V_NUM_CHUNKS; ++chunk) {
        pipeline_load_mma_do.producer_acquire(pipeline_load_mma_do_producer_state);
        auto tma_bar_c = pipeline_load_mma_do.producer_get_barrier(pipeline_load_mma_do_producer_state);
        // Expect V gather bytes (manual, since dO bytes auto-tracked by pipeline).
        pipeline_load_mma_do.producer_expect_transaction(
            pipeline_load_mma_do_producer_state, kSparseVGatherBytes);
        // V gather at chunk's d_v slice.
        if (cute::elect_one_sync()) {
          using bf16_pc = Element;
          static constexpr int B_TOPK_pc       = decltype(get<1>(TileShape{}))::value;
          static constexpr int D_V_CHUNK_pc    = decltype(get<3>(TileShape{}))::value;
          static constexpr int kNumRowChunks_pc = B_TOPK_pc / 4;
          const int* gIndices_pc = mainloop_args.ptr_indices
                                 + sq_idx * mainloop_args.topk;
          bf16_pc* sV_dst_pc = reinterpret_cast<bf16_pc*>(shared_tensors.smem_v.begin());
          static constexpr int kCols_pc        = 64;
          static constexpr int kNumColItersV_pc = D_V_CHUNK_pc / kCols_pc;
          CUTLASS_PRAGMA_UNROLL
          for (int rc = 0; rc < kNumRowChunks_pc; ++rc) {
            int4 row_idxs_pc = *reinterpret_cast<const int4*>(
                gIndices_pc + iter_index * B_TOPK_pc + rc * 4);
            if (row_idxs_pc.x < 0 || row_idxs_pc.x >= K) row_idxs_pc.x = 0;
            if (row_idxs_pc.y < 0 || row_idxs_pc.y >= K) row_idxs_pc.y = 0;
            if (row_idxs_pc.z < 0 || row_idxs_pc.z >= K) row_idxs_pc.z = 0;
            if (row_idxs_pc.w < 0 || row_idxs_pc.w >= K) row_idxs_pc.w = 0;
            CUTLASS_PRAGMA_UNROLL
            for (int cc = 0; cc < kNumColItersV_pc; ++cc) {
              ku::tma_gather4_cta_group_1_pipe(
                  &mainloop_params.tensor_map_kv,
                  tma_bar_c,
                  sV_dst_pc + cc * B_TOPK_pc * kCols_pc + rc * 4 * kCols_pc,
                  /*col_idx=*/chunk * D_V_CHUNK_pc + cc * kCols_pc,
                  row_idxs_pc,
                  static_cast<int64_t>(cute::TMA::CacheHintSm90::EVICT_LAST));
            }
          }
        }
        // dO gather at chunk's d_v slice (existing).
        if (cute::elect_one_sync()) {
          cute::copy(
              mainloop_params.tma_load_do.with(*tma_bar_c, mcast_mask),
              tDOgDO_mkl(_, sq_idx, chunk, blk_coord_batch),
              tDOsDO(_, pipeline_load_mma_do_producer_state.index())
          );
        }
        ++pipeline_load_mma_do_producer_state;
      }
    }

    iter_count -= 1;
    iter_index += 1;

    while (iter_count > 0) {
      // D path: per K-iter, Load produces 6 Q+K chunks (3 for S phase + 3
      // reload for dK/dQ phase). Each chunk index in [0,3) cols [c*192,(c+1)*192).
      // Reload pattern needed because kStages=2 < 3 chunks in flight.
      static constexpr int kSparseKGatherBytesIter =
          4 * TileShapeDQK::value * sizeof(Element) * (TileShapeK::value / 4);

      CUTLASS_PRAGMA_NO_UNROLL
      for (int phase_idx = 0; phase_idx < 6; ++phase_idx) {
        const int chunk_idx = phase_idx % 3;
        // phases 0..2 = S phase (uses CURRENT iter's K_N for S = Q @ K_N^T).
        // phases 3..5 = dK/dQ reload (uses PREV iter's K_{N-1} to pair with
        // dS_{N-1} that arrives in dQ/dK MMA). Previously this used iter_index for
        // both phases -> dQ off-by-one: dS_{N-1} @ K_N (autograd expects K_{N-1}).
        // Fix: reload phase uses (iter_index - 1) -> matches trailing block convention.
        const int K_iter_idx = (phase_idx >= 3) ? (iter_index - 1) : iter_index;
        pipeline_load_mma_q.producer_acquire(pipeline_load_mma_q_producer_state);
        tma_barrier = pipeline_load_mma_q.producer_get_barrier(pipeline_load_mma_q_producer_state);
        pipeline_load_mma_q.producer_expect_transaction(pipeline_load_mma_q_producer_state, kSparseKGatherBytesIter);

        if (cute::elect_one_sync()) {
          using bf16 = Element;
          static constexpr int B_TOPK_         = decltype(get<1>(TileShape{}))::value;
          static constexpr int D_QK_PARTIAL_   = decltype(get<2>(TileShape{}))::value;
          static constexpr int kNumRowChunks   = B_TOPK_ / 4;
          const int* gIndices_in = mainloop_args.ptr_indices
                                 + sq_idx * mainloop_args.topk;
          bf16* sK_dst_in_base = reinterpret_cast<bf16*>(shared_tensors.smem_k.begin());
          const int sK_slot_in = pipeline_load_mma_q_producer_state.index();
          bf16* sK_dst_in = sK_dst_in_base + sK_slot_in * (B_TOPK_ * D_QK_PARTIAL_);
          static constexpr int kCols_in       = 64;
          static constexpr int kNumColIters_in = D_QK_PARTIAL_ / kCols_in;
          CUTLASS_PRAGMA_UNROLL
          for (int rc = 0; rc < kNumRowChunks; ++rc) {
            int4 row_idxs_in = *reinterpret_cast<const int4*>(
                gIndices_in + K_iter_idx * B_TOPK_ + rc * 4);
            if (row_idxs_in.x < 0 || row_idxs_in.x >= K) row_idxs_in.x = 0;
            if (row_idxs_in.y < 0 || row_idxs_in.y >= K) row_idxs_in.y = 0;
            if (row_idxs_in.z < 0 || row_idxs_in.z >= K) row_idxs_in.z = 0;
            if (row_idxs_in.w < 0 || row_idxs_in.w >= K) row_idxs_in.w = 0;
            CUTLASS_PRAGMA_UNROLL
            for (int cc = 0; cc < kNumColIters_in; ++cc) {
              ku::tma_gather4_cta_group_1_pipe(
                  &mainloop_params.tensor_map_kv,
                  tma_barrier,
                  sK_dst_in + cc * B_TOPK_ * kCols_in + rc * 4 * kCols_in,
                  /*col_idx=*/chunk_idx * D_QK_PARTIAL_ + cc * kCols_in,
                  row_idxs_in,
                  static_cast<int64_t>(cute::TMA::CacheHintSm90::EVICT_LAST));
            }
          }
        }

        // load Q chunk_idx
        if (cute::elect_one_sync()) {
          cute::copy(
              mainloop_params.tma_load_q.with(*tma_barrier, mcast_mask),
              tQgQ_mkl(_, sq_idx, chunk_idx, blk_coord_batch),
              tQsQ(_, pipeline_load_mma_q_producer_state.index())
          );
        }

        ++pipeline_load_mma_q_producer_state;
      }

      pipeline_load_compute_lse.producer_acquire(pipeline_load_compute_lse_producer_state);

      // load LSE -- sparse: same sq_idx.
      smem_idx = TileShapeQ{} * pipeline_load_compute_lse_producer_state.index() + thread_idx * kLoadPerThread;
      gmem_idx = TileShapeQ{} * sq_idx + thread_idx * kLoadPerThread;
      for (int i = 0; i < kLoadPerThread; i++) {
        cutlass::arch::cp_async_zfill<4>(
            shared_tensors.smem_lse.begin() + smem_idx + i,
            &mLSE(gmem_idx + i, blk_coord_batch),
            gmem_idx + i < Q
        );
      }

      pipeline_load_compute_lse.producer_commit(pipeline_load_compute_lse_producer_state, cutlass::arch::cpasync_barrier_arrive);
      ++pipeline_load_compute_lse_producer_state;

      pipeline_load_mma_do.producer_acquire(pipeline_load_mma_do_producer_state);
      tma_barrier = pipeline_load_mma_do.producer_get_barrier(pipeline_load_mma_do_producer_state);

      // the sparse variant sparse: in-loop V re-gather (was missing -- iters >0 reused
      // stale slot 0 V data from iter 0). V is gathered at indices[iter_index*B_TOPK..]
      // and d_v chunk 0 (col_idx=0).
      static constexpr int kSparseVGatherBytesIter =
          4 * TileShapeDVO::value * sizeof(Element) * (TileShapeK::value / 4);
      pipeline_load_mma_do.producer_expect_transaction(pipeline_load_mma_do_producer_state, kSparseVGatherBytesIter);

      if (cute::elect_one_sync()) {
        using bf16 = Element;
        static constexpr int B_TOPK_VI    = decltype(get<1>(TileShape{}))::value;
        static constexpr int D_V_CHUNK_VI = decltype(get<3>(TileShape{}))::value;
        static constexpr int kNumRowChunksVI = B_TOPK_VI / 4;
        const int* gIndicesVI = mainloop_args.ptr_indices
                              + sq_idx * mainloop_args.topk;
        bf16* sV_dst_in = reinterpret_cast<bf16*>(shared_tensors.smem_v.begin());
        static constexpr int kCols_VI       = 64;
        static constexpr int kNumColItersVI = D_V_CHUNK_VI / kCols_VI;
        CUTLASS_PRAGMA_UNROLL
        for (int rc = 0; rc < kNumRowChunksVI; ++rc) {
          int4 row_idxs_vi = *reinterpret_cast<const int4*>(
              gIndicesVI + iter_index * B_TOPK_VI + rc * 4);
          if (row_idxs_vi.x < 0 || row_idxs_vi.x >= K) row_idxs_vi.x = 0;
          if (row_idxs_vi.y < 0 || row_idxs_vi.y >= K) row_idxs_vi.y = 0;
          if (row_idxs_vi.z < 0 || row_idxs_vi.z >= K) row_idxs_vi.z = 0;
          if (row_idxs_vi.w < 0 || row_idxs_vi.w >= K) row_idxs_vi.w = 0;
          CUTLASS_PRAGMA_UNROLL
          for (int cc = 0; cc < kNumColItersVI; ++cc) {
            ku::tma_gather4_cta_group_1_pipe(
                &mainloop_params.tensor_map_kv,
                tma_barrier,
                sV_dst_in + cc * B_TOPK_VI * kCols_VI + rc * 4 * kCols_VI,
                /*col_idx=*/cc * kCols_VI,
                row_idxs_vi,
                static_cast<int64_t>(cute::TMA::CacheHintSm90::EVICT_LAST));
          }
        }
      }

      // load dO chunk 0 -- sparse: same sq_idx, d_v tile 0.
      if (cute::elect_one_sync()) {
        cute::copy(
            mainloop_params.tma_load_do.with(*tma_barrier, mcast_mask),
            tDOgDO_mkl(_, sq_idx, _0{}, blk_coord_batch),
            tDOsDO(_, pipeline_load_mma_do_producer_state.index())
        );
      }

      ++pipeline_load_mma_do_producer_state;

      // hang-fix: load sum_OdO BEFORE dO chunks 1..3 (see pre-loop).
      pipeline_load_compute_sum_odo.producer_acquire(pipeline_load_compute_sum_odo_producer_state);

      // load sum_OdO -- sparse: same sq_idx.
      smem_idx = TileShapeQ{} * pipeline_load_compute_sum_odo_producer_state.index() + thread_idx * kLoadPerThread;
      gmem_idx = TileShapeQ{} * sq_idx + thread_idx * kLoadPerThread;
      for (int i = 0; i < kLoadPerThread; i++) {
        cutlass::arch::cp_async_zfill<4>(
            shared_tensors.smem_sum_odo.begin() + smem_idx + i,
            &mSumOdO(gmem_idx + i, blk_coord_batch),
            gmem_idx + i < Q
        );
      }

      pipeline_load_compute_sum_odo.producer_commit(pipeline_load_compute_sum_odo_producer_state, cutlass::arch::cpasync_barrier_arrive);
      ++pipeline_load_compute_sum_odo_producer_state;

      // Note: in-loop V+dO chunks 1..3 for D_V multi-chunk.
      {
        static constexpr int D_V_NUM_CHUNKS_IL = /*D_V=512 / D_V_CHUNK=128 = */ 4;
        CUTLASS_PRAGMA_NO_UNROLL
        for (int chunk = 1; chunk < D_V_NUM_CHUNKS_IL; ++chunk) {
          pipeline_load_mma_do.producer_acquire(pipeline_load_mma_do_producer_state);
          auto tma_bar_il = pipeline_load_mma_do.producer_get_barrier(pipeline_load_mma_do_producer_state);
          pipeline_load_mma_do.producer_expect_transaction(
              pipeline_load_mma_do_producer_state, kSparseVGatherBytesIter);
          // V gather at chunk d_v slice.
          if (cute::elect_one_sync()) {
            using bf16_il = Element;
            static constexpr int B_TOPK_il        = decltype(get<1>(TileShape{}))::value;
            static constexpr int D_V_CHUNK_il     = decltype(get<3>(TileShape{}))::value;
            static constexpr int kNumRowChunks_il = B_TOPK_il / 4;
            const int* gIndices_il = mainloop_args.ptr_indices
                                   + sq_idx * mainloop_args.topk;
            bf16_il* sV_dst_il = reinterpret_cast<bf16_il*>(shared_tensors.smem_v.begin());
            static constexpr int kCols_il        = 64;
            static constexpr int kNumColItersV_il = D_V_CHUNK_il / kCols_il;
            CUTLASS_PRAGMA_UNROLL
            for (int rc = 0; rc < kNumRowChunks_il; ++rc) {
              int4 row_idxs_il = *reinterpret_cast<const int4*>(
                  gIndices_il + iter_index * B_TOPK_il + rc * 4);
              if (row_idxs_il.x < 0 || row_idxs_il.x >= K) row_idxs_il.x = 0;
              if (row_idxs_il.y < 0 || row_idxs_il.y >= K) row_idxs_il.y = 0;
              if (row_idxs_il.z < 0 || row_idxs_il.z >= K) row_idxs_il.z = 0;
              if (row_idxs_il.w < 0 || row_idxs_il.w >= K) row_idxs_il.w = 0;
              CUTLASS_PRAGMA_UNROLL
              for (int cc = 0; cc < kNumColItersV_il; ++cc) {
                ku::tma_gather4_cta_group_1_pipe(
                    &mainloop_params.tensor_map_kv,
                    tma_bar_il,
                    sV_dst_il + cc * B_TOPK_il * kCols_il + rc * 4 * kCols_il,
                    /*col_idx=*/chunk * D_V_CHUNK_il + cc * kCols_il,
                    row_idxs_il,
                    static_cast<int64_t>(cute::TMA::CacheHintSm90::EVICT_LAST));
              }
            }
          }
          // dO at chunk d_v slice.
          if (cute::elect_one_sync()) {
            cute::copy(
                mainloop_params.tma_load_do.with(*tma_bar_il, mcast_mask),
                tDOgDO_mkl(_, sq_idx, chunk, blk_coord_batch),
                tDOsDO(_, pipeline_load_mma_do_producer_state.index())
            );
          }
          ++pipeline_load_mma_do_producer_state;
        }
      }

      iter_count -= 1;
      iter_index += 1;
    }

    // D path: trailing 3 chunks of K+Q for MMA's trailing dK/dQ (last K-tile's
    // 3 d_qk slices). After iter loop, iter_index = N. Trailing uses iter_index-1
    // for indices to fetch last K-tile's rows.
    {
      static constexpr int kSparseKGatherBytesTrail =
          4 * TileShapeDQK::value * sizeof(Element) * (TileShapeK::value / 4);
      const int trail_iter_index = iter_index - 1;

      CUTLASS_PRAGMA_NO_UNROLL
      for (int chunk_idx = 0; chunk_idx < 3; ++chunk_idx) {
        pipeline_load_mma_q.producer_acquire(pipeline_load_mma_q_producer_state);
        tma_barrier = pipeline_load_mma_q.producer_get_barrier(pipeline_load_mma_q_producer_state);
        pipeline_load_mma_q.producer_expect_transaction(pipeline_load_mma_q_producer_state, kSparseKGatherBytesTrail);

        if (cute::elect_one_sync()) {
          using bf16 = Element;
          static constexpr int B_TOPK_         = decltype(get<1>(TileShape{}))::value;
          static constexpr int D_QK_PARTIAL_   = decltype(get<2>(TileShape{}))::value;
          static constexpr int kNumRowChunks   = B_TOPK_ / 4;
          const int* gIndices_tr = mainloop_args.ptr_indices
                                 + sq_idx * mainloop_args.topk;
          bf16* sK_dst_tr_base = reinterpret_cast<bf16*>(shared_tensors.smem_k.begin());
          const int sK_slot_tr = pipeline_load_mma_q_producer_state.index();
          bf16* sK_dst_tr = sK_dst_tr_base + sK_slot_tr * (B_TOPK_ * D_QK_PARTIAL_);
          static constexpr int kCols_tr       = 64;
          static constexpr int kNumColIters_tr = D_QK_PARTIAL_ / kCols_tr;
          CUTLASS_PRAGMA_UNROLL
          for (int rc = 0; rc < kNumRowChunks; ++rc) {
            int4 row_idxs_tr = *reinterpret_cast<const int4*>(
                gIndices_tr + trail_iter_index * B_TOPK_ + rc * 4);
            if (row_idxs_tr.x < 0 || row_idxs_tr.x >= K) row_idxs_tr.x = 0;
            if (row_idxs_tr.y < 0 || row_idxs_tr.y >= K) row_idxs_tr.y = 0;
            if (row_idxs_tr.z < 0 || row_idxs_tr.z >= K) row_idxs_tr.z = 0;
            if (row_idxs_tr.w < 0 || row_idxs_tr.w >= K) row_idxs_tr.w = 0;
            CUTLASS_PRAGMA_UNROLL
            for (int cc = 0; cc < kNumColIters_tr; ++cc) {
              ku::tma_gather4_cta_group_1_pipe(
                  &mainloop_params.tensor_map_kv,
                  tma_barrier,
                  sK_dst_tr + cc * B_TOPK_ * kCols_tr + rc * 4 * kCols_tr,
                  /*col_idx=*/chunk_idx * D_QK_PARTIAL_ + cc * kCols_tr,
                  row_idxs_tr,
                  static_cast<int64_t>(cute::TMA::CacheHintSm90::EVICT_LAST));
            }
          }
        }

        if (cute::elect_one_sync()) {
          cute::copy(
              mainloop_params.tma_load_q.with(*tma_barrier, mcast_mask),
              tQgQ_mkl(_, sq_idx, chunk_idx, blk_coord_batch),
              tQsQ(_, pipeline_load_mma_q_producer_state.index())
          );
        }

        ++pipeline_load_mma_q_producer_state;
      }
    }
  }


  template<class BlkCoord, class ProblemShape_>
  CUTLASS_DEVICE void mma(
      BlkCoord const& blk_coord,
      ProblemShape_ const& problem_shape,
      int iter_index,
      int iter_count,
      MainloopArguments const& mainloop_args,
      TensorStorage& shared_tensors,
      PipelineLoadMmaQ& pipeline_load_mma_q,
      typename PipelineLoadMmaQ::PipelineState& pipeline_load_mma_q_consumer_state,
      PipelineLoadMmaDO& pipeline_load_mma_do,
      typename PipelineLoadMmaDO::PipelineState& pipeline_load_mma_do_consumer_state,
      PipelineMmaComputeS& pipeline_mma_compute_s,
      typename PipelineMmaComputeS::PipelineState& pipeline_mma_compute_s_producer_state,
      PipelineMmaComputeDP& pipeline_mma_compute_dp,
      typename PipelineMmaComputeDP::PipelineState& pipeline_mma_compute_dp_producer_state,
      PipelineMmaReduceDQ& pipeline_mma_reduce_dq,
      typename PipelineMmaReduceDQ::PipelineState& pipeline_mma_reduce_dq_producer_state,
      PipelineComputeMmaP& pipeline_compute_mma_p,
      typename PipelineComputeMmaP::PipelineState& pipeline_compute_mma_p_consumer_state,
      PipelineComputeMmaDS& pipeline_compute_mma_ds,
      typename PipelineComputeMmaDS::PipelineState& pipeline_compute_mma_ds_consumer_state,
      PipelineMmaComputeDKDV& pipeline_mma_compute_dkdv,
      typename PipelineMmaComputeDKDV::PipelineState& pipeline_mma_compute_dkdv_producer_state) {

    auto [Q, K, D, D_VO, HB] = problem_shape;

    auto sQ = make_tensor(make_smem_ptr(shared_tensors.smem_q.begin()), SmemLayoutQ{});
    auto sK = make_tensor(make_smem_ptr(shared_tensors.smem_k.begin()), SmemLayoutK{});
    auto sV = make_tensor(make_smem_ptr(shared_tensors.smem_v.begin()), SmemLayoutV{});
    auto sDO = make_tensor(make_smem_ptr(shared_tensors.smem_do.begin()), SmemLayoutDO{});

    auto sQT = make_tensor(make_smem_ptr(shared_tensors.smem_q_t.begin()), SmemLayoutQT{});
    auto sKT = make_tensor(make_smem_ptr(shared_tensors.smem_k_t.begin()), SmemLayoutKT{});
    auto sDS = make_tensor(make_smem_ptr(shared_tensors.smem_ds.begin()), SmemLayoutDS{});
    auto sDST = make_tensor(make_smem_ptr(shared_tensors.smem_ds_t.begin()), SmemLayoutDST{});
    auto sP = make_tensor(make_smem_ptr(shared_tensors.smem_p.begin()), SmemLayoutP{});
    auto sDOT = make_tensor(make_smem_ptr(shared_tensors.smem_do_t.begin()), SmemLayoutDOT{});

    Tensor tSTrK = TiledMmaQK::make_fragment_B(sK);
    Tensor tSTrQ = TiledMmaQK::make_fragment_A(sQ);

    Tensor tDPTrV = TiledMmaDOV::make_fragment_B(sV);
    Tensor tDPTrDO = TiledMmaDOV::make_fragment_A(sDO);

    Tensor tDQrDS = TiledMmaDSK::make_fragment_A(sDS);
    Tensor tDQrKT = TiledMmaDSK::make_fragment_B(sKT);

    Tensor tDKrDST = TiledMmaDSQ::make_fragment_A(sDST);
    Tensor tDKrQT = TiledMmaDSQ::make_fragment_B(sQT);

    Tensor tDVrP = TiledMmaPDO::make_fragment_A(sP);
    Tensor tDVrDOT = TiledMmaPDO::make_fragment_B(sDOT);

    TiledMmaQK tiled_mma_qk;
    TiledMmaDOV tiled_mma_dov;
    TiledMmaDSK tiled_mma_dsk;
    TiledMmaDSQ tiled_mma_dsq;
    TiledMmaPDO tiled_mma_pdo;

    tiled_mma_dsq.accumulate_ = UMMA::ScaleOut::Zero;
    tiled_mma_pdo.accumulate_ = UMMA::ScaleOut::Zero;

    Tensor tSTtST =  partition_fragment_C(tiled_mma_qk, select<0,1>(TileShapeQK{}));
    tSTtST.data() = TmemAllocation::kS;

    Tensor tDPTtDPT = partition_fragment_C(tiled_mma_dov, select<0,1>(TileShapeDOV{}));
    tDPTtDPT.data() = TmemAllocation::kDP;

    Tensor tDQtDQ = partition_fragment_C(tiled_mma_dsk, select<0,1>(TileShapeDSK{}));
    tDQtDQ.data() = TmemAllocation::kDQ;

    Tensor tDKtDK = partition_fragment_C(tiled_mma_dsq, select<0,1>(TileShapeDSQ{}));
    tDKtDK.data() = TmemAllocation::kDK;

    Tensor tDVtDV = partition_fragment_C(tiled_mma_pdo, select<0,1>(TileShapePDO{}));
    tDVtDV.data() = TmemAllocation::kDV;

    auto pipeline_load_mma_q_release_state = pipeline_load_mma_q_consumer_state;

    pipeline_mma_compute_s.producer_acquire(pipeline_mma_compute_s_producer_state);

    // D path setup: S accumulate over 3 d_qk chunks. Zero on chunk 0, One after.
    // Stage release happens here (within chunk loop) since kStages=2 < 3 chunks.
    // dK MMA in iter loop must use FRESH stages (reload pattern), not release_state.
    tiled_mma_qk.accumulate_ = UMMA::ScaleOut::Zero;
    CUTLASS_PRAGMA_NO_UNROLL
    for (int chunk_idx = 0; chunk_idx < 3; ++chunk_idx) {
      pipeline_load_mma_q.consumer_wait(pipeline_load_mma_q_consumer_state);
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tSTrQ); ++k_block) {
        cute::gemm(tiled_mma_qk,
                   tSTrQ(_,_,k_block,pipeline_load_mma_q_consumer_state.index()),
                   tSTrK(_,_,k_block,pipeline_load_mma_q_consumer_state.index()),
                   tSTtST);
        tiled_mma_qk.accumulate_ = UMMA::ScaleOut::One;
      }
      pipeline_load_mma_q.consumer_release(pipeline_load_mma_q_consumer_state);
      ++pipeline_load_mma_q_consumer_state;
      ++pipeline_load_mma_q_release_state;
    }

    pipeline_mma_compute_s.producer_commit(pipeline_mma_compute_s_producer_state);
    ++pipeline_mma_compute_s_producer_state;

    pipeline_load_mma_do.consumer_wait(pipeline_load_mma_do_consumer_state);

    pipeline_mma_compute_dp.producer_acquire(pipeline_mma_compute_dp_producer_state);
    pipeline_mma_reduce_dq.producer_acquire(pipeline_mma_reduce_dq_producer_state);

    pipeline_compute_mma_p.consumer_wait(pipeline_compute_mma_p_consumer_state);

    // Note: V multi-chunk dP fused with dV per chunk.
    //   dP += dO[c] @ V[c]^T  (Zero on c=0, One after) -> kDP
    //   dV_c = P^T @ dO[c]    (Zero per chunk)         -> kDV scatter
    // V[c] is in sV slot 0 (Load overwrites slot 0 each chunk; SMEM offset has
    // no chunk index). dP commit deferred until after all 4 chunks accumulated.
    {
      static constexpr int D_V_NUM_CHUNKS_M = /*D_V=512 / D_V_CHUNK=128 = */ 4;
      tiled_mma_dov.accumulate_ = UMMA::ScaleOut::Zero;
      CUTLASS_PRAGMA_NO_UNROLL
      for (int chunk = 0; chunk < D_V_NUM_CHUNKS_M; ++chunk) {
        if (chunk > 0) {
          pipeline_load_mma_do.consumer_wait(pipeline_load_mma_do_consumer_state);
        }
        // dP partial chunk c: dO[c] @ V[c]^T -> kDP
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tDPTrV); ++k_block) {
          cute::gemm(tiled_mma_dov,
                     tDPTrDO(_,_,k_block,pipeline_load_mma_do_consumer_state.index()),
                     tDPTrV(_,_,k_block,_0{}),
                     tDPTtDPT);
          tiled_mma_dov.accumulate_ = UMMA::ScaleOut::One;
        }

        // dV chunk c: P^T @ dO[c] -> kDV (Zero per chunk)
        pipeline_mma_compute_dkdv.producer_acquire(pipeline_mma_compute_dkdv_producer_state);
        tiled_mma_pdo.accumulate_ = UMMA::ScaleOut::Zero;
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tDVrP); ++k_block) {
          cute::gemm(tiled_mma_pdo,
                     tDVrP(_,_,k_block,_0{}),
                     tDVrDOT(_,_,k_block,pipeline_load_mma_do_consumer_state.index()),
                     tDVtDV);
          tiled_mma_pdo.accumulate_ = UMMA::ScaleOut::One;
        }
        pipeline_mma_compute_dkdv.producer_commit(pipeline_mma_compute_dkdv_producer_state);
        ++pipeline_mma_compute_dkdv_producer_state;
        pipeline_load_mma_do.consumer_release(pipeline_load_mma_do_consumer_state);
        ++pipeline_load_mma_do_consumer_state;
      }
    }
    pipeline_mma_compute_dp.producer_commit(pipeline_mma_compute_dp_producer_state);
    ++pipeline_mma_compute_dp_producer_state;

    pipeline_compute_mma_p.consumer_release(pipeline_compute_mma_p_consumer_state);
    ++pipeline_compute_mma_p_consumer_state;

    iter_count -= 1;

    // in tmem, S & P overlap
    // and dP and dQ overlap
    // so we need to acquire dQ and dP at the same time
    while (iter_count > 0) {
      // D path S phase: accumulate S over 3 d_qk chunks. Each chunk consumes
      // one Q+K stage and releases it. After 3 chunks, S is fully accumulated.
      pipeline_mma_compute_s.producer_acquire(pipeline_mma_compute_s_producer_state);
      tiled_mma_qk.accumulate_ = UMMA::ScaleOut::Zero;
      CUTLASS_PRAGMA_NO_UNROLL
      for (int chunk_idx = 0; chunk_idx < 3; ++chunk_idx) {
        pipeline_load_mma_q.consumer_wait(pipeline_load_mma_q_consumer_state);
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tSTrQ); ++k_block) {
          cute::gemm(tiled_mma_qk,
                     tSTrQ(_,_,k_block,pipeline_load_mma_q_consumer_state.index()),
                     tSTrK(_,_,k_block,pipeline_load_mma_q_consumer_state.index()),
                     tSTtST);
          tiled_mma_qk.accumulate_ = UMMA::ScaleOut::One;
        }
        pipeline_load_mma_q.consumer_release(pipeline_load_mma_q_consumer_state);
        ++pipeline_load_mma_q_consumer_state;
        ++pipeline_load_mma_q_release_state;
      }

      pipeline_mma_compute_s.producer_commit(pipeline_mma_compute_s_producer_state);
      ++pipeline_mma_compute_s_producer_state;

      pipeline_compute_mma_ds.consumer_wait(pipeline_compute_mma_ds_consumer_state);

      // D path dK/dQ phase: 3 RELOADED Q+K chunks. Per chunk c:
      //   dQ_c = dS @ K_c (TMEM kDQ, signaled to reduce -> scatter to dq_acc[:, c*192:..])
      //   dK_c = dS^T @ Q_c (TMEM kDK, signaled to compute -> scatter to dkv[:, c*192:..])
      // dP and dV pipelines unchanged (dV has its own d_v chunked).
      pipeline_mma_compute_dp.producer_acquire(pipeline_mma_compute_dp_producer_state);

      CUTLASS_PRAGMA_NO_UNROLL
      for (int chunk_idx = 0; chunk_idx < 3; ++chunk_idx) {
        pipeline_load_mma_q.consumer_wait(pipeline_load_mma_q_consumer_state);

        // dQ chunk_c (Zero accumulator: TMEM kDQ holds only this chunk's slice)
        if (chunk_idx > 0) {
          pipeline_mma_reduce_dq.producer_acquire(pipeline_mma_reduce_dq_producer_state);
        }
        tiled_mma_dsk.accumulate_ = UMMA::ScaleOut::Zero;
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tDQrDS); ++k_block) {
          cute::gemm(tiled_mma_dsk,
                     tDQrDS(_,_,k_block,pipeline_compute_mma_ds_consumer_state.index()),
                     tDQrKT(_,_,k_block,pipeline_load_mma_q_consumer_state.index()),
                     tDQtDQ);
          tiled_mma_dsk.accumulate_ = UMMA::ScaleOut::One;
        }
        pipeline_mma_reduce_dq.producer_commit(pipeline_mma_reduce_dq_producer_state);
        ++pipeline_mma_reduce_dq_producer_state;

        // dK chunk_c (Zero accumulator: TMEM kDK holds only this chunk's slice)
        pipeline_mma_compute_dkdv.producer_acquire(pipeline_mma_compute_dkdv_producer_state);
        tiled_mma_dsq.accumulate_ = UMMA::ScaleOut::Zero;
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tDKrDST); ++k_block) {
          cute::gemm(tiled_mma_dsq,
                     tDKrDST(_,_,k_block,pipeline_compute_mma_ds_consumer_state.index()),
                     tDKrQT(_,_,k_block,pipeline_load_mma_q_consumer_state.index()),
                     tDKtDK);
          tiled_mma_dsq.accumulate_ = UMMA::ScaleOut::One;
        }
        pipeline_mma_compute_dkdv.producer_commit(pipeline_mma_compute_dkdv_producer_state);
        ++pipeline_mma_compute_dkdv_producer_state;

        pipeline_load_mma_q.consumer_release(pipeline_load_mma_q_consumer_state);
        ++pipeline_load_mma_q_consumer_state;
        ++pipeline_load_mma_q_release_state;
      }

      pipeline_compute_mma_ds.consumer_release(pipeline_compute_mma_ds_consumer_state);
      ++pipeline_compute_mma_ds_consumer_state;

      // re-acquire reduce_dq for next iter's first chunk
      pipeline_mma_reduce_dq.producer_acquire(pipeline_mma_reduce_dq_producer_state);

      pipeline_load_mma_do.consumer_wait(pipeline_load_mma_do_consumer_state);

      pipeline_compute_mma_p.consumer_wait(pipeline_compute_mma_p_consumer_state);

      // Note: V multi-chunk dP fused with dV per chunk (mirrors setup).
      {
        static constexpr int D_V_NUM_CHUNKS_MI = /*D_V=512 / D_V_CHUNK=128 = */ 4;
        tiled_mma_dov.accumulate_ = UMMA::ScaleOut::Zero;
        CUTLASS_PRAGMA_NO_UNROLL
        for (int chunk = 0; chunk < D_V_NUM_CHUNKS_MI; ++chunk) {
          if (chunk > 0) {
            pipeline_load_mma_do.consumer_wait(pipeline_load_mma_do_consumer_state);
          }
          // dP partial chunk c
          CUTLASS_PRAGMA_UNROLL
          for (int k_block = 0; k_block < size<2>(tDPTrV); ++k_block) {
            cute::gemm(tiled_mma_dov,
                       tDPTrDO(_,_,k_block,pipeline_load_mma_do_consumer_state.index()),
                       tDPTrV(_,_,k_block,_0{}),
                       tDPTtDPT);
            tiled_mma_dov.accumulate_ = UMMA::ScaleOut::One;
          }
          // dV chunk c
          pipeline_mma_compute_dkdv.producer_acquire(pipeline_mma_compute_dkdv_producer_state);
          tiled_mma_pdo.accumulate_ = UMMA::ScaleOut::Zero;
          CUTLASS_PRAGMA_UNROLL
          for (int k_block = 0; k_block < size<2>(tDVrP); ++k_block) {
            cute::gemm(tiled_mma_pdo,
                       tDVrP(_,_,k_block,_0{}),
                       tDVrDOT(_,_,k_block,pipeline_load_mma_do_consumer_state.index()),
                       tDVtDV);
            tiled_mma_pdo.accumulate_ = UMMA::ScaleOut::One;
          }
          pipeline_mma_compute_dkdv.producer_commit(pipeline_mma_compute_dkdv_producer_state);
          ++pipeline_mma_compute_dkdv_producer_state;
          pipeline_load_mma_do.consumer_release(pipeline_load_mma_do_consumer_state);
          ++pipeline_load_mma_do_consumer_state;
        }
      }
      pipeline_mma_compute_dp.producer_commit(pipeline_mma_compute_dp_producer_state);
      ++pipeline_mma_compute_dp_producer_state;

      pipeline_compute_mma_p.consumer_release(pipeline_compute_mma_p_consumer_state);
      ++pipeline_compute_mma_p_consumer_state;

      iter_count -= 1;
    }

    // D path trailing: dK + dQ for LAST K-tile across 3 d_qk chunks.
    // Load produced 3 trailing chunks (chunks 0,1,2) post-loop. Each chunk:
    //   dQ_c = dS_last @ K_c -> reduce scatter to dQ_acc[:, c*192:(c+1)*192]
    //   dK_c = dS_last^T @ Q_c -> compute scatter to dKV[:, c*192:(c+1)*192]
    pipeline_compute_mma_ds.consumer_wait(pipeline_compute_mma_ds_consumer_state);

    CUTLASS_PRAGMA_NO_UNROLL
    for (int chunk_idx = 0; chunk_idx < 3; ++chunk_idx) {
      pipeline_load_mma_q.consumer_wait(pipeline_load_mma_q_consumer_state);

      // dQ chunk_c: re-acquire for chunks > 0 (chunk 0 used the end-of-loop acquire).
      if (chunk_idx > 0) {
        pipeline_mma_reduce_dq.producer_acquire(pipeline_mma_reduce_dq_producer_state);
      }
      tiled_mma_dsk.accumulate_ = UMMA::ScaleOut::Zero;
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tDQrDS); ++k_block) {
        cute::gemm(tiled_mma_dsk,
                   tDQrDS(_,_,k_block,pipeline_compute_mma_ds_consumer_state.index()),
                   tDQrKT(_,_,k_block,pipeline_load_mma_q_consumer_state.index()),
                   tDQtDQ);
        tiled_mma_dsk.accumulate_ = UMMA::ScaleOut::One;
      }
      pipeline_mma_reduce_dq.producer_commit(pipeline_mma_reduce_dq_producer_state);
      ++pipeline_mma_reduce_dq_producer_state;

      // dK chunk_c
      pipeline_mma_compute_dkdv.producer_acquire(pipeline_mma_compute_dkdv_producer_state);
      tiled_mma_dsq.accumulate_ = UMMA::ScaleOut::Zero;
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tDKrDST); ++k_block) {
        cute::gemm(tiled_mma_dsq,
                   tDKrDST(_,_,k_block,pipeline_compute_mma_ds_consumer_state.index()),
                   tDKrQT(_,_,k_block,pipeline_load_mma_q_consumer_state.index()),
                   tDKtDK);
        tiled_mma_dsq.accumulate_ = UMMA::ScaleOut::One;
      }
      pipeline_mma_compute_dkdv.producer_commit(pipeline_mma_compute_dkdv_producer_state);
      ++pipeline_mma_compute_dkdv_producer_state;

      pipeline_load_mma_q.consumer_release(pipeline_load_mma_q_consumer_state);
      ++pipeline_load_mma_q_consumer_state;
      ++pipeline_load_mma_q_release_state;
    }

    pipeline_compute_mma_ds.consumer_release(pipeline_compute_mma_ds_consumer_state);
    ++pipeline_compute_mma_ds_consumer_state;
  }



  template<class TensorG, class TensorR, class TensorC, class TensorShape>
  CUTLASS_DEVICE void store(
      TensorG gmem,
      TensorR const& regs,
      TensorC const& coord,
      TensorShape const& tensor_shape) {
  
    Tensor preds = cute::lazy::transform(coord, [&](auto const& c) { return elem_less(c, tensor_shape); });

    auto copy_op = make_cotiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, Element>{},
        make_layout(make_shape(_1{}, Int<sizeof(uint128_t) / sizeof(Element)>{})),
        regs.layout()
    );
    auto thr_copy = copy_op.get_slice(_0{});

    Tensor quantized_regs = quantize(regs);
    Tensor tCr = thr_copy.partition_S(quantized_regs);
    Tensor tCg = thr_copy.partition_D(gmem);
    Tensor tPc = thr_copy.partition_D(preds);
 
    copy_if(copy_op, tPc, tCr, tCg);
  }


  template<class BlkCoord, class BlkOffset, class ProblemShape_>
  CUTLASS_DEVICE void epilogue_clear(
      BlkCoord const& blk_coord,
      BlkOffset const& blk_offset,
      ProblemShape_ const& problem_shape,
      MainloopArguments const& mainloop_args,
      EpilogueArguments const& epilogue_args) {

    auto [Q, K, D, D_VO, HB] = problem_shape;
    auto [blk_coord_q, blk_coord_k, blk_coord_d, blk_coord_dv, blk_coord_batch] = blk_coord;

    auto mDK_in = make_tensor(make_gmem_ptr(epilogue_args.ptr_dk), make_shape(K, TileShapeDQK{}, HB), epilogue_args.stride_dk);
    auto mDK = domain_offset(select<1,2,4>(blk_offset), mDK_in);
    auto gDK = local_tile(mDK, TileShapeDSQ{}, make_coord(_,_,_), Step<_1, _1, X>{})
        (_, _, blk_coord_k, _0{}, blk_coord_batch);

    Tensor cDK = domain_offset(
        make_coord(get<1>(blk_coord) * TileShapeK{}, _0{}),
        make_identity_tensor(take<0,2>(TileShapeDSQ{}))
    );

    auto mDV_in = make_tensor(make_gmem_ptr(epilogue_args.ptr_dv), make_shape(K, TileShapeDVO{}, HB), epilogue_args.stride_dv);
    auto mDV = domain_offset(select<1,3,4>(blk_offset), mDV_in);
    auto gDV = local_tile(mDV, TileShapePDO{}, make_coord(_,_,_), Step<_1, _1, X>{})
        (_, _, blk_coord_k, _0{}, blk_coord_batch);

    Tensor cDV = domain_offset(
        make_coord(blk_coord_k * TileShapeK{}, _0{}),
        make_identity_tensor(take<0,2>(TileShapePDO{}))
    );
    
    for (int i = threadIdx.x; i < size(gDK); i += blockDim.x) {
      if (elem_less(cDK(i), select<1,2>(problem_shape))) {
        gDK(i) = Element(0);
      }
    }
    for (int i = threadIdx.x; i < size(gDV); i += blockDim.x) {
      if (elem_less(cDV(i), select<1,3>(problem_shape))) {
        gDV(i) = Element(0);
      }
    }

  }


  template<class BlkCoord, class BlkOffset, class ProblemShape_>
  CUTLASS_DEVICE void epilogue(
      BlkCoord const& blk_coord,
      BlkOffset const& blk_offset,
      ProblemShape_ const& problem_shape,
      MainloopArguments const& mainloop_args,
      EpilogueArguments const& epilogue_args,
      PipelineMmaComputeDKDV& pipeline_mma_compute_dkdv,
      typename PipelineMmaComputeDKDV::PipelineState& pipeline_mma_compute_dkdv_consumer_state) {

    auto [Q, K, D, D_VO, HB] = problem_shape;
    auto [blk_coord_q, blk_coord_k, blk_coord_d, blk_coord_dv, blk_coord_batch] = blk_coord;

    auto load_op = SM100_TMEM_LOAD_32dp32b16x{};

    auto tDKtDK = partition_fragment_C(TiledMmaDSQ{}, select<0,1>(TileShapeDSQ{}))(make_coord(_,_),_0{},_0{});
    tDKtDK.data() = TmemAllocation::kDK;

    auto mDK_in = make_tensor(make_gmem_ptr(epilogue_args.ptr_dk), make_shape(K, TileShapeDQK{}, HB), epilogue_args.stride_dk);
    auto mDK = domain_offset(select<1,2,4>(blk_offset), mDK_in);
    auto gDK = local_tile(mDK, TileShapeDSQ{}, make_coord(_,_,_), Step<_1, _1, X>{})
        (_, _, blk_coord_k, _0{}, blk_coord_batch);

    Tensor cDK = domain_offset(
        make_coord(get<1>(blk_coord) * TileShapeK{}, _0{}),
        make_identity_tensor(take<0,2>(TileShapeDSQ{}))
    );

    constexpr int kNumWarpgroups = kNumComputeWarps / 4;
    int dp_idx = threadIdx.x % 128;
    int wg_idx = (threadIdx.x % (kNumComputeWarps * NumThreadsPerWarp)) / 128;

    auto split_wg = [&](auto const& t) {
      if constexpr (decltype(rank(t))::value == 3) {
        auto p = t.compose(make_layout(make_shape(size<0>(t), size<1>(t), make_shape(Int<kNumWarpgroups>{}, size<2>(t) / Int<kNumWarpgroups>{}))));
        return p(_, _, make_coord(wg_idx, _));
      }
      else {
        auto p = t.compose(make_layout(make_shape(size<0>(t), size<1>(t), size<2>(t), make_shape(Int<kNumWarpgroups>{}, size<3>(t) / Int<kNumWarpgroups>{}))));
        return p(_, _, _, make_coord(wg_idx, _));
      }
    };

    auto tiled_t2r_dk = make_tmem_copy(load_op, tDKtDK);
    auto thread_t2r_dk = tiled_t2r_dk.get_slice(dp_idx);

    Tensor tTR_cDK   = split_wg(thread_t2r_dk.partition_D(cDK));
    Tensor tTR_gDK   = split_wg(thread_t2r_dk.partition_D(gDK));
    Tensor tTR_rDK = make_tensor<ElementAcc>(shape(tTR_cDK));
    Tensor tTR_tDK = split_wg(thread_t2r_dk.partition_S(tDKtDK));

    auto tDVtDV = partition_fragment_C(TiledMmaPDO{}, select<0,1>(TileShapePDO{}))(make_coord(_,_),_0{},_0{});
    tDVtDV.data() = TmemAllocation::kDV;

    auto mDV_in = make_tensor(make_gmem_ptr(epilogue_args.ptr_dv), make_shape(K, TileShapeDVO{}, HB), epilogue_args.stride_dv);
    auto mDV = domain_offset(select<1,3,4>(blk_offset), mDV_in);
    auto gDV = local_tile(mDV, TileShapePDO{}, make_coord(_,_,_), Step<_1, _1, X>{})
        (_, _, blk_coord_k, _0{}, blk_coord_batch);

    Tensor cDV = domain_offset(
        make_coord(blk_coord_k * TileShapeK{}, _0{}),
        make_identity_tensor(take<0,2>(TileShapePDO{}))
    );

    auto tiled_t2r_dv = make_tmem_copy(load_op, tDVtDV);
    auto thread_t2r_dv = tiled_t2r_dv.get_slice(dp_idx);

    Tensor tTR_cDV   = split_wg(thread_t2r_dv.partition_D(cDV));
    Tensor tTR_gDV   = split_wg(thread_t2r_dv.partition_D(gDV));
    Tensor tTR_rDV = make_tensor<ElementAcc>(shape(tTR_cDV));
    Tensor tTR_tDV = split_wg(thread_t2r_dv.partition_S(tDVtDV));

    // dV consumer_wait/release removed -- dV is now consumed per-iter
    // inside compute()'s while loop (scatter via atomicAdd to indices[k_pos]).
    // Only the trailing dK signal from MMA's after-loop block remains.

    pipeline_mma_compute_dkdv.consumer_wait(pipeline_mma_compute_dkdv_consumer_state);

    // load tDKtDK (TMEM -> regs).
    cute::copy(tiled_t2r_dk, tTR_tDK, tTR_rDK);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tTR_rDK); i++) {
      tTR_rDK(i) = mainloop_args.softmax_scale * tTR_rDK(i);
    }

    // dKV dense store DISABLED -- isolate dQ.
    // store(tTR_gDK, tTR_rDK, tTR_cDK, select<1,2>(problem_shape));

    cutlass::arch::fence_view_async_tmem_load();
    pipeline_mma_compute_dkdv.consumer_release(pipeline_mma_compute_dkdv_consumer_state);
    ++pipeline_mma_compute_dkdv_consumer_state;

  }


  template<class BlkCoord, class BlkOffset, class ProblemShape_>
  CUTLASS_DEVICE void compute(
      BlkCoord const& blk_coord,
      BlkOffset const& blk_offset,
      ProblemShape_ const& problem_shape,
      int iter_index,
      int iter_count,
      MainloopArguments const& mainloop_args,
      EpilogueArguments const& epilogue_args,
      TensorStorage& shared_tensors,
      PipelineLoadComputeLSE& pipeline_load_compute_lse,
      typename PipelineLoadComputeLSE::PipelineState& pipeline_load_compute_lse_consumer_state,
      PipelineLoadComputeSumOdO& pipeline_load_compute_sum_odo,
      typename PipelineLoadComputeSumOdO::PipelineState& pipeline_load_compute_sum_odo_consumer_state,
      PipelineMmaComputeS& pipeline_mma_compute_s,
      typename PipelineMmaComputeS::PipelineState& pipeline_mma_compute_s_consumer_state,
      PipelineMmaComputeDP& pipeline_mma_compute_dp,
      typename PipelineMmaComputeDP::PipelineState& pipeline_mma_compute_dp_consumer_state,
      PipelineComputeMmaP& pipeline_compute_mma_p,
      typename PipelineComputeMmaP::PipelineState& pipeline_compute_mma_p_producer_state,
      PipelineComputeMmaDS& pipeline_compute_mma_ds,
      typename PipelineComputeMmaDS::PipelineState& pipeline_compute_mma_ds_producer_state,
      PipelineMmaComputeDKDV& pipeline_mma_compute_dkdv,
      typename PipelineMmaComputeDKDV::PipelineState& pipeline_mma_compute_dkdv_consumer_state) {


    auto [Q, K, D, D_VO, HB] = problem_shape;

    // in tmem, S & P overlap
    // and dP and dQ overlap

    // there are two compute wg's that cooperatively compute softmax
    // they are striped by this tmem atom, i.e. wg0 has 16 elems, then wg1 etc

    auto load_op = SM100_TMEM_LOAD_16dp32b32x{};

    Tensor tSTtST =  partition_fragment_C(TiledMmaQK{}, select<0,1>(TileShapeQK{}))(make_coord(_,_),_0{},_0{});
    tSTtST.data() = TmemAllocation::kS;

    Tensor tDPTtDPT =  partition_fragment_C(TiledMmaDOV{}, select<0,1>(TileShapeDOV{}))(make_coord(_,_),_0{},_0{});
    tDPTtDPT.data() = TmemAllocation::kDP;

    Tensor cST = make_identity_tensor(take<0,2>(TileShapeQK{}));
    Tensor cDPT = make_identity_tensor(take<0,2>(TileShapeDOV{}));
    Tensor cPT = make_identity_tensor(take<0,2>(TileShapeQK{}));

    constexpr int kNumWarpgroups = kNumComputeWarps / 4;
    int dp_idx = threadIdx.x % 128;
    int wg_idx = (threadIdx.x % (kNumComputeWarps * NumThreadsPerWarp)) / 128;
    auto tiled_t2r = make_tmem_copy(load_op, tSTtST);
    auto thread_t2r = tiled_t2r.get_slice(dp_idx);

    auto split_wg = [&](auto const& t) {
      if constexpr (decltype(size<1>(t))::value > 1) {
        if constexpr (decltype(rank(t))::value == 3) {
          auto p = t.compose(make_layout(make_shape(size<0>(t), make_shape(Int<kNumWarpgroups>{}, size<1>(t) / Int<kNumWarpgroups>{}), size<2>(t))));
          return p(_, make_coord(wg_idx, _), _);
        }
        else {
          auto p = t.compose(make_layout(make_shape(size<0>(t), make_shape(Int<kNumWarpgroups>{}, size<1>(t) / Int<kNumWarpgroups>{}), size<2>(t), size<3>(t))));
          return p(_, make_coord(wg_idx, _), _, _);
        }
      }
      else {
        if constexpr (decltype(rank(t))::value == 3) {
          auto p = t.compose(make_layout(make_shape(size<0>(t), size<1>(t), make_shape(Int<kNumWarpgroups>{}, size<2>(t) / Int<kNumWarpgroups>{}))));
          return p(_, _, make_coord(wg_idx, _));
        }
        else {
          auto p = t.compose(make_layout(make_shape(size<0>(t), size<1>(t), size<2>(t), make_shape(Int<kNumWarpgroups>{}, size<3>(t) / Int<kNumWarpgroups>{}))));
          return p(_, _, _, make_coord(wg_idx, _));
        }
      }
    };

    Tensor tTR_cST_p = thread_t2r.partition_D(cST);
    Tensor tTR_cST   = split_wg(tTR_cST_p);
    Tensor tTR_rST = make_tensor<ElementAcc>(shape(tTR_cST));
    Tensor tTR_tST = split_wg(thread_t2r.partition_S(tSTtST));

    Tensor tTR_cDPT_p = thread_t2r.partition_D(cDPT);
    Tensor tTR_cPT_p = thread_t2r.partition_D(cPT);
    Tensor tTR_cDPT = split_wg(tTR_cDPT_p);
    Tensor tTR_rDPT = make_tensor<ElementAcc>(shape(tTR_cDPT));
    Tensor tTR_tDPT = split_wg(thread_t2r.partition_S(tDPTtDPT));

    Tensor sLSE = make_tensor(make_smem_ptr(shared_tensors.smem_lse.begin()), SmemLayoutLSE{});
    Tensor sSumOdO = make_tensor(make_smem_ptr(shared_tensors.smem_sum_odo.begin()), SmemLayoutSumOdO{});

    // sparse semantics -- indices[] already encodes causality, so
    // no Q-vs-K triangular masking. The `is_residual_k` check (originally
    // "are we at the right edge of the K dim?") doesn't apply either since
    // K is gather-by-indices. Force both masks off; the validity mask for
    // indices == -1 will be applied separately in the softmax block.
    bool is_residual_k = false;
    int last_iter = iter_count - 1 + iter_index;

    CUTLASS_PRAGMA_NO_UNROLL
    while (iter_count > 0) {
      // wait for S and P
      pipeline_mma_compute_s.consumer_wait(pipeline_mma_compute_s_consumer_state);
      pipeline_compute_mma_p.producer_acquire(pipeline_compute_mma_p_producer_state);
      // wait for LSE
      pipeline_load_compute_lse.consumer_wait(pipeline_load_compute_lse_consumer_state);

      auto dispatch_bool = [](bool b, auto fn) {
        if (b) {
          fn(cute::true_type{});
        }
        else {
          fn(cute::false_type{});
        }
      };

      // sparse path skips all dense causal/residual masking.
      bool leading_causal_masking = false;
      bool trailing_residual_masking = false;

      dispatch_bool(leading_causal_masking || trailing_residual_masking, [&](auto is_masked_tile) {

        // compute P = softmax(S, LSE)
        cute::copy(tiled_t2r, tTR_tST, tTR_rST);

        if constexpr (decltype(is_masked_tile)::value) {
          Mask{}.apply_mask(tTR_rST, [&](int i) {
            auto c_transpose = tTR_cST(i);
            return make_coord(get<0>(c_transpose) + iter_index * TileShapeQ{}, get<1>(c_transpose) + get<1>(blk_coord) * TileShapeK{});
          }, problem_shape);
        }

        ElementAcc log2_e = static_cast<ElementAcc>(M_LOG2E);
        float2 softmax_scale_log2_e;
        softmax_scale_log2_e.x = mainloop_args.softmax_scale * log2_e;
        softmax_scale_log2_e.y = mainloop_args.softmax_scale * log2_e;

        // sparse invalid-k mask. Per-element check whether indices[k]
        // is valid; if not, force P=0 so the slot contributes nothing to dV/dK
        // and dQ. K=K[0] (we set indices[invalid]=0 in Load) yields valid score
        // otherwise, breaking dQ magnitude.
        const int sq_idx_c = get<1>(blk_coord);
        const int K_ps = get<1>(problem_shape);
        const int* gIndices_compute =
            mainloop_args.ptr_indices + sq_idx_c * mainloop_args.topk;
        const int k_tile_base = iter_index * int(TileShapeK{});

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tTR_rST); i += 2) {
          float2 acc;
          float2 lse;
          float2 out;
          acc.x = tTR_rST(i);
          acc.y = tTR_rST(i + 1);
          // Fix: caller's LSE is positive (base-2 log-sum-exp normalizer).
          // Softmax: P = exp2(S * scale * log2(e) - LSE_base2). Negate LSE so
          // FMA = scale*acc + (-LSE) computes the correct subtraction.
          lse.x = -sLSE(get<0>(tTR_cST(i)), pipeline_load_compute_lse_consumer_state.index());
          lse.y = -sLSE(get<0>(tTR_cST(i+1)), pipeline_load_compute_lse_consumer_state.index());
          cute::fma(out, softmax_scale_log2_e, acc, lse);
          tTR_rST(i) = ::exp2f(out.x);
          tTR_rST(i+1) = ::exp2f(out.y);

          // Apply invalid-k mask. k_local = get<1>(tTR_cST(i)) in [0, TileShapeK).
          int k_local_x = get<1>(tTR_cST(i));
          int k_local_y = get<1>(tTR_cST(i+1));
          int kidx_x = gIndices_compute[k_tile_base + k_local_x];
          int kidx_y = gIndices_compute[k_tile_base + k_local_y];
          if (kidx_x < 0 || kidx_x >= K_ps) tTR_rST(i) = 0.0f;
          if (kidx_y < 0 || kidx_y >= K_ps) tTR_rST(i+1) = 0.0f;
        }

        auto tRT_rST = quantize(tTR_rST);

        Tensor sP = make_tensor(make_smem_ptr((Element*) shared_tensors.smem_p.begin()), SmemLayoutP{})
          (_, _, _, pipeline_compute_mma_p_producer_state.index());

        cutlass::arch::fence_view_async_tmem_load();
        cutlass::arch::NamedBarrier(
          kNumComputeWarps * NumThreadsPerWarp,
          cutlass::arch::ReservedNamedBarriers::TransformBarrier
        ).arrive_and_wait();

        auto sP_pi = as_position_independent_swizzle_tensor(sP);

        auto thread_layout = make_ordered_layout(
            make_shape(_64{}, _32{}, _2{}, _2{}),
            make_stride(_3{}, _0{}, _1{}, _2{})
            );
        auto sP_pi_slice_p = sP_pi.compose(thread_layout)(((dp_idx/32) * 16) + (dp_idx % 16) , _, (dp_idx % 32 / 16), _).compose(make_layout(shape(tTR_cPT_p)));
        auto sP_pi_slice = split_wg(sP_pi_slice_p);
        copy_aligned(tRT_rST, sP_pi_slice);
      });

      // notify for P
      cutlass::arch::fence_view_async_shared();
      pipeline_compute_mma_p.producer_commit(pipeline_compute_mma_p_producer_state);
      ++pipeline_compute_mma_p_producer_state;
      // release S
      pipeline_mma_compute_s.consumer_release(pipeline_mma_compute_s_consumer_state);
      ++pipeline_mma_compute_s_consumer_state;
      // release LSE
      pipeline_load_compute_lse.consumer_release(pipeline_load_compute_lse_consumer_state);
      ++pipeline_load_compute_lse_consumer_state;

      // reorder: scatter dK + dV BEFORE wait dP/OdO. With
      // pipeline_mma_compute_dkdv kStages=1, MMA chunk c+1 producer_acquire
      // BLOCKS until compute releases chunk c. If compute were at wait dP
      // (which fires only after MMA's full dP+dV phase including 4 dV chunk
      // commits), MMA dK chunk 1 acquire would deadlock. Moving scatter here
      // gives strict serial: MMA c commit -> compute c read+scatter+release
      // -> MMA c+1 acquire unblocks. Eliminates TMEM kDK/kDV race.

      // dK scatter (3 chunks, iter_index > 0 only -- setup phase has no dK).
      if (iter_index > 0) {
        CUTLASS_PRAGMA_NO_UNROLL
        for (int chunk_idx = 0; chunk_idx < 3; ++chunk_idx) {
          pipeline_mma_compute_dkdv.consumer_wait(pipeline_mma_compute_dkdv_consumer_state);

          auto load_op_dk_m11 = SM100_TMEM_LOAD_32dp32b16x{};
          auto tDKtDK_scatter_m11 = partition_fragment_C(
              TiledMmaDSQ{}, select<0,1>(TileShapeDSQ{}))(make_coord(_,_),_0{},_0{});
          tDKtDK_scatter_m11.data() = TmemAllocation::kDK;

          Tensor cDK_full_m11 = domain_offset(
              make_coord(get<1>(blk_coord) * TileShapeK{}, _0{}),
              make_identity_tensor(take<0,2>(TileShapeDSQ{}))
          );

          auto tiled_t2r_dk_m11 = make_tmem_copy(load_op_dk_m11, tDKtDK_scatter_m11);
          auto thread_t2r_dk_m11 = tiled_t2r_dk_m11.get_slice(dp_idx);

          Tensor tTR_cDK_p2_m11 = thread_t2r_dk_m11.partition_D(cDK_full_m11);
          Tensor tTR_cDK_dk_m11 = split_wg(tTR_cDK_p2_m11);
          Tensor tTR_rDK_dk_m11 = make_tensor<ElementAcc>(shape(tTR_cDK_dk_m11));
          Tensor tTR_tDK_dk_m11 = split_wg(thread_t2r_dk_m11.partition_S(tDKtDK_scatter_m11));
          cute::copy(tiled_t2r_dk_m11, tTR_tDK_dk_m11, tTR_rDK_dk_m11);

          const int K_total_dk_m11 = get<1>(problem_shape);
          const int sq_idx_dk_m11 = get<1>(blk_coord);
          const int* gIndices_dk_m11 =
              mainloop_args.ptr_indices + sq_idx_dk_m11 * mainloop_args.topk;
          const int k_tile_base_dk_m11 = (iter_index - 1) * int(TileShapeK{});
          const int sq_block_base_dk_m11 = sq_idx_dk_m11 * int(TileShapeK{});
          const int row_stride_dk_m11 = (int)get<0>(epilogue_args.stride_dk);
          ElementAcc* ptr_dkv_acc_dk_m11 = epilogue_args.ptr_dkv_acc;
          const int cta_d_offset_m11 = chunk_idx * int(TileShapeDQK{});
          const float softmax_scale_dk_m11 = mainloop_args.softmax_scale;

          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tTR_rDK_dk_m11); i++) {
            auto c = tTR_cDK_dk_m11(i);
            int k_dense = get<0>(c);
            int d_local = get<1>(c);
            int k_local = k_dense - sq_block_base_dk_m11;
            if (k_local < 0 || k_local >= int(TileShapeK{})) continue;
            if (d_local < 0 || d_local >= int(TileShapeDQK{})) continue;
            int kidx = gIndices_dk_m11[k_tile_base_dk_m11 + k_local];
            if (kidx < 0 || kidx >= K_total_dk_m11) continue;
            const size_t _off_dk = (size_t)kidx * row_stride_dk_m11
                                  + cta_d_offset_m11 + d_local;
            if (_off_dk >= (size_t)K_total_dk_m11 * row_stride_dk_m11) continue;
            float v = tTR_rDK_dk_m11(i) * softmax_scale_dk_m11;
            atomicAdd(&ptr_dkv_acc_dk_m11[_off_dk], v);
          }

          cutlass::arch::fence_view_async_tmem_load();
          pipeline_mma_compute_dkdv.consumer_release(pipeline_mma_compute_dkdv_consumer_state);
          ++pipeline_mma_compute_dkdv_consumer_state;
        }
      }

      // dV scatter (4 chunks, D_V_CHUNK=128 each).
      {
        static constexpr int D_V_NUM_CHUNKS_M11 = /*D_V=512 / D_V_CHUNK=128 = */ 4;
        CUTLASS_PRAGMA_NO_UNROLL
        for (int chunk = 0; chunk < D_V_NUM_CHUNKS_M11; ++chunk) {
          pipeline_mma_compute_dkdv.consumer_wait(pipeline_mma_compute_dkdv_consumer_state);

          auto load_op_dv_m11 = SM100_TMEM_LOAD_32dp32b16x{};
          auto tDVtDV_scatter_m11 = partition_fragment_C(
              TiledMmaPDO{}, select<0,1>(TileShapePDO{}))(make_coord(_,_),_0{},_0{});
          tDVtDV_scatter_m11.data() = TmemAllocation::kDV;

          Tensor cDV_full_m11 = domain_offset(
              make_coord(get<1>(blk_coord) * TileShapeK{}, _0{}),
              make_identity_tensor(take<0,2>(TileShapePDO{}))
          );

          auto tiled_t2r_dv_m11 = make_tmem_copy(load_op_dv_m11, tDVtDV_scatter_m11);
          auto thread_t2r_dv_m11 = tiled_t2r_dv_m11.get_slice(dp_idx);

          Tensor tTR_cDV_p_m11 = thread_t2r_dv_m11.partition_D(cDV_full_m11);
          Tensor tTR_cDV_m11   = split_wg(tTR_cDV_p_m11);
          Tensor tTR_rDV_m11   = make_tensor<ElementAcc>(shape(tTR_cDV_m11));
          Tensor tTR_tDV_m11   = split_wg(thread_t2r_dv_m11.partition_S(tDVtDV_scatter_m11));
          cute::copy(tiled_t2r_dv_m11, tTR_tDV_m11, tTR_rDV_m11);

          const int K_total_dv_m11 = get<1>(problem_shape);
          const int sq_idx_dv_m11 = get<1>(blk_coord);
          const int* gIndices_dv_m11 =
              mainloop_args.ptr_indices + sq_idx_dv_m11 * mainloop_args.topk;
          const int k_tile_base_dv_m11 = iter_index * int(TileShapeK{});
          const int sq_block_base_dv_m11  = sq_idx_dv_m11 * int(TileShapeK{});
          const int row_stride_dv_m11 = (int)get<0>(epilogue_args.stride_dv);
          ElementAcc* ptr_dkv_acc_dv_m11 = epilogue_args.ptr_dkv_acc;
          const int chunk_d_offset_m11 = chunk * int(TileShapeDVO{});

          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tTR_rDV_m11); i++) {
            auto cc = tTR_cDV_m11(i);
            int k_dense = get<0>(cc);
            int d_local = get<1>(cc);
            int k_local = k_dense - sq_block_base_dv_m11;
            if (k_local < 0 || k_local >= int(TileShapeK{})) continue;
            if (d_local < 0 || d_local >= int(TileShapeDVO{})) continue;
            int kidx = gIndices_dv_m11[k_tile_base_dv_m11 + k_local];
            if (kidx < 0 || kidx >= K_total_dv_m11) continue;
            const size_t _off_dv = (size_t)kidx * row_stride_dv_m11
                                  + chunk_d_offset_m11 + d_local;
            if (_off_dv >= (size_t)K_total_dv_m11 * row_stride_dv_m11) continue;
            float v = tTR_rDV_m11(i);
            atomicAdd(&ptr_dkv_acc_dv_m11[_off_dv], v);
          }

          cutlass::arch::fence_view_async_tmem_load();
          pipeline_mma_compute_dkdv.consumer_release(pipeline_mma_compute_dkdv_consumer_state);
          ++pipeline_mma_compute_dkdv_consumer_state;
        }
      }

      // wait for OdO
      pipeline_load_compute_sum_odo.consumer_wait(pipeline_load_compute_sum_odo_consumer_state);
      // wait for dP
      pipeline_mma_compute_dp.consumer_wait(pipeline_mma_compute_dp_consumer_state);

      // wait for dS
      // in principle, we could defer waiting for dS, and move in the freeing of dP
      // however, that would force us to keep dS in registers longer
      pipeline_compute_mma_ds.producer_acquire(pipeline_compute_mma_ds_producer_state);

      // compute dS = dsoftmax(P, dP, sum_OdO)
      cute::copy(tiled_t2r, tTR_tDPT, tTR_rDPT);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tTR_rDPT); i += 2) {
        float2 st;
        st.x = tTR_rST(i);
        st.y = tTR_rST(i+1);
        float2 dpt;
        dpt.x = tTR_rDPT(i);
        dpt.y = tTR_rDPT(i+1);
        float2 odo;
        odo.x = sSumOdO(get<0>(tTR_cDPT(i)), pipeline_load_compute_sum_odo_consumer_state.index());
        odo.y = sSumOdO(get<0>(tTR_cDPT(i+1)), pipeline_load_compute_sum_odo_consumer_state.index());
        float2 dif;
        // sum odo is negated during preprocess
        cute::add(dif, dpt, odo);
        float2 out;
        cute::mul(out, dif, st);
        tTR_rDPT(i) = out.x;
        tTR_rDPT(i+1) = out.y;
      }

      auto tTR_rDST = quantize(tTR_rDPT);

      // release dP
      cutlass::arch::fence_view_async_tmem_load();
      pipeline_mma_compute_dp.consumer_release(pipeline_mma_compute_dp_consumer_state);
      ++pipeline_mma_compute_dp_consumer_state;

      Tensor sDS = make_tensor(make_smem_ptr((Element*) shared_tensors.smem_ds_t.begin()), SmemLayoutDST{})
          (_, _, _, pipeline_compute_mma_ds_producer_state.index());

      auto thread_layout = make_ordered_layout(
          make_shape(_64{}, _32{}, _2{}, _2{}),
          make_stride(_3{}, _0{}, _1{}, _2{})
          );
      auto sDS_pi = as_position_independent_swizzle_tensor(sDS);
      auto sDS_pi_slice_p = sDS_pi.compose(thread_layout)(((dp_idx/32) * 16) + (dp_idx % 16) , _, (dp_idx % 32 / 16), _).compose(make_layout(shape      (tTR_cDPT_p)));
      auto sDS_pi_slice = split_wg(sDS_pi_slice_p);

      copy_aligned(tTR_rDST, sDS_pi_slice);

      // notify for dS
      cutlass::arch::fence_view_async_shared();
      pipeline_compute_mma_ds.producer_commit(pipeline_compute_mma_ds_producer_state);
      ++pipeline_compute_mma_ds_producer_state;
      // release OdO
      pipeline_load_compute_sum_odo.consumer_release(pipeline_load_compute_sum_odo_consumer_state);
      ++pipeline_load_compute_sum_odo_consumer_state;

      // dK + dV scatter MOVED to before "wait for OdO" above.
      // The previous location had TMEM kDK/kDV race because compute warp was
      // blocked at wait dP while MMA raced ahead overwriting TMEM.

      iter_count -= 1;
      iter_index += 1;
    }

    // D path trailing: dK for LAST K-tile across 3 d_qk chunks. Each chunk
    // scatters to dKV[:, chunk_idx*192:(chunk_idx+1)*192].
    CUTLASS_PRAGMA_NO_UNROLL
    for (int chunk_idx = 0; chunk_idx < 3; ++chunk_idx) {
      pipeline_mma_compute_dkdv.consumer_wait(pipeline_mma_compute_dkdv_consumer_state);

      auto load_op_dk_trail = SM100_TMEM_LOAD_32dp32b16x{};
      auto tDKtDK_trail = partition_fragment_C(
          TiledMmaDSQ{}, select<0,1>(TileShapeDSQ{}))(make_coord(_,_),_0{},_0{});
      tDKtDK_trail.data() = TmemAllocation::kDK;

      Tensor cDK_trail = domain_offset(
          make_coord(get<1>(blk_coord) * TileShapeK{}, _0{}),
          make_identity_tensor(take<0,2>(TileShapeDSQ{}))
      );

      auto tiled_t2r_trail = make_tmem_copy(load_op_dk_trail, tDKtDK_trail);
      auto thread_t2r_trail = tiled_t2r_trail.get_slice(dp_idx);

      Tensor tTR_cDK_tr = split_wg(thread_t2r_trail.partition_D(cDK_trail));
      Tensor tTR_rDK_tr = make_tensor<ElementAcc>(shape(tTR_cDK_tr));
      Tensor tTR_tDK_tr = split_wg(thread_t2r_trail.partition_S(tDKtDK_trail));
      cute::copy(tiled_t2r_trail, tTR_tDK_tr, tTR_rDK_tr);

      const int K_total_tr = get<1>(problem_shape);
      const int sq_idx_tr = get<1>(blk_coord);
      const int* gIndices_tr =
          mainloop_args.ptr_indices + sq_idx_tr * mainloop_args.topk;
      const int k_tile_base_tr = (iter_index - 1) * int(TileShapeK{});
      const int sq_block_base_tr = sq_idx_tr * int(TileShapeK{});
      const int row_stride_dk_tr = (int)get<0>(epilogue_args.stride_dk);
      ElementAcc* ptr_dkv_acc_tr = epilogue_args.ptr_dkv_acc;
      const int cta_d_offset_tr = chunk_idx * int(TileShapeDQK{});
      const float softmax_scale_tr = mainloop_args.softmax_scale;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tTR_rDK_tr); i++) {
        auto c = tTR_cDK_tr(i);
        int k_dense = get<0>(c);
        int d_local = get<1>(c);
        int k_local = k_dense - sq_block_base_tr;
        if (k_local < 0 || k_local >= int(TileShapeK{})) continue;
        if (d_local < 0 || d_local >= int(TileShapeDQK{})) continue;
        int kidx = gIndices_tr[k_tile_base_tr + k_local];
        if (kidx < 0 || kidx >= K_total_tr) continue;
        const size_t _off_dk_tr = (size_t)kidx * row_stride_dk_tr
                                 + cta_d_offset_tr + d_local;
        if (_off_dk_tr >= (size_t)K_total_tr * row_stride_dk_tr) continue;
        float v = tTR_rDK_tr(i) * softmax_scale_tr;
        atomicAdd(&ptr_dkv_acc_tr[_off_dk_tr], v);
      }

      cutlass::arch::fence_view_async_tmem_load();
      pipeline_mma_compute_dkdv.consumer_release(pipeline_mma_compute_dkdv_consumer_state);
      ++pipeline_mma_compute_dkdv_consumer_state;
    }
  }

  template<class BlkCoord, class ProblemShape_>
  CUTLASS_DEVICE void reduce(
      BlkCoord const& blk_coord,
      ProblemShape_ const& problem_shape,
      int iter_index,
      int iter_count,
      MainloopArguments const& mainloop_args,
      MainloopParams const& mainloop_params,
      TensorStorage& shared_tensors,
      PipelineMmaReduceDQ& pipeline_mma_reduce_dq,
      typename PipelineMmaReduceDQ::PipelineState& pipeline_mma_reduce_dq_consumer_state,
      PipelineReduceTmaStore& pipeline_reduce_tma_store,
      typename PipelineReduceTmaStore::PipelineState& pipeline_reduce_tma_store_producer_state) {

    using X = Underscore;

    auto [Q, K, D, D_VO, HB] = problem_shape;

    auto [blk_coord_q, blk_coord_k, blk_coord_d, blk_coord_dv, blk_coord_batch] = blk_coord;
    // sparse semantics -- get<1>(blk_coord) is the Q-token index.
    // All K-tile iters within this CTA reduce-add dQ into the same Q-token row.
    const int sq_idx = blk_coord_k;

    // must match TileShapeDQ
    auto load_op = SM100_TMEM_LOAD_16dp32b16x{};

    auto tDQtDQ = partition_fragment_C(TiledMmaDSK{}, select<0,1>(TileShapeDSK{}))(make_coord(_,_),_0{},_0{});
    tDQtDQ.data() = TmemAllocation::kDQ;

    Tensor mDQ = mainloop_params.tma_red_dq.get_tma_tensor(make_shape(Q, D, HB));
    // D path: KEEP num_D_tiles dim (3 tiles of 192 cols each). Previous code
    // sliced to _0{} which only exposed cols [0, 192) -- chunks 1,2 wrote OOB.
    auto gDQ_outer = local_tile(mDQ, TileShapeQK{}, make_coord(_,_,_), Step<_1, X, _1>{})
        (_, _, _, _, blk_coord_batch);
    // gDQ_outer shape: (TileShapeQ=64, TileShapeDQK=192, num_Q_tiles, num_D_tiles=3)

    Tensor cDQ = make_identity_tensor(take<0,2>(TileShapeDSK{}));

    Tensor sDQ = make_tensor(make_smem_ptr(shared_tensors.smem_dq.begin()), SmemLayoutDQ{});

    int thread_idx = threadIdx.x % (kNumReduceWarps * NumThreadsPerWarp);
    auto tiled_t2r = make_tmem_copy(load_op, tDQtDQ);
    auto thread_t2r = tiled_t2r.get_slice(thread_idx);

    Tensor tTR_cDQ   = thread_t2r.partition_D(cDQ);
    Tensor tTR_sDQ   = thread_t2r.partition_D(sDQ);
    Tensor tTR_tDQ = thread_t2r.partition_S(tDQtDQ);

    auto block_tma = mainloop_params.tma_red_dq.get_slice(_0{});

    Tensor tDQsDQ = block_tma.partition_S(sDQ);
    Tensor tDQcDQ = block_tma.partition_S(cDQ);

    int lane_predicate = (threadIdx.x % (kNumReduceWarps * NumThreadsPerWarp)) == 0;

    // D path: loop N iters consuming 3 dQ per iter (3N total).
    // MMA produces: 3(N-1) iter loop + 3 trailing = 3N. Balanced.
    while (iter_count > 0) {
      // D path: per K-iter, consume 3 dQ chunks (chunks 0..2). Each chunk
      // scatters to dQ_acc[:, chunk_idx*192:(chunk_idx+1)*192].
      CUTLASS_PRAGMA_NO_UNROLL
      for (int chunk_idx = 0; chunk_idx < 3; ++chunk_idx) {
        pipeline_mma_reduce_dq.consumer_wait(pipeline_mma_reduce_dq_consumer_state);

        Tensor tTR_rDQ = make_tensor<ElementAcc>(shape(tTR_cDQ));

        // load dQ from tmem to rmem
        cute::copy(tiled_t2r, tTR_tDQ, tTR_rDQ);

        cutlass::arch::fence_view_async_tmem_load();
        pipeline_mma_reduce_dq.consumer_release(pipeline_mma_reduce_dq_consumer_state);
        ++pipeline_mma_reduce_dq_consumer_state;

        // D path: select chunk_idx's d_qk tile (cols [chunk_idx*192, (chunk_idx+1)*192)).
        // partition_D over this tile gives 6 partitions of TileShapeDQ=32 cols each.
        auto gDQ_chunk = gDQ_outer(_, _, _, chunk_idx);
        Tensor tDQgDQ = block_tma.partition_D(gDQ_chunk);

        // we don't have enough smem to dump it all to smem, so we do it in stages
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<2>(tTR_cDQ); i++) {
          if (lane_predicate) {
            pipeline_reduce_tma_store.producer_acquire(pipeline_reduce_tma_store_producer_state);
          }
          // wait in all threads for the acquire to complete
          cutlass::arch::NamedBarrier(
              kNumReduceWarps * NumThreadsPerWarp,
              cutlass::arch::ReservedNamedBarriers::TransposeBarrier
          ).arrive_and_wait();

          cute::copy(tTR_rDQ(_, _, i), tTR_sDQ(_, _, _0{}, pipeline_reduce_tma_store_producer_state.index()));

          // wait for the stores to all be visible to the TMA
          cutlass::arch::fence_view_async_shared();
          cutlass::arch::NamedBarrier(
              kNumReduceWarps * NumThreadsPerWarp,
              cutlass::arch::ReservedNamedBarriers::TransposeBarrier
          ).arrive_and_wait();
          if (lane_predicate) {
            // i indexes into the 6 partitions of chunk's d_qk tile.
            copy(mainloop_params.tma_red_dq,
                 tDQsDQ(_,_,_0{}, pipeline_reduce_tma_store_producer_state.index()),
                 tDQgDQ(_,_, i, sq_idx));
            pipeline_reduce_tma_store.producer_commit(pipeline_reduce_tma_store_producer_state);
          }

          ++pipeline_reduce_tma_store_producer_state;
        }
      }

      iter_count -= 1;
      iter_index += 1;
    }
  }


  CUTLASS_DEVICE void operator()(Params const& params, char* smem) {
#if defined(KERUTILS_ENABLE_SM100A)
    int warp_idx = cutlass::canonical_warp_idx_sync();
    auto role = warp_idx_to_role(warp_idx);
    uint32_t lane_predicate = cute::elect_one_sync();

    // prefetch_tma_descriptor was the source of illegal access for
    // our sparse setup (binary search confirmed via build 614 vs 612).
    // The dense TMA descriptors (tma_load_q/k/v/do) are constructed but the
    // sparse path doesn't actually use them for K/V (gather4 path); even Q/dO
    // use direct TMA load via the descriptor object, not via the prefetched
    // L2 cache hint. Skipping prefetch is purely a perf cost (cold descriptor
    // fetch on first use) -- correctness unaffected.
    // if (role == WarpRole::Load && lane_predicate) {
    //   prefetch_tma_descriptor(params.mainloop_params.tma_load_q.get_tma_descriptor());
    //   prefetch_tma_descriptor(params.mainloop_params.tma_load_k.get_tma_descriptor());
    //   prefetch_tma_descriptor(params.mainloop_params.tma_load_v.get_tma_descriptor());
    //   prefetch_tma_descriptor(params.mainloop_params.tma_load_do.get_tma_descriptor());
    // }

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem);

    int initializing_warp = 0;
    typename PipelineLoadMmaQ::Params pipeline_load_mma_q_params;
    if (role == WarpRole::Load) {
      pipeline_load_mma_q_params.role = PipelineLoadMmaQ::ThreadCategory::Producer;
    }
    if (role == WarpRole::Mma) {
      pipeline_load_mma_q_params.role = PipelineLoadMmaQ::ThreadCategory::Consumer;
    }
    pipeline_load_mma_q_params.is_leader = lane_predicate && (role == WarpRole::Load);
    // Also loads K in the first iteration
    pipeline_load_mma_q_params.transaction_bytes = kTransactionsBytesLoadQ;
    pipeline_load_mma_q_params.initializing_warp = initializing_warp++;
    PipelineLoadMmaQ pipeline_load_mma_q(shared_storage.pipelines.load_mma_q, pipeline_load_mma_q_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{});

    typename PipelineLoadMmaDO::Params pipeline_load_mma_do_params;
    if (role == WarpRole::Load) {
      pipeline_load_mma_do_params.role = PipelineLoadMmaDO::ThreadCategory::Producer;
    }
    if (role == WarpRole::Mma) {
      pipeline_load_mma_do_params.role = PipelineLoadMmaDO::ThreadCategory::Consumer;
    }
    pipeline_load_mma_do_params.is_leader = lane_predicate && (role == WarpRole::Load);
    // Also loads V in the first iteration
    pipeline_load_mma_do_params.transaction_bytes = kTransactionsBytesLoadDO;
    pipeline_load_mma_do_params.initializing_warp = initializing_warp++;
    PipelineLoadMmaDO pipeline_load_mma_do(shared_storage.pipelines.load_mma_do, pipeline_load_mma_do_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{});

    typename PipelineLoadComputeLSE::Params pipeline_load_compute_lse_params;
    if (role == WarpRole::Load) {
      pipeline_load_compute_lse_params.role = PipelineLoadComputeLSE::ThreadCategory::Producer;
    }
    if (role == WarpRole::Compute) {
      pipeline_load_compute_lse_params.role = PipelineLoadComputeLSE::ThreadCategory::Consumer;
    }
    pipeline_load_compute_lse_params.producer_arv_count = NumThreadsPerWarp;
    pipeline_load_compute_lse_params.consumer_arv_count = kNumComputeWarps * NumThreadsPerWarp;
    pipeline_load_compute_lse_params.initializing_warp = initializing_warp++;
    PipelineLoadComputeLSE pipeline_load_compute_lse(
      shared_storage.pipelines.load_compute_lse,
      pipeline_load_compute_lse_params,
      /*barrier init*/ cute::true_type{});

    typename PipelineLoadComputeSumOdO::Params pipeline_load_compute_sum_odo_params;
    if (role == WarpRole::Load) {
      pipeline_load_compute_sum_odo_params.role = PipelineLoadComputeSumOdO::ThreadCategory::Producer;
    }
    if (role == WarpRole::Compute) {
      pipeline_load_compute_sum_odo_params.role = PipelineLoadComputeSumOdO::ThreadCategory::Consumer;
    }
    pipeline_load_compute_sum_odo_params.producer_arv_count = NumThreadsPerWarp;
    pipeline_load_compute_sum_odo_params.consumer_arv_count = kNumComputeWarps * NumThreadsPerWarp;
    pipeline_load_compute_sum_odo_params.initializing_warp = initializing_warp++;
    PipelineLoadComputeSumOdO pipeline_load_compute_sum_odo(
      shared_storage.pipelines.load_compute_sum_odo,
      pipeline_load_compute_sum_odo_params,
      /*barrier init*/ cute::true_type{});

    typename PipelineMmaComputeS::Params pipeline_mma_compute_s_params;
    if (role == WarpRole::Mma) {
      pipeline_mma_compute_s_params.role = PipelineMmaComputeS::ThreadCategory::Producer;
    }
    if (role == WarpRole::Compute) {
      pipeline_mma_compute_s_params.role = PipelineMmaComputeS::ThreadCategory::Consumer;
    }
    pipeline_mma_compute_s_params.consumer_arv_count = kNumComputeWarps * cutlass::NumThreadsPerWarp;
    pipeline_mma_compute_s_params.initializing_warp = initializing_warp++;
    PipelineMmaComputeS pipeline_mma_compute_s(
      shared_storage.pipelines.mma_compute_s,
      pipeline_mma_compute_s_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{});

    typename PipelineMmaComputeDP::Params pipeline_mma_compute_dp_params;
    if (role == WarpRole::Mma) {
      pipeline_mma_compute_dp_params.role = PipelineMmaComputeDP::ThreadCategory::Producer;
    }
    if (role == WarpRole::Compute) {
      pipeline_mma_compute_dp_params.role = PipelineMmaComputeDP::ThreadCategory::Consumer;
    }
    pipeline_mma_compute_dp_params.consumer_arv_count = kNumComputeWarps * cutlass::NumThreadsPerWarp;
    pipeline_mma_compute_dp_params.initializing_warp = initializing_warp++;
    PipelineMmaComputeDP pipeline_mma_compute_dp(
      shared_storage.pipelines.mma_compute_dp,
      pipeline_mma_compute_dp_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{});

    typename PipelineMmaReduceDQ::Params pipeline_mma_reduce_dq_params;
    if (role == WarpRole::Mma) {
      pipeline_mma_reduce_dq_params.role = PipelineMmaReduceDQ::ThreadCategory::Producer;
    }
    if (role == WarpRole::Reduce) {
      pipeline_mma_reduce_dq_params.role = PipelineMmaReduceDQ::ThreadCategory::Consumer;
    }
    pipeline_mma_reduce_dq_params.consumer_arv_count = kNumReduceWarps * cutlass::NumThreadsPerWarp;
    pipeline_mma_reduce_dq_params.initializing_warp = initializing_warp++;
    PipelineMmaReduceDQ pipeline_mma_reduce_dq(
      shared_storage.pipelines.mma_reduce_dq,
      pipeline_mma_reduce_dq_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{});

    typename PipelineComputeMmaP::Params pipeline_compute_mma_p_params;
    if (role == WarpRole::Mma) {
      pipeline_compute_mma_p_params.role = PipelineComputeMmaP::ThreadCategory::Consumer;
    }
    if (role == WarpRole::Compute) {
      pipeline_compute_mma_p_params.role = PipelineComputeMmaP::ThreadCategory::Producer;
    }
    pipeline_compute_mma_p_params.producer_arv_count = kNumComputeWarps * cutlass::NumThreadsPerWarp;
    pipeline_compute_mma_p_params.consumer_arv_count = 1;
    pipeline_compute_mma_p_params.initializing_warp = initializing_warp++;
    PipelineComputeMmaP pipeline_compute_mma_p(
      shared_storage.pipelines.compute_mma_p,
      pipeline_compute_mma_p_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{});

    typename PipelineComputeMmaDS::Params pipeline_compute_mma_ds_params;
    if (role == WarpRole::Mma) {
      pipeline_compute_mma_ds_params.role = PipelineComputeMmaDS::ThreadCategory::Consumer;
    }
    if (role == WarpRole::Compute) {
      pipeline_compute_mma_ds_params.role = PipelineComputeMmaDS::ThreadCategory::Producer;
    }
    pipeline_compute_mma_ds_params.producer_arv_count = kNumComputeWarps * cutlass::NumThreadsPerWarp;
    pipeline_compute_mma_ds_params.consumer_arv_count = 1;
    pipeline_compute_mma_ds_params.initializing_warp = initializing_warp++;
    PipelineComputeMmaDS pipeline_compute_mma_ds(
      shared_storage.pipelines.compute_mma_ds,
      pipeline_compute_mma_ds_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{});

    typename PipelineMmaComputeDKDV::Params pipeline_mma_compute_dkdv_params;
    if (role == WarpRole::Mma) {
      pipeline_mma_compute_dkdv_params.role = PipelineMmaComputeDKDV::ThreadCategory::Producer;
    }
    if (role == WarpRole::Compute) {
      pipeline_mma_compute_dkdv_params.role = PipelineMmaComputeDKDV::ThreadCategory::Consumer;
    }
    pipeline_mma_compute_dkdv_params.consumer_arv_count = kNumComputeWarps * cutlass::NumThreadsPerWarp;
    pipeline_mma_compute_dkdv_params.initializing_warp = initializing_warp++;
    PipelineMmaComputeDKDV pipeline_mma_compute_dkdv(
      shared_storage.pipelines.mma_compute_dkdv,
      pipeline_mma_compute_dkdv_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{});
    PipelineReduceTmaStore pipeline_reduce_tma_store;

    TmemAllocator tmem_allocator;

    pipeline_init_arrive_relaxed(size(ClusterShape{}));

    pipeline_load_mma_q.init_masks(ClusterShape{});
    pipeline_load_mma_do.init_masks(ClusterShape{});
    pipeline_mma_compute_s.init_masks(ClusterShape{});
    pipeline_mma_compute_dp.init_masks(ClusterShape{});
    pipeline_mma_reduce_dq.init_masks(ClusterShape{});
    pipeline_compute_mma_p.init_masks(ClusterShape{});
    pipeline_compute_mma_ds.init_masks(ClusterShape{});
    pipeline_mma_compute_dkdv.init_masks(ClusterShape{});

    typename decltype(pipeline_load_mma_q)::PipelineState pipeline_load_mma_q_consumer_state;
    typename decltype(pipeline_load_mma_do)::PipelineState pipeline_load_mma_do_consumer_state;
    typename decltype(pipeline_load_compute_lse)::PipelineState pipeline_load_compute_lse_consumer_state;
    typename decltype(pipeline_load_compute_sum_odo)::PipelineState pipeline_load_compute_sum_odo_consumer_state;
    typename decltype(pipeline_mma_compute_s)::PipelineState pipeline_mma_compute_s_consumer_state;
    typename decltype(pipeline_mma_compute_dp)::PipelineState pipeline_mma_compute_dp_consumer_state;
    typename decltype(pipeline_mma_reduce_dq)::PipelineState pipeline_mma_reduce_dq_consumer_state;
    typename decltype(pipeline_compute_mma_p)::PipelineState pipeline_compute_mma_p_consumer_state;
    typename decltype(pipeline_compute_mma_ds)::PipelineState pipeline_compute_mma_ds_consumer_state;
    typename decltype(pipeline_mma_compute_dkdv)::PipelineState pipeline_mma_compute_dkdv_consumer_state;

    auto pipeline_load_mma_q_producer_state = make_producer_start_state<decltype(pipeline_load_mma_q)>();
    auto pipeline_load_mma_do_producer_state = make_producer_start_state<decltype(pipeline_load_mma_do)>();
    auto pipeline_load_compute_lse_producer_state = make_producer_start_state<decltype(pipeline_load_compute_lse)>();
    auto pipeline_load_compute_sum_odo_producer_state = make_producer_start_state<decltype(pipeline_load_compute_sum_odo)>();
    auto pipeline_mma_compute_s_producer_state = make_producer_start_state<decltype(pipeline_mma_compute_s)>();
    auto pipeline_mma_compute_dp_producer_state = make_producer_start_state<decltype(pipeline_mma_compute_dp)>();
    auto pipeline_mma_reduce_dq_producer_state = make_producer_start_state<decltype(pipeline_mma_reduce_dq)>();
    auto pipeline_compute_mma_p_producer_state = make_producer_start_state<decltype(pipeline_compute_mma_p)>();
    auto pipeline_compute_mma_ds_producer_state = make_producer_start_state<decltype(pipeline_compute_mma_ds)>();
    auto pipeline_mma_compute_dkdv_producer_state = make_producer_start_state<decltype(pipeline_mma_compute_dkdv)>();
    auto pipeline_reduce_tma_store_producer_state = make_producer_start_state<decltype(pipeline_reduce_tma_store)>();

    pipeline_init_wait(size(ClusterShape{}));

    // sparse iteration topology.
    //
    // Dense MLA bwd: grid = (s_kv/TileShapeK, H, B) -- each CTA owns a fixed
    // K-block and iterates over Q-blocks. iter_index there is the Q-block.
    //
    // D path: grid = (s_q, 1, 1) -- one CTA per Q-token, no cluster.
    // Each CTA processes ALL 3 d_qk chunks sequentially (loops in Load/MMA).
    //
    // Re-purposed blk_coord fields:
    //   get<0>(blk_coord) -- always 0 (single Q-token per CTA; B_H=64 covers H)
    //   get<1>(blk_coord) -- Q-token index (was K-block index in dense)
    //   get<4>(blk_coord) -- (H_idx=0, B_idx=0)  (DSA: B=1, H baked into B_H)
    //
    // iter_index = K-tile index (0..ceil(topk/B_TOPK)-1). iter_start = 0
    // (sparse indices already encode causality; no extra Q-vs-K mask needed).
    const int sq_idx_in_cluster = blockIdx.x;
    auto blk_coord = make_coord(_0{}, sq_idx_in_cluster, _0{}, _0{},
                                make_coord(blockIdx.y, blockIdx.z));
    auto [problem_shape, blk_offset] = apply_variable_length_offset(
        params.problem_shape,
        blk_coord
    );
    // iter_count = number of K-tiles per Q-token = ceil_div(topk, B_TOPK).
    int iter_count = (params.mainloop.topk + int(TileShapeK{}) - 1) / int(TileShapeK{});
    int iter_start = 0;

    // Q-token bounds check: skip if Q-token is outside [0, s_q).
    if (sq_idx_in_cluster >= get<0>(problem_shape)) {
      return;
    }

    if (iter_count <= 0) {
      epilogue_clear(
          blk_coord,
          blk_offset,
          problem_shape,
          params.mainloop,
          params.epilogue
      );
      return;
    }

    if (role == WarpRole::Load) {
      warpgroup_reg_set<RegisterAllocation::kLoad>();

      load(
          blk_coord,
          blk_offset,
          problem_shape,
          iter_start,
          iter_count,
          params.mainloop,
          params.mainloop_params,
          shared_storage.tensors,
          pipeline_load_mma_q, pipeline_load_mma_q_producer_state,
          pipeline_load_mma_do, pipeline_load_mma_do_producer_state,
          pipeline_load_compute_lse, pipeline_load_compute_lse_producer_state,
          pipeline_load_compute_sum_odo, pipeline_load_compute_sum_odo_producer_state
      );

    }
    else if (role == WarpRole::Mma) {
      warpgroup_reg_set<RegisterAllocation::kMma>();

      tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
      __syncwarp();

      mma(
          blk_coord,
          problem_shape,
          iter_start,
          iter_count,
          params.mainloop,
          shared_storage.tensors,
          pipeline_load_mma_q, pipeline_load_mma_q_consumer_state,
          pipeline_load_mma_do, pipeline_load_mma_do_consumer_state,
          pipeline_mma_compute_s, pipeline_mma_compute_s_producer_state,
          pipeline_mma_compute_dp, pipeline_mma_compute_dp_producer_state,
          pipeline_mma_reduce_dq, pipeline_mma_reduce_dq_producer_state,
          pipeline_compute_mma_p, pipeline_compute_mma_p_consumer_state,
          pipeline_compute_mma_ds, pipeline_compute_mma_ds_consumer_state,
          pipeline_mma_compute_dkdv, pipeline_mma_compute_dkdv_producer_state
      );

    }
    else if (role == WarpRole::Compute) {
      warpgroup_reg_set<RegisterAllocation::kCompute>();

      compute(
          blk_coord,
          blk_offset,
          problem_shape,
          iter_start,
          iter_count,
          params.mainloop,
          params.epilogue,
          shared_storage.tensors,
          pipeline_load_compute_lse, pipeline_load_compute_lse_consumer_state,
          pipeline_load_compute_sum_odo, pipeline_load_compute_sum_odo_consumer_state,
          pipeline_mma_compute_s, pipeline_mma_compute_s_consumer_state,
          pipeline_mma_compute_dp, pipeline_mma_compute_dp_consumer_state,
          pipeline_compute_mma_p, pipeline_compute_mma_p_producer_state,
          pipeline_compute_mma_ds, pipeline_compute_mma_ds_producer_state,
          pipeline_mma_compute_dkdv, pipeline_mma_compute_dkdv_consumer_state
      );

      cutlass::arch::NamedBarrier(
          kNumComputeWarps * NumThreadsPerWarp,
          cutlass::arch::ReservedNamedBarriers::EpilogueBarrier
      ).arrive_and_wait();

      if (warp_idx % kNumComputeWarps == 0) {
        uint32_t free_stage_ptr = shared_storage.tmem_base_ptr;
        tmem_allocator.free(free_stage_ptr, TmemAllocator::Sm100TmemCapacityColumns);
      }

    }
    else if (role == WarpRole::Reduce) {
      warpgroup_reg_set<RegisterAllocation::kReduce>();

      reduce(
          blk_coord,
          problem_shape,
          iter_start,
          iter_count,
          params.mainloop,
          params.mainloop_params,
          shared_storage.tensors,
          pipeline_mma_reduce_dq, pipeline_mma_reduce_dq_consumer_state,
          pipeline_reduce_tma_store, pipeline_reduce_tma_store_producer_state
      );

      pipeline_reduce_tma_store.producer_tail(pipeline_reduce_tma_store_producer_state);
    }
    else {
      warpgroup_reg_set<RegisterAllocation::kEmpty>();

      /* no-op */

    }
#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm100\n");
    }
#endif
  }

  static dim3 get_block_shape() {
    dim3 block(MaxThreadsPerBlock, 1, 1);
    return block;
  }

  static dim3 get_grid_shape(Params const& params) {
    auto [Q, K, D, D_VO, HB] = params.problem_shape;
    auto [H, B] = HB;
    dim3 grid(ceil_div(K, TileShapeK{}), H, B);
    return grid;
  }
};

}  // namespace cutlass::fmha::kernel
