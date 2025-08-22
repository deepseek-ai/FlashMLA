#pragma once

#include "collective/fmha_fusion.hpp"
#include "collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp"
#include "collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"
#include "collective/sm100_fmha_mla_fwd_mainloop_tma_warpspecialized.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "device/fmha.hpp"
#include "kernel/fmha_causal_tile_scheduler.hpp"
#include "kernel/fmha_options.hpp"
#include "kernel/fmha_tile_scheduler.hpp"
#include "kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp"

#include <torch/library.h>
#include <c10/cuda/CUDAStream.h>

using namespace cute;
using namespace cutlass::fmha::collective;
using namespace cutlass::fmha::kernel;
using namespace cutlass::fmha::device;

struct FmhaOptions {
  int b = 1;
  int h = 1;
  int h_k = 1;
  int q = 256;
  int k = 256;
  int d = 128;
};

struct MlaOptions {
  int b = 1;
  int h = 1;
  int h_k = 1;
  int q = 256;
  int k = 256;
  int dl = 128; // headdim latent
  int dr = 64;  // headdim rope
};

template <bool kIsMla, bool kIsMaskTileSchedulerValid, bool kIsVarlen, class Element_,
          class ElementOut_, class ActiveMask, class... KernelOptions>
struct FwdRunner {

  using Element = Element_;
  using ElementAccumulatorQK = float;
  using ElementAccumulatorPV = float;
  using ElementOut = ElementOut_;

  using HeadDimLatent = _128;
  using HeadDim = Shape<HeadDimLatent, _64>;
  using TileShapeMla = Shape<_256, _128, HeadDim>;
  using TileShapeFmha = Shape<_256, _128, _128>;
  using TileShape = std::conditional_t<kIsMla, TileShapeMla, TileShapeFmha>;

  using ProblemShapeRegular = std::conditional_t<
      kIsMla,
      cute::tuple<int, int, cute::tuple<int, int>, cute::tuple<cute::tuple<int, int>, int>>,
      cute::tuple<int, int, int, cute::tuple<cute::tuple<int, int>, int>>>;

  using ProblemShapeVarlen =
      std::conditional_t<kIsMla,
                         cute::tuple<VariableLength, VariableLength, cute::tuple<int, int>,
                                     cute::tuple<cute::tuple<int, int>, int>>,
                         cute::tuple<VariableLength, VariableLength, int,
                                     cute::tuple<cute::tuple<int, int>, int>>>;

  using ProblemShapeType =
      std::conditional_t<kIsVarlen, ProblemShapeVarlen, ProblemShapeRegular>;

  using StrideQ = cute::tuple<int, _1, cute::tuple<cute::tuple<int, int>, int>>;
  using StrideK = cute::tuple<int, _1, cute::tuple<cute::tuple<_0, int>, int>>;
  using StrideV = StrideK;
  using StrideO = StrideQ;
  using StrideLSE = cute::tuple<_1, cute::tuple<cute::tuple<int, int>, int>>;

  static constexpr bool kIsPersistent =
      find_option_t<Tag::kIsPersistent, true_type, KernelOptions...>::value;

  using TileScheduler = std::conditional_t<
      kIsPersistent,
      std::conditional_t<std::is_same_v<ActiveMask, CausalMask<false>> ||
                             std::is_same_v<ActiveMask, CausalMask<true>>,
                         cutlass::fmha::kernel::CausalPersistentTileScheduler,
                         cutlass::fmha::kernel::PersistentTileScheduler>,
      std::conditional_t<kIsMaskTileSchedulerValid,
                         cutlass::fmha::kernel::CausalIndividualTileScheduler,
                         cutlass::fmha::kernel::IndividualTileScheduler>>;

  static constexpr bool IsOrderLoadEpilogue =
      kIsPersistent && (sizeof(Element) == sizeof(ElementOut));
  using OrderLoadEpilogue = std::conditional_t<IsOrderLoadEpilogue, true_type, false_type>;

  using MainloopMla = cutlass::fmha::collective::Sm100MlaFwdMainloopTmaWarpspecialized<
      Element, ElementAccumulatorQK, ElementAccumulatorPV, TileShapeMla, StrideQ, StrideK,
      StrideV, ActiveMask, Shape<_2, _1, _1>, OrderLoadEpilogue>;

  using OperationMla =
      cutlass::fmha::device::FMHA<cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<
          ProblemShapeType, MainloopMla,
          cutlass::fmha::collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<
              ElementOut, ElementAccumulatorPV, typename MainloopMla::TileShapePV, StrideO,
              StrideLSE, OrderLoadEpilogue>,
          TileScheduler, cutlass::fmha::kernel::Sm100MlaFwdCtxKernelWarpspecializedSchedule>>;

  using MainloopFmha = cutlass::fmha::collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
      Element, ElementAccumulatorQK, ElementAccumulatorPV, TileShapeFmha, StrideQ, StrideK,
      StrideV, ActiveMask>;

  using OperationFmha =
      cutlass::fmha::device::FMHA<cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<
          ProblemShapeType, MainloopFmha,
          cutlass::fmha::collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<
              ElementOut, ElementAccumulatorPV, typename MainloopFmha::TileShapePV, StrideO,
              StrideLSE>,
          TileScheduler>>;

  using Mainloop = std::conditional_t<kIsMla, MainloopMla, MainloopFmha>;
  using Operation = std::conditional_t<kIsMla, OperationMla, OperationFmha>;

  //
  // Data members
  //

  /// Initialization
  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;
  StrideLSE stride_LSE;

  template <class ProblemShape>
  auto initialize_varlen(const ProblemShape &problem_size, int max_seqlen_q, int max_seqlen_kv,
                         int total_seqlen_q, int total_seqlen_kv) {

    int num_batches = get<3, 1>(problem_size);

    ProblemShape problem_size_for_init = problem_size;
    get<3, 1>(problem_size_for_init) = 1;
    get<0>(problem_size_for_init) = total_seqlen_q;
    get<1>(problem_size_for_init) = total_seqlen_kv;

    ProblemShapeType problem_size_for_launch;

    get<0>(problem_size_for_launch) = VariableLength{max_seqlen_q, nullptr, total_seqlen_q};
    get<1>(problem_size_for_launch) = VariableLength{max_seqlen_kv, nullptr, total_seqlen_kv};
    get<2>(problem_size_for_launch) = get<2>(problem_size);
    get<3>(problem_size_for_launch) = get<3>(problem_size);

    return cute::make_tuple(problem_size_for_init, problem_size_for_launch);
  }

  template <class Options>
  static constexpr auto get_problem_shape(const Options &options) {
    int h_r = options.h / options.h_k;
    if constexpr (std::is_same_v<Options, MlaOptions>) {
      return cute::make_tuple(options.q, options.k, cute::make_tuple(options.dl, options.dr),
                              cute::make_tuple(cute::make_tuple(h_r, options.h_k), options.b));
    } else {
      return cute::make_tuple(options.q, options.k, options.d,
                              cute::make_tuple(cute::make_tuple(h_r, options.h_k), options.b));
    }
  }

  template <class Options>
  ProblemShapeType initialize(const Options &options, int max_seqlen_q, int max_seqlen_kv,
                                   int total_seqlen_q, int total_seqlen_kv,
                                   void *cumulative_length_q, void *cumulative_length_kv) {
    assert(options.h % options.h_k == 0);
    auto problem_shape_in = get_problem_shape(options);

    ProblemShapeType problem_shape;
    decltype(problem_shape_in) problem_size;

    if constexpr (kIsVarlen) {
      auto [problem_shape_init, problem_shape_launch] = initialize_varlen(
          problem_shape_in, max_seqlen_q, max_seqlen_kv, total_seqlen_q, total_seqlen_kv);
      problem_shape = problem_shape_launch;
      problem_size = problem_shape_init;
    } else {
      problem_size = problem_shape_in;
      problem_shape = problem_shape_in;
    }

    auto get_head_dimension = [&]() {
      if constexpr (rank_v<decltype(get<2>(problem_shape))> == 2) {
        return cute::make_tuple(size<2, 0>(problem_shape) + size<2, 1>(problem_shape),
                                size<2, 0>(problem_shape));
      } else {
        return cute::make_tuple(size<2>(problem_size), size<2>(problem_size));
      }
    };


    if constexpr (kIsVarlen) {
      get<0>(problem_shape).cumulative_length = static_cast<int *>(cumulative_length_q);
      get<1>(problem_shape).cumulative_length = static_cast<int *>(cumulative_length_kv);
    }

    return problem_shape;
  }

  auto get_arguments(const ProblemShapeType &problem_shape,
                     const cutlass::KernelHardwareInfo &hw_info, float scale_softmax,
                     void *q_ptr, void *k_ptr, void *v_ptr, void *o_ptr, void *lse_ptr,
                     void *cumulative_length_q, void *cumulative_length_kv) {
    auto problem_shape_ = problem_shape;

    typename Operation::Arguments arguments{
        problem_shape_,
        {static_cast<Element *>(q_ptr), stride_Q, static_cast<Element *>(k_ptr), stride_K,
         static_cast<Element *>(v_ptr), stride_V, scale_softmax},
        {static_cast<ElementOut *>(o_ptr), stride_O,
         static_cast<ElementAccumulatorPV *>(lse_ptr), stride_LSE},
        hw_info};

    return arguments;
  }

  template <class Options>
  void run(const Options &options, const cutlass::KernelHardwareInfo &hw_info, at::Tensor q,
           at::Tensor k, at::Tensor v, at::Tensor o, at::Tensor lse, float scale_softmax,
           at::Tensor workspace, at::Tensor cumulative_seqlen_q,
           at::Tensor cumulative_seqlen_kv, int max_seqlen_q, int max_seqlen_kv) {

    int total_seqlen_q = q.size(0);
    int total_seqlen_kv = k.size(0);

    ProblemShapeType problem_shape =
        initialize(options, max_seqlen_q, max_seqlen_kv, total_seqlen_q, total_seqlen_kv,
                        cumulative_seqlen_q.data_ptr(), cumulative_seqlen_kv.data_ptr());
    
    int SQ = size<0>(problem_shape);
    int SK = size<1>(problem_shape);
    int B = size<3, 1>(problem_shape);
    int H = size<3, 0>(problem_shape);
    int H_K = size<3, 0, 1>(problem_shape);
    int H_Q = size<3, 0, 0>(problem_shape);

    int q_stride0 = q.stride(0), q_stride1 = q.stride(1), q_stride2 = q.stride(2);
    int k_stride0 = k.stride(0), k_stride1 = k.stride(1), k_stride2 = k.stride(2);
    int v_stride0 = v.stride(0), v_stride1 = v.stride(1), v_stride2 = v.stride(2);
    int o_stride0 = o.stride(0), o_stride1 = o.stride(1), o_stride2 = o.stride(2);
    int lse_stride0 = lse.stride(0), lse_stride1 = lse.stride(1);
    TORCH_CHECK(q_stride2 == 1);
    TORCH_CHECK(k_stride2 == 1);
    TORCH_CHECK(v_stride2 == 1);
    TORCH_CHECK(o_stride2 == 1);
    TORCH_CHECK(lse_stride0 == 1);

    stride_Q = make_stride(q_stride0, _1{}, make_stride(make_stride(q_stride1, H_Q * q_stride1), SQ * q_stride0));
    stride_O = make_stride(o_stride0, _1{}, make_stride(make_stride(o_stride1, H_Q * o_stride1), SQ * o_stride0));
    stride_K = make_stride(k_stride0, _1{}, make_stride(make_stride(_0{}, k_stride1), SK * k_stride0));
    stride_V = make_stride(v_stride0, _1{}, make_stride(make_stride(_0{}, v_stride1), SK * v_stride0));
    stride_LSE = make_stride(_1{}, make_stride(make_stride(lse_stride1, lse_stride1 * H_Q), SQ));

    if constexpr (kIsVarlen) {
      get<2, 1>(stride_Q) = 0;
      get<2, 1>(stride_K) = 0;
      get<2, 1>(stride_V) = 0;
      get<2, 1>(stride_O) = 0;
      get<1, 1>(stride_LSE) = 0;
    }

    typename Operation::Arguments arguments =
        get_arguments(problem_shape, hw_info, scale_softmax, q.data_ptr(), k.data_ptr(),
                      v.data_ptr(), o.data_ptr(), lse.data_ptr(),
                      cumulative_seqlen_q.data_ptr(), cumulative_seqlen_kv.data_ptr());

    Operation op;

    // size_t workspace_size = 0;
    // workspace_size = Operation::get_workspace_size(arguments);

    // todo: if use workspace, need check workspace size first.
    // we don't use workspace in current version.

    CUTLASS_CHECK(op.can_implement(arguments));
    CUTLASS_CHECK(op.initialize(arguments, nullptr));
    CUTLASS_CHECK(op.run(at::cuda::getCurrentCUDAStream()));
  }
};

template <class DTypeIn, class DTypeOut, bool kIsVarlen, bool kIsMla, class ActiveMask,
          class... KernelOptions>
void run_fmha_fwd(at::Tensor workspace, at::Tensor q, at::Tensor k, at::Tensor v,
                  at::Tensor cumulative_seqlen_q, at::Tensor cumulative_seqlen_kv, at::Tensor o,
                  at::Tensor lse, float scale_softmax, int max_seqlen_q, int max_seqlen_kv) {

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  auto get_options = [&]() {
    if constexpr (kIsMla) {
      MlaOptions options;
      options.b = cumulative_seqlen_q.size(0) - 1;
      options.h = q.size(1);
      options.h_k = k.size(1);
      options.q = q.size(0) / options.b;
      options.k = k.size(0) / options.b;
      options.dl = v.size(-1);
      options.dr = q.size(-1) - v.size(-1);
      return options;
    } else {
      FmhaOptions options;
      options.b = cumulative_seqlen_q.size(0) - 1;
      options.h = q.size(1);
      options.h_k = k.size(1);
      options.q = q.size(0) / options.b;
      options.k = k.size(0) / options.b;
      options.d = q.size(-1);
      return options;
    }
  };

  auto options = get_options();

  if (options.h % cutlass::fmha::kernel::CausalIndividualTileScheduler::TileH == 0 &&
      (std::is_same_v<ActiveMask, CausalMask<false>> || std::is_same_v<ActiveMask, CausalMask<true>>)) {
    FwdRunner<kIsMla, true, kIsVarlen, DTypeIn, DTypeOut, ActiveMask, KernelOptions...> runner;
    runner.run(options, hw_info, q, k, v, o, lse, scale_softmax, workspace, cumulative_seqlen_q,
               cumulative_seqlen_kv, max_seqlen_q, max_seqlen_kv);
  } else {
    FwdRunner<kIsMla, false, kIsVarlen, DTypeIn, DTypeOut, ActiveMask, KernelOptions...> runner;
    runner.run(options, hw_info, q, k, v, o, lse, scale_softmax, workspace, cumulative_seqlen_q,
               cumulative_seqlen_kv, max_seqlen_q, max_seqlen_kv);
  }
}
