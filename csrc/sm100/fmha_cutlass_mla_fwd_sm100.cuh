#pragma once

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"

#include "device/fmha.hpp"
#include "collective/fmha_fusion.hpp"
#include "collective/sm100_fmha_mla_fwd_mainloop_tma_warpspecialized.hpp"
#include "collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp"
#include "kernel/fmha_options.hpp"
#include "kernel/fmha_tile_scheduler.hpp"
#include "kernel/fmha_causal_tile_scheduler.hpp"
#include "kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp"

using namespace cute;
using namespace cutlass::fmha::collective;
using namespace cutlass::fmha::kernel;
using namespace cutlass::fmha::device;

struct Options {
  int b = 1;
  int h = 1;
  int h_k = 1;
  int q = 256;
  int k = 256;
  int dl = 128; // headdim latent
  int dr = 64;  // headdim rope
};

template<
  bool kIsMaskTileSchedulerValid,
  bool kIsVarlen,
  class Element_,
  class ElementOut_,
  class TileShape,
  class ActiveMask,
  class... KernelOptions
>
struct MlaFwdRunner {

  using Element = Element_;
  using ElementAccumulatorQK = float;
  using ElementAccumulatorPV = float;
  using ElementOut = ElementOut_;

  // Q K (D_latent D_rope) (H B)
  using ProblemShapeRegular = cute::tuple<int, int, cute::tuple<int, int>, cute::tuple<cute::tuple<int, int>, int>>;
  using ProblemShapeVarlen = cute::tuple<VariableLength, VariableLength, cute::tuple<int, int>, cute::tuple<cute::tuple<int, int>, int>>;
  using ProblemShapeType = std::conditional_t<kIsVarlen, ProblemShapeVarlen, ProblemShapeRegular>;
  
  using StrideQ = cute::tuple<int, _1, cute::tuple<cute::tuple<int, int>, int>>;  // Q D (H_G H_R B)
  using StrideK = cute::tuple<int, _1, cute::tuple<cute::tuple<_0, int>, int>>;  // K D (H_G H_R B)
  using StrideV = StrideK;
  using StrideO = StrideQ;
  using StrideLSE = cute::tuple<_1, cute::tuple<cute::tuple<int, int>, int>>;     // Q   (H_G H_R B)

  static constexpr bool kIsPersistent = find_option_t<Tag::kIsPersistent, true_type, KernelOptions...>::value;

  
  using TileScheduler = std::conditional_t<kIsPersistent, 
                                          std::conditional_t<std::is_same_v<ActiveMask, CausalMask<false>> 
                                                                          || std::is_same_v<ActiveMask, CausalMask<true>>, 
                                                            cutlass::fmha::kernel::CausalPersistentTileScheduler,
                                                            cutlass::fmha::kernel::PersistentTileScheduler>,
                                          std::conditional_t<kIsMaskTileSchedulerValid, 
                                                            cutlass::fmha::kernel::CausalIndividualTileScheduler,
                                                            cutlass::fmha::kernel::IndividualTileScheduler>>;

  static constexpr bool IsOrderLoadEpilogue = kIsPersistent && (sizeof(Element) == sizeof(ElementOut));
  using OrderLoadEpilogue = std::conditional_t<IsOrderLoadEpilogue, true_type, false_type>;

  using Mainloop = 
    cutlass::fmha::collective::Sm100MlaFwdMainloopTmaWarpspecialized<
      Element, ElementAccumulatorQK, ElementAccumulatorPV,
      TileShape, StrideQ, StrideK, StrideV,
      ActiveMask, Shape<_2, _1, _1>, OrderLoadEpilogue
    >;
  using Operation = cutlass::fmha::device::FMHA<
    cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<
      ProblemShapeType,
      Mainloop,
      cutlass::fmha::collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<
        ElementOut, ElementAccumulatorPV,
        typename Mainloop::TileShapePV,
        StrideO, StrideLSE, OrderLoadEpilogue
      >,
      TileScheduler,
      cutlass::fmha::kernel::Sm100MlaFwdCtxKernelWarpspecializedSchedule
    >>;

  //
  // Data members
  //

  /// Initialization
  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;
  StrideLSE stride_LSE;


  template<class ProblemShape>
  auto initialize_varlen(
      const Options& options, const ProblemShape& problem_size,
      int max_seqlen_q, int max_seqlen_kv,
      int total_seqlen_q, int total_seqlen_kv ) {

    int num_batches = get<3,1>(problem_size);

    ProblemShape problem_size_for_init = problem_size;
    get<3,1>(problem_size_for_init) = 1;
    get<0>(problem_size_for_init) = total_seqlen_q;
    get<1>(problem_size_for_init) = total_seqlen_kv;

    ProblemShapeType problem_size_for_launch;

    get<0>(problem_size_for_launch) = VariableLength{max_seqlen_q};
    get<1>(problem_size_for_launch) = VariableLength{max_seqlen_kv};
    get<2>(problem_size_for_launch) = get<2>(problem_size);
    get<3>(problem_size_for_launch) = get<3>(problem_size);

    return cute::make_tuple(problem_size_for_init, problem_size_for_launch);
  }

  ProblemShapeType initialize(const Options& options, int max_seqlen_q, int max_seqlen_kv,
                              int total_seqlen_q, int total_seqlen_kv,
                              void * cumulative_length_q,
                              void * cumulative_length_kv) {
    int h_r = options.h / options.h_k;
    assert(options.h % options.h_k == 0);
    auto problem_shape_in = cute::make_tuple(options.q, options.k, cute::make_tuple(options.dl, options.dr), cute::make_tuple(cute::make_tuple(h_r, options.h_k), options.b));
    
    ProblemShapeType problem_shape;
    decltype(problem_shape_in) problem_size;

    if constexpr (kIsVarlen) {
      auto [problem_shape_init, problem_shape_launch] = initialize_varlen(options, problem_shape_in, max_seqlen_q, 
                                                                          max_seqlen_kv, total_seqlen_q, total_seqlen_kv);
      problem_shape = problem_shape_launch;
      problem_size = problem_shape_init;
    }
    else {
      problem_size = problem_shape_in;
      problem_shape = problem_shape_in;
    }

    int D_latent_rope = size<2, 0>(problem_shape) + size<2, 1>(problem_shape);
    auto shape_Q = replace<1>(select<0,2,3>(problem_shape), D_latent_rope);
    auto shape_K = replace<1>(select<1,2,3>(problem_shape), D_latent_rope);

    auto shape_O = replace<1>(select<0,2,3>(problem_shape), get<2, 0>(problem_shape));
    auto shape_V = replace<1>(select<1,2,3>(problem_shape), get<2, 0>(problem_shape));

    auto shape_LSE = select<0,3>(problem_size);

    int SQ = size<0>(problem_size);
    int SK = size<1>(problem_size);
    int D = size<2, 0>(problem_size);
    int H  = size<3,0>(problem_size);
    int H_K = size<3,0,1>(problem_size);
    int H_Q = size<3,0,0>(problem_size);
    int B = size<3,1>(problem_size);

    stride_Q = make_stride(H*D_latent_rope , _1{}, make_stride(make_stride(D_latent_rope, H_Q*D_latent_rope), H*D_latent_rope*SQ));
    stride_O = make_stride(H*D , _1{}, make_stride(make_stride(D, H_Q*D), H*D*SQ));
    stride_K = make_stride(H_K*D_latent_rope , _1{}, make_stride(make_stride(_0{}, D_latent_rope), H_K*D_latent_rope*SK));
    stride_V = make_stride(H_K*D , _1{}, make_stride(make_stride(_0{}, D), H_K*D*SK));
    stride_LSE = make_stride(_1{}, make_stride(make_stride(SQ, SQ*H_Q), SQ*H));

    if (kIsVarlen) {
      get<2,1>(stride_Q) = 0;
      get<2,1>(stride_K) = 0;
      get<2,1>(stride_V) = 0;
      get<2,1>(stride_O) = 0;
      get<1,1>(stride_LSE) = 0;
    }

    if constexpr (kIsVarlen) {
      get<0>(problem_shape).cumulative_length = static_cast<int *>(cumulative_length_q);
      get<1>(problem_shape).cumulative_length = static_cast<int *>(cumulative_length_kv);
    }

    return problem_shape;
  }


  auto get_arguments(const ProblemShapeType& problem_shape, const cutlass::KernelHardwareInfo& hw_info,
                      float scale_softmax,
                      void * q_ptr, void * k_ptr, void * v_ptr, 
                      void * o_ptr, void * lse_ptr,
                      void * cumulative_length_q, void * cumulative_length_kv
                      ) {
    auto problem_shape_ = problem_shape;
    if constexpr (kIsVarlen) {
      get<0>(problem_shape_).cumulative_length = static_cast<int *>(cumulative_length_q);
      get<1>(problem_shape_).cumulative_length = static_cast<int *>(cumulative_length_kv);
    }

    typename Operation::Arguments arguments{
      problem_shape_,
      { static_cast<Element*>(q_ptr), stride_Q, 
        static_cast<Element*>(k_ptr), stride_K, 
        static_cast<Element*>(v_ptr), stride_V, scale_softmax},
      { static_cast<ElementOut*>(o_ptr), stride_O,
        static_cast<ElementAccumulatorPV*>(lse_ptr), stride_LSE },
      hw_info
    };

    return arguments;
  }


  void run(const Options& options, const cutlass::KernelHardwareInfo& hw_info, 
                  at::Tensor q, at::Tensor k, at::Tensor v,
                  at::Tensor o, at::Tensor lse, 
                  float scale_softmax, at::Tensor workspace,
                  at::Tensor cumulative_seqlen_q, at::Tensor cumulative_seqlen_kv,
                  int max_seqlen_q, int max_seqlen_kv) {
    
    int total_seqlen_q = q.size(0);
    int total_seqlen_kv = k.size(0);

    ProblemShapeType problem_shape = initialize(options, max_seqlen_q, max_seqlen_kv, 
                                                total_seqlen_q, total_seqlen_kv, 
                                                cumulative_seqlen_q.data_ptr(), 
                                                cumulative_seqlen_kv.data_ptr());

    typename Operation::Arguments arguments = get_arguments(problem_shape, hw_info, scale_softmax,
                                                            q.data_ptr(), k.data_ptr(), v.data_ptr(),
                                                            o.data_ptr(), lse.data_ptr(), 
                                                            cumulative_seqlen_q.data_ptr(),
                                                            cumulative_seqlen_kv.data_ptr());

    Operation op;

    // size_t workspace_size = 0;
    // workspace_size = Operation::get_workspace_size(arguments);

    // todo: if use workspace, need check workspace size first.
    // we don't use workspace in current version.

    cutlass::Status status = cutlass::Status::kSuccess;
    status = op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "This kernel is not supported. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return;
    }

    // status = op.initialize(arguments, workspace.data_ptr());
    status = op.initialize(arguments, nullptr);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to initialize the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return;
    }

    status = op.run(at::cuda::getCurrentCUDAStream());
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to launch the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
    }            
  }
};

template <typename DTypeIn, typename DTypeOut, bool kIsVarlen, class ActiveMask, class... KernelOptions>
void run_mla_fwd(at::Tensor workspace, at::Tensor q, at::Tensor k, at::Tensor v,
                  at::Tensor cumulative_seqlen_q, at::Tensor cumulative_seqlen_kv,
                  at::Tensor o, at::Tensor lse,
                  float scale_softmax, int max_seqlen_q, int max_seqlen_kv) {

  Options options;
  options.b = cumulative_seqlen_q.size(0) - 1;
  options.h =  q.size(1);
  options.h_k = k.size(1);
  options.q = q.size(0) / options.b; 
  options.k = k.size(0) / options.b; 
  options.dl = v.size(-1);
  options.dr = q.size(-1) - v.size(-1);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  using HeadDimLatent = _128;
  using HeadDim = Shape<HeadDimLatent, _64>;
  using TileShape = Shape<_256, _128, HeadDim>;

  if(options.h % cutlass::fmha::kernel::CausalIndividualTileScheduler::TileH == 0 && (!std::is_same_v<ActiveMask, NoMask>)) {
    MlaFwdRunner<true, kIsVarlen, DTypeIn, DTypeOut, TileShape, ActiveMask, KernelOptions...> runner;
    runner.run(options, hw_info, q, k, v, o, lse, scale_softmax, workspace,
            cumulative_seqlen_q, cumulative_seqlen_kv,
            max_seqlen_q, max_seqlen_kv);
  } else {
    MlaFwdRunner<false, kIsVarlen, DTypeIn, DTypeOut, TileShape, ActiveMask, KernelOptions...> runner;
    runner.run(options, hw_info, q, k, v, o, lse, scale_softmax, workspace,
            cumulative_seqlen_q, cumulative_seqlen_kv,
            max_seqlen_q, max_seqlen_kv);
  }
}
