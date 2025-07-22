#pragma once

#include <cstdint>

#include "cutlass/kernel_hardware_info.h"

#include "collective/fmha_fusion.hpp"
#include "collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp"
#include "collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"
#include "device/fmha.hpp"
#include "kernel/fmha_tile_scheduler.hpp"
#include "kernel/fmha_causal_tile_scheduler.hpp"
#include "kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp"

using namespace cute;
using namespace cutlass::fmha::collective;
using namespace cutlass::fmha::kernel;
using namespace cutlass::fmha::device;


template <typename DTypeIn, 
          typename DTypeOut, 
          bool kIsVarlen,
          bool kIsMaskTileSchedulerValid,
          class TileShape, 
          class ActiveMask,
          class... KernelOptions>
struct FwdRunner {
    using Element = DTypeIn;
    using ElementAccumulatorQK = float;
    using ElementAccumulatorPV = float;
    using ElementOut = DTypeOut;

    // Q K D ((H_R, H_KV), B)
    using ProblemShapeRegular = cute::tuple<int, int, int, cute::tuple<cute::tuple<int, int>, int>>;
    using ProblemShapeVarlen = cute::tuple<VariableLength, VariableLength, int, cute::tuple<cute::tuple<int, int>, int>>;
    using ProblemShapeType = std::conditional_t<kIsVarlen, ProblemShapeVarlen, ProblemShapeRegular>;

    using StrideQ = cute::tuple<int, _1, cute::tuple<cute::tuple<int, int>, int>>;  // Q D ((H_G H_R), B)
    using StrideK = cute::tuple<int, _1, cute::tuple<cute::tuple<_0, int>, int>>;  // K D ((H_G H_R), B)
    using StrideV = StrideK;
    // NOTE(Zihao): use markus's trick for tma store
    using StrideO = StrideQ;
    using StrideLSE = cute::tuple<_1, cute::tuple<cute::tuple<int, int>, int>>;  // Q ((H_G H_R), B)

    static constexpr bool kIsPersistent = find_option_t<Tag::kIsPersistent, true_type, KernelOptions...>::value;

  using TileScheduler = std::conditional_t<kIsPersistent, 
                                          std::conditional_t<std::is_same_v<ActiveMask, CausalMask<false>> 
                                                                          || std::is_same_v<ActiveMask, CausalMask<true>>, 
                                                            cutlass::fmha::kernel::CausalPersistentTileScheduler,
                                                            cutlass::fmha::kernel::PersistentTileScheduler>,
                                          std::conditional_t<kIsMaskTileSchedulerValid, 
                                                            cutlass::fmha::kernel::CausalIndividualTileScheduler,
                                                            cutlass::fmha::kernel::IndividualTileScheduler>>;

    using Mainloop = 
        cutlass::fmha::collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
        Element, ElementAccumulatorQK, ElementAccumulatorPV,
        TileShape, StrideQ, StrideK, StrideV,
        ActiveMask
        >;
    using Operation = cutlass::fmha::device::FMHA<
        cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<
        ProblemShapeType,
        Mainloop,
        cutlass::fmha::collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<
            ElementOut, ElementAccumulatorPV,
            typename Mainloop::TileShapePV,
            StrideO, StrideLSE
        >,
        TileScheduler
        >>;

    static void run(at::Tensor workspace_buffer, at::Tensor q, at::Tensor k, at::Tensor v,
                  at::Tensor cumulative_seqlen_q, at::Tensor cumulative_seqlen_kv,
                  at::Tensor o, at::Tensor lse,
                  float softmax_scale, int max_seqlen_q, int max_seqlen_kv) {
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    ProblemShapeRegular problem_size;
    ProblemShapeType problem_shape;
    //varlen: q: [Q, H, D]
    //fixedlen: q: [B, H, Q, D] 
    if constexpr (kIsVarlen) {
        int d = q.size(-1);
        int d_vo = v.size(-1);
        int batch_size = cumulative_seqlen_q.size(0) - 1;
        int num_qo_heads = q.size(1);
        int num_kv_heads = k.size(1);
        int h_r = num_qo_heads / num_kv_heads;
        int total_seqlen_q = q.size(0);
        int total_seqlen_kv = k.size(0);

        problem_size = make_shape(total_seqlen_q, total_seqlen_kv, d, make_shape(make_shape(h_r,num_kv_heads), 1));
        problem_shape = make_shape(
            VariableLength{max_seqlen_q, static_cast<int*>(cumulative_seqlen_q.data_ptr()), total_seqlen_q},
            VariableLength{max_seqlen_kv, static_cast<int*>(cumulative_seqlen_kv.data_ptr()), total_seqlen_kv}, 
            d, make_shape(make_shape(h_r, num_kv_heads), batch_size));
    } else {
        int q_len = q.size(1);
        int kv_len = k.size(1);
        int d = q.size(-1); 
        int batch_size = q.size(0);
        int num_qo_heads = q.size(2);
        int num_kv_heads = k.size(2);
        int h_r = num_qo_heads / num_kv_heads;

        problem_size = make_shape(q_len, kv_len, d, make_shape(make_shape(h_r, num_kv_heads), batch_size));
        problem_shape = problem_size;
    }

    get<2>(problem_size) = cutlass::round_up(get<2>(problem_size), 8);  // alignment

    int SQ = size<0>(problem_size);
    int SK = size<1>(problem_size);
    int D = size<2>(problem_size);
    int H  = size<3,0>(problem_size);
    int H_K = size<3,0,1>(problem_size);
    int H_Q = size<3,0,0>(problem_size);
    int B = size<3,1>(problem_size);

    StrideQ stride_Q = make_stride(H*D , _1{}, make_stride(make_stride(D, H_Q*D), H*D*SQ));
    StrideO stride_O = stride_Q;
    StrideK stride_K = make_stride(H_K*D , _1{}, make_stride(make_stride(_0{}, D), H_K*D*SK));
    StrideV stride_V = stride_K;
    StrideLSE stride_LSE = make_stride(_1{}, make_stride(make_stride(SQ, SQ*H_Q), SQ*H));

    if (kIsVarlen) {
        get<2,1>(stride_Q) = 0;
        get<2,1>(stride_K) = 0;
        get<2,1>(stride_V) = 0;
        get<2,1>(stride_O) = 0;
        get<1,1>(stride_LSE) = 0;
    }
    typename Operation::Arguments arguments{
        problem_shape,
        {static_cast<Element*>(q.data_ptr()), stride_Q, 
         static_cast<Element*>(k.data_ptr()), stride_K, 
         static_cast<Element*>(v.data_ptr()), stride_V, 
         softmax_scale},
        {static_cast<ElementOut*>(o.data_ptr()), stride_O,
         static_cast<ElementAccumulatorPV*>(lse.data_ptr()), stride_LSE},
        hw_info};
    Operation op;

    CUTLASS_CHECK(op.can_implement(arguments));
    // CUTLASS_CHECK(op.initialize(arguments, workspace.get()));
    CUTLASS_CHECK(op.initialize(arguments, nullptr));
    CUTLASS_CHECK(op.run(at::cuda::getCurrentCUDAStream()));
  }
};

template <typename DTypeIn, typename DTypeOut, bool kIsVarlen, class TileShape, 
          class ActiveMask, class... KernelOptions>
void run_fmha_fwd(at::Tensor workspace_buffer, at::Tensor q, at::Tensor k, at::Tensor v,
                  at::Tensor cumulative_seqlen_q, at::Tensor cumulative_seqlen_kv,
                  at::Tensor o, at::Tensor lse,
                  float softmax_scale, int max_seqlen_q, int max_seqlen_kv) {

    if(q.size(1) % cutlass::fmha::kernel::CausalIndividualTileScheduler::TileH == 0 && (!std::is_same_v<ActiveMask, NoMask>)) {
        FwdRunner<DTypeIn, DTypeOut, kIsVarlen, true, TileShape, ActiveMask, KernelOptions...>::run(
        workspace_buffer, q, k, v, cumulative_seqlen_q, cumulative_seqlen_kv, o, lse,
        softmax_scale, max_seqlen_q, max_seqlen_kv);
    } else {
        FwdRunner<DTypeIn, DTypeOut, kIsVarlen, false, TileShape, ActiveMask, KernelOptions...>::run(
        workspace_buffer, q, k, v, cumulative_seqlen_q, cumulative_seqlen_kv, o, lse,
        softmax_scale, max_seqlen_q, max_seqlen_kv);
    }
}
