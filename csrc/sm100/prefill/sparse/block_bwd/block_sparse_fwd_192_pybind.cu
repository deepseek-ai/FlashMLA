// Standalone host + pybind for the 192/128 block-sparse MLA forward (training path).
// Forks the dense MLA fwd runner (csrc/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cuh)
// but (1) swaps the mainloop to the block-sparse fork (per-q-block selected-K-block
// iteration via q2k) and (2) threads q2k into the load Arguments. Non-varlen, MLA
// 192/128 (D_qk = dl128 + dr64, D_v = dl128), CausalMask. Emits O + LSE in the dense
// fwd convention so it feeds the matched 192/128 block-sparse KV-outer bwd.
//
// q2k: [num_q_blocks, topk] int32, -1 padded, SHARED across heads (one selection per
// q-block of TileShape Q = 256 rows). kv_block_size = 128 (= TileShapeQK K).

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/kernel_hardware_info.h>

#include "../../dense/collective/fmha_fusion.hpp"
#include "../../dense/collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp"
#include "../../dense/kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp"
#include "../../dense/kernel/fmha_causal_tile_scheduler.hpp"
#include "../../dense/kernel/fmha_tile_scheduler.hpp"
#include "../../dense/device/fmha.hpp"

#include "block_sparse_fwd_192_mainloop.hpp"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "../../../../api/block_sparse_fwd.h"

namespace bsfwd192 {

using namespace cute;
using namespace cutlass::fmha;
using namespace cutlass::fmha::collective;
using namespace cutlass::fmha::kernel;

#define CUTLASS_CHECK(status)                                                              \
  {                                                                                        \
    cutlass::Status error = status;                                                        \
    TORCH_CHECK(error == cutlass::Status::kSuccess,                                        \
                "cutlass error: ", cutlassGetStatusString(error));                         \
  }

template <bool kIsMaskTileSchedulerValid>
struct BlockSparseFwd192Runner {
  using Element = cutlass::bfloat16_t;
  using ElementOut = cutlass::bfloat16_t;
  using ElementAcc = float;
  using ActiveMask = CausalMask</*kIsQBegin=*/false>;

  using HeadDimLatent = _128;
  using HeadDim = Shape<HeadDimLatent, _64>;          // (dl=128, dr=64) -> D_qk=192
  using TileShapeMla = Shape<_256, _128, HeadDim>;

  // non-varlen MLA problem shape: (q_len, k_len, (dl,dr), ((h_r,h_k), b))
  using ProblemShapeType =
      cute::tuple<int, int, cute::tuple<int, int>, cute::tuple<cute::tuple<int, int>, int>>;

  using StrideQ = cute::tuple<int, _1, cute::tuple<cute::tuple<int, int>, int>>;
  using StrideK = cute::tuple<int, _1, cute::tuple<cute::tuple<_0, int>, int>>;
  using StrideV = StrideK;
  using StrideO = StrideQ;
  using StrideLSE = cute::tuple<_1, cute::tuple<cute::tuple<int, int>, int>>;

  using Mainloop = Sm100BlockSparseFwdMainloopTmaWarpspecialized<
      Element, ElementAcc, ElementAcc, TileShapeMla, StrideQ, StrideK, StrideV,
      ActiveMask, Shape<_2, _1, _1>, cute::false_type>;

  using TileScheduler = std::conditional_t<kIsMaskTileSchedulerValid,
                                           CausalIndividualTileScheduler,
                                           IndividualTileScheduler>;

  using Operation = device::FMHA<Sm100FmhaFwdKernelTmaWarpspecialized<
      ProblemShapeType, Mainloop,
      Sm100FmhaFwdEpilogueTmaWarpspecialized<ElementOut, ElementAcc,
                                             typename Mainloop::TileShapePV, StrideO,
                                             StrideLSE, cute::false_type>,
      TileScheduler, Sm100MlaFwdCtxKernelWarpspecializedSchedule>>;

  void run(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor o, at::Tensor lse,
           const int* q2k_ptr, int topk, float scale_softmax, const float* sink_ptr = nullptr) {
    const at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = q.get_device();
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    int b = 1;
    int SQ = q.size(0);              // seqlen_q (b=1)
    int SK = k.size(0);              // seqlen_k
    int H = q.size(1);               // num query heads
    int H_K = k.size(1);             // num kv heads
    int H_Q = H / H_K;               // q heads per kv head
    int dl = v.size(-1);             // 128
    int dr = q.size(-1) - dl;        // 64

    ProblemShapeType problem_shape = cute::make_tuple(
        SQ, SK, cute::make_tuple(dl, dr),
        cute::make_tuple(cute::make_tuple(H_Q, H_K), b));

    int q_s0 = q.stride(0), q_s1 = q.stride(1);
    int k_s0 = k.stride(0), k_s1 = k.stride(1);
    int v_s0 = v.stride(0), v_s1 = v.stride(1);
    int o_s0 = o.stride(0), o_s1 = o.stride(1);
    int lse_s1 = lse.stride(1);
    TORCH_CHECK(q.stride(2) == 1 && k.stride(2) == 1 && v.stride(2) == 1 && o.stride(2) == 1);
    TORCH_CHECK(lse.stride(0) == 1);

    StrideQ stride_Q = make_stride(q_s0, _1{}, make_stride(make_stride(q_s1, H_Q * q_s1), SQ * q_s0));
    StrideO stride_O = make_stride(o_s0, _1{}, make_stride(make_stride(o_s1, H_Q * o_s1), SQ * o_s0));
    StrideK stride_K = make_stride(k_s0, _1{}, make_stride(make_stride(_0{}, k_s1), SK * k_s0));
    StrideV stride_V = make_stride(v_s0, _1{}, make_stride(make_stride(_0{}, v_s1), SK * v_s0));
    StrideLSE stride_LSE = make_stride(_1{}, make_stride(make_stride(lse_s1, lse_s1 * H_Q), SQ));

    // NB: explicit inner braces around the Load::Arguments fields -- Load::Arguments now
    // has ptr_q2k/topk after ptr_V/dV, so without these braces scale_softmax would land
    // on ptr_q2k (brace elision). ptr_q2k/topk take their defaults and are set below.
    typename Operation::Arguments arguments{
        problem_shape,
        {{static_cast<Element*>(q.data_ptr()), stride_Q,
          static_cast<Element*>(k.data_ptr()), stride_K,
          static_cast<Element*>(v.data_ptr()), stride_V},
         scale_softmax},
        {static_cast<ElementOut*>(o.data_ptr()), stride_O,
         static_cast<ElementAcc*>(lse.data_ptr()), stride_LSE},
        hw_info};

    // block-sparse: thread q2k into the load args (set by member so we don't depend
    // on the positional order of the scale fields).
    arguments.mainloop.load.ptr_q2k = q2k_ptr;
    arguments.mainloop.load.topk = topk;
    arguments.mainloop.window_size = -1;
    arguments.mainloop.sink_bias = sink_ptr;  // gpt-oss attention sink (nullptr disables in-kernel)

    Operation op;
    CUTLASS_CHECK(op.can_implement(arguments));
    CUTLASS_CHECK(op.initialize(arguments, nullptr));
    CUTLASS_CHECK(op.run(at::cuda::getCurrentCUDAStream()));
  }
};

// q: [s_q, h, 192] bf16; k: [s_k, h_k, 192] bf16; v: [s_k, h_k, 128] bf16
// o: [s_q, h, 128] bf16 (out); lse: [s_q, h] fp32 (out, stride(0)==1)
// q2k: [num_q_blocks, topk] int32, -1 padded (num_q_blocks = ceil(s_q/256))
void block_sparse_fwd_192(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor o,
                          at::Tensor lse, at::Tensor q2k, double scale_softmax,
                          std::optional<at::Tensor> attn_sink = std::nullopt) {
  int topk = q2k.size(1);
  int H = q.size(1);
  // gpt-oss per-head attention sink ([h_q], fp32). nullptr disables (in-kernel fold-in).
  const float* sink_ptr = attn_sink.has_value() ? attn_sink->data_ptr<float>() : nullptr;
  // CausalIndividualTileScheduler valid when h is a multiple of its TileH.
  if (H % CausalIndividualTileScheduler::TileH == 0) {
    BlockSparseFwd192Runner</*valid=*/true> runner;
    runner.run(q, k, v, o, lse, static_cast<const int*>(q2k.data_ptr()), topk, (float)scale_softmax, sink_ptr);
  } else {
    BlockSparseFwd192Runner</*valid=*/false> runner;
    runner.run(q, k, v, o, lse, static_cast<const int*>(q2k.data_ptr()), topk, (float)scale_softmax, sink_ptr);
  }
}

}  // namespace bsfwd192

// Free function registered in csrc/api/api.cpp (main module). Forwards to the
// namespaced implementation. Declared in csrc/api/block_sparse_fwd.h.
void block_sparse_prefill_fwd(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor o,
                              at::Tensor lse, at::Tensor q2k_blocks, double softmax_scale,
                              std::optional<at::Tensor> attn_sink) {
  bsfwd192::block_sparse_fwd_192(q, k, v, o, lse, q2k_blocks, softmax_scale, attn_sink);
}

#ifdef BLOCK_SPARSE_FWD192_STANDALONE
// Standalone JIT module (tests/bench). The setup.py build does NOT define
// BLOCK_SPARSE_FWD192_STANDALONE -> only the free function above is compiled,
// registered via api.cpp (so no second pybind module).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("block_sparse_fwd_192", &bsfwd192::block_sparse_fwd_192,
        "192/128 block-sparse MLA forward (training)");
}
#endif
