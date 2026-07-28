// Standalone pybind entry for the block-sparse KV-outer MLA backward.
// Builds the k2q CSR from per-Q-block selection, then runs Sm100BlockSparseBwd
// (MLA 192/128, TileShape<64,128,192,128>, CausalForBackwardMask, B arbitrary).
// Mirrors BwdRunner's stride setup (fmha_cutlass_bwd_sm100.cuh).
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/kernel_hardware_info.h>

#include "../../../../api/block_sparse_bwd.h"
#include "../../dense/common/utils.hpp"
#include "../../dense/collective/fmha_fusion.hpp"

// Bring is_variable_length_v etc. into global scope BEFORE the device wrapper /
// sum_OdO / convert kernels are included (they use it unqualified) -- mirrors the
// using-directive order in fmha_cutlass_bwd_sm100.cuh.
using namespace cute;
using namespace cutlass::fmha::collective;

#include "block_sparse_bwd_host.cuh"
#include "k2q_csr.cuh"

// q,k,v,o,d_o: [S, H, D]; lse: [S, H] with stride(0)==1 (i.e. pass lse_hq.transpose(0,1));
// q2k_blocks: int32 [num_q_blocks, topk] (per-Q-block selected KV-block ids, -1 pad).
// dq,dk,dv: [S, H, D] outputs. B=1 (single sequence).
void block_sparse_prefill_bwd(at::Tensor workspace, at::Tensor d_o, at::Tensor q, at::Tensor k,
                      at::Tensor v, at::Tensor o, at::Tensor lse, at::Tensor q2k_blocks,
                      at::Tensor dq, at::Tensor dk, at::Tensor dv,
                      double softmax_scale, int64_t window_size,
                      int64_t kv_block_size, int64_t q_block_size,
                      std::optional<at::Tensor> attn_sink,
                      std::optional<at::Tensor> d_sink) {
  const at::cuda::CUDAGuard device_guard{(char)q.get_device()};
  using Element = cutlass::bfloat16_t;
  using ElementAccumulator = float;
  using TileShape = Shape<_64, _128, _192, _128>;
  using Mask = CausalForBackwardMask<false>;
  using ProblemShape = cute::tuple<int, int, int, int, cute::tuple<int, int>>;
  using Operation = cutlass::fmha::device::Sm100BlockSparseBwd<
      ProblemShape, Element, ElementAccumulator, TileShape, Mask>;
  using TensorStride = Stride<int, _1, Stride<int, int>>;

  int D = q.size(-1), D_VO = v.size(-1);
  int H = q.size(1);
  int B = 1;
  int Q = q.size(0), K = k.size(0);
  TORCH_CHECK(kv_block_size == 128 && q_block_size == 64, "block sizes must match TileShapeK/Q");

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = q.get_device();
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  // gpt-oss attention sink: per-head [h] fp32 logit (sink_ptr) + its gradient output (d_sink_ptr,
  // [h], pre-zeroed). d_sink folded into the reused dense sum_OdO pass. nullptr disables.
  const float* sink_ptr = attn_sink.has_value() ? attn_sink->data_ptr<float>() : nullptr;
  float* d_sink_ptr = d_sink.has_value() ? d_sink->data_ptr<float>() : nullptr;

  ProblemShape problem_shape = cute::make_tuple(Q, K, D, D_VO, cute::make_tuple(H, B));

  auto S = [](at::Tensor t, int s0, int s1, int len) {
    return make_stride(s0, _1{}, make_stride(s1, len));
  };
  TORCH_CHECK(q.stride(2) == 1 && k.stride(2) == 1 && v.stride(2) == 1 && o.stride(2) == 1);
  TORCH_CHECK(lse.stride(0) == 1, "lse must have stride(0)==1 (pass lse_hq.transpose(0,1))");
  TensorStride stride_Q  = make_stride((int)q.stride(0),  _1{}, make_stride((int)q.stride(1),  0));
  TensorStride stride_K  = make_stride((int)k.stride(0),  _1{}, make_stride((int)k.stride(1),  0));
  TensorStride stride_V  = make_stride((int)v.stride(0),  _1{}, make_stride((int)v.stride(1),  0));
  TensorStride stride_O  = make_stride((int)o.stride(0),  _1{}, make_stride((int)o.stride(1),  0));
  Stride<_1, Stride<int, int>> stride_LSE = make_stride(_1{}, make_stride((int)lse.stride(1), 0));
  TensorStride stride_dO = make_stride((int)d_o.stride(0),_1{}, make_stride((int)d_o.stride(1),0));
  TensorStride stride_dQ = make_stride((int)dq.stride(0), _1{}, make_stride((int)dq.stride(1), 0));
  TensorStride stride_dK = make_stride((int)dk.stride(0), _1{}, make_stride((int)dk.stride(1), 0));
  TensorStride stride_dV = make_stride((int)dv.stride(0), _1{}, make_stride((int)dv.stride(1), 0));

  // ---- build k2q CSR from q2k_blocks ----
  int num_q_blocks = q2k_blocks.size(0);
  int topk = q2k_blocks.size(1);
  int num_kv_blocks = (K + (int)kv_block_size - 1) / (int)kv_block_size;
  auto iopt = q2k_blocks.options();
  auto counts   = at::zeros({num_kv_blocks}, iopt);
  auto cursor   = at::zeros({num_kv_blocks}, iopt);
  auto row_ptr  = at::zeros({num_kv_blocks + 1}, iopt);
  auto q_idx    = at::full({(int64_t)num_q_blocks * topk}, -1, iopt);
  auto stream = at::cuda::getCurrentCUDAStream();
  sm100::block_bwd::build_k2q_csr(
      q2k_blocks.contiguous().data_ptr<int>(), num_q_blocks, topk, num_kv_blocks,
      counts.data_ptr<int>(), cursor.data_ptr<int>(),
      row_ptr.data_ptr<int>(), q_idx.data_ptr<int>(), stream);
  // build_k2q_csr's atomic-cursor scatter leaves each KV-block's attending-Q-block
  // order racy -> the dK/dV TMEM accumulate-store order varies -> bitwise nondeterminism
  // for some selections. Sort each CSR row to a stable ascending order -> deterministic.
  sm100::block_bwd::sort_k2q_csr(
      row_ptr.data_ptr<int>(), q_idx.data_ptr<int>(), num_kv_blocks, stream);

  typename Operation::Arguments args{
    problem_shape,
    static_cast<Element*>(q.data_ptr()), stride_Q,
    static_cast<Element*>(k.data_ptr()), stride_K,
    static_cast<Element*>(v.data_ptr()), stride_V,
    static_cast<Element*>(o.data_ptr()), stride_O,
    static_cast<ElementAccumulator*>(lse.data_ptr()), stride_LSE,
    static_cast<Element*>(d_o.data_ptr()), stride_dO,
    static_cast<Element*>(dq.data_ptr()), stride_dQ,
    static_cast<Element*>(dk.data_ptr()), stride_dK,
    static_cast<Element*>(dv.data_ptr()), stride_dV,
    static_cast<ElementAccumulator>(softmax_scale),
    (int)window_size,
    row_ptr.data_ptr<int>(), q_idx.data_ptr<int>(), num_kv_blocks,
    sink_ptr, d_sink_ptr,   // gpt-oss attention sink (d_sink folded into sum_OdO; nullptr disables)
    hw_info
  };

  Operation op;
  CUTLASS_CHECK(op.can_implement(args));
  CUTLASS_CHECK(op.run(args, workspace.data_ptr(), stream));
}

int64_t block_sparse_bwd_workspace_size(int64_t Q, int64_t H, int64_t B, int64_t D) {
  int64_t Qr = (Q + 7) / 8 * 8;
  int64_t Dr = (D + 7) / 8 * 8;
  return (int64_t)sizeof(float) * B * H * Qr * 2 + (int64_t)sizeof(float) * B * H * Qr * Dr;
}

#ifdef BLOCK_SPARSE_BWD_STANDALONE
// Standalone JIT module (tests/bench). The setup.py build instead registers
// block_sparse_prefill_bwd in csrc/api/api.cpp (so no second pybind module).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("block_sparse_bwd", &block_sparse_prefill_bwd, "block-sparse KV-outer MLA bwd");
  m.def("workspace_size", &block_sparse_bwd_workspace_size, "bwd workspace bytes");
}
#endif
