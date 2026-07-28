// Device wrapper for the block-sparse KV-outer MLA backward. Fork of the dense
// Sm100FmhaBwd (device/fmha_device_bwd.hpp): MLA-only, and threads the k2q CSR
// (k2q_row_ptr / k2q_q_indices / num_kv_blocks) into the bwd kernel's
// MainloopArguments. Reuses the dense sum_OdO + convert kernels unchanged.
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cute/tensor.hpp"

#include "../../dense/device/fmha.hpp"
// Include the (forked dense MLA) kernel first: it pulls cutlass/arch/arch.h etc.
// that the sum_OdO/convert kernels rely on transitively (e.g. cutlass::arch::Sm90).
#include "block_sparse_bwd_kernel.hpp"
#include "../../dense/kernel/fmha_kernel_bwd_sum_OdO.hpp"
#include "../../dense/kernel/fmha_kernel_bwd_convert.hpp"

namespace cutlass::fmha::device {

template<class ProblemShape, class Element, class ElementAccumulator, class TileShape, class Mask>
class Sm100BlockSparseBwd {
public:
  struct Arguments {
    ProblemShape problem_shape;                                    // Q K D D_VO HB
    const Element* ptr_Q;  cute::tuple<int, cute::_1, cute::tuple<int, int>> stride_Q;
    const Element* ptr_K;  cute::tuple<int, cute::_1, cute::tuple<int, int>> stride_K;
    const Element* ptr_V;  cute::tuple<int, cute::_1, cute::tuple<int, int>> stride_V;
    const Element* ptr_O;  cute::tuple<int, cute::_1, cute::tuple<int, int>> stride_O;
    const ElementAccumulator* ptr_LSE;  cute::tuple<cute::_1, cute::tuple<int, int>> stride_LSE;
    const Element* ptr_dO; cute::tuple<int, cute::_1, cute::tuple<int, int>> stride_dO;
    Element* ptr_dQ; cute::tuple<int, cute::_1, cute::tuple<int, int>> stride_dQ;
    Element* ptr_dK; cute::tuple<int, cute::_1, cute::tuple<int, int>> stride_dK;
    Element* ptr_dV; cute::tuple<int, cute::_1, cute::tuple<int, int>> stride_dV;
    ElementAccumulator softmax_scale;
    int window_size = -1;
    // Block-sparse k2q CSR (built by build_k2q_csr). nullptr => dense.
    const int* k2q_row_ptr = nullptr;
    const int* k2q_q_indices = nullptr;
    int num_kv_blocks = 0;
    // gpt-oss attention sink: per-head [h_q] logit + its gradient output (zeroed, atomic-accumulated),
    // folded into the (reused dense) sum_OdO pass: d_sink[h] = -sum_q exp(sink[h]-lse[q,h])*(O.dO)_q.
    // dQ/dK/dV need no change: the fwd writes a sink-aware LSE, so the bwd's softmax recompute is
    // already sink-correct (the value-less sink leaves delta = O.dO unchanged). nullptr disables.
    const ElementAccumulator* sink_bias = nullptr;
    ElementAccumulator* ptr_d_sink = nullptr;
    cutlass::KernelHardwareInfo hw_info;
  };

  using OperationSumOdO = cutlass::fmha::device::FMHA<
    cutlass::fmha::kernel::FmhaKernelBwdSumOdO<ProblemShape, Element, ElementAccumulator>>;
  using OperationConvert = cutlass::fmha::device::FMHA<
    cutlass::fmha::kernel::FmhaKernelBwdConvert<ProblemShape, Element, ElementAccumulator>>;
  using Operation = cutlass::fmha::device::FMHA<
    cutlass::fmha::kernel::Sm100BlockSparseBwdMlaKernelTmaWarpSpecialized<
      ProblemShape, Element, ElementAccumulator, TileShape, Mask>>;
  using Kernel = typename Operation::Kernel;

  struct Params {
    OperationSumOdO op_sum_OdO;
    Operation op;
    OperationConvert op_convert;
    ElementAccumulator* dQ_acc;
    size_t dQ_acc_size;
  };

private:
  Params params_;

  static typename OperationSumOdO::Arguments to_sum_OdO_arguments(
        Arguments const& args, ElementAccumulator* sum_odo = nullptr, ElementAccumulator* scaled_lse = nullptr) {
    using namespace cute;
    auto [Q_, K, D, D_VO, HB] = args.problem_shape;
    auto [H, B] = HB;
    D = cutlass::round_up(D, 8);
    int Q = cutlass::round_up(static_cast<int>(Q_), 8);
    auto stride_sum_OdO = make_stride(_1{}, make_stride(Q, Q*H));
    auto stride_scaled_lse = make_stride(_1{}, make_stride(Q, Q*H));
    auto log2_e = log2f(expf(1.0f));
    return typename OperationSumOdO::Arguments {
      args.problem_shape, args.ptr_O, args.stride_O, args.ptr_dO, args.stride_dO,
      sum_odo, stride_sum_OdO, args.ptr_LSE, args.stride_LSE,
      scaled_lse, stride_scaled_lse, -1.0f, -log2_e,
      args.sink_bias, args.ptr_d_sink };  // gpt-oss attention sink: d_sink folded into this pass
  }

  static typename OperationConvert::Arguments to_convert_arguments(Arguments const& args, ElementAccumulator* src = nullptr) {
    using namespace cute;
    auto [Q_, K, D, D_VO, HB] = args.problem_shape;
    auto [H, B] = HB;
    D = cutlass::round_up(D, 8);
    int Q = cutlass::round_up(static_cast<int>(Q_), 8);
    auto stride_src_dQ = make_stride(D, _1{}, make_stride(D*Q, D*Q*H));
    return typename OperationConvert::Arguments {
      args.problem_shape, src, stride_src_dQ, nullptr, stride_src_dQ, nullptr, stride_src_dQ,
      args.ptr_dQ, args.stride_dQ, nullptr, args.stride_dK, nullptr, args.stride_dV, args.softmax_scale };
  }

  static typename Operation::Arguments to_bwd_arguments(
      Arguments const& args,
      ElementAccumulator* sum_OdO = nullptr, cute::tuple<cute::_1, cute::tuple<int, int>> const& stride_sum_OdO = {},
      ElementAccumulator* scaled_lse = nullptr, cute::tuple<cute::_1, cute::tuple<int, int>> const& stride_scaled_lse = {},
      ElementAccumulator* dQ_acc = nullptr, cute::tuple<int, cute::_1, cute::tuple<int, int>> const& stride_dQ = {}) {
    return typename Operation::Arguments{
      args.problem_shape,
      { args.ptr_Q, args.stride_Q, args.ptr_K, args.stride_K, args.ptr_V, args.stride_V,
        args.ptr_dO, args.stride_dO, scaled_lse, stride_scaled_lse, sum_OdO, stride_sum_OdO,
        dQ_acc, stride_dQ, args.softmax_scale, args.window_size,
        args.k2q_row_ptr, args.k2q_q_indices, args.num_kv_blocks },
      { args.ptr_dK, args.stride_dK, args.ptr_dV, args.stride_dV },
      args.hw_info };
  }

public:
  static Status can_implement(Arguments const& args) {
    Status status = OperationSumOdO::can_implement(to_sum_OdO_arguments(args));
    if (status != Status::kSuccess) return status;
    status = OperationConvert::can_implement(to_convert_arguments(args));
    if (status != Status::kSuccess) return status;
    return Operation::can_implement(to_bwd_arguments(args));
  }

  static size_t get_workspace_size(Arguments const& args) {
    auto [Q_, K, D, D_VO, HB] = args.problem_shape;
    auto [H, B] = HB;
    D = cutlass::round_up(D, 8);
    int Q = cutlass::round_up(static_cast<int>(Q_), 8);
    size_t bytes = 0;
    bytes += sizeof(ElementAccumulator) * B*H*Q;          // sum_OdO
    bytes += sizeof(ElementAccumulator) * B*H*Q;          // scaled LSE
    bytes += sizeof(ElementAccumulator) * B*H*Q*D;        // dQ FP32 acc
    return bytes;
  }

  Status initialize(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
    auto [Q_, K, D, D_VO, HB] = args.problem_shape;
    auto [H, B] = HB;
    D = cutlass::round_up(D, 8);
    int Q = cutlass::round_up(static_cast<int>(Q_), 8);
    char* wc = reinterpret_cast<char*>(workspace);
    ElementAccumulator* sum_OdO = reinterpret_cast<ElementAccumulator*>(wc);
    wc += sizeof(ElementAccumulator) * B*H*Q;
    ElementAccumulator* scaled_lse = reinterpret_cast<ElementAccumulator*>(wc);
    wc += sizeof(ElementAccumulator) * B*H*Q;
    ElementAccumulator* dQ_acc = reinterpret_cast<ElementAccumulator*>(wc);
    params_.dQ_acc = dQ_acc;
    params_.dQ_acc_size = sizeof(ElementAccumulator) * B*H*Q*D;
    auto args_sum_OdO = to_sum_OdO_arguments(args, sum_OdO, scaled_lse);
    auto args_convert = to_convert_arguments(args, dQ_acc);
    params_.op_sum_OdO.initialize(args_sum_OdO, nullptr, stream);
    params_.op_convert.initialize(args_convert, nullptr, stream);
    auto args_bwd = to_bwd_arguments(args, sum_OdO, args_sum_OdO.stride_sum_OdO,
        scaled_lse, args_sum_OdO.stride_scaled_lse, dQ_acc, args_convert.stride_src_dQ);
    params_.op.initialize(args_bwd, nullptr, stream);
    return Status::kSuccess;
  }

  static Status run(Params& params, cudaStream_t stream = nullptr) {
    Status r = params.op_sum_OdO.run(stream);
    if (r != Status::kSuccess) return r;
    if (cudaMemsetAsync(params.dQ_acc, 0, params.dQ_acc_size, stream) != cudaSuccess) return Status::kErrorInternal;
    r = params.op.run(stream);
    if (r != Status::kSuccess) return r;
    return params.op_convert.run(stream);
  }

  Status run(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
    Status status = initialize(args, workspace, stream);
    if (Status::kSuccess == status) status = run(params_, stream);
    return status;
  }
};

} // namespace cutlass::fmha::device
