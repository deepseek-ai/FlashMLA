#include "interface.h"

#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include <cuda_bf16.h>
#include "common/mask.cuh"
#include "common/utils.hpp"

#include "fmha_cutlass_bwd_sm100.cuh"

template<class Mask, class Varlen, class Element, class ElementOut, class Mla>
void call_run_fmha_bwd([[maybe_unused]] Mask mask, [[maybe_unused]] Varlen is_varlen,
                      [[maybe_unused]] Element in, [[maybe_unused]] ElementOut out, [[maybe_unused]] Mla mla,
                  torch::stable::Tensor workspace_buffer, torch::stable::Tensor d_o, torch::stable::Tensor q, torch::stable::Tensor k,
                  torch::stable::Tensor v, torch::stable::Tensor o, torch::stable::Tensor lse,
                  torch::stable::Tensor cumulative_seqlen_q, torch::stable::Tensor cumulative_seqlen_kv,
                  torch::stable::Tensor dq, torch::stable::Tensor dk, torch::stable::Tensor dv,
                  float softmax_scale, int max_seqlen_q, int total_seqlen_kv) {
  static constexpr bool IsVarlen = std::is_same_v<Varlen, true_type>;
  static constexpr bool IsMla = std::is_same_v<Mla, true_type>;
  using TileShape = std::conditional_t<IsMla, Shape<_64, _128, _192, _128>, Shape<_128, _128, _128, _128>>;
  run_fmha_bwd<Element, IsVarlen, IsMla, TileShape, Mask>(workspace_buffer, d_o, q, k, v, o, lse,
                          cumulative_seqlen_q, cumulative_seqlen_kv,
                          dq, dk, dv,
                          softmax_scale, max_seqlen_q, total_seqlen_kv);
}


void FMHACutlassSM100BwdRun(torch::stable::Tensor workspace_buffer, torch::stable::Tensor d_o, torch::stable::Tensor q, torch::stable::Tensor k,
                            torch::stable::Tensor v, torch::stable::Tensor o, torch::stable::Tensor lse,
                            torch::stable::Tensor cumulative_seqlen_q, torch::stable::Tensor cumulative_seqlen_kv,
                            torch::stable::Tensor dq, torch::stable::Tensor dk, torch::stable::Tensor dv,
                            int64_t mask_mode_code, double softmax_scale, int64_t max_seqlen_q, int64_t max_seqlen_kv, bool is_varlen) {

  const torch::stable::accelerator::DeviceGuard device_guard(q.get_device_index());

  int head_dim_qk = q.size(-1);
  int head_dim_vo = v.size(-1);
  MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  auto scalar_type_in = q.scalar_type();
  auto scalar_type_out = o.scalar_type();

  if(scalar_type_in == torch::headeronly::ScalarType::BFloat16 && scalar_type_out == torch::headeronly::ScalarType::BFloat16) {
    using Element = cutlass::bfloat16_t;
    using ElementOut = cutlass::bfloat16_t;

    auto apply_config = [&](auto fn) {
      if (mask_mode == MaskMode::kCausal) {
        if(is_varlen) {
          fn(CausalForBackwardMask<false>{}, cute::true_type{}, Element{}, ElementOut{});
        } else {
          fn(CausalForBackwardMask<false>{}, cute::false_type{}, Element{}, ElementOut{});
        }
      }
      else {
        if(is_varlen) {
          fn(ResidualMaskForBackward{}, cute::true_type{}, Element{}, ElementOut{});
        } else {
          fn(ResidualMaskForBackward{}, cute::false_type{}, Element{}, ElementOut{});
        }
      }
    };

    apply_config([&](auto mask, auto varlen, auto in, auto out) {
      if (head_dim_qk == 192 && head_dim_vo == 128) {
        call_run_fmha_bwd(mask, varlen, in, out, true_type{}, workspace_buffer, d_o, q, k, v, o, lse,
                          cumulative_seqlen_q, cumulative_seqlen_kv,
                          dq, dk, dv,
                          softmax_scale, max_seqlen_q, max_seqlen_kv);
      } else if (head_dim_qk == 128 && head_dim_vo == 128) {
        call_run_fmha_bwd(mask, varlen, in, out, false_type{}, workspace_buffer, d_o, q, k, v, o, lse,
                          cumulative_seqlen_q, cumulative_seqlen_kv,
                          dq, dk, dv,
                          softmax_scale, max_seqlen_q, max_seqlen_kv);      }
      else {
        std::cout << "No kernel instantiated for head_dim_qk=" << head_dim_qk << " head_dim_vo=" << head_dim_vo << std::endl;
      }
    });

  } else {
    FLASH_MLA_ASSERT(false);
  }
}
