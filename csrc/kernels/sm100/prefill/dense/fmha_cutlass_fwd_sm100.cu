#include "interface.h"

#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include <cuda_bf16.h>

#include "common/mask.cuh"
#include "common/utils.hpp"

#include "fmha_cutlass_fwd_sm100.cuh"

template <class Mask, class Varlen, class Element, class ElementOut, class Mla>
void call_run_fmha_fwd([[maybe_unused]] Mask mask, [[maybe_unused]] Varlen is_varlen,
                       [[maybe_unused]] Element in, [[maybe_unused]] ElementOut out,
                       [[maybe_unused]] Mla mla, torch::stable::Tensor workspace_buffer, torch::stable::Tensor q,
                       torch::stable::Tensor k, torch::stable::Tensor v, torch::stable::Tensor cumulative_seqlen_q,
                       torch::stable::Tensor cumulative_seqlen_kv, torch::stable::Tensor o, torch::stable::Tensor lse,
                       float softmax_scale, int max_seqlen_q, int max_seqlen_kv) {
  static constexpr bool IsVarlen = std::is_same_v<Varlen, true_type>;
  static constexpr bool IsMla = std::is_same_v<Mla, true_type>;
  static constexpr bool IsCausalMask = std::is_same_v<Mask, CausalMask<false>>;
  using Option =
      std::conditional_t<IsCausalMask || (IsVarlen), Option<Tag::kIsPersistent, false_type>,
                         Option<Tag::kIsPersistent, true_type>>;

  run_fmha_fwd<Element, ElementOut, IsVarlen, IsMla, Mask, Option>(
      workspace_buffer, q, k, v, cumulative_seqlen_q, cumulative_seqlen_kv, o, lse,
      softmax_scale, max_seqlen_q, max_seqlen_kv);
}

void FMHACutlassSM100FwdRun(torch::stable::Tensor workspace_buffer, torch::stable::Tensor q, torch::stable::Tensor k,
                            torch::stable::Tensor v, torch::stable::Tensor cumulative_seqlen_q,
                            torch::stable::Tensor cumulative_seqlen_kv, torch::stable::Tensor o, torch::stable::Tensor lse,
                            int64_t mask_mode_code, double sm_scale, int64_t max_seqlen_q,
                            int64_t max_seqlen_kv, bool is_varlen) {
  const torch::stable::accelerator::DeviceGuard device_guard(q.get_device_index());
  STD_TORCH_CHECK(q.scalar_type() == k.scalar_type());
  auto scalar_type_in = q.scalar_type();
  auto scalar_type_out = o.scalar_type();
  int head_dim_qk = q.size(-1);
  int head_dim_vo = v.size(-1);
  MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);

  if (scalar_type_in == torch::headeronly::ScalarType::BFloat16 &&
      scalar_type_out == torch::headeronly::ScalarType::BFloat16) {
    using Element = cutlass::bfloat16_t;
    using ElementOut = cutlass::bfloat16_t;

    auto apply_config = [&](auto fn) {
      if (mask_mode == MaskMode::kCausal) {
        if (is_varlen) {
          fn(CausalMask<false>{}, cute::true_type{}, Element{}, ElementOut{});
        } else {
          fn(CausalMask<false>{}, cute::false_type{}, Element{}, ElementOut{});
        }
      } else {
        if (is_varlen) {
          fn(ResidualMask{}, cute::true_type{}, Element{}, ElementOut{});
        } else {
          fn(ResidualMask{}, cute::false_type{}, Element{}, ElementOut{});
        }
      }
    };

    apply_config([&](auto mask, auto varlen, auto in, auto out) {
      if (head_dim_qk == 192 && head_dim_vo == 128) {
        call_run_fmha_fwd(mask, varlen, in, out, true_type{}, workspace_buffer, q, k, v,
                          cumulative_seqlen_q, cumulative_seqlen_kv, o, lse, sm_scale,
                          max_seqlen_q, max_seqlen_kv);
      } else if (head_dim_qk == 128 && head_dim_vo == 128) {
        call_run_fmha_fwd(mask, varlen, in, out, false_type{}, workspace_buffer, q, k, v,
                          cumulative_seqlen_q, cumulative_seqlen_kv, o, lse, sm_scale,
                          max_seqlen_q, max_seqlen_kv);
      } else {
        std::cout << "No kernel instantiated for head_dim_qk=" << head_dim_qk
                  << " head_dim_vo=" << head_dim_vo << std::endl;
      }
    });

  } else {
    FLASH_MLA_ASSERT(false);
  }
}
