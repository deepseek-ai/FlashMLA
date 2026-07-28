#include "interface.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include "common/mask.cuh"
#include "common/utils.hpp"

#include "fmha_cutlass_bwd_sm100.cuh"

template<class Mask, class Varlen, class Element, class ElementOut, class Mla>
void call_run_fmha_bwd([[maybe_unused]] Mask mask, [[maybe_unused]] Varlen is_varlen,
                      [[maybe_unused]] Element in, [[maybe_unused]] ElementOut out, [[maybe_unused]] Mla mla,
                  at::Tensor workspace_buffer, at::Tensor d_o, at::Tensor q, at::Tensor k,
                  at::Tensor v, at::Tensor o, at::Tensor lse,
                  at::Tensor cumulative_seqlen_q, at::Tensor cumulative_seqlen_kv,
                  at::Tensor dq, at::Tensor dk, at::Tensor dv,
                  float softmax_scale, int max_seqlen_q, int total_seqlen_kv,
                  int window_size, const float *sink_ptr, float *d_sink_ptr) {
  static constexpr bool IsVarlen = std::is_same_v<Varlen, true_type>;
  static constexpr bool IsMla = std::is_same_v<Mla, true_type>;
  using TileShape = std::conditional_t<IsMla, Shape<_64, _128, _192, _128>, Shape<_128, _128, _128, _128>>;
  run_fmha_bwd<Element, IsVarlen, IsMla, TileShape, Mask>(workspace_buffer, d_o, q, k, v, o, lse,
                          cumulative_seqlen_q, cumulative_seqlen_kv,
                          dq, dk, dv,
                          softmax_scale, max_seqlen_q, total_seqlen_kv, window_size,
                          sink_ptr, d_sink_ptr);
}


void FMHACutlassSM100BwdRun(at::Tensor workspace_buffer, at::Tensor d_o, at::Tensor q, at::Tensor k,
                            at::Tensor v, at::Tensor o, at::Tensor lse,
                            at::Tensor cumulative_seqlen_q, at::Tensor cumulative_seqlen_kv,
                            at::Tensor dq, at::Tensor dk, at::Tensor dv,
                            int mask_mode_code, float softmax_scale, int max_seqlen_q, int max_seqlen_kv, bool is_varlen,
                            int window_size, std::optional<at::Tensor> attn_sink,
                            std::optional<at::Tensor> d_sink) {

  const c10::cuda::OptionalCUDAGuard device_guard(q.device());

  // gpt-oss attention sink (MLA only): per-head [h_q] logit + its output grad d_sink ([h_q],
  // pre-zeroed). d_sink is folded into the sum_OdO pass (zero extra cost).
  const float *sink_ptr = attn_sink.has_value() ? attn_sink->data_ptr<float>() : nullptr;
  float *d_sink_ptr = d_sink.has_value() ? d_sink->data_ptr<float>() : nullptr;

  int head_dim_qk = q.size(-1);
  int head_dim_vo = v.size(-1);
  MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  auto scalar_type_in = q.scalar_type();
  auto scalar_type_out = o.scalar_type();

  if(scalar_type_in == at::ScalarType::BFloat16 && scalar_type_out == at::ScalarType::BFloat16) {
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
                          softmax_scale, max_seqlen_q, max_seqlen_kv, window_size,
                          sink_ptr, d_sink_ptr);   // MLA: attention-sink fold-in
      } else if (head_dim_qk == 128 && head_dim_vo == 128) {
        call_run_fmha_bwd(mask, varlen, in, out, false_type{}, workspace_buffer, d_o, q, k, v, o, lse,
                          cumulative_seqlen_q, cumulative_seqlen_kv,
                          dq, dk, dv,
                          softmax_scale, max_seqlen_q, max_seqlen_kv, window_size,
                          nullptr, nullptr);        }
      else {
        std::cout << "No kernel instantiated for head_dim_qk=" << head_dim_qk << " head_dim_vo=" << head_dim_vo << std::endl;
      }
    });

  } else {
    FLASH_MLA_ASSERT(false);
  }
}
