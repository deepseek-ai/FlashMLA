#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>

#include "attention_output_quant/attention_output_quant.h"
#include "combine_indices/combine_indices.h"
#include "faithful_leaves/binding.h"
#include "index_cache_gather/index_cache_gather.h"
#include "learned_topk/learned_topk.h"
#include "main_compress/main_compress.h"
#include "output_quant/output_quant.h"
#include "reset_ready/reset_ready.h"
#ifdef FLASH_MLA_CSA_WITH_ROOT_FUSION
#include "root_fusion/root_fusion.h"
#endif
#include "state_save/state_save.h"
#include "swa_cache_insert/swa_cache_insert.h"
#include "pipeline_bindings.h"
#ifdef FLASH_MLA_CSA_WITH_NATIVE_BASELINE
#include "activation_quant/activation_quant.h"
#include "attention_prepare/attention_prepare.h"
#include "learned_topk.h"
#endif
#ifdef FLASH_MLA_CSA_WITH_FLASHMLA_BASELINE
#include "flashmla/flashmla_out.h"
#endif
#ifdef FLASH_MLA_CSA_WITH_DEEPGEMM
#include "deepgemm/adapters.h"
#endif

namespace py = pybind11;

namespace {

using torch::Tensor;

void check_cuda(const Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.layout() == torch::kStrided, name, " must be strided");
}

void check_tensor(
    const Tensor& tensor,
    c10::ScalarType dtype,
    const char* name) {
  check_cuda(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
}

void check_same_device(
    const Tensor& reference,
    std::initializer_list<const Tensor*> tensors) {
  for (const Tensor* tensor : tensors) {
    TORCH_CHECK(
        tensor->device() == reference.device(),
        "all tensors must be on one CUDA device");
  }
}

void check_positive_strides(const Tensor& tensor, const char* name) {
  for (int64_t dimension = 0; dimension < tensor.dim(); ++dimension) {
    TORCH_CHECK(tensor.stride(dimension) > 0, name, " must have positive strides");
  }
}

void check_not_alias(
    const Tensor& lhs,
    const Tensor& rhs,
    const char* message) {
  TORCH_CHECK(!lhs.is_alias_of(rhs), message);
}

void check_grid(
    const Tensor& tensor,
    int64_t grid) {
  TORCH_CHECK(grid > 0, "grid size must be positive");
  TORCH_CHECK(
      grid <=
          at::cuda::getDeviceProperties(tensor.get_device())
              ->multiProcessorCount,
      "grid size exceeds the physical SM count");
}

cudaStream_t stream_for(const Tensor& tensor) {
  return at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
}

void check_launch(cudaError_t status, const char* operation) {
  TORCH_CHECK(
      status == cudaSuccess,
      operation,
      " launch failed: ",
      cudaGetErrorString(status));
}

size_t tensor_bytes(const Tensor& tensor) {
  TORCH_CHECK(
      tensor.numel() >= 0 &&
          static_cast<uint64_t>(tensor.numel()) <=
              std::numeric_limits<size_t>::max() / tensor.element_size(),
      "tensor byte size overflows size_t");
  return static_cast<size_t>(tensor.numel()) * tensor.element_size();
}

void state_save(
    const Tensor& kv,
    const Tensor& score,
    const Tensor& ape,
    const Tensor& positions,
    const Tensor& state_cache,
    const Tensor& slot_mapping,
    int64_t block_size,
    int64_t num_sms) {
  check_cuda(kv, "kv");
  TORCH_CHECK(
      kv.scalar_type() == torch::kFloat32 ||
          kv.scalar_type() == torch::kBFloat16,
      "kv must be float32 or bfloat16");
  check_tensor(score, kv.scalar_type(), "score");
  check_tensor(ape, kv.scalar_type(), "ape");
  check_tensor(state_cache, kv.scalar_type(), "state_cache");
  check_tensor(positions, torch::kInt64, "positions");
  check_tensor(slot_mapping, torch::kInt64, "slot_mapping");
  constexpr int64_t kWidth = 1024;
  TORCH_CHECK(
      kv.dim() == 2 && kv.size(1) == kWidth && kv.stride(1) == 1,
      "kv must have contiguous rows [tokens,1024]");
  const int64_t tokens = kv.size(0);
  TORCH_CHECK(
      score.sizes() == kv.sizes() && score.stride(1) == 1,
      "score must match kv with contiguous rows");
  TORCH_CHECK(
      ape.dim() == 2 && ape.size(0) == 4 && ape.size(1) == kWidth &&
          ape.stride(1) == 1,
      "ape must have contiguous rows [4,1024]");
  TORCH_CHECK(
      positions.dim() == 1 && positions.numel() == tokens &&
          positions.stride(0) == 1,
      "positions must be contiguous int64 [tokens]");
  TORCH_CHECK(
      slot_mapping.dim() == 1 && slot_mapping.numel() == tokens &&
          slot_mapping.stride(0) == 1,
      "slot_mapping must be contiguous int64 [tokens]");
  TORCH_CHECK(block_size > 0, "block_size must be positive");
  TORCH_CHECK(
      tokens <= std::numeric_limits<uint32_t>::max() &&
          block_size <= std::numeric_limits<uint32_t>::max(),
      "state-save token count or block_size exceeds uint32");
  TORCH_CHECK(
      state_cache.dim() == 3 && state_cache.size(1) == block_size &&
          state_cache.size(2) >= 2 * kWidth && state_cache.stride(2) == 1 &&
          state_cache.stride(1) >= 2 * kWidth &&
          state_cache.stride(0) >= block_size * state_cache.stride(1),
      "state_cache must be non-overlapping [pages,block_size,>=2048]");
  TORCH_CHECK(
      (reinterpret_cast<std::uintptr_t>(kv.data_ptr()) & 15u) == 0 &&
          (reinterpret_cast<std::uintptr_t>(score.data_ptr()) & 15u) == 0 &&
          (reinterpret_cast<std::uintptr_t>(ape.data_ptr()) & 15u) == 0 &&
          (reinterpret_cast<std::uintptr_t>(state_cache.data_ptr()) & 15u) == 0 &&
          (kv.stride(0) * kv.element_size()) % 16 == 0 &&
          (score.stride(0) * score.element_size()) % 16 == 0 &&
          (ape.stride(0) * ape.element_size()) % 16 == 0 &&
          (state_cache.stride(0) * state_cache.element_size()) % 16 == 0 &&
          (state_cache.stride(1) * state_cache.element_size()) % 16 == 0,
      "state-save vector surfaces must be 16-byte aligned");
  check_same_device(kv, {&score, &ape, &positions, &state_cache, &slot_mapping});
  check_not_alias(kv, state_cache, "kv and state_cache must not alias");
  check_not_alias(score, state_cache, "score and state_cache must not alias");
  check_not_alias(ape, state_cache, "ape and state_cache must not alias");
  check_grid(kv, num_sms);
  c10::cuda::CUDAGuard guard(kv.device());
  const auto stream = stream_for(kv);
  cudaError_t status;
  if (kv.scalar_type() == torch::kFloat32) {
    status = launch_state_save_fp32(
        kv.const_data_ptr<float>(), kv.stride(0),
        score.const_data_ptr<float>(), score.stride(0),
        ape.const_data_ptr<float>(), ape.stride(0),
        positions.const_data_ptr<int64_t>(), state_cache.data_ptr<float>(),
        state_cache.stride(0), state_cache.stride(1),
        slot_mapping.const_data_ptr<int64_t>(), static_cast<uint32_t>(tokens),
        static_cast<uint32_t>(block_size), static_cast<uint32_t>(num_sms), stream);
  } else {
    status = launch_state_save_bf16(
        kv.const_data_ptr(), kv.stride(0), score.const_data_ptr(),
        score.stride(0), ape.const_data_ptr(), ape.stride(0),
        positions.const_data_ptr<int64_t>(), state_cache.data_ptr(),
        state_cache.stride(0), state_cache.stride(1),
        slot_mapping.const_data_ptr<int64_t>(), static_cast<uint32_t>(tokens),
        static_cast<uint32_t>(block_size), static_cast<uint32_t>(num_sms), stream);
  }
  check_launch(status, "state_save");
}

void main_compress(
    const Tensor& state_cache,
    const Tensor& token_to_req,
    const Tensor& positions,
    const Tensor& state_slot_mapping,
    const Tensor& state_block_table,
    const Tensor& rms_weight,
    double rms_norm_eps,
    const Tensor& cos_sin_cache,
    const Tensor& kv_cache,
    const Tensor& kv_slot_mapping,
    int64_t num_sms) {
  check_tensor(state_cache, torch::kFloat32, "state_cache");
  check_tensor(token_to_req, torch::kInt32, "token_to_req");
  check_tensor(positions, torch::kInt64, "positions");
  check_tensor(state_slot_mapping, torch::kInt64, "state_slot_mapping");
  check_tensor(state_block_table, torch::kInt32, "state_block_table");
  check_cuda(rms_weight, "rms_weight");
  TORCH_CHECK(
      rms_weight.scalar_type() == torch::kFloat32 ||
          rms_weight.scalar_type() == torch::kBFloat16,
      "rms_weight must be float32 or bfloat16");
  check_tensor(cos_sin_cache, torch::kFloat32, "cos_sin_cache");
  check_tensor(kv_cache, torch::kUInt8, "kv_cache");
  check_tensor(kv_slot_mapping, torch::kInt64, "kv_slot_mapping");
  const int64_t tokens = positions.numel();
  TORCH_CHECK(
      token_to_req.dim() == 1 && token_to_req.numel() == tokens &&
          token_to_req.stride(0) == 1 && positions.dim() == 1 &&
          positions.stride(0) == 1 && state_slot_mapping.dim() == 1 &&
          state_slot_mapping.numel() == tokens && state_slot_mapping.stride(0) == 1 &&
          kv_slot_mapping.dim() == 1 && kv_slot_mapping.numel() == tokens &&
          kv_slot_mapping.stride(0) == 1,
      "token metadata must be contiguous one-dimensional tensors of equal length");
  TORCH_CHECK(
      state_cache.dim() == 3 && state_cache.size(1) > 0 &&
          state_cache.size(2) >= 2048 && state_cache.stride(2) == 1 &&
          state_cache.stride(1) >= 2048 &&
          state_cache.stride(0) >= state_cache.size(1) * state_cache.stride(1),
      "state_cache must be non-overlapping float32 [pages,block_size,>=2048]");
  TORCH_CHECK(
      state_block_table.dim() == 2 && state_block_table.stride(1) == 1,
      "state_block_table must have contiguous int32 rows");
  TORCH_CHECK(
      rms_weight.dim() == 1 && rms_weight.numel() == 512 &&
          rms_weight.is_contiguous(),
      "rms_weight must be contiguous [512]");
  TORCH_CHECK(
      rms_norm_eps >= 0.0 && rms_norm_eps <= std::numeric_limits<float>::max(),
      "rms_norm_eps must be a finite non-negative float");
  TORCH_CHECK(
      cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) >= 64 &&
          cos_sin_cache.stride(1) == 1,
      "cos_sin_cache must have contiguous rows [positions,>=64]");
  constexpr int64_t kPageBytes = 64 * 576 + 64 * 8;
  TORCH_CHECK(
      kv_cache.dim() == 2 && kv_cache.size(1) >= kPageBytes &&
          kv_cache.stride(1) == 1 && kv_cache.stride(0) >= kPageBytes,
      "kv_cache must provide at least 37376 bytes per page");
  TORCH_CHECK(
      (reinterpret_cast<std::uintptr_t>(state_cache.data_ptr()) & 15u) == 0 &&
          (reinterpret_cast<std::uintptr_t>(kv_cache.data_ptr()) & 15u) == 0 &&
          (state_cache.stride(0) * state_cache.element_size()) % 16 == 0 &&
          (state_cache.stride(1) * state_cache.element_size()) % 16 == 0 &&
          kv_cache.stride(0) % 16 == 0,
      "main-compress vector surfaces must be 16-byte aligned");
  TORCH_CHECK(
      tokens <= std::numeric_limits<uint32_t>::max() &&
          state_cache.stride(0) <= std::numeric_limits<uint32_t>::max() &&
          state_cache.stride(1) <= std::numeric_limits<uint32_t>::max() &&
          state_cache.size(1) <= std::numeric_limits<uint32_t>::max() &&
          state_block_table.stride(0) <= std::numeric_limits<uint32_t>::max() &&
          cos_sin_cache.stride(0) <= std::numeric_limits<uint32_t>::max() &&
          kv_cache.stride(0) <= std::numeric_limits<uint32_t>::max(),
      "main-compress dimensions or strides exceed uint32");
  check_same_device(state_cache, {&token_to_req, &positions, &state_slot_mapping,
      &state_block_table, &rms_weight, &cos_sin_cache, &kv_cache, &kv_slot_mapping});
  check_not_alias(state_cache, kv_cache, "state_cache and kv_cache must not alias");
  check_grid(state_cache, num_sms);
  c10::cuda::CUDAGuard guard(state_cache.device());
  check_launch(
      launch_main_compress(
          static_cast<uint32_t>(tokens), state_cache.const_data_ptr<float>(),
          static_cast<uint32_t>(state_cache.stride(0)),
          static_cast<uint32_t>(state_cache.stride(1)),
          static_cast<uint32_t>(state_cache.size(1)),
          token_to_req.const_data_ptr<int32_t>(), positions.const_data_ptr<int64_t>(),
          state_slot_mapping.const_data_ptr<int64_t>(),
          state_block_table.const_data_ptr<int32_t>(),
          static_cast<uint32_t>(state_block_table.stride(0)),
          rms_weight.const_data_ptr(), static_cast<float>(rms_norm_eps),
          cos_sin_cache.const_data_ptr<float>(),
          static_cast<uint32_t>(cos_sin_cache.stride(0)),
          kv_cache.data_ptr<uint8_t>(), kv_slot_mapping.const_data_ptr<int64_t>(),
          static_cast<uint32_t>(kv_cache.stride(0)),
          static_cast<uint32_t>(num_sms),
          rms_weight.scalar_type() == torch::kBFloat16, stream_for(state_cache)),
      "main_compress");
}

void swa_cache_insert(
    const Tensor& kv,
    const Tensor& cache,
    const Tensor& slots,
    const Tensor& positions,
    const Tensor& cos_sin,
    int64_t block_size,
    int64_t num_sms) {
  check_tensor(kv, torch::kBFloat16, "kv");
  check_tensor(cache, torch::kUInt8, "cache");
  check_tensor(slots, torch::kInt64, "slot_mapping");
  check_tensor(positions, torch::kInt64, "position_ids");
  check_tensor(cos_sin, torch::kFloat32, "cos_sin_cache");
  TORCH_CHECK(
      kv.dim() == 2 && kv.size(1) == 512 && kv.stride(1) == 1,
      "kv must be BF16 [tokens,512] with contiguous rows");
  const int64_t tokens = kv.size(0);
  TORCH_CHECK(
      slots.dim() == 1 && slots.size(0) == tokens && slots.stride(0) > 0,
      "slot_mapping must be int64 [tokens]");
  TORCH_CHECK(
      positions.dim() == 1 && positions.size(0) == tokens &&
          positions.stride(0) > 0,
      "position_ids must be int64 [tokens]");
  TORCH_CHECK(
      cos_sin.dim() == 2 && cos_sin.size(1) >= 64 &&
          cos_sin.stride(1) == 1,
      "cos_sin_cache must be float32 [positions,>=64]");
  TORCH_CHECK(
      block_size > 0 && block_size <= std::numeric_limits<int32_t>::max(),
      "block_size must be a positive int32");
  TORCH_CHECK(
      cache.dim() == 2 && cache.stride(1) == 1 &&
          cache.stride(0) >= block_size * 584,
      "cache pages must provide at least 584 bytes per token");
  TORCH_CHECK(
      (reinterpret_cast<std::uintptr_t>(kv.data_ptr()) & 15u) == 0 &&
          (reinterpret_cast<std::uintptr_t>(cache.data_ptr()) & 15u) == 0 &&
          (reinterpret_cast<std::uintptr_t>(cos_sin.data_ptr()) & 15u) == 0 &&
          kv.stride(0) % 8 == 0 && cache.stride(0) % 16 == 0 &&
          cos_sin.stride(0) % 4 == 0,
      "kv, cache, and cos_sin_cache must satisfy vector alignment");
  TORCH_CHECK(
      tokens <= std::numeric_limits<int32_t>::max(),
      "token count exceeds int32");
  check_same_device(kv, {&cache, &slots, &positions, &cos_sin});
  check_not_alias(kv, cache, "kv and cache must not alias");
  check_grid(kv, num_sms);
  c10::cuda::CUDAGuard guard(kv.device());
  if (tokens == 0) {
    return;
  }
  launch_swa_cache_insert(
      kv.const_data_ptr(),
      kv.stride(0),
      cache.data_ptr<uint8_t>(),
      cache.stride(0),
      slots.const_data_ptr<int64_t>(),
      slots.stride(0),
      positions.const_data_ptr<int64_t>(),
      positions.stride(0),
      cos_sin.const_data_ptr<float>(),
      cos_sin.stride(0),
      static_cast<int32_t>(tokens),
      static_cast<int32_t>(block_size),
      static_cast<int32_t>(num_sms),
      stream_for(kv));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void index_cache_gather(
    const Tensor& cache,
    const Tensor& dst_k,
    const Tensor& dst_scale,
    const Tensor& block_table,
    const Tensor& cu_lens,
    int64_t num_sms,
    bool vectorized) {
  check_tensor(cache, torch::kUInt8, "kv_cache");
  check_tensor(dst_k, torch::kUInt8, "dst_k");
  check_tensor(dst_scale, torch::kUInt8, "dst_scale");
  check_tensor(block_table, torch::kInt32, "block_table");
  check_tensor(cu_lens, torch::kInt32, "cu_seq_lens");
  constexpr int64_t kPageBytes = 64 * (64 + 4);
  TORCH_CHECK(
      cache.dim() == 2 && cache.stride(1) == 1 &&
          cache.stride(0) >= kPageBytes,
      "kv_cache must be uint8 [blocks,>=4352]");
  TORCH_CHECK(
      dst_k.dim() == 2 && dst_k.size(1) >= 64 &&
          dst_k.stride(1) == 1 && dst_k.stride(0) >= 64,
      "dst_k must be uint8 [capacity,>=64]");
  TORCH_CHECK(
      dst_scale.dim() == 2 && dst_scale.size(0) == dst_k.size(0) &&
          dst_scale.size(1) >= 4 && dst_scale.stride(1) == 1 &&
          dst_scale.stride(0) >= 4,
      "dst_scale must be uint8 [capacity,>=4]");
  TORCH_CHECK(
      block_table.dim() == 2 && block_table.stride(1) == 1,
      "block_table must have contiguous int32 rows");
  TORCH_CHECK(
      cu_lens.dim() == 1 && cu_lens.is_contiguous() &&
          cu_lens.numel() == block_table.size(0) + 1,
      "cu_seq_lens must be contiguous int32 [batch+1]");
  TORCH_CHECK(
      cache.size(0) <= std::numeric_limits<uint32_t>::max() &&
          dst_k.size(0) <= std::numeric_limits<uint32_t>::max() &&
          block_table.size(0) <= std::numeric_limits<uint32_t>::max() &&
          block_table.size(1) <= std::numeric_limits<uint32_t>::max(),
      "surface dimensions exceed uint32");
  check_same_device(cache, {&dst_k, &dst_scale, &block_table, &cu_lens});
  check_not_alias(cache, dst_k, "cache and dst_k must not alias");
  check_not_alias(cache, dst_scale, "cache and dst_scale must not alias");
  check_not_alias(dst_k, dst_scale, "gather outputs must not alias");
  if (vectorized) {
    TORCH_CHECK(
        (reinterpret_cast<std::uintptr_t>(cache.data_ptr()) & 15u) == 0 &&
            (reinterpret_cast<std::uintptr_t>(dst_k.data_ptr()) & 15u) == 0 &&
            (reinterpret_cast<std::uintptr_t>(dst_scale.data_ptr()) & 3u) == 0 &&
            cache.stride(0) % 16 == 0 && dst_k.stride(0) % 16 == 0 &&
            dst_scale.stride(0) % 4 == 0,
        "vectorized gather surfaces do not satisfy alignment requirements");
  }
  check_grid(cache, num_sms);
  c10::cuda::CUDAGuard guard(cache.device());
  check_launch(
      launch_index_cache_gather(
          cache.const_data_ptr<uint8_t>(),
          dst_k.data_ptr<uint8_t>(),
          dst_scale.data_ptr<uint8_t>(),
          block_table.const_data_ptr<int32_t>(),
          cu_lens.const_data_ptr<int32_t>(),
          cache.stride(0),
          dst_k.stride(0),
          dst_scale.stride(0),
          block_table.stride(0),
          static_cast<uint32_t>(block_table.size(0)),
          static_cast<uint32_t>(dst_k.size(0)),
          static_cast<uint32_t>(cache.size(0)),
          static_cast<uint32_t>(block_table.size(1)),
          static_cast<uint32_t>(num_sms),
          vectorized,
          stream_for(cache)),
      "index_cache_gather");
}

void learned_topk(
    const Tensor& logits,
    const Tensor& starts,
    const Tensor& ends,
    const Tensor& output,
    int64_t grid) {
  check_tensor(logits, torch::kFloat32, "logits");
  check_tensor(starts, torch::kInt32, "row_starts");
  check_tensor(ends, torch::kInt32, "row_ends");
  check_tensor(output, torch::kInt32, "output");
  TORCH_CHECK(
      logits.dim() == 2 && logits.stride(1) == 1 &&
          logits.stride(0) >= logits.size(1),
      "logits must have non-overlapping contiguous rows");
  const int64_t rows = logits.size(0);
  TORCH_CHECK(
      starts.dim() == 1 && starts.is_contiguous() &&
          starts.numel() == rows,
      "row_starts must be contiguous int32 [rows]");
  TORCH_CHECK(
      ends.dim() == 1 && ends.is_contiguous() && ends.numel() == rows,
      "row_ends must be contiguous int32 [rows]");
  TORCH_CHECK(
      output.dim() == 2 && output.is_contiguous() &&
          output.sizes() == torch::IntArrayRef({rows, 1024}),
      "output must be contiguous int32 [rows,1024]");
  TORCH_CHECK(
      rows <= std::numeric_limits<int32_t>::max() &&
          logits.size(1) <= std::numeric_limits<int32_t>::max(),
      "logit dimensions exceed int32");
  check_same_device(logits, {&starts, &ends, &output});
  check_not_alias(logits, output, "logits and output must not alias");
  check_grid(logits, grid);
  c10::cuda::CUDAGuard guard(logits.device());
  if (rows == 0) {
    return;
  }
  check_launch(
      flash_mla_csa::learned_topk::launch_learned_topk(
          logits.const_data_ptr<float>(),
          starts.const_data_ptr<int32_t>(),
          ends.const_data_ptr<int32_t>(),
          output.data_ptr<int32_t>(),
          logits.stride(0),
          static_cast<int32_t>(rows),
          static_cast<int32_t>(logits.size(1)),
          static_cast<int32_t>(grid),
          stream_for(logits)),
      "learned_topk");
}

void combine_indices(
    const Tensor& topk,
    const Tensor& combined,
    const Tensor& lengths,
    int64_t compressed_rows,
    int64_t num_sms) {
  check_tensor(topk, torch::kInt32, "topk");
  check_tensor(combined, torch::kInt32, "combined");
  check_tensor(lengths, torch::kInt32, "lengths");
  TORCH_CHECK(
      topk.dim() == 2 && topk.size(1) >= 1024 &&
          topk.stride(1) == 1 && topk.stride(0) >= 1024,
      "topk must be int32 [tokens,>=1024]");
  const int64_t tokens = topk.size(0);
  TORCH_CHECK(
      combined.dim() == 2 && combined.size(0) >= tokens &&
          combined.size(1) >= 1152 && combined.stride(1) == 1 &&
          combined.stride(0) >= 1152,
      "combined must be int32 [tokens,>=1152]");
  TORCH_CHECK(
      lengths.dim() == 1 && lengths.numel() >= tokens &&
          lengths.stride(0) > 0,
      "lengths must be int32 [tokens]");
  TORCH_CHECK(
      compressed_rows >= 0 &&
          compressed_rows + tokens + 127 <=
              std::numeric_limits<int32_t>::max(),
      "combined index range exceeds int32");
  check_same_device(topk, {&combined, &lengths});
  check_not_alias(topk, combined, "topk and combined must not alias");
  check_not_alias(topk, lengths, "topk and lengths must not alias");
  check_not_alias(combined, lengths, "combined and lengths must not alias");
  check_grid(topk, num_sms);
  c10::cuda::CUDAGuard guard(topk.device());
  launch_combine_indices(
      topk.const_data_ptr<int32_t>(),
      topk.stride(0),
      combined.data_ptr<int32_t>(),
      combined.stride(0),
      lengths.data_ptr<int32_t>(),
      lengths.stride(0),
      static_cast<int32_t>(compressed_rows),
      static_cast<int32_t>(tokens),
      static_cast<int32_t>(num_sms),
      stream_for(topk));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void attention_output_quant(
    const Tensor& input,
    const Tensor& positions,
    const Tensor& cos_sin,
    const Tensor& output,
    const Tensor& scales,
    int64_t num_sms) {
  check_tensor(input, torch::kBFloat16, "input");
  check_tensor(positions, torch::kInt64, "positions");
  check_tensor(cos_sin, torch::kFloat32, "cos_sin_cache");
  check_cuda(output, "output");
  check_tensor(scales, torch::kInt32, "scales");
  TORCH_CHECK(
      output.scalar_type() == torch::kUInt8 ||
          output.scalar_type() == torch::kFloat8_e4m3fn,
      "output must be uint8 or float8_e4m3fn");
  TORCH_CHECK(
      input.dim() == 3 && input.size(1) == 128 && input.size(2) == 512,
      "input must be BF16 [tokens,128,512]");
  const int64_t tokens = input.size(0);
  TORCH_CHECK(
      tokens > 0 && tokens <= std::numeric_limits<int32_t>::max() - 3,
      "token count must be in [1,INT32_MAX-3]");
  TORCH_CHECK(
      positions.dim() == 1 && positions.size(0) == tokens,
      "positions must be int64 [tokens]");
  TORCH_CHECK(
      cos_sin.dim() == 2 && cos_sin.size(1) == 64,
      "cos_sin_cache must be float32 [positions,64]");
  TORCH_CHECK(
      output.dim() == 3 &&
          output.sizes() == torch::IntArrayRef({tokens, 16, 4096}),
      "output must have shape [tokens,16,4096]");
  TORCH_CHECK(
      scales.dim() == 3 &&
          scales.sizes() == torch::IntArrayRef({tokens, 16, 8}),
      "scales must have shape [tokens,16,8]");
  const int64_t aligned_tokens = (tokens + 3) / 4 * 4;
  TORCH_CHECK(
      output.stride(0) == 4096 &&
          output.stride(1) == tokens * 4096 &&
          output.stride(2) == 1,
      "output must use group-major strides");
  TORCH_CHECK(
      scales.stride(0) == 1 &&
          scales.stride(1) == 8 * aligned_tokens &&
          scales.stride(2) == aligned_tokens,
      "scales must use the padded production strides");
  check_positive_strides(input, "input");
  check_positive_strides(positions, "positions");
  check_positive_strides(cos_sin, "cos_sin_cache");
  check_same_device(input, {&positions, &cos_sin, &output, &scales});
  check_not_alias(input, output, "input and output must not alias");
  check_not_alias(input, scales, "input and scales must not alias");
  check_not_alias(output, scales, "output and scales must not alias");
  check_grid(input, num_sms);
  c10::cuda::CUDAGuard guard(input.device());
  check_launch(
      launch_attention_output_quant(
          input.const_data_ptr(),
          positions.const_data_ptr<int64_t>(),
          cos_sin.const_data_ptr<float>(),
          output.data_ptr(),
          scales.data_ptr<int32_t>(),
          static_cast<int32_t>(tokens),
          input.stride(0),
          input.stride(1),
          input.stride(2),
          positions.stride(0),
          cos_sin.stride(0),
          cos_sin.stride(1),
          output.stride(0),
          output.stride(1),
          output.stride(2),
          scales.stride(0),
          scales.stride(1),
          scales.stride(2),
          static_cast<int32_t>(num_sms),
          stream_for(input)),
      "attention_output_quant");
}

void output_quant(
    const Tensor& input,
    const Tensor& output,
    const Tensor& scales,
    int64_t num_sms) {
  check_tensor(input, torch::kBFloat16, "input");
  check_cuda(output, "output");
  check_tensor(scales, torch::kInt32, "scales");
  TORCH_CHECK(
      output.scalar_type() == torch::kUInt8 ||
          output.scalar_type() == torch::kFloat8_e4m3fn,
      "output must be uint8 or float8_e4m3fn");
  constexpr int64_t kRows = 8192;
  constexpr int64_t kHidden = 16384;
  TORCH_CHECK(
      input.sizes() == torch::IntArrayRef({kRows, kHidden}) &&
          input.is_contiguous(),
      "input must be contiguous BF16 [8192,16384]");
  TORCH_CHECK(
      output.sizes() == input.sizes() && output.is_contiguous(),
      "output must be contiguous [8192,16384]");
  const bool physical =
      scales.sizes() == torch::IntArrayRef({32, kRows}) &&
      scales.is_contiguous();
  const bool logical =
      scales.sizes() == torch::IntArrayRef({kRows, 32}) &&
      scales.stride(0) == 1 && scales.stride(1) == kRows;
  TORCH_CHECK(
      physical || logical,
      "scales must be physical [32,8192] or its logical transpose");
  check_same_device(input, {&output, &scales});
  check_not_alias(input, output, "input and output must not alias");
  check_not_alias(input, scales, "input and scales must not alias");
  check_not_alias(output, scales, "output and scales must not alias");
  check_grid(input, num_sms);
  c10::cuda::CUDAGuard guard(input.device());
  check_launch(
      launch_output_quant(
          input.const_data_ptr(),
          static_cast<uint8_t*>(output.data_ptr()),
          scales.data_ptr<int32_t>(),
          static_cast<uint32_t>(num_sms),
          stream_for(input)),
      "output_quant");
}

void check_reset(
    const Tensor& counters,
    const Tensor& ready,
    const Tensor& producer,
    int64_t num_sms) {
  check_tensor(counters, torch::kInt32, "counters");
  check_tensor(ready, torch::kInt32, "ready");
  check_tensor(producer, torch::kInt64, "producer_clocks");
  TORCH_CHECK(
      counters.dim() == 1 && counters.is_contiguous() &&
          counters.numel() > 0,
      "counters must be non-empty contiguous int32 [N]");
  TORCH_CHECK(
      ready.dim() == 1 && ready.is_contiguous() &&
          ready.numel() == counters.numel(),
      "ready must be contiguous int32 [N]");
  TORCH_CHECK(
      producer.dim() == 1 && producer.is_contiguous() &&
          producer.numel() == counters.numel() + 1,
      "producer_clocks must be contiguous int64 [N+1]");
  check_same_device(counters, {&ready, &producer});
  check_not_alias(counters, ready, "reset buffers must not alias");
  check_not_alias(counters, producer, "reset buffers must not alias");
  check_not_alias(ready, producer, "reset buffers must not alias");
  check_grid(counters, num_sms);
}

void reset_ready_three(
    const Tensor& counters,
    const Tensor& ready,
    const Tensor& producer,
    int64_t num_sms) {
  check_reset(counters, ready, producer, num_sms);
  c10::cuda::CUDAGuard guard(counters.device());
  check_launch(
      launch_reset_ready(
          counters.data_ptr(),
          tensor_bytes(counters),
          ready.data_ptr(),
          tensor_bytes(ready),
          producer.data_ptr(),
          tensor_bytes(producer),
          nullptr,
          0,
          nullptr,
          0,
          3,
          static_cast<int>(num_sms),
          stream_for(counters)),
      "reset_ready");
}

void reset_ready_five(
    const Tensor& counters,
    const Tensor& ready,
    const Tensor& producer,
    const Tensor& consumer,
    const Tensor& qnorm,
    int64_t num_sms) {
  check_reset(counters, ready, producer, num_sms);
  check_tensor(consumer, torch::kInt64, "consumer_clocks");
  check_tensor(qnorm, torch::kInt64, "qnorm_clocks");
  TORCH_CHECK(
      consumer.dim() == 1 && consumer.is_contiguous() &&
          consumer.numel() == counters.numel(),
      "consumer_clocks must be contiguous int64 [N]");
  TORCH_CHECK(
      qnorm.dim() == 1 && qnorm.is_contiguous() &&
          qnorm.numel() == counters.numel(),
      "qnorm_clocks must be contiguous int64 [N]");
  check_same_device(counters, {&consumer, &qnorm});
  check_not_alias(counters, consumer, "reset buffers must not alias");
  check_not_alias(counters, qnorm, "reset buffers must not alias");
  check_not_alias(ready, consumer, "reset buffers must not alias");
  check_not_alias(ready, qnorm, "reset buffers must not alias");
  check_not_alias(producer, consumer, "reset buffers must not alias");
  check_not_alias(producer, qnorm, "reset buffers must not alias");
  check_not_alias(consumer, qnorm, "reset buffers must not alias");
  c10::cuda::CUDAGuard guard(counters.device());
  check_launch(
      launch_reset_ready(
          counters.data_ptr(),
          tensor_bytes(counters),
          ready.data_ptr(),
          tensor_bytes(ready),
          producer.data_ptr(),
          tensor_bytes(producer),
          consumer.data_ptr(),
          tensor_bytes(consumer),
          qnorm.data_ptr(),
          tensor_bytes(qnorm),
          5,
          static_cast<int>(num_sms),
          stream_for(counters)),
      "reset_ready");
}

#ifdef FLASH_MLA_CSA_WITH_ROOT_FUSION
void root_fusion(
    const Tensor& hidden,
    const Tensor& hidden_fp8,
    const Tensor& hidden_scale,
    const Tensor& fused_wqa_wkv,
    const Tensor& fused_wqa_wkv_scale,
    const Tensor& qkv,
    const Tensor& q_norm_weight,
    const Tensor& kv_norm_weight,
    const Tensor& q_norm,
    const Tensor& kv_norm,
    const Tensor& q_norm_fp8,
    const Tensor& q_norm_scale,
    const Tensor& projection_barrier,
    const Tensor& index_compressor_weight,
    const Tensor& index_kv_score,
    const Tensor& index_weight_projection,
    const Tensor& index_weights,
    const Tensor& index_barrier,
    const Tensor& row_ready,
    const Tensor& c4_workspace,
    const Tensor& state_cache,
    const Tensor& token_to_request,
    const Tensor& positions,
    const Tensor& state_slot_mapping,
    const Tensor& state_block_table,
    const Tensor& rms_weight,
    const Tensor& cos_sin_cache,
    const Tensor& kv_cache,
    const Tensor& kv_slot_mapping,
    const Tensor& ape,
    int64_t grid_sms,
    int64_t projection_sms) {
  constexpr int64_t kTokens = 8192;
  constexpr int64_t kHidden = 7168;
  constexpr int64_t kProjectionWidth = 2048;
  check_tensor(hidden, torch::kBFloat16, "hidden");
  check_tensor(hidden_fp8, torch::kFloat8_e4m3fn, "hidden_fp8");
  check_tensor(hidden_scale, torch::kInt32, "hidden_scale");
  check_tensor(fused_wqa_wkv, torch::kFloat8_e4m3fn, "fused_wqa_wkv");
  check_tensor(fused_wqa_wkv_scale, torch::kInt32, "fused_wqa_wkv_scale");
  check_tensor(qkv, torch::kBFloat16, "qkv");
  check_tensor(q_norm_weight, torch::kBFloat16, "q_norm_weight");
  check_tensor(kv_norm_weight, torch::kBFloat16, "kv_norm_weight");
  check_tensor(q_norm, torch::kBFloat16, "q_norm");
  check_tensor(kv_norm, torch::kBFloat16, "kv_norm");
  check_tensor(q_norm_fp8, torch::kFloat8_e4m3fn, "q_norm_fp8");
  check_tensor(q_norm_scale, torch::kInt32, "q_norm_scale");
  check_tensor(projection_barrier, torch::kInt32, "projection_barrier");
  check_tensor(index_compressor_weight, torch::kBFloat16,
               "index_compressor_weight");
  check_tensor(index_kv_score, torch::kFloat32, "index_kv_score");
  check_tensor(index_weight_projection, torch::kBFloat16,
               "index_weight_projection");
  check_tensor(index_weights, torch::kBFloat16, "index_weights");
  check_tensor(index_barrier, torch::kInt32, "index_barrier");
  check_tensor(row_ready, torch::kInt64, "row_ready");
  check_tensor(c4_workspace, torch::kFloat32, "c4_workspace");
  check_tensor(state_cache, torch::kFloat32, "state_cache");
  check_tensor(token_to_request, torch::kInt32, "token_to_request");
  check_tensor(positions, torch::kInt64, "positions");
  check_tensor(state_slot_mapping, torch::kInt64, "state_slot_mapping");
  check_tensor(state_block_table, torch::kInt32, "state_block_table");
  check_tensor(rms_weight, torch::kFloat32, "rms_weight");
  check_tensor(cos_sin_cache, torch::kFloat32, "cos_sin_cache");
  check_tensor(kv_cache, torch::kUInt8, "kv_cache");
  check_tensor(kv_slot_mapping, torch::kInt64, "kv_slot_mapping");
  check_tensor(ape, torch::kFloat32, "ape");

  TORCH_CHECK(hidden.sizes() == torch::IntArrayRef({kTokens, kHidden}) &&
                  hidden.is_contiguous(),
              "hidden must be contiguous BF16 [8192,7168]");
  TORCH_CHECK(hidden_fp8.sizes() == hidden.sizes() &&
                  hidden_fp8.is_contiguous(),
              "hidden_fp8 must be contiguous FP8 [8192,7168]");
  TORCH_CHECK(hidden_scale.sizes() ==
                  torch::IntArrayRef({kTokens, kHidden / 512}) &&
                  hidden_scale.stride(0) == 1 &&
                  hidden_scale.stride(1) >= kTokens,
              "hidden_scale must be MN-major int32 [8192,14]");
  TORCH_CHECK(fused_wqa_wkv.sizes() ==
                  torch::IntArrayRef({kProjectionWidth, kHidden}) &&
                  fused_wqa_wkv.is_contiguous(),
              "fused_wqa_wkv must be contiguous FP8 [2048,7168]");
  TORCH_CHECK(fused_wqa_wkv_scale.sizes() ==
                  torch::IntArrayRef({kProjectionWidth, kHidden / 512}) &&
                  fused_wqa_wkv_scale.stride(0) == 1 &&
                  fused_wqa_wkv_scale.stride(1) >= kProjectionWidth,
              "fused_wqa_wkv_scale must be MN-major int32 [2048,14]");
  TORCH_CHECK(qkv.sizes() ==
                  torch::IntArrayRef({kTokens, kProjectionWidth}) &&
                  qkv.is_contiguous(),
              "qkv must be contiguous BF16 [8192,2048]");
  TORCH_CHECK(q_norm_weight.sizes() == torch::IntArrayRef({1536}) &&
                  q_norm_weight.is_contiguous(),
              "q_norm_weight must be contiguous BF16 [1536]");
  TORCH_CHECK(kv_norm_weight.sizes() == torch::IntArrayRef({512}) &&
                  kv_norm_weight.is_contiguous(),
              "kv_norm_weight must be contiguous BF16 [512]");
  TORCH_CHECK(q_norm.sizes() == torch::IntArrayRef({kTokens, 1536}) &&
                  q_norm.is_contiguous(),
              "q_norm must be contiguous BF16 [8192,1536]");
  TORCH_CHECK(kv_norm.sizes() == torch::IntArrayRef({kTokens, 512}) &&
                  kv_norm.is_contiguous(),
              "kv_norm must be contiguous BF16 [8192,512]");
  TORCH_CHECK(q_norm_fp8.sizes() == q_norm.sizes() &&
                  q_norm_fp8.is_contiguous(),
              "q_norm_fp8 must be contiguous FP8 [8192,1536]");
  TORCH_CHECK(q_norm_scale.sizes() == torch::IntArrayRef({kTokens, 3}) &&
                  q_norm_scale.stride(0) == 1 &&
                  q_norm_scale.stride(1) >= kTokens,
              "q_norm_scale must be MN-major int32 [8192,3]");
  TORCH_CHECK(projection_barrier.numel() >= 1 &&
                  projection_barrier.is_contiguous(),
              "projection_barrier must be nonempty contiguous int32");
  TORCH_CHECK(index_compressor_weight.sizes() ==
                  torch::IntArrayRef({512, kHidden}) &&
                  index_compressor_weight.is_contiguous(),
              "index_compressor_weight must be contiguous BF16 [512,7168]");
  TORCH_CHECK(index_kv_score.sizes() ==
                  torch::IntArrayRef({kTokens, 512}) &&
                  index_kv_score.is_contiguous(),
              "index_kv_score must be contiguous float32 [8192,512]");
  TORCH_CHECK(index_weight_projection.sizes() ==
                  torch::IntArrayRef({64, kHidden}) &&
                  index_weight_projection.is_contiguous(),
              "index_weight_projection must be contiguous BF16 [64,7168]");
  TORCH_CHECK(index_weights.sizes() == torch::IntArrayRef({kTokens, 64}) &&
                  index_weights.is_contiguous(),
              "index_weights must be contiguous BF16 [8192,64]");
  TORCH_CHECK(index_barrier.numel() >= 8 && index_barrier.is_contiguous(),
              "index_barrier must contain at least 8 contiguous int32 values");
  TORCH_CHECK(row_ready.numel() > 0 && row_ready.is_contiguous(),
              "row_ready must be nonempty contiguous int64");
  check_grid(hidden, grid_sms);
  TORCH_CHECK(grid_sms == 148 || grid_sms == 152,
              "csa_root_fusion grid_sms must select 148 or 152");
  const bool valid_projection =
      (grid_sms == 148 && projection_sms >= 96 && projection_sms <= 108 &&
       projection_sms % 2 == 0) ||
      (grid_sms == 152 && projection_sms >= 104 && projection_sms <= 108 &&
       projection_sms % 2 == 0);
  TORCH_CHECK(valid_projection, "csa_root_fusion projection_sms is not precompiled");
  const int64_t index_sms = grid_sms - projection_sms;
  TORCH_CHECK(c4_workspace.dim() == 3 &&
                  c4_workspace.size(0) >= index_sms &&
                  c4_workspace.size(1) == 8 && c4_workspace.size(2) == 128 &&
                  c4_workspace.is_contiguous(),
              "c4_workspace must cover the selected index CTA lane");
  TORCH_CHECK(state_cache.dim() == 3 && state_cache.size(1) == 4 &&
                  state_cache.size(2) == 512 && state_cache.is_contiguous(),
              "state_cache must be contiguous float32 [blocks,4,512]");
  for (const Tensor* tensor :
       {&token_to_request, &positions, &state_slot_mapping, &kv_slot_mapping}) {
    TORCH_CHECK(tensor->sizes() == torch::IntArrayRef({kTokens}) &&
                    tensor->is_contiguous(),
                "ROOT token metadata must be contiguous [8192]");
  }
  TORCH_CHECK(state_block_table.dim() == 2 &&
                  state_block_table.is_contiguous(),
              "state_block_table must be contiguous int32 [requests,blocks]");
  TORCH_CHECK(rms_weight.sizes() == torch::IntArrayRef({128}) &&
                  rms_weight.is_contiguous(),
              "rms_weight must be contiguous float32 [128]");
  TORCH_CHECK(cos_sin_cache.dim() == 2 && cos_sin_cache.size(0) >= kTokens &&
                  cos_sin_cache.size(1) == 64 && cos_sin_cache.is_contiguous(),
              "cos_sin_cache must be contiguous float32 [>=8192,64]");
  TORCH_CHECK(kv_cache.dim() == 2 && kv_cache.is_contiguous(),
              "kv_cache must be contiguous uint8 [blocks,block_bytes]");
  TORCH_CHECK(ape.sizes() == torch::IntArrayRef({4, 256}) &&
                  ape.is_contiguous(),
              "ape must be contiguous float32 [4,256]");
  TORCH_CHECK(row_ready.numel() <= std::numeric_limits<uint32_t>::max(),
              "row_ready is too large");
  check_same_device(
      hidden,
      {&hidden_fp8, &hidden_scale, &fused_wqa_wkv,
       &fused_wqa_wkv_scale, &qkv, &q_norm_weight, &kv_norm_weight,
       &q_norm, &kv_norm, &q_norm_fp8, &q_norm_scale,
       &projection_barrier, &index_compressor_weight, &index_kv_score,
       &index_weight_projection, &index_weights, &index_barrier,
       &row_ready, &c4_workspace, &state_cache, &token_to_request,
       &positions, &state_slot_mapping, &state_block_table, &rms_weight,
       &cos_sin_cache, &kv_cache, &kv_slot_mapping, &ape});

  const RootFusionParams params = {
      hidden.data_ptr(), hidden_fp8.data_ptr(),
      reinterpret_cast<uint32_t*>(hidden_scale.data_ptr<int32_t>()),
      static_cast<uint32_t>(hidden_scale.stride(1)),
      fused_wqa_wkv.data_ptr(),
      reinterpret_cast<const uint32_t*>(
          fused_wqa_wkv_scale.data_ptr<int32_t>()),
      static_cast<uint32_t>(fused_wqa_wkv_scale.stride(1)),
      qkv.data_ptr(), q_norm_weight.data_ptr(), kv_norm_weight.data_ptr(),
      q_norm.data_ptr(), kv_norm.data_ptr(), q_norm_fp8.data_ptr(),
      reinterpret_cast<uint32_t*>(q_norm_scale.data_ptr<int32_t>()),
      static_cast<uint32_t>(q_norm_scale.stride(1)),
      reinterpret_cast<uint32_t*>(projection_barrier.data_ptr<int32_t>()),
      index_compressor_weight.data_ptr(), index_kv_score.data_ptr<float>(),
      index_weight_projection.data_ptr(), index_weights.data_ptr(),
      reinterpret_cast<uint32_t*>(index_barrier.data_ptr<int32_t>()),
      reinterpret_cast<uint64_t*>(row_ready.data_ptr<int64_t>()),
      static_cast<uint32_t>(row_ready.numel()), c4_workspace.data_ptr<float>(),
      state_cache.data_ptr<float>(), static_cast<uint32_t>(state_cache.stride(0)),
      static_cast<uint32_t>(state_cache.stride(1)),
      token_to_request.data_ptr<int32_t>(), positions.data_ptr<int64_t>(),
      state_slot_mapping.data_ptr<int64_t>(),
      state_block_table.data_ptr<int32_t>(),
      static_cast<uint32_t>(state_block_table.stride(0)),
      rms_weight.data_ptr<float>(), cos_sin_cache.data_ptr<float>(),
      static_cast<uint32_t>(cos_sin_cache.stride(0)),
      kv_cache.data_ptr<uint8_t>(), kv_slot_mapping.data_ptr<int64_t>(),
      static_cast<uint32_t>(kv_cache.stride(0)), ape.data_ptr<float>()};
  c10::cuda::CUDAGuard guard(hidden.device());
  check_launch(
      launch_root_fusion(
          params,
          static_cast<uint32_t>(grid_sms),
          static_cast<uint32_t>(projection_sms),
          stream_for(hidden)),
      "csa_root_fusion");
}
#endif

void root_fusion_unavailable() {
  TORCH_CHECK(
      false,
      "csa_root_fusion is disabled in this extension; rebuild with "
      "FLASH_MLA_BUILD_CSA=1 and an explicitly prepared official "
      "ROOT dependency checkout");
}

#ifdef FLASH_MLA_CSA_WITH_NATIVE_BASELINE
void native_activation_quant_out(
    const Tensor& input, const Tensor& values_out, const Tensor& scales_out) {
  constexpr int64_t kGroupSize = 128;
  check_tensor(input, torch::kBFloat16, "input");
  check_tensor(values_out, torch::kFloat8_e4m3fn, "values_out");
  check_tensor(scales_out, torch::kInt32, "scales_out");
  TORCH_CHECK(input.dim() == 2 && input.size(0) > 0 &&
                  input.size(1) > 0 && input.size(1) % kGroupSize == 0 &&
                  input.is_contiguous(),
              "input must be contiguous BF16 [rows,hidden] with hidden "
              "divisible by 128");
  TORCH_CHECK(values_out.sizes() == input.sizes() &&
                  values_out.is_contiguous(),
              "values_out must be caller-owned contiguous FP8 [rows,hidden]");
  const int64_t rows = input.size(0);
  const int64_t hidden = input.size(1);
  const int64_t packed_columns = (hidden / kGroupSize + 3) / 4;
  const int64_t tma_aligned_rows = (rows + 3) / 4 * 4;
  TORCH_CHECK(scales_out.sizes() ==
                  torch::IntArrayRef({rows, packed_columns}) &&
                  scales_out.stride(0) == 1 &&
                  scales_out.stride(1) == tma_aligned_rows,
              "scales_out must be caller-owned MN-major packed int32 "
              "[rows,ceil(hidden/512)] with TMA-aligned row extent");
  TORCH_CHECK(rows <= std::numeric_limits<int>::max() &&
                  hidden <= std::numeric_limits<int>::max() &&
                  tma_aligned_rows <= std::numeric_limits<int>::max(),
              "native activation quant dimensions exceed int32");
  TORCH_CHECK(
      (reinterpret_cast<std::uintptr_t>(input.data_ptr()) & 15u) == 0 &&
          (reinterpret_cast<std::uintptr_t>(values_out.data_ptr()) & 15u) == 0 &&
          (reinterpret_cast<std::uintptr_t>(scales_out.data_ptr()) & 3u) == 0,
      "native activation quant tensors do not satisfy vector alignment");
  check_same_device(input, {&values_out, &scales_out});
  check_not_alias(input, values_out, "input and values_out must not alias");
  check_not_alias(input, scales_out, "input and scales_out must not alias");
  check_not_alias(values_out, scales_out,
                  "values_out and scales_out must not alias");
  TORCH_CHECK(at::cuda::getDeviceProperties(input.get_device())->major == 10,
              "native activation quant requires an SM100-class CUDA device");
  c10::cuda::CUDAGuard guard(input.device());
  check_launch(
      launch_native_activation_quant_out(
          input.data_ptr(), values_out.data_ptr(), scales_out.data_ptr<int>(),
          static_cast<int>(rows), static_cast<int>(hidden),
          static_cast<int>(tma_aligned_rows), stream_for(input)),
      "native_activation_quant_out");
}

void native_learned_topk(
    const Tensor& logits, const Tensor& row_starts, const Tensor& row_ends,
    const Tensor& indices) {
  check_tensor(logits, torch::kFloat32, "logits");
  check_tensor(row_starts, torch::kInt32, "row_starts");
  check_tensor(row_ends, torch::kInt32, "row_ends");
  check_tensor(indices, torch::kInt32, "indices");
  TORCH_CHECK(logits.dim() == 2, "logits must be float32 [rows, columns]");
  TORCH_CHECK(row_starts.sizes() == torch::IntArrayRef({logits.size(0)}),
              "row_starts must match logits rows");
  TORCH_CHECK(row_ends.sizes() == row_starts.sizes(),
              "row_ends must match row_starts");
  TORCH_CHECK(indices.dim() == 2 && indices.size(0) == logits.size(0),
              "indices must be int32 [rows, top_k]");
  TORCH_CHECK(logits.stride(0) <= std::numeric_limits<int>::max() &&
                  logits.stride(1) <= std::numeric_limits<int>::max() &&
                  logits.size(0) <= std::numeric_limits<int>::max() &&
                  indices.size(1) <= std::numeric_limits<int>::max(),
              "native TopK dimensions exceed int32");
  check_same_device(logits, {&row_starts, &row_ends, &indices});
  c10::cuda::CUDAGuard guard(logits.device());
  check_launch(
      launch_learned_topk(
          logits.data_ptr<float>(), row_starts.data_ptr<int>(),
          row_ends.data_ptr<int>(), indices.data_ptr<int>(),
          static_cast<int>(logits.size(0)), static_cast<int>(logits.stride(0)),
          static_cast<int>(logits.stride(1)), static_cast<int>(indices.size(1)),
          stream_for(logits)),
      "native_learned_topk");
}

void native_attention_prepare(
    const Tensor& query, const Tensor& query_output, const Tensor& key_value,
    const Tensor& cache, const Tensor& slot_mapping, const Tensor& positions,
    const Tensor& cos_sin_cache, double epsilon, int64_t padded_heads,
    int64_t cache_block_size) {
  check_tensor(query, torch::kBFloat16, "query");
  check_tensor(query_output, torch::kBFloat16, "query_output");
  check_tensor(key_value, torch::kBFloat16, "key_value");
  check_tensor(cache, torch::kUInt8, "cache");
  check_tensor(slot_mapping, torch::kInt64, "slot_mapping");
  check_tensor(positions, torch::kInt64, "positions");
  check_tensor(cos_sin_cache, torch::kFloat32, "cos_sin_cache");
  TORCH_CHECK(query.dim() == 3 && query.size(2) == 512 && query.is_contiguous(),
              "query must be contiguous BF16 [tokens, heads, 512]");
  TORCH_CHECK(query_output.dim() == 3 && query_output.size(0) == query.size(0) &&
                  query_output.size(1) == padded_heads &&
                  query_output.size(2) == 512 && query_output.is_contiguous(),
              "query_output must be contiguous BF16 [tokens, padded_heads, 512]");
  TORCH_CHECK(key_value.sizes() == torch::IntArrayRef({query.size(0), 512}) &&
                  key_value.is_contiguous(),
              "key_value must be contiguous BF16 [tokens, 512]");
  TORCH_CHECK(cache.dim() == 2 && cache.is_contiguous(),
              "cache must be contiguous uint8 [blocks, block_bytes]");
  TORCH_CHECK(slot_mapping.dim() == 1 && slot_mapping.is_contiguous() &&
                  positions.sizes() == torch::IntArrayRef({query.size(0)}) &&
                  positions.is_contiguous(),
              "slot_mapping/positions have invalid layouts");
  TORCH_CHECK(cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == 64 &&
                  cos_sin_cache.is_contiguous(),
              "cos_sin_cache must be contiguous float32 [max_position,64]");
  check_same_device(query, {&query_output, &key_value, &cache, &slot_mapping,
                            &positions, &cos_sin_cache});
  c10::cuda::CUDAGuard guard(query.device());
  check_launch(
      launch_attention_prepare(
          query.data_ptr(), query_output.data_ptr(), key_value.data_ptr(),
          cache.data_ptr<uint8_t>(), slot_mapping.data_ptr<int64_t>(),
          positions.data_ptr<int64_t>(), cos_sin_cache.data_ptr<float>(),
          static_cast<float>(epsilon), static_cast<int>(query.size(0)),
          static_cast<int>(slot_mapping.numel()), static_cast<int>(query.size(1)),
          static_cast<int>(padded_heads), static_cast<int>(cache_block_size),
          static_cast<int>(cache.stride(0)), stream_for(query)),
      "native_attention_prepare");
}
#endif

}  // namespace

void persistent_sparse_attention::register_pipeline_apis(
    pybind11::module_& module) {
  register_faithful_leaves(module);
#ifdef FLASH_MLA_CSA_WITH_DEEPGEMM
  flash_mla::csa::deepgemm_adapters::register_apis(module);
#endif
#ifdef FLASH_MLA_CSA_WITH_FLASHMLA_BASELINE
  module.def(
      "native_flashmla_sparse_prefill_out",
      &flash_mla_csa::baseline::flashmla_sparse_prefill_out,
      py::arg("q"), py::arg("kv"), py::arg("indices"), py::arg("sm_scale"),
      py::arg("out"), py::arg("max_logits"), py::arg("lse"),
      py::arg("attn_sink") = py::none(),
      py::arg("topk_length") = py::none());
#endif
#ifdef FLASH_MLA_CSA_WITH_NATIVE_BASELINE
  module.def(
      "native_activation_quant_out", &native_activation_quant_out,
      py::arg("input"), py::arg("values_out"), py::arg("scales_out"));
  module.def("native_learned_topk", &native_learned_topk);
  module.def(
      "native_attention_prepare", &native_attention_prepare,
      py::arg("query"), py::arg("query_output"), py::arg("key_value"),
      py::arg("cache"), py::arg("slot_mapping"), py::arg("positions"),
      py::arg("cos_sin_cache"), py::arg("epsilon") = 1.0e-6,
      py::arg("padded_heads") = 128, py::arg("cache_block_size") = 64);
#endif
  module.def(
      "swa_cache_insert", &swa_cache_insert, py::arg("kv"), py::arg("cache"),
      py::arg("slot_mapping"), py::arg("position_ids"),
      py::arg("cos_sin_cache"), py::arg("cache_block_size") = 64,
      py::arg("num_sms") = 4);
  module.def(
      "index_cache_gather", &index_cache_gather, py::arg("kv_cache"),
      py::arg("dst_k"), py::arg("dst_scale"), py::arg("block_table"),
      py::arg("cu_seq_lens"), py::arg("num_sms") = 148,
      py::arg("vectorized") = true);
  module.def(
      "learned_topk", &learned_topk, py::arg("logits"),
      py::arg("row_starts"), py::arg("row_ends"), py::arg("output"),
      py::arg("grid_size") = 148);
  module.def(
      "combine_indices", &combine_indices, py::arg("topk"),
      py::arg("combined"), py::arg("lengths"), py::arg("compressed_rows"),
      py::arg("num_sms") = 148);
  module.def(
      "attention_output_quant", &attention_output_quant, py::arg("input"),
      py::arg("positions"), py::arg("cos_sin_cache"), py::arg("output"),
      py::arg("scales"), py::arg("num_sms") = 68);
  module.def(
      "output_quant", &output_quant, py::arg("input"), py::arg("output"),
      py::arg("scales"), py::arg("num_sms") = 148);
  module.def(
      "reset_ready", &reset_ready_three, py::arg("counters"),
      py::arg("ready"), py::arg("producer_clocks"),
      py::arg("num_sms") = 148);
  module.def(
      "reset_ready", &reset_ready_five, py::arg("counters"),
      py::arg("ready"), py::arg("producer_clocks"),
      py::arg("consumer_clocks"), py::arg("qnorm_clocks"),
      py::arg("num_sms") = 148);
#ifdef FLASH_MLA_CSA_WITH_ROOT_FUSION
  module.def(
      "csa_root_fusion", &root_fusion,
      py::arg("hidden"), py::arg("hidden_fp8"), py::arg("hidden_scale"),
      py::arg("fused_wqa_wkv"), py::arg("fused_wqa_wkv_scale"),
      py::arg("qkv"), py::arg("q_norm_weight"), py::arg("kv_norm_weight"),
      py::arg("q_norm"), py::arg("kv_norm"), py::arg("q_norm_fp8"),
      py::arg("q_norm_scale"), py::arg("projection_barrier"),
      py::arg("index_compressor_weight"), py::arg("index_kv_score"),
      py::arg("index_weight_projection"), py::arg("index_weights"),
      py::arg("index_barrier"), py::arg("row_ready"),
      py::arg("c4_workspace"), py::arg("state_cache"),
      py::arg("token_to_request"), py::arg("positions"),
      py::arg("state_slot_mapping"), py::arg("state_block_table"),
      py::arg("rms_weight"), py::arg("cos_sin_cache"),
      py::arg("kv_cache"), py::arg("kv_slot_mapping"), py::arg("ape"),
      py::arg("grid_sms") = 148,
      py::arg("projection_sms") = 104);
#else
  module.def(
      "csa_root_fusion",
      [](py::args, py::kwargs) { root_fusion_unavailable(); });
#endif
  module.def(
      "state_save", &state_save, py::arg("kv"), py::arg("score"),
      py::arg("ape"), py::arg("positions"), py::arg("state_cache"),
      py::arg("slot_mapping"), py::arg("block_size") = 4,
      py::arg("num_sms") = 108);
  module.def(
      "main_compress", &main_compress, py::arg("state_cache"),
      py::arg("token_to_req"), py::arg("positions"),
      py::arg("state_slot_mapping"), py::arg("state_block_table"),
      py::arg("rms_weight"), py::arg("rms_norm_eps"),
      py::arg("cos_sin_cache"), py::arg("kv_cache"),
      py::arg("kv_slot_mapping"), py::arg("num_sms") = 108);
}
