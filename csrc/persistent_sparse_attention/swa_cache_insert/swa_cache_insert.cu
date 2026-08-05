// SPDX-License-Identifier: Apache-2.0
// Derived from accepted CSA K3; vendored to avoid source-tree dependencies.
#include "swa_cache_insert.h"

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace {

constexpr int32_t kHeadDim = 512;
constexpr int32_t kRopeDim = 64;
constexpr int32_t kNopeDim = kHeadDim - kRopeDim;
constexpr int32_t kElemsPerLane = kHeadDim / 32;
constexpr int32_t kQuantBlock = 64;
constexpr int32_t kNumQuantBlocks = kNopeDim / kQuantBlock;
constexpr int32_t kScaleBytesPerToken = kNumQuantBlocks + 1;
constexpr int32_t kTokenDataBytes = kNopeDim + kRopeDim * 2;
constexpr float kFp8Max = 448.0f;

// Warp 0 owns scheduling and prefetches metadata. Warps 1-31 each process
// one complete token. A 1024-thread CTA fills the SM while preserving the
// fixed-grid one-CTA-per-SM contract needed by the four-SM Subwave lane.
constexpr int32_t kNumThreads = 1024;
constexpr int32_t kFirstConsumerWarp = 1;
constexpr int32_t kNumConsumerWarps = 31;

struct alignas(16) TaskDescriptor {
  int64_t slot_id;
  int64_t position;
  int32_t token_idx;
  int32_t valid;
};

__device__ __forceinline__ float warp4_max_abs(float value) {
  float peer = __shfl_xor_sync(0xffffffffu, value, 1);
  value = fmaxf(value, peer);
  peer = __shfl_xor_sync(0xffffffffu, value, 2);
  return fmaxf(value, peer);
}

__device__ __forceinline__ void process_token(
    __nv_bfloat16 const* __restrict__ kv, int64_t const kv_stride0,
    uint8_t* __restrict__ cache, int64_t const cache_stride0,
    float const* __restrict__ cos_sin_cache,
    int64_t const cos_sin_stride0, TaskDescriptor const task,
    int32_t const cache_block_size, int32_t const lane_idx) {
  if (task.valid == 0 or task.slot_id < 0) {
    return;
  }

  int32_t const dim_base = lane_idx * kElemsPerLane;
  auto const src =
      kv + static_cast<int64_t>(task.token_idx) * kv_stride0 + dim_base;
  uint4 const input0 = *reinterpret_cast<uint4 const*>(src);
  uint4 const input1 = *reinterpret_cast<uint4 const*>(src + 8);

  float elements[kElemsPerLane];
  auto const packed0 = reinterpret_cast<__nv_bfloat162 const*>(&input0);
  auto const packed1 = reinterpret_cast<__nv_bfloat162 const*>(&input1);
#pragma unroll
  for (int32_t idx = 0; idx < 4; ++idx) {
    float2 const value = __bfloat1622float2(packed0[idx]);
    elements[2 * idx] = value.x;
    elements[2 * idx + 1] = value.y;
  }
#pragma unroll
  for (int32_t idx = 0; idx < 4; ++idx) {
    float2 const value = __bfloat1622float2(packed1[idx]);
    elements[8 + 2 * idx] = value.x;
    elements[8 + 2 * idx + 1] = value.y;
  }

  bool const is_rope_lane = dim_base >= kNopeDim;
  if (is_rope_lane) {
    constexpr int32_t kHalfRope = kRopeDim / 2;
    auto const cos_ptr =
        cos_sin_cache + task.position * cos_sin_stride0;
    auto const sin_ptr = cos_ptr + kHalfRope;
    int32_t const rope_local_base = dim_base - kNopeDim;
    int32_t const half_base = rope_local_base >> 1;
    float4 const cos0 =
        *reinterpret_cast<float4 const*>(cos_ptr + half_base);
    float4 const cos1 =
        *reinterpret_cast<float4 const*>(cos_ptr + half_base + 4);
    float4 const sin0 =
        *reinterpret_cast<float4 const*>(sin_ptr + half_base);
    float4 const sin1 =
        *reinterpret_cast<float4 const*>(sin_ptr + half_base + 4);
    float const cos_values[8] = {
        cos0.x, cos0.y, cos0.z, cos0.w, cos1.x, cos1.y, cos1.z, cos1.w};
    float const sin_values[8] = {
        sin0.x, sin0.y, sin0.z, sin0.w, sin1.x, sin1.y, sin1.z, sin1.w};
#pragma unroll
    for (int32_t pair_idx = 0; pair_idx < kElemsPerLane / 2; ++pair_idx) {
      float const even = elements[2 * pair_idx];
      float const odd = elements[2 * pair_idx + 1];
      elements[2 * pair_idx] =
          even * cos_values[pair_idx] - odd * sin_values[pair_idx];
      elements[2 * pair_idx + 1] =
          even * sin_values[pair_idx] + odd * cos_values[pair_idx];
    }
  }

  // Match processDeepseekV4Slot exactly: every value takes a BF16
  // round-trip before block-scale selection and cache storage.
#pragma unroll
  for (int32_t idx = 0; idx < kElemsPerLane; ++idx) {
    elements[idx] = __bfloat162float(__float2bfloat16(elements[idx]));
  }

  int64_t const block_idx = task.slot_id / cache_block_size;
  int64_t const position_in_block = task.slot_id % cache_block_size;
  auto const block_base = cache + block_idx * cache_stride0;
  auto const token_fp8 =
      block_base + position_in_block * kTokenDataBytes;
  auto const token_bf16 = token_fp8 + kNopeDim;
  auto const token_scale =
      block_base + static_cast<int64_t>(cache_block_size) * kTokenDataBytes +
      position_in_block * kScaleBytesPerToken;

  float local_absmax = 0.0f;
#pragma unroll
  for (int32_t idx = 0; idx < kElemsPerLane; ++idx) {
    local_absmax = fmaxf(local_absmax, fabsf(elements[idx]));
  }
  float const absmax = fmaxf(warp4_max_abs(local_absmax), 1.0e-4f);
  float const exponent = ceilf(log2f(absmax / kFp8Max));
  float const inv_scale = exp2f(-exponent);

  if (not is_rope_lane) {
    alignas(16) uint8_t output[kElemsPerLane];
#pragma unroll
    for (int32_t idx = 0; idx < kElemsPerLane; ++idx) {
      float scaled = elements[idx] * inv_scale;
      scaled = fminf(fmaxf(scaled, -kFp8Max), kFp8Max);
      __nv_fp8_storage_t const fp8 =
          __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
      output[idx] = static_cast<uint8_t>(fp8);
    }
    *reinterpret_cast<uint4*>(token_fp8 + dim_base) =
        *reinterpret_cast<uint4 const*>(output);

    if ((lane_idx & 3) == 0) {
      int32_t const quant_block_idx = lane_idx >> 2;
      float const encoded =
          fmaxf(fminf(exponent + 127.0f, 255.0f), 0.0f);
      token_scale[quant_block_idx] = static_cast<uint8_t>(encoded);
    }
    if (lane_idx == 0) {
      token_scale[kNumQuantBlocks] = 0;
    }
  } else {
    uint4 output0;
    uint4 output1;
    auto const packed_output0 =
        reinterpret_cast<__nv_bfloat162*>(&output0);
    auto const packed_output1 =
        reinterpret_cast<__nv_bfloat162*>(&output1);
#pragma unroll
    for (int32_t idx = 0; idx < 4; ++idx) {
      packed_output0[idx] = __float22bfloat162_rn(
          make_float2(elements[2 * idx], elements[2 * idx + 1]));
    }
#pragma unroll
    for (int32_t idx = 0; idx < 4; ++idx) {
      packed_output1[idx] = __float22bfloat162_rn(
          make_float2(elements[8 + 2 * idx], elements[8 + 2 * idx + 1]));
    }
    int32_t const rope_local_base = dim_base - kNopeDim;
    auto const dst =
        reinterpret_cast<__nv_bfloat16*>(token_bf16) + rope_local_base;
    *reinterpret_cast<uint4*>(dst) = output0;
    *reinterpret_cast<uint4*>(dst + 8) = output1;
  }
}

__global__ __launch_bounds__(kNumThreads, 1)
void swa_cache_insert_kernel(
    __nv_bfloat16 const* __restrict__ kv, int64_t const kv_stride0,
    uint8_t* __restrict__ cache, int64_t const cache_stride0,
    int64_t const* __restrict__ slot_mapping,
    int64_t const slot_stride0,
    int64_t const* __restrict__ position_ids,
    int64_t const position_stride0,
    float const* __restrict__ cos_sin_cache,
    int64_t const cos_sin_stride0, int32_t const num_tokens,
    int32_t const cache_block_size) {
  // The producer fills the next descriptor bank while consumers process the
  // current one. The single barrier at each bank handoff both publishes the
  // next bank and proves the old bank is safe to reuse.
  __shared__ TaskDescriptor tasks[2][kNumConsumerWarps];

  int32_t const warp_idx = static_cast<int32_t>(threadIdx.x) >> 5;
  int32_t const lane_idx = static_cast<int32_t>(threadIdx.x) & 31;
  int32_t const consumer_idx = warp_idx - kFirstConsumerWarp;

  int64_t task_base =
      static_cast<int64_t>(blockIdx.x) * kNumConsumerWarps;
  int64_t const task_stride =
      static_cast<int64_t>(gridDim.x) * kNumConsumerWarps;

  if (task_base >= num_tokens) {
    return;
  }

  // Prime bank zero before entering the overlapped work loop.
  if (warp_idx == 0 and lane_idx < kNumConsumerWarps) {
    int64_t const token_idx = task_base + lane_idx;
    TaskDescriptor task;
    task.valid = token_idx < num_tokens;
    task.token_idx = static_cast<int32_t>(token_idx);
    if (task.valid != 0) {
      task.slot_id = slot_mapping[token_idx * slot_stride0];
      task.position = position_ids[token_idx * position_stride0];
    } else {
      task.slot_id = -1;
      task.position = 0;
    }
    tasks[0][lane_idx] = task;
  }
  __syncthreads();

  int32_t task_bank = 0;
  while (true) {
    int64_t const next_task_base = task_base + task_stride;
    int32_t const next_task_bank = task_bank ^ 1;

    // Metadata producer: prefetch the next work round into the bank
    // consumers are not using. Its loads overlap the current token work.
    if (next_task_base < num_tokens and warp_idx == 0 and
        lane_idx < kNumConsumerWarps) {
      int64_t const token_idx = next_task_base + lane_idx;
      TaskDescriptor task;
      task.valid = token_idx < num_tokens;
      task.token_idx = static_cast<int32_t>(token_idx);
      if (task.valid != 0) {
        task.slot_id = slot_mapping[token_idx * slot_stride0];
        task.position = position_ids[token_idx * position_stride0];
      } else {
        task.slot_id = -1;
        task.position = 0;
      }
      tasks[next_task_bank][lane_idx] = task;
    }

    if (consumer_idx >= 0 and consumer_idx < kNumConsumerWarps) {
      process_token(kv, kv_stride0, cache, cache_stride0, cos_sin_cache,
                    cos_sin_stride0, tasks[task_bank][consumer_idx],
                    cache_block_size, lane_idx);
    }

    if (next_task_base >= num_tokens) {
      return;
    }

    // Publish the prefetched bank and release the current bank for reuse.
    __syncthreads();
    task_base = next_task_base;
    task_bank = next_task_bank;
  }
}

}  // namespace

void launch_swa_cache_insert(
    void const* kv, int64_t const kv_stride0, uint8_t* cache,
    int64_t const cache_stride0, int64_t const* slot_mapping,
    int64_t const slot_stride0, int64_t const* position_ids,
    int64_t const position_stride0, float const* cos_sin_cache,
    int64_t const cos_sin_stride0, int32_t const num_tokens,
    int32_t const cache_block_size, int32_t const num_sms,
    cudaStream_t const stream) {
  swa_cache_insert_kernel<<<num_sms, kNumThreads, 0, stream>>>(
      static_cast<__nv_bfloat16 const*>(kv), kv_stride0, cache, cache_stride0,
      slot_mapping, slot_stride0, position_ids, position_stride0,
      cos_sin_cache, cos_sin_stride0, num_tokens, cache_block_size);
}
