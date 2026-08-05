// SPDX-License-Identifier: Apache-2.0
// Derived from the manifest-selected accepted K11 implementation. The final
// all-consumer and exact-scale branches are intentionally baked into this TU.

/*
 * Inverse-RoPE + E4M3/UE8M0 output quantization for DSv4/B200.
 *
 * CTA topology (1024 threads):
 *   WG0 warp 0: the only task/position/cos-sin producer.
 *   WG0 warps 1-3: deliberately register-light and otherwise idle.
 *   WG1-WG7: 28 consumer warps.  A consumer owns heads
 *            consumer_idx + n * 28 for one complete token.
 *
 * A task is one complete aligned token.  On the production vector path, the
 * producer prefetches positions/cos-sin into L2 while consumers load the same
 * token metadata directly into registers; there is no per-token handoff or
 * spin.  The generic positive-stride fallback retains two shared stages.  CTAs
 * walk tokens in a grid-stride loop and launch exactly num_sms blocks.
 */

#include "attention_output_quant.h"

#include <cstdint>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace {

constexpr int32_t kNumThreads = 1024;
constexpr int32_t kWarpSize = 32;
constexpr int32_t kFirstConsumerWarp = 4;
constexpr int32_t kNumConsumerWarps = 28;
constexpr int32_t kNumPipelineStages = 2;

constexpr int32_t kNumHeads = 128;
constexpr int32_t kNumGroups = 16;
constexpr int32_t kHeadsPerGroup = 8;
constexpr int32_t kHeadDim = 512;
constexpr int32_t kNopeDim = 448;
constexpr int32_t kRopeDim = 64;
constexpr int32_t kHalfRope = kRopeDim / 2;
constexpr int32_t kQuantGroupSize = 128;
constexpr int32_t kChunksPerHead = kHeadDim / kQuantGroupSize;
constexpr int32_t kBaseHeadsPerConsumer =
    kNumHeads / kNumConsumerWarps;
constexpr int32_t kNumRotatingExtraHeads =
    kNumHeads % kNumConsumerWarps;
constexpr float kFp8Max = 448.0f;
constexpr float kScaleEpsilon = 1.0e-10f;

static_assert(kNumHeads == kNumGroups * kHeadsPerGroup);
static_assert(kChunksPerHead == 4);
static_assert(kBaseHeadsPerConsumer == 4);
static_assert(kNumRotatingExtraHeads == 16);
static_assert(kNumThreads == 8 * 128);
static_assert(kFirstConsumerWarp + kNumConsumerWarps == 32);

struct alignas(16) TaskDescriptor {
  int64_t position;
  int32_t token_idx;
  int32_t valid_token;
};

struct alignas(16) PipelineStorage {
  TaskDescriptor descriptors[kNumPipelineStages];
  float cos_sin[kNumPipelineStages][kRopeDim];
  uint32_t ready_generation[kNumPipelineStages];
  uint32_t done_count[kNumPipelineStages];
};

__device__ __forceinline__ float multiply_rn(
    float const lhs, float const rhs) {
  float result;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(lhs), "f"(rhs));
  return result;
}

__device__ __forceinline__ float fma_rn(
    float const lhs, float const rhs, float const addend) {
  float result;
  asm("fma.rn.f32 %0, %1, %2, %3;"
      : "=f"(result)
      : "f"(lhs), "f"(rhs), "f"(addend));
  return result;
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int32_t offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(
        value,
        __shfl_xor_sync(0xffffffffu, value, offset));
  }
  return value;
}

__device__ __forceinline__ uint32_t atomic_load_shared(
    uint32_t* const address) {
  return atomicAdd(address, 0u);
}

__device__ __forceinline__ void wait_until_equal(
    uint32_t* const address, uint32_t const expected) {
  while (atomic_load_shared(address) != expected) {
    __nanosleep(32);
  }
}

template <bool kVectorized>
__device__ __forceinline__ void process_head(
    __nv_bfloat16 const* __restrict__ o,
    uint8_t* __restrict__ fp8_output,
    int32_t* __restrict__ scale_output,
    TaskDescriptor const task,
    float const* const cos_sin,
    float const vector_cosine_0,
    float const vector_cosine_1,
    float const vector_sine_0,
    float const vector_sine_1,
    int64_t const o_stride_token,
    int64_t const o_stride_head,
    int64_t const o_stride_dim,
    int64_t const fp8_stride_token,
    int64_t const fp8_stride_group,
    int64_t const fp8_stride_dim,
    int64_t const scale_stride_token,
    int64_t const scale_stride_group,
    int64_t const scale_stride_head,
    int32_t const global_head_idx,
    int32_t const lane_idx) {
  int32_t const group_idx = global_head_idx / kHeadsPerGroup;
  int32_t const head_in_group_idx =
      global_head_idx % kHeadsPerGroup;

  int64_t const scale_offset =
      static_cast<int64_t>(task.token_idx) * scale_stride_token +
      static_cast<int64_t>(group_idx) * scale_stride_group +
      static_cast<int64_t>(head_in_group_idx) * scale_stride_head;

  if (task.valid_token == 0) {
    if (lane_idx == 0) {
      scale_output[scale_offset] = 0;
    }
    return;
  }

  // The production fast path assigns four adjacent elements to each lane, so
  // one uint64 load moves four BF16 values and one uint32 store moves four FP8
  // bytes.  The generic path preserves support for arbitrary positive input
  // dimension strides.
  float values[kChunksPerHead][4];
  int64_t const input_base =
      static_cast<int64_t>(task.token_idx) * o_stride_token +
      static_cast<int64_t>(global_head_idx) * o_stride_head;
#pragma unroll
  for (int32_t chunk_idx = 0;
       chunk_idx < kChunksPerHead;
       ++chunk_idx) {
    if constexpr (kVectorized) {
      int32_t const first_dim_idx =
          chunk_idx * kQuantGroupSize + lane_idx * 4;
      uint64_t const packed = *reinterpret_cast<uint64_t const*>(
          o + input_base + first_dim_idx);
#pragma unroll
      for (int32_t element_idx = 0; element_idx < 4; ++element_idx) {
        uint16_t const bits = static_cast<uint16_t>(
            packed >> (element_idx * 16));
        values[chunk_idx][element_idx] = __bfloat162float(
            __ushort_as_bfloat16(bits));
      }
    } else {
#pragma unroll
      for (int32_t quarter_idx = 0; quarter_idx < 4; ++quarter_idx) {
        int32_t const dim_idx =
            chunk_idx * kQuantGroupSize +
            quarter_idx * kWarpSize + lane_idx;
        values[chunk_idx][quarter_idx] = __bfloat162float(
            o[input_base +
              static_cast<int64_t>(dim_idx) * o_stride_dim]);
      }
    }
  }

  // The 64-value RoPE tail is the upper half of the last quant group.
  if constexpr (kVectorized) {
    if (lane_idx >= 16) {
#pragma unroll
      for (int32_t pair_idx = 0; pair_idx < 2; ++pair_idx) {
        int32_t const even_element_idx = pair_idx * 2;
        int32_t const odd_element_idx = even_element_idx + 1;
        float const even_value = values[3][even_element_idx];
        float const odd_value = values[3][odd_element_idx];
        float const cosine =
            pair_idx == 0 ? vector_cosine_0 : vector_cosine_1;
        float const sine =
            pair_idx == 0 ? vector_sine_0 : vector_sine_1;
        values[3][even_element_idx] = fma_rn(
            even_value, cosine, multiply_rn(odd_value, sine));
        values[3][odd_element_idx] = fma_rn(
            odd_value, cosine, -multiply_rn(even_value, sine));
      }
    }
  } else {
#pragma unroll
    for (int32_t quarter_idx = 2; quarter_idx < 4; ++quarter_idx) {
      int32_t const dim_idx =
          3 * kQuantGroupSize + quarter_idx * kWarpSize + lane_idx;
      int32_t const rope_local_idx = dim_idx - kNopeDim;
      int32_t const partner_dim_idx = dim_idx ^ 1;
      float const partner = __bfloat162float(
          o[input_base +
            static_cast<int64_t>(partner_dim_idx) * o_stride_dim]);
      float const cosine = cos_sin[rope_local_idx >> 1];
      float const sine =
          cos_sin[kHalfRope + (rope_local_idx >> 1)];
      float const partner_sine = multiply_rn(partner, sine);
      values[3][quarter_idx] =
          (rope_local_idx & 1) == 0
          ? fma_rn(values[3][quarter_idx], cosine, partner_sine)
          : fma_rn(values[3][quarter_idx], cosine, -partner_sine);
    }
  }

  float scales[kChunksPerHead];
  float inverse_scales[kChunksPerHead];
#pragma unroll
  for (int32_t chunk_idx = 0;
       chunk_idx < kChunksPerHead;
       ++chunk_idx) {
    float local_absmax = 0.0f;
#pragma unroll
    for (int32_t quarter_idx = 0; quarter_idx < 4; ++quarter_idx) {
      local_absmax =
          fmaxf(local_absmax, fabsf(values[chunk_idx][quarter_idx]));
    }
    scales[chunk_idx] =
        fmaxf(warp_max(local_absmax), kScaleEpsilon);
  }

  // The warp reductions already broadcast all four absmax values.  Assign
  // one scale exponent to each of lanes 0..3 instead of redundantly issuing
  // four log2/ceil/exp2 chains on all 32 lanes.
  float owned_scale = 0.0f;
  float owned_inverse_scale = 0.0f;
#pragma unroll
  for (int32_t chunk_idx = 0;
       chunk_idx < kChunksPerHead;
       ++chunk_idx) {
    if (lane_idx == chunk_idx) {
      float const scale_raw =
          multiply_rn(scales[chunk_idx], 1.0f / kFp8Max);
      // Preserve the exact log2/ceil boundary decision, then
      // construct the two exact powers of two directly from exponent bits.
      // This removes both exp2 SFU operations without changing scale values.
      float const scale_exponent_float = ceilf(log2f(scale_raw));
      int32_t const scale_exponent =
          static_cast<int32_t>(scale_exponent_float);
      owned_scale = __uint_as_float(
          static_cast<uint32_t>(scale_exponent + 127) << 23);
      owned_inverse_scale = __uint_as_float(
          static_cast<uint32_t>(127 - scale_exponent) << 23);
    }
  }
#pragma unroll
  for (int32_t chunk_idx = 0;
       chunk_idx < kChunksPerHead;
       ++chunk_idx) {
    scales[chunk_idx] = __shfl_sync(
        0xffffffffu, owned_scale, chunk_idx);
    inverse_scales[chunk_idx] = __shfl_sync(
        0xffffffffu, owned_inverse_scale, chunk_idx);
  }

  int64_t const fp8_base =
      static_cast<int64_t>(task.token_idx) * fp8_stride_token +
      static_cast<int64_t>(group_idx) * fp8_stride_group +
      static_cast<int64_t>(head_in_group_idx * kHeadDim) *
          fp8_stride_dim;
#pragma unroll
  for (int32_t chunk_idx = 0;
       chunk_idx < kChunksPerHead;
       ++chunk_idx) {
    if constexpr (kVectorized) {
      uint32_t packed_fp8 = 0;
#pragma unroll
      for (int32_t pair_idx = 0; pair_idx < 2; ++pair_idx) {
        int32_t const element_idx = pair_idx * 2;
        float scaled_0 = multiply_rn(
            values[chunk_idx][element_idx],
            inverse_scales[chunk_idx]);
        float scaled_1 = multiply_rn(
            values[chunk_idx][element_idx + 1],
            inverse_scales[chunk_idx]);
        scaled_0 = fminf(fmaxf(scaled_0, -kFp8Max), kFp8Max);
        scaled_1 = fminf(fmaxf(scaled_1, -kFp8Max), kFp8Max);
        // CUDA's packed intrinsic applies the same RNE/SATFINITE E4M3
        // conversion as two scalar calls and returns x in the low byte and y
        // in the high byte.  It therefore preserves the raw FP8 contract
        // while issuing one x2 conversion instead of two zero-padded x2
        // conversions.
        __nv_fp8x2_storage_t const fp8_pair =
            __nv_cvt_float2_to_fp8x2(
                make_float2(scaled_0, scaled_1),
                __NV_SATFINITE,
                __NV_E4M3);
        packed_fp8 |= static_cast<uint32_t>(fp8_pair) <<
            (pair_idx * 16);
      }
      int32_t const first_dim_idx =
          chunk_idx * kQuantGroupSize + lane_idx * 4;
      *reinterpret_cast<uint32_t*>(
          fp8_output + fp8_base + first_dim_idx) = packed_fp8;
    } else {
#pragma unroll
      for (int32_t quarter_idx = 0; quarter_idx < 4; ++quarter_idx) {
        int32_t const dim_idx =
            chunk_idx * kQuantGroupSize +
            quarter_idx * kWarpSize + lane_idx;
        float scaled = multiply_rn(
            values[chunk_idx][quarter_idx],
            inverse_scales[chunk_idx]);
        scaled = fminf(fmaxf(scaled, -kFp8Max), kFp8Max);
        __nv_fp8_storage_t const fp8 = __nv_cvt_float_to_fp8(
            scaled, __NV_SATFINITE, __NV_E4M3);
        fp8_output[
            fp8_base +
            static_cast<int64_t>(dim_idx) * fp8_stride_dim] =
            static_cast<uint8_t>(fp8);
      }
    }
  }

  if (lane_idx == 0) {
    uint32_t packed_scale = 0;
#pragma unroll
    for (int32_t chunk_idx = 0;
         chunk_idx < kChunksPerHead;
         ++chunk_idx) {
      uint32_t const ue8m0 =
          (__float_as_uint(scales[chunk_idx]) >> 23) & 0xffu;
      packed_scale |= ue8m0 << (chunk_idx * 8);
    }
    scale_output[scale_offset] =
        static_cast<int32_t>(packed_scale);
  }
}

template <bool kVectorized>
__global__ __launch_bounds__(kNumThreads, 1)
void attention_output_quant_kernel(
    __nv_bfloat16 const* __restrict__ o,
    int64_t const* __restrict__ positions,
    float const* __restrict__ cos_sin_cache,
    uint8_t* __restrict__ fp8_output,
    int32_t* __restrict__ scale_output,
    int32_t const num_tokens,
    int64_t const o_stride_token,
    int64_t const o_stride_head,
    int64_t const o_stride_dim,
    int64_t const position_stride,
    int64_t const cache_stride_position,
    int64_t const cache_stride_dim,
    int64_t const fp8_stride_token,
    int64_t const fp8_stride_group,
    int64_t const fp8_stride_dim,
    int64_t const scale_stride_token,
    int64_t const scale_stride_group,
    int64_t const scale_stride_head) {
  __shared__ PipelineStorage pipeline;

  int32_t const thread_idx = static_cast<int32_t>(threadIdx.x);
  int32_t const warp_idx = thread_idx / kWarpSize;
  int32_t const lane_idx = thread_idx % kWarpSize;

  if (thread_idx == 0) {
#pragma unroll
    for (int32_t stage_idx = 0;
         stage_idx < kNumPipelineStages;
         ++stage_idx) {
      pipeline.ready_generation[stage_idx] = 0;
      pipeline.done_count[stage_idx] = kNumConsumerWarps;
    }
  }
  __syncthreads();

  bool const is_producer = warp_idx == 0;
  bool const is_consumer =
      warp_idx >= kFirstConsumerWarp and
      warp_idx < kFirstConsumerWarp + kNumConsumerWarps;
  if (not is_producer and not is_consumer) {
    return;
  }

  int32_t const aligned_tokens = (num_tokens + 3) & ~3;
  int32_t token_idx = static_cast<int32_t>(blockIdx.x);
  uint32_t local_generation = 1;

  if (is_producer) {
    for (; token_idx < aligned_tokens;
         token_idx += static_cast<int32_t>(gridDim.x),
         ++local_generation) {
      int32_t const stage_idx =
          static_cast<int32_t>((local_generation - 1) & 1u);
      if (lane_idx == 0) {
        wait_until_equal(
            pipeline.done_count + stage_idx,
            kNumConsumerWarps);
        atomicExch(pipeline.done_count + stage_idx, 0u);
      }
      __syncwarp();

      bool const valid_token = token_idx < num_tokens;

      if (lane_idx == 0) {
        TaskDescriptor descriptor;
        descriptor.position = valid_token
            ? positions[
                  static_cast<int64_t>(token_idx) *
                  position_stride]
            : 0;
        descriptor.token_idx = token_idx;
        descriptor.valid_token = valid_token ? 1 : 0;
        pipeline.descriptors[stage_idx] = descriptor;
      }
      __syncwarp();
      // Lane 0 performs the only global position load.  The other producer
      // lanes consume the position from the published shared descriptor.
      int64_t const position =
          pipeline.descriptors[stage_idx].position;
      pipeline.cos_sin[stage_idx][lane_idx] = valid_token
          ? cos_sin_cache[
                position * cache_stride_position +
                static_cast<int64_t>(lane_idx) *
                    cache_stride_dim]
          : 0.0f;
      pipeline.cos_sin[stage_idx][lane_idx + kWarpSize] =
          valid_token
          ? cos_sin_cache[
                position * cache_stride_position +
                static_cast<int64_t>(lane_idx + kWarpSize) *
                    cache_stride_dim]
          : 0.0f;

      __syncwarp();
      __threadfence_block();
      if (lane_idx == 0) {
        atomicExch(
            pipeline.ready_generation + stage_idx,
            local_generation);
      }
    }
    return;
  }

  int32_t const consumer_warp_idx =
      warp_idx - kFirstConsumerWarp;
  for (; token_idx < aligned_tokens;
       token_idx += static_cast<int32_t>(gridDim.x),
       ++local_generation) {
    int32_t const stage_idx =
        static_cast<int32_t>((local_generation - 1) & 1u);
    if (lane_idx == 0) {
      wait_until_equal(
          pipeline.ready_generation + stage_idx,
          local_generation);
    }
    __syncwarp();

    TaskDescriptor const descriptor =
        pipeline.descriptors[stage_idx];
    for (int32_t global_head_idx = consumer_warp_idx;
         global_head_idx < kNumHeads;
         global_head_idx += kNumConsumerWarps) {
      process_head<kVectorized>(
          o, fp8_output, scale_output, descriptor,
          pipeline.cos_sin[stage_idx], 0.0f, 0.0f, 0.0f, 0.0f,
          o_stride_token, o_stride_head,
          o_stride_dim, fp8_stride_token, fp8_stride_group,
          fp8_stride_dim, scale_stride_token, scale_stride_group,
          scale_stride_head, global_head_idx, lane_idx);
    }

    __syncwarp();
    if (lane_idx == 0) {
      atomicAdd(pipeline.done_count + stage_idx, 1u);
    }
  }
}

__device__ __forceinline__ void prefetch_global_l2(
    void const* const address) {
#if defined(__CUDA_ARCH__)
  asm volatile(
      "prefetch.global.L2 [%0];\n" :: "l"(address) : "memory");
#endif
}

// Fixed-shape path: all 32 warps are consumers. Each owns exactly four heads,
// eliminating tail-head rotation and per-token shared handoff on the hot path.
template <int32_t kGridSize>
__global__ __maxnreg__(56)
void attention_output_quant_vector_kernel(
    __nv_bfloat16 const* __restrict__ o,
    int64_t const* __restrict__ positions,
    float const* __restrict__ cos_sin_cache,
    uint8_t* __restrict__ fp8_output,
    int32_t* __restrict__ scale_output,
    int32_t const num_tokens,
    int64_t const o_stride_token,
    int64_t const o_stride_head,
    int64_t const o_stride_dim,
    int64_t const position_stride,
    int64_t const cache_stride_position,
    int64_t const cache_stride_dim,
    int64_t const fp8_stride_token,
    int64_t const fp8_stride_group,
    int64_t const fp8_stride_dim,
    int64_t const scale_stride_token,
    int64_t const scale_stride_group,
    int64_t const scale_stride_head) {
  static_assert(
      kGridSize == 40 or kGridSize == 64 or kGridSize == 68 or
      kGridSize == 84 or kGridSize == 108 or kGridSize == 148);
  constexpr int32_t kVectorFirstConsumerWarp = 0;
  constexpr int32_t kVectorNumConsumerWarps = 32;
  constexpr int32_t kVectorBaseHeadsPerConsumer = 4;
  constexpr int32_t kVectorNumRotatingExtraHeads = 0;

  int32_t const thread_idx = static_cast<int32_t>(threadIdx.x);
  int32_t const warp_idx = thread_idx / kWarpSize;
  int32_t const lane_idx = thread_idx % kWarpSize;
  int32_t const aligned_tokens = (num_tokens + 3) & ~3;

  int32_t const consumer_warp_idx =
      warp_idx - kVectorFirstConsumerWarp;
  constexpr int32_t rotating_extra_head = 0;
  for (int32_t token_idx = static_cast<int32_t>(blockIdx.x);
       token_idx < aligned_tokens;
       token_idx += kGridSize) {
    bool const valid_token = token_idx < num_tokens;
    int64_t position = 0;
    if (valid_token and lane_idx == 0) {
      position = positions[
          static_cast<int64_t>(token_idx) * position_stride];
    }
    position = __shfl_sync(0xffffffffu, position, 0);

    float vector_cosine_0 = 0.0f;
    float vector_cosine_1 = 0.0f;
    float vector_sine_0 = 0.0f;
    float vector_sine_1 = 0.0f;
    if (valid_token and lane_idx >= 16) {
      int32_t const first_pair_idx = (lane_idx - 16) * 2;
      int64_t const cache_base = position * cache_stride_position;
      vector_cosine_0 = cos_sin_cache[
          cache_base +
          static_cast<int64_t>(first_pair_idx) * cache_stride_dim];
      vector_cosine_1 = cos_sin_cache[
          cache_base +
          static_cast<int64_t>(first_pair_idx + 1) * cache_stride_dim];
      vector_sine_0 = cos_sin_cache[
          cache_base +
          static_cast<int64_t>(kHalfRope + first_pair_idx) *
              cache_stride_dim];
      vector_sine_1 = cos_sin_cache[
          cache_base +
          static_cast<int64_t>(kHalfRope + first_pair_idx + 1) *
              cache_stride_dim];
    }

    TaskDescriptor descriptor;
    descriptor.position = position;
    descriptor.token_idx = token_idx;
    descriptor.valid_token = valid_token ? 1 : 0;
    int32_t const heads_this_token =
        kVectorBaseHeadsPerConsumer +
        (rotating_extra_head <
             kVectorNumRotatingExtraHeads ? 1 : 0);
    for (int32_t head_round = 0;
         head_round < heads_this_token;
         ++head_round) {
      int32_t const global_head_idx =
          head_round < kVectorBaseHeadsPerConsumer
          ? consumer_warp_idx +
              head_round * kVectorNumConsumerWarps
          : kVectorBaseHeadsPerConsumer *
                kVectorNumConsumerWarps +
              rotating_extra_head;
      process_head<true>(
          o, fp8_output, scale_output, descriptor, nullptr,
          vector_cosine_0, vector_cosine_1,
          vector_sine_0, vector_sine_1,
          o_stride_token, o_stride_head, o_stride_dim,
          fp8_stride_token, fp8_stride_group, fp8_stride_dim,
          scale_stride_token, scale_stride_group, scale_stride_head,
          global_head_idx, lane_idx);
    }
  }
}

}  // namespace

cudaError_t launch_attention_output_quant(
    const void* o,
    const int64_t* positions,
    const float* cos_sin_cache,
    void* fp8_output,
    int32_t* scale_output,
    int32_t num_tokens,
    int64_t o_stride_token,
    int64_t o_stride_head,
    int64_t o_stride_dim,
    int64_t position_stride,
    int64_t cache_stride_position,
    int64_t cache_stride_dim,
    int64_t fp8_stride_token,
    int64_t fp8_stride_group,
    int64_t fp8_stride_dim,
    int64_t scale_stride_token,
    int64_t scale_stride_group,
    int64_t scale_stride_head,
    int32_t num_sms,
    cudaStream_t stream) {
  auto const input = reinterpret_cast<__nv_bfloat16 const*>(o);
  auto const output = reinterpret_cast<uint8_t*>(fp8_output);
  bool const vectorized = o_stride_dim == 1 and fp8_stride_dim == 1 and
      (reinterpret_cast<std::uintptr_t>(input) & 7u) == 0u and
      (reinterpret_cast<std::uintptr_t>(output) & 3u) == 0u and
      (o_stride_token & 3) == 0 and (o_stride_head & 3) == 0 and
      (fp8_stride_token & 3) == 0 and (fp8_stride_group & 3) == 0;

  if (vectorized) {
    auto launch_vector = [&](auto grid_constant) {
      constexpr int32_t kGridSize = decltype(grid_constant)::value;
      attention_output_quant_vector_kernel<kGridSize><<<
          kGridSize, kNumThreads, 0, stream>>>(
          input, positions, cos_sin_cache, output, scale_output,
          num_tokens, o_stride_token, o_stride_head, o_stride_dim,
          position_stride, cache_stride_position, cache_stride_dim,
          fp8_stride_token, fp8_stride_group, fp8_stride_dim,
          scale_stride_token, scale_stride_group, scale_stride_head);
    };
    if (num_sms == 40) {
      launch_vector(std::integral_constant<int32_t, 40>{});
    } else if (num_sms == 64) {
      launch_vector(std::integral_constant<int32_t, 64>{});
    } else if (num_sms == 68) {
      launch_vector(std::integral_constant<int32_t, 68>{});
    } else if (num_sms == 84) {
      launch_vector(std::integral_constant<int32_t, 84>{});
    } else if (num_sms == 108) {
      launch_vector(std::integral_constant<int32_t, 108>{});
    } else {
      launch_vector(std::integral_constant<int32_t, 148>{});
    }
  } else {
    attention_output_quant_kernel<false><<<
        num_sms, kNumThreads, 0, stream>>>(
        input, positions, cos_sin_cache, output, scale_output,
        num_tokens, o_stride_token, o_stride_head, o_stride_dim,
        position_stride, cache_stride_position, cache_stride_dim,
        fp8_stride_token, fp8_stride_group, fp8_stride_dim,
        scale_stride_token, scale_stride_group, scale_stride_head);
  }
  return cudaGetLastError();
}
