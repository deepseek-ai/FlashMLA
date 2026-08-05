// SPDX-License-Identifier: Apache-2.0

/*
 * Warp-specialized persistent IQ1-W scale kernel for B200/sm_100a.
 *
 * CTA topology (1,024 threads, eight warp groups):
 *   WG0 warp 0: asynchronous BF16 global -> shared producer.
 *   WG0 warps 1-3: intentionally idle; the producer role is one warp.
 *   WG1 onward: grid-specific 16/20/28 active consumer warps, BF16 -> FP32,
 *               two ordered FP32 multiplies, and vectorized FP32 stores.
 *
 * The vector range is evenly partitioned over exactly num_sms persistent CTAs.
 * Each CTA walks its range in double-buffered 8,192-element chunks.  The
 * producer can fill the next slot while all consumer warp groups calculate
 * the current slot.  Slots use split arrive/wait named barriers, so production
 * 44/108/148-SM launches need only one ready rendezvous per CTA; the 40-SM launch
 * needs two ready rendezvous and no consumed rendezvous.  TMA is deliberately
 * not required for these <=16-KiB BF16 chunks: cp.async gives the dedicated
 * producer warp the same asynchronous hand-off property without descriptors.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr uint32_t kNumThreads = 1024;
constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kConsumerThreadStart = 128;
constexpr uint32_t kMaxNumConsumerThreads =
    kNumThreads - kConsumerThreadStart;
constexpr uint32_t kNumConsumerThreadsSm108 = 20 * kWarpSize;
constexpr uint32_t kNumConsumerThreadsSm148 = 16 * kWarpSize;
constexpr uint32_t kNumElementsPerVector = sizeof(int4) / sizeof(__nv_bfloat16);
constexpr uint32_t kChunkVectors = 1024;
constexpr uint32_t kChunkElements =
    kChunkVectors * kNumElementsPerVector;
constexpr uint32_t kNumPipelineSlots = 2;

static_assert(kNumElementsPerVector == 8);
static_assert(kMaxNumConsumerThreads == 28 * kWarpSize);
static_assert(kNumConsumerThreadsSm108 == 20 * kWarpSize);
static_assert(kNumConsumerThreadsSm148 == 16 * kWarpSize);
static_assert(kChunkElements == 8192);

__device__ __forceinline__ void producer_arrive_ready(
    uint32_t const slot,
    uint32_t const num_pipeline_threads) {
  uint32_t const barrier_id = 1 + slot;
  asm volatile(
      "bar.arrive %0, %1;"
      :
      : "r"(barrier_id), "r"(num_pipeline_threads)
      : "memory");
}

__device__ __forceinline__ void consumer_wait_ready(
    uint32_t const slot,
    uint32_t const num_pipeline_threads) {
  uint32_t const barrier_id = 1 + slot;
  asm volatile(
      "bar.sync %0, %1;"
      :
      : "r"(barrier_id), "r"(num_pipeline_threads)
      : "memory");
}

__device__ __forceinline__ void consumer_arrive_consumed(
    uint32_t const slot,
    uint32_t const num_pipeline_threads) {
  uint32_t const barrier_id = 3 + slot;
  asm volatile(
      "bar.arrive %0, %1;"
      :
      : "r"(barrier_id), "r"(num_pipeline_threads)
      : "memory");
}

__device__ __forceinline__ void producer_wait_consumed(
    uint32_t const slot,
    uint32_t const num_pipeline_threads) {
  uint32_t const barrier_id = 3 + slot;
  asm volatile(
      "bar.sync %0, %1;"
      :
      : "r"(barrier_id), "r"(num_pipeline_threads)
      : "memory");
}

__device__ __forceinline__ void copy_async_16(
    void* const shared_destination,
    void const* const global_source,
    uint32_t const valid_bytes) {
  uint32_t const shared_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(shared_destination));
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16, %2;"
      :
      : "r"(shared_address), "l"(global_source), "r"(valid_bytes)
      : "memory");
}

__device__ __forceinline__ void copy_async_commit_and_wait() {
  asm volatile("cp.async.commit_group;" ::: "memory");
  asm volatile("cp.async.wait_group 0;" ::: "memory");
}

__device__ __forceinline__ float multiply_rn(
    float const lhs, float const rhs) {
  float result;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(lhs), "f"(rhs));
  return result;
}

__device__ __forceinline__ float2 scale_pair(
    __nv_bfloat162 const packed_values,
    float const softmax_scale,
    float const head_scale) {
  float2 values = __bfloat1622float2(packed_values);
  values.x = multiply_rn(
      head_scale, multiply_rn(softmax_scale, values.x));
  values.y = multiply_rn(
      head_scale, multiply_rn(softmax_scale, values.y));
  return values;
}

union PackedBfloat16Vector {
  int4 raw;
  __nv_bfloat162 pairs[4];
};

static_assert(sizeof(PackedBfloat16Vector) == sizeof(int4));

__global__ __launch_bounds__(kNumThreads, 1)
void scale_index_weights_persistent_kernel(
    __nv_bfloat16 const* __restrict__ index_weights,
    float* __restrict__ weights_out,
    uint32_t const num_elements,
    float const softmax_scale,
    float const head_scale) {
  __shared__ __align__(16)
      __nv_bfloat16 shared_weights[kNumPipelineSlots][kChunkElements];

  uint32_t const thread_idx = static_cast<uint32_t>(threadIdx.x);
  uint32_t const num_consumer_threads =
      gridDim.x == 40
      ? kMaxNumConsumerThreads
      : (gridDim.x == 96 or gridDim.x == 108
         ? kNumConsumerThreadsSm108
         : kNumConsumerThreadsSm148);
  uint32_t const num_pipeline_threads =
      kWarpSize + num_consumer_threads;
  bool const is_producer = thread_idx < kWarpSize;
  bool const is_consumer =
      thread_idx >= kConsumerThreadStart and
      thread_idx < kConsumerThreadStart + num_consumer_threads;
  if (not is_producer and not is_consumer) {
    return;
  }

  uint32_t const role_thread_idx =
      is_producer ? thread_idx : thread_idx - kConsumerThreadStart;
  uint32_t const num_vectors =
      (num_elements - 1) / kNumElementsPerVector + 1;
  uint32_t const vectors_per_cta =
      (num_vectors - 1) / static_cast<uint32_t>(gridDim.x) + 1;
  uint32_t const cta_vector_start =
      static_cast<uint32_t>(blockIdx.x) * vectors_per_cta;
  if (cta_vector_start >= num_vectors) {
    return;
  }
  uint32_t const cta_vector_end = min(
      cta_vector_start + vectors_per_cta, num_vectors);
  uint32_t const cta_num_vectors =
      cta_vector_end - cta_vector_start;
  uint32_t const num_chunks =
      (cta_num_vectors - 1) / kChunkVectors + 1;

  for (uint32_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    uint32_t const slot = chunk_idx % kNumPipelineSlots;
    uint32_t const chunk_vector_start =
        cta_vector_start + chunk_idx * kChunkVectors;
    uint32_t const chunk_num_vectors = min(
        kChunkVectors, cta_vector_end - chunk_vector_start);

    if (is_producer) {
      if (chunk_idx >= kNumPipelineSlots) {
        producer_wait_consumed(slot, num_pipeline_threads);
      }
      auto const input_vectors =
          reinterpret_cast<int4 const*>(index_weights);
      auto const shared_vectors =
          reinterpret_cast<int4*>(shared_weights[slot]);

      #pragma unroll
      for (uint32_t vector_idx = role_thread_idx;
           vector_idx < chunk_num_vectors;
           vector_idx += kWarpSize) {
        uint32_t const global_vector_idx =
            chunk_vector_start + vector_idx;
        uint32_t const global_element_idx =
            global_vector_idx * kNumElementsPerVector;
        uint32_t const valid_elements = min(
            kNumElementsPerVector, num_elements - global_element_idx);
        copy_async_16(
            shared_vectors + vector_idx,
            input_vectors + global_vector_idx,
            valid_elements * sizeof(__nv_bfloat16));
      }
      copy_async_commit_and_wait();
      producer_arrive_ready(slot, num_pipeline_threads);
    }

    if (is_consumer) {
      consumer_wait_ready(slot, num_pipeline_threads);
      auto const shared_vectors =
          reinterpret_cast<int4 const*>(shared_weights[slot]);

      #pragma unroll
      for (uint32_t vector_idx = role_thread_idx;
           vector_idx < chunk_num_vectors;
           vector_idx += num_consumer_threads) {
        PackedBfloat16Vector packed;
        packed.raw = shared_vectors[vector_idx];
        uint32_t const global_vector_idx =
            chunk_vector_start + vector_idx;
        uint32_t const global_element_idx =
            global_vector_idx * kNumElementsPerVector;
        uint32_t const valid_elements = min(
            kNumElementsPerVector, num_elements - global_element_idx);

        float2 const values01 =
            scale_pair(packed.pairs[0], softmax_scale, head_scale);
        float2 const values23 =
            scale_pair(packed.pairs[1], softmax_scale, head_scale);
        float2 const values45 =
            scale_pair(packed.pairs[2], softmax_scale, head_scale);
        float2 const values67 =
            scale_pair(packed.pairs[3], softmax_scale, head_scale);
        if (valid_elements == kNumElementsPerVector) {
          auto const output_vectors =
              reinterpret_cast<float4*>(weights_out + global_element_idx);
          output_vectors[0] = make_float4(
              values01.x, values01.y, values23.x, values23.y);
          output_vectors[1] = make_float4(
              values45.x, values45.y, values67.x, values67.y);
        } else {
          if (valid_elements > 0) {
            weights_out[global_element_idx] = values01.x;
          }
          if (valid_elements > 1) {
            weights_out[global_element_idx + 1] = values01.y;
          }
          if (valid_elements > 2) {
            weights_out[global_element_idx + 2] = values23.x;
          }
          if (valid_elements > 3) {
            weights_out[global_element_idx + 3] = values23.y;
          }
          if (valid_elements > 4) {
            weights_out[global_element_idx + 4] = values45.x;
          }
          if (valid_elements > 5) {
            weights_out[global_element_idx + 5] = values45.y;
          }
          if (valid_elements > 6) {
            weights_out[global_element_idx + 6] = values67.x;
          }
        }
      }
      if (chunk_idx + kNumPipelineSlots < num_chunks) {
        consumer_arrive_consumed(slot, num_pipeline_threads);
      }
    }
  }
}

}  // namespace

extern "C" cudaError_t launch_index_weight_scale(
    const void* index_weights, float* weights_out, uint32_t num_elements,
    float softmax_scale, float head_scale, uint32_t num_sms,
    cudaStream_t stream) {
  if (num_elements == 0) return cudaSuccess;
  scale_index_weights_persistent_kernel<<<num_sms, kNumThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(index_weights), weights_out,
      num_elements, softmax_scale, head_scale);
  return cudaGetLastError();
}
