// SPDX-License-Identifier: Apache-2.0
// Derived from accepted CSA K13; vendored to avoid source-tree dependencies.

#include "reset_ready.h"

/*
 * Warp-specialized fixed-grid zero reset for CSA ready-protocol metadata.
 *
 * The launch has exactly num_sms CTAs and 256 threads per CTA:
 *
 *   WG0 warp 0     producer: publishes one heterogeneous buffer descriptor
 *                              at a time into shared memory.
 *   WG0 warps 1-3            intentionally idle.
 *   WG1 warps 4-7 consumers: a complete warpgroup performs vectorized,
 *                              grid-stride zero stores.
 *
 * There is deliberately no global scheduler counter to initialize.  Every CTA
 * consumes the same descriptor stream and owns a disjoint slice selected by
 * blockIdx.x.  This makes all 40/108/148 CTAs safe even though the usual
 * metadata is small enough that most high-index CTAs have no stores.
 */

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kConsumerStart = 128;
constexpr int kConsumerThreads = 128;
constexpr int kPipelineThreads = kWarpSize + kConsumerThreads;
constexpr int kMaxDescriptors = 5;
constexpr size_t kResidencySharedBytes = 120 * 1024;

static_assert(kPipelineThreads == 160);

struct alignas(16) ResetDescriptor {
  uint32_t* pointer;
  size_t num_words;
};

__device__ __forceinline__ void descriptor_ready() {
  asm volatile("bar.sync 1, 160;" ::: "memory");
}

__device__ __forceinline__ void descriptor_consumed() {
  asm volatile("bar.sync 2, 160;" ::: "memory");
}

__global__ __launch_bounds__(kThreads, 1)
void reset_ready_kernel(
    uint32_t* counters,
    size_t counter_words,
    uint32_t* ready,
    size_t ready_words,
    uint32_t* producer_clocks,
    size_t producer_clock_words,
    uint32_t* consumer_clocks,
    size_t consumer_clock_words,
    uint32_t* qnorm_clocks,
    size_t qnorm_clock_words,
    int num_descriptors) {
  __shared__ ResetDescriptor descriptor;

  int const thread_idx = static_cast<int>(threadIdx.x);
  bool const is_producer = thread_idx < kWarpSize;
  bool const is_consumer = thread_idx >= kConsumerStart;
  if (!is_producer && !is_consumer) {
    return;
  }

  int const consumer_idx = thread_idx - kConsumerStart;
  size_t const consumer_offset =
      static_cast<size_t>(blockIdx.x) * kConsumerThreads +
      static_cast<size_t>(consumer_idx);
  size_t const consumer_stride =
      static_cast<size_t>(gridDim.x) * kConsumerThreads;

  #pragma unroll
  for (int descriptor_idx = 0;
       descriptor_idx < kMaxDescriptors;
       ++descriptor_idx) {
    if (descriptor_idx >= num_descriptors) {
      break;
    }

    if (is_producer && thread_idx == 0) {
      if (descriptor_idx == 0) {
        descriptor = {counters, counter_words};
      } else if (descriptor_idx == 1) {
        descriptor = {ready, ready_words};
      } else if (descriptor_idx == 2) {
        descriptor = {producer_clocks, producer_clock_words};
      } else if (descriptor_idx == 3) {
        descriptor = {consumer_clocks, consumer_clock_words};
      } else {
        descriptor = {qnorm_clocks, qnorm_clock_words};
      }
    }

    descriptor_ready();

    if (is_consumer && descriptor.num_words != 0) {
      uintptr_t const address =
          reinterpret_cast<uintptr_t>(descriptor.pointer);
      if ((address & (alignof(int4) - 1)) == 0) {
        auto* vectors = reinterpret_cast<int4*>(descriptor.pointer);
        size_t const num_vectors = descriptor.num_words / 4;
        int4 const zero = {0, 0, 0, 0};
        for (size_t vector_idx = consumer_offset;
             vector_idx < num_vectors;
             vector_idx += consumer_stride) {
          vectors[vector_idx] = zero;
        }

        size_t const tail_start = num_vectors * 4;
        size_t const word_idx = tail_start + consumer_offset;
        if (word_idx < descriptor.num_words) {
          descriptor.pointer[word_idx] = 0;
        }
      } else {
        for (size_t word_idx = consumer_offset;
             word_idx < descriptor.num_words;
             word_idx += consumer_stride) {
          descriptor.pointer[word_idx] = 0;
        }
      }
    }

    descriptor_consumed();
  }
}

}  // namespace

cudaError_t launch_reset_ready(
    void* counters,
    size_t counter_bytes,
    void* ready,
    size_t ready_bytes,
    void* producer_clocks,
    size_t producer_clock_bytes,
    void* consumer_clocks,
    size_t consumer_clock_bytes,
    void* qnorm_clocks,
    size_t qnorm_clock_bytes,
    int num_descriptors,
    int num_sms,
    cudaStream_t stream) {
  static cudaError_t const residency_status = cudaFuncSetAttribute(
      reset_ready_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kResidencySharedBytes));
  if (residency_status != cudaSuccess) {
    return residency_status;
  }

  reset_ready_kernel<<<
      num_sms, kThreads, kResidencySharedBytes, stream>>>(
      static_cast<uint32_t*>(counters),
      counter_bytes / sizeof(uint32_t),
      static_cast<uint32_t*>(ready),
      ready_bytes / sizeof(uint32_t),
      static_cast<uint32_t*>(producer_clocks),
      producer_clock_bytes / sizeof(uint32_t),
      static_cast<uint32_t*>(consumer_clocks),
      consumer_clock_bytes / sizeof(uint32_t),
      static_cast<uint32_t*>(qnorm_clocks),
      qnorm_clock_bytes / sizeof(uint32_t),
      num_descriptors);
  return cudaGetLastError();
}

cudaError_t query_reset_ready_max_active_blocks_per_sm(
    int* max_active_blocks) {
  cudaError_t const residency_status = cudaFuncSetAttribute(
      reset_ready_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kResidencySharedBytes));
  if (residency_status != cudaSuccess) {
    return residency_status;
  }
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      max_active_blocks,
      reset_ready_kernel,
      kThreads,
      kResidencySharedBytes);
}

size_t reset_ready_residency_shared_bytes() {
  return kResidencySharedBytes;
}
