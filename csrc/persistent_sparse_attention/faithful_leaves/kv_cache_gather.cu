// SPDX-License-Identifier: Apache-2.0

/*
 * Warp-specialized persistent DSv4 packed-K gather for B200/sm_100a.
 *
 * CTA topology (256 threads):
 *   WG0 warp 0: descriptor producer.  Four lanes resolve request length,
 *               logical position, optional table-base offset, physical page,
 *               token/scales address, validity, and output address.
 *   WG0 warps 1-3: intentionally idle.
 *   WG1 warps 4-7: one complete consumer warp group.  Each warp consumes one
 *                  row descriptor, converts 448 E4M3 values with seven UE8M0
 *                  scales, and raw-copies the final 64 BF16 values.
 *
 * The two descriptor slots form a software pipeline: while WG1 computes slot
 * N, the producer resolves slot N+1.  Each CTA persistently walks row quads in
 * a grid-stride loop for every request.  The launch grid is exactly num_sms,
 * allowing 16/40/108/148-SM stream partitions without a Green Context.
 */

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace {

constexpr uint32_t kNumThreads = 256;
constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kProducerThreadStart = 0;
constexpr uint32_t kConsumerThreadStart = 128;
constexpr uint32_t kNumConsumerWarps = 4;
constexpr uint32_t kNumConsumerThreads = kNumConsumerWarps * kWarpSize;
constexpr uint32_t kNumPipelineThreads = kWarpSize + kNumConsumerThreads;
constexpr uint32_t kRowsPerTask = kNumConsumerWarps;
constexpr uint32_t kNumDescriptorStages = 2;

constexpr int32_t kHeadDim = 512;
constexpr int32_t kFp8Dim = 448;
constexpr int32_t kBf16Dim = 64;
constexpr int32_t kQuantBlock = 64;
constexpr int32_t kNumQuantBlocks = kFp8Dim / kQuantBlock;
constexpr int32_t kTokenDataBytes = kFp8Dim + kBf16Dim * 2;
constexpr int32_t kScaleBytes = kNumQuantBlocks + 1;

static_assert(kNumPipelineThreads == 160);
static_assert(kRowsPerTask == 4);
static_assert(kNumQuantBlocks == 7);
static_assert(kTokenDataBytes == 576);
static_assert(kScaleBytes == 8);

struct alignas(16) RowDescriptor {
  int64_t token_data_byte_offset;
  int64_t token_scale_byte_offset;
  int64_t output_element_offset;
  uint32_t active;
  uint32_t valid_block;
};

__device__ __forceinline__ void descriptor_ready(uint32_t const stage_idx) {
  if (stage_idx == 0) {
    asm volatile("bar.sync 1, 160;" ::: "memory");
  } else {
    asm volatile("bar.sync 2, 160;" ::: "memory");
  }
}

// Prevent the producer from reusing descriptor stage 0 for the next request
// while consumers are still reading the final (possibly stage-0) task of the
// current request.  Production prefill has B=1, so this is one final barrier;
// multi-request tests need it for an odd task count.
__device__ __forceinline__ void request_complete() {
  asm volatile("bar.sync 3, 160;" ::: "memory");
}

__device__ __forceinline__ float multiply_rn(
    float const lhs, float const rhs) {
  float result;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(lhs), "f"(rhs));
  return result;
}

__device__ __forceinline__ float e4m3_to_float(uint8_t const bits) {
  __nv_fp8_e4m3 value;
  value.__x = bits;
  return static_cast<float>(value);
}

__device__ __forceinline__ void store_bf16_pair(
    __nv_bfloat16* const output,
    int64_t const element_idx,
    __nv_bfloat16 const value0,
    __nv_bfloat16 const value1) {
  if ((element_idx & 1) == 0) {
    auto const output_pair =
        reinterpret_cast<__nv_bfloat162*>(output + element_idx);
    *output_pair = __halves2bfloat162(value0, value1);
  } else {
    output[element_idx] = value0;
    output[element_idx + 1] = value1;
  }
}

__device__ __forceinline__ void store_raw_bf16_pair(
    __nv_bfloat16* const output,
    int64_t const element_idx,
    uint32_t const raw_pair) {
  if ((element_idx & 1) == 0) {
    *reinterpret_cast<uint32_t*>(output + element_idx) = raw_pair;
  } else {
    auto const raw_halves = reinterpret_cast<uint16_t const*>(&raw_pair);
    *reinterpret_cast<uint16_t*>(output + element_idx) = raw_halves[0];
    *reinterpret_cast<uint16_t*>(output + element_idx + 1) = raw_halves[1];
  }
}

__global__ __launch_bounds__(kNumThreads, 1)
void dequantize_and_gather_k_cache_persistent_kernel(
    __nv_bfloat16* __restrict__ output,
    uint8_t const* __restrict__ cache,
    int32_t const* __restrict__ seq_lens,
    int32_t const* __restrict__ gather_lens,
    int32_t const* __restrict__ block_table,
    int32_t const* __restrict__ block_table_base_offsets,
    int64_t const output_stride0,
    int64_t const output_stride1,
    int64_t const cache_block_stride,
    int64_t const seq_lens_stride,
    int64_t const gather_lens_stride,
    int64_t const block_table_stride0,
    int64_t const block_table_stride1,
    int64_t const block_table_base_offsets_stride,
    int32_t const num_requests,
    int32_t const max_blocks_per_sequence,
    int32_t const cache_block_size,
    int32_t const output_offset) {
  __shared__ alignas(16)
      RowDescriptor descriptors[kNumDescriptorStages][kRowsPerTask];

  uint32_t const thread_idx = static_cast<uint32_t>(threadIdx.x);
  bool const is_producer =
      thread_idx >= kProducerThreadStart and thread_idx < kWarpSize;
  bool const is_consumer =
      thread_idx >= kConsumerThreadStart and
      thread_idx < kConsumerThreadStart + kNumConsumerThreads;
  if (not is_producer and not is_consumer) {
    return;
  }

  uint32_t const lane_idx = thread_idx % kWarpSize;
  uint32_t const consumer_warp_idx =
      is_consumer ? (thread_idx - kConsumerThreadStart) / kWarpSize : 0;

  for (int32_t request_idx = 0;
       request_idx < num_requests;
       ++request_idx) {
    int32_t const seq_len =
        seq_lens[static_cast<int64_t>(request_idx) * seq_lens_stride];
    int32_t const gather_len = gather_lens == nullptr
        ? seq_len
        : gather_lens[
              static_cast<int64_t>(request_idx) * gather_lens_stride];
    uint32_t const num_tasks = gather_len > 0
        ? (static_cast<uint32_t>(gather_len) + kRowsPerTask - 1) /
              kRowsPerTask
        : 0;

    uint32_t task_idx = static_cast<uint32_t>(blockIdx.x);
    uint32_t local_task_idx = 0;
    while (task_idx < num_tasks) {
      uint32_t const stage_idx = local_task_idx % kNumDescriptorStages;

      if (is_producer and lane_idx < kRowsPerTask) {
        uint32_t const gather_idx =
            task_idx * kRowsPerTask + lane_idx;
        RowDescriptor descriptor;
        descriptor.active =
            gather_idx < static_cast<uint32_t>(gather_len) ? 1u : 0u;
        descriptor.valid_block = 0;
        descriptor.token_data_byte_offset = 0;
        descriptor.token_scale_byte_offset = 0;
        descriptor.output_element_offset =
            static_cast<int64_t>(request_idx) * output_stride0 +
            (static_cast<int64_t>(output_offset) + gather_idx) *
                output_stride1;

        if (descriptor.active != 0) {
          int32_t const position =
              seq_len - gather_len + static_cast<int32_t>(gather_idx);
          int32_t block_in_sequence =
              position >= 0 ? position / cache_block_size : -1;
          if (block_table_base_offsets != nullptr) {
            block_in_sequence -= block_table_base_offsets[
                static_cast<int64_t>(request_idx) *
                block_table_base_offsets_stride];
          }
          bool const logical_block_valid =
              block_in_sequence >= 0 and
              block_in_sequence < max_blocks_per_sequence;
          int32_t physical_block_idx = -1;
          if (logical_block_valid) {
            physical_block_idx = block_table[
                static_cast<int64_t>(request_idx) * block_table_stride0 +
                static_cast<int64_t>(block_in_sequence) *
                    block_table_stride1];
          }
          descriptor.valid_block =
              logical_block_valid and physical_block_idx >= 0 ? 1u : 0u;
          if (descriptor.valid_block != 0) {
            int32_t const position_in_block =
                position % cache_block_size;
            int64_t const cache_block_byte_offset =
                static_cast<int64_t>(physical_block_idx) *
                cache_block_stride;
            descriptor.token_data_byte_offset =
                cache_block_byte_offset +
                static_cast<int64_t>(position_in_block) *
                    kTokenDataBytes;
            descriptor.token_scale_byte_offset =
                cache_block_byte_offset +
                static_cast<int64_t>(cache_block_size) *
                    kTokenDataBytes +
                static_cast<int64_t>(position_in_block) * kScaleBytes;
          }
        }
        descriptors[stage_idx][lane_idx] = descriptor;
      }

      descriptor_ready(stage_idx);

      if (is_consumer) {
        RowDescriptor const descriptor =
            descriptors[stage_idx][consumer_warp_idx];
        if (descriptor.active != 0) {
          bool const valid_block = descriptor.valid_block != 0;
          uint8_t const* const token_data =
              cache + descriptor.token_data_byte_offset;
          uint8_t const* const token_scales =
              cache + descriptor.token_scale_byte_offset;

          #pragma unroll
          for (uint32_t quant_block_idx = 0;
               quant_block_idx < kNumQuantBlocks;
               ++quant_block_idx) {
            uint32_t const fp8_element_idx =
                quant_block_idx * kQuantBlock + lane_idx * 2;
            uint16_t raw_pair = 0;
            uint8_t encoded_scale = 127;
            if (valid_block) {
              raw_pair = *reinterpret_cast<uint16_t const*>(
                  token_data + fp8_element_idx);
              if (lane_idx == 0) {
                encoded_scale = token_scales[quant_block_idx];
              }
            }
            encoded_scale = static_cast<uint8_t>(
                __shfl_sync(0xffffffffu, encoded_scale, 0));
            float const scale = exp2f(
                static_cast<float>(encoded_scale) - 127.0f);
            uint8_t const value0_bits =
                static_cast<uint8_t>(raw_pair & 0xffu);
            uint8_t const value1_bits =
                static_cast<uint8_t>(raw_pair >> 8);
            __nv_bfloat16 const value0 = __float2bfloat16_rn(
                multiply_rn(e4m3_to_float(value0_bits), scale));
            __nv_bfloat16 const value1 = __float2bfloat16_rn(
                multiply_rn(e4m3_to_float(value1_bits), scale));
            int64_t const output_element_idx =
                descriptor.output_element_offset + fp8_element_idx;
            store_bf16_pair(
                output, output_element_idx, value0, value1);
          }

          uint32_t raw_bf16_pair = 0;
          if (valid_block) {
            raw_bf16_pair = *reinterpret_cast<uint32_t const*>(
                token_data + kFp8Dim + lane_idx * sizeof(uint32_t));
          }
          int64_t const output_element_idx =
              descriptor.output_element_offset +
              kFp8Dim + lane_idx * 2;
          store_raw_bf16_pair(
              output, output_element_idx, raw_bf16_pair);
        }
      }

      task_idx += static_cast<uint32_t>(gridDim.x);
      ++local_task_idx;
    }
    request_complete();
  }
}

}  // namespace

extern "C" cudaError_t launch_kv_cache_gather(
    void* output, const uint8_t* cache, const int32_t* seq_lens,
    const int32_t* gather_lens, const int32_t* block_table,
    const int32_t* block_table_base_offsets, int64_t output_stride0,
    int64_t output_stride1, int64_t cache_block_stride,
    int64_t seq_lens_stride, int64_t gather_lens_stride,
    int64_t block_table_stride0, int64_t block_table_stride1,
    int64_t block_table_base_offsets_stride, int32_t num_requests,
    int32_t max_blocks_per_sequence, int32_t cache_block_size,
    int32_t output_offset, uint32_t num_sms, cudaStream_t stream) {
  if (num_requests == 0) return cudaSuccess;
  dequantize_and_gather_k_cache_persistent_kernel<<<
      num_sms, kNumThreads, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(output), cache, seq_lens, gather_lens,
      block_table, block_table_base_offsets, output_stride0, output_stride1,
      cache_block_stride, seq_lens_stride, gather_lens_stride,
      block_table_stride0, block_table_stride1,
      block_table_base_offsets_stride, num_requests, max_blocks_per_sequence,
      cache_block_size, output_offset);
  return cudaGetLastError();
}
