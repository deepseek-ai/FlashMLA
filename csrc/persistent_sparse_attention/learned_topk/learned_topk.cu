#include "learned_topk.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace flash_mla_csa::learned_topk {
namespace {

constexpr int kWarpsPerTeam = kThreadsPerTeam / 32;
constexpr int kNumBins = 2048;
constexpr int kNumFinalItems = 2048;

struct FinalItems {
  int indices[kNumFinalItems];
  float logits[kNumFinalItems];
};

struct Histogram {
  int data[kNumBins];
};

union alignas(16) FinalStorage {
  FinalItems items;
  Histogram histogram;
};

struct alignas(16) TeamStorage {
  FinalStorage final;
  int threshold_bin;
  int final_dst;
  int final_bin_size;
  int found_topk;
  int warp_scan[kWarpsPerTeam];
  int scan_total;
};

static_assert(kWarpsPerTeam == 16);
static_assert(sizeof(FinalStorage) == 16384);
static_assert(sizeof(TeamStorage) == 16480);

__device__ __forceinline__ void team_barrier(int barrier_id) {
  asm volatile("bar.sync %0, 512;" : : "r"(barrier_id) : "memory");
}

__device__ __forceinline__ int team_exclusive_sum(
    int value,
    TeamStorage* storage,
    int logical_thread,
    int barrier_id) {
  const int lane = logical_thread & 31;
  const int warp = logical_thread >> 5;
  int inclusive = value;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int incoming = __shfl_up_sync(0xffffffffu, inclusive, offset);
    if (lane >= offset) {
      inclusive += incoming;
    }
  }
  if (lane == 31) {
    storage->warp_scan[warp] = inclusive;
  }
  team_barrier(barrier_id);

  if (warp == 0) {
    const int warp_value =
        lane < kWarpsPerTeam ? storage->warp_scan[lane] : 0;
    int warp_inclusive = warp_value;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const int incoming =
          __shfl_up_sync(0xffffffffu, warp_inclusive, offset);
      if (lane >= offset) {
        warp_inclusive += incoming;
      }
    }
    if (lane < kWarpsPerTeam) {
      storage->warp_scan[lane] = warp_inclusive - warp_value;
    }
    if (lane == kWarpsPerTeam - 1) {
      storage->scan_total = warp_inclusive;
    }
  }
  team_barrier(barrier_id);
  return storage->warp_scan[warp] + inclusive - value;
}

__device__ __forceinline__ uint32_t ordered_float_bits(float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? bits : (~bits & 0x7fffffffu);
}

template <int step>
__device__ __forceinline__ uint32_t extract_bin(float value) {
  if constexpr (step == 0) {
    uint16_t bits = __half_as_ushort(__float2half(value));
    bits = (bits & 0x8000u) ? bits : (~bits & 0x7fffu);
    return bits >> 5;
  } else if constexpr (step == 1) {
    return ordered_float_bits(value) >> 21;
  } else if constexpr (step == 2) {
    return (ordered_float_bits(value) >> 10) & 0x7ffu;
  } else {
    return ordered_float_bits(value) & 0x3ffu;
  }
}

template <int shift>
__device__ __forceinline__ bool matches_prefix(
    float value,
    uint32_t pattern) {
  if constexpr (shift == 0) {
    return true;
  }
  return ((ordered_float_bits(value) ^ pattern) >> shift) == 0;
}

template <typename Function>
__device__ __forceinline__ void process_vectorized(
    const float* input,
    int length,
    int logical_thread,
    Function function) {
  constexpr int kItemsPerVector = 4;
  const uintptr_t address = reinterpret_cast<uintptr_t>(input);
  int skip = address % sizeof(float4)
      ? (sizeof(float4) - address % sizeof(float4)) / sizeof(float)
      : 0;
  if (skip > length) {
    skip = length;
  }
  const float4* vectors = reinterpret_cast<const float4*>(input + skip);
  const int vector_count = (length - skip) / kItemsPerVector;
  for (int vector = logical_thread; vector < vector_count;
       vector += kThreadsPerTeam) {
    const float4 values = vectors[vector];
    const int index = skip + vector * kItemsPerVector;
    function(values.x, index);
    function(values.y, index + 1);
    function(values.z, index + 2);
    function(values.w, index + 3);
  }
  if (logical_thread < skip) {
    function(input[logical_thread], logical_thread);
  }
  const int tail =
      skip + vector_count * kItemsPerVector + logical_thread;
  if (tail < length) {
    function(input[tail], tail);
  }
}

template <int step>
__device__ __forceinline__ bool process_histogram_step(
    const float* logits,
    int row_end,
    uint32_t& logit_pattern,
    int& threshold_bin,
    TeamStorage* storage,
    int* shared_output,
    int column_stride,
    int row_start,
    int logical_thread,
    int barrier_id) {
  for (int bin = logical_thread; bin < kNumBins; bin += kThreadsPerTeam) {
    storage->final.histogram.data[bin] = 0;
  }
  if (logical_thread == 0) {
    storage->threshold_bin = -1;
  }
  team_barrier(barrier_id);

  constexpr int pattern_shift = step < 2 ? 0 : step == 2 ? 21 : 10;
  if constexpr (step == 2) {
    logit_pattern =
        static_cast<uint32_t>(threshold_bin & 0x7ff) << pattern_shift;
  } else if constexpr (step == 3) {
    logit_pattern |=
        static_cast<uint32_t>(threshold_bin & 0x7ff) << pattern_shift;
  }

  auto count_bins = [=] __device__(float logit, int) {
    if (matches_prefix<pattern_shift>(logit, logit_pattern)) {
      atomicAdd(&storage->final.histogram.data[extract_bin<step>(logit)], 1);
    }
  };
  if (column_stride == 1) {
    process_vectorized(
        logits + row_start,
        row_end - row_start,
        logical_thread,
        count_bins);
  } else {
    for (int index = row_start + logical_thread; index < row_end;
         index += kThreadsPerTeam) {
      count_bins(logits[index * column_stride], index);
    }
  }
  team_barrier(barrier_id);

  int last_value = storage->found_topk;
#pragma unroll
  for (int round = 0; round < kNumBins / kThreadsPerTeam; ++round) {
    const int bin = logical_thread + kThreadsPerTeam * round;
    const int bin_count = storage->final.histogram.data[bin];
    team_barrier(barrier_id);

    int prefix_sum =
        team_exclusive_sum(bin_count, storage, logical_thread, barrier_id);
    const int total_sum = storage->scan_total + last_value;
    prefix_sum += last_value;
    storage->final.histogram.data[bin] = prefix_sum;
    team_barrier(barrier_id);

    const int next_prefix_sum = logical_thread == kThreadsPerTeam - 1
        ? total_sum
        : storage->final.histogram.data[bin + 1];
    if (prefix_sum < kTopK && next_prefix_sum >= kTopK) {
      storage->threshold_bin = bin;
      storage->final_bin_size = next_prefix_sum - prefix_sum;
    }
    team_barrier(barrier_id);
    if (storage->threshold_bin >= 0) {
      break;
    }
    last_value = total_sum;
  }
  team_barrier(barrier_id);
  threshold_bin = storage->threshold_bin;

  auto collect_bins = [=] __device__(float logit, int index) {
    if (!matches_prefix<pattern_shift>(logit, logit_pattern)) {
      return;
    }
    const uint32_t bin = extract_bin<step>(logit);
    const bool write_direct =
        (step == 0 && storage->final_bin_size <= kNumFinalItems) ||
        step >= 1;
    if (bin < static_cast<uint32_t>(threshold_bin) && write_direct) {
      const int destination = atomicAdd(&storage->found_topk, 1);
      shared_output[destination] = index;
    }
    if constexpr (step < 3) {
      if (bin == static_cast<uint32_t>(threshold_bin) &&
          storage->final_bin_size <= kNumFinalItems) {
        const int destination = atomicAdd(&storage->final_dst, 1);
        storage->final.items.logits[destination] = logit;
        storage->final.items.indices[destination] = index;
      }
    } else {
      if (bin == static_cast<uint32_t>(threshold_bin)) {
        const int destination =
            atomicAdd(&storage->final.histogram.data[bin], 1);
        if (destination < kTopK) {
          shared_output[destination] = index;
        }
      }
    }
  };
  if (column_stride == 1) {
    process_vectorized(
        logits + row_start,
        row_end - row_start,
        logical_thread,
        collect_bins);
  } else {
    for (int index = row_start + logical_thread; index < row_end;
         index += kThreadsPerTeam) {
      collect_bins(logits[index * column_stride], index);
    }
  }
  team_barrier(barrier_id);
  return storage->final_bin_size > kNumFinalItems;
}

__device__ __forceinline__ void canonicalize_cutoff(
    const float* logits,
    int row_start,
    int row_end,
    int column_stride,
    uint32_t logit_pattern,
    int threshold_bin,
    TeamStorage* storage,
    int* shared_output,
    int logical_thread,
    int barrier_id) {
  const int base = storage->found_topk;
  const int row_length = row_end - row_start;
  int threshold_seen = 0;

  for (int chunk = 0; chunk < row_length; chunk += kThreadsPerTeam) {
    const int logical_index = chunk + logical_thread;
    const bool valid = logical_index < row_length;
    const int input_index = row_start + logical_index;
    const float logit =
        valid ? logits[input_index * column_stride] : 0.0f;
    const bool cutoff_match =
        valid &&
        matches_prefix<10>(logit, logit_pattern) &&
        extract_bin<3>(logit) == static_cast<uint32_t>(threshold_bin);
    const int chunk_rank = team_exclusive_sum(
        cutoff_match ? 1 : 0,
        storage,
        logical_thread,
        barrier_id);
    const int chunk_count = storage->scan_total;
    const int destination = base + threshold_seen + chunk_rank;
    if (cutoff_match && destination < kTopK) {
      shared_output[destination] =
          column_stride == 1 ? logical_index : input_index;
    }
    threshold_seen += chunk_count;
  }
  team_barrier(barrier_id);
}

__device__ __forceinline__ void select_row(
    const float* logits,
    int row_start,
    int row_end,
    int* output,
    int column_stride,
    TeamStorage* storage,
    int* shared_output,
    int logical_thread,
    int barrier_id) {
  const int row_length = row_end - row_start;
  if (row_length <= kTopK) {
    for (int index = logical_thread; index < row_length;
         index += kThreadsPerTeam) {
      output[index] = index;
    }
    for (int index = row_length + logical_thread; index < kTopK;
         index += kThreadsPerTeam) {
      output[index] = -1;
    }
    return;
  }

  if (logical_thread == 0) {
    storage->final_dst = 0;
    storage->found_topk = 0;
  }
  team_barrier(barrier_id);
  int threshold_bin = -1;
  uint32_t logit_pattern = 0;

  bool refine = process_histogram_step<0>(
      logits,
      row_end,
      logit_pattern,
      threshold_bin,
      storage,
      shared_output,
      column_stride,
      row_start,
      logical_thread,
      barrier_id);
  if (refine) {
    refine = process_histogram_step<1>(
        logits,
        row_end,
        logit_pattern,
        threshold_bin,
        storage,
        shared_output,
        column_stride,
        row_start,
        logical_thread,
        barrier_id);
  }
  if (refine) {
    refine = process_histogram_step<2>(
        logits,
        row_end,
        logit_pattern,
        threshold_bin,
        storage,
        shared_output,
        column_stride,
        row_start,
        logical_thread,
        barrier_id);
  }
  if (refine) {
    process_histogram_step<3>(
        logits,
        row_end,
        logit_pattern,
        threshold_bin,
        storage,
        shared_output,
        column_stride,
        row_start,
        logical_thread,
        barrier_id);
    canonicalize_cutoff(
        logits,
        row_start,
        row_end,
        column_stride,
        logit_pattern,
        threshold_bin,
        storage,
        shared_output,
        logical_thread,
        barrier_id);
  }

  if (!refine) {
    const int base = storage->found_topk;
    const int final_count = storage->final_dst;
    for (int item = logical_thread; item < final_count;
         item += kThreadsPerTeam) {
      int rank = 0;
      const float logit = storage->final.items.logits[item];
      const int input_index = storage->final.items.indices[item];
      for (int other_item = 0; other_item < final_count; ++other_item) {
        const float other_logit =
            storage->final.items.logits[other_item];
        const int other_index =
            storage->final.items.indices[other_item];
        if (logit < other_logit ||
            (logit == other_logit && input_index > other_index)) {
          ++rank;
        }
      }
      if (rank + base < kTopK) {
        shared_output[rank + base] = input_index;
      }
    }
    team_barrier(barrier_id);
  }

  for (int index = logical_thread; index < kTopK;
       index += kThreadsPerTeam) {
    output[index] = column_stride == 1
        ? shared_output[index]
        : shared_output[index] - row_start;
  }
}

__global__ __launch_bounds__(kThreadsPerBlock, 1)
void learned_topk_kernel(
    const float* __restrict__ logits,
    const int* __restrict__ row_starts,
    const int* __restrict__ row_ends,
    int* __restrict__ output,
    int row_stride,
    int column_stride,
    int num_rows) {
  __shared__ TeamStorage team_storage[kTeamsPerBlock];
  extern __shared__ int shared_output_all[];
  const int team = threadIdx.x >> 9;
  const int logical_thread = threadIdx.x & (kThreadsPerTeam - 1);
  const int barrier_id = team + 1;
  TeamStorage* storage = &team_storage[team];
  int* shared_output = shared_output_all + team * kTopK;

  const int first_row = kTeamsPerBlock * static_cast<int>(blockIdx.x) + team;
  const int row_step = kTeamsPerBlock * static_cast<int>(gridDim.x);
#pragma unroll 1
  for (int row = first_row; row < num_rows; row += row_step) {
    const int row_start = row_starts[row];
    const int row_end = row_ends[row];
    select_row(
        logits + static_cast<int64_t>(row) * row_stride,
        row_start,
        row_end,
        output + static_cast<int64_t>(row) * kTopK,
        column_stride,
        storage,
        shared_output,
        logical_thread,
        barrier_id);
  }
}

}  // namespace

cudaError_t launch_learned_topk(
    const float* logits,
    const int32_t* row_starts,
    const int32_t* row_ends,
    int32_t* output,
    int64_t row_stride,
    int32_t num_rows,
    int32_t num_columns,
    int32_t grid_size,
    cudaStream_t stream) {
  if (logits == nullptr || row_starts == nullptr || row_ends == nullptr ||
      output == nullptr || row_stride < num_columns || num_rows < 0 ||
      num_columns < 0 || row_stride > INT32_MAX ||
      !is_supported_grid(grid_size)) {
    return cudaErrorInvalidValue;
  }
  if (num_rows == 0) {
    return cudaSuccess;
  }
  learned_topk_kernel<<<
      grid_size,
      kThreadsPerBlock,
      kTeamsPerBlock * kTopK * sizeof(int),
      stream>>>(
      logits,
      row_starts,
      row_ends,
      output,
      static_cast<int>(row_stride),
      1,
      num_rows);
  return cudaGetLastError();
}

}  // namespace flash_mla_csa::learned_topk
