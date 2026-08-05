#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace flash_mla_csa::learned_topk {

inline constexpr int kTopK = 1024;
inline constexpr int kThreadsPerTeam = 512;
inline constexpr int kTeamsPerBlock = 2;
inline constexpr int kThreadsPerBlock = kThreadsPerTeam * kTeamsPerBlock;

inline constexpr bool is_supported_grid(int grid_size) {
  return grid_size > 0;
}

cudaError_t launch_learned_topk(
    const float* logits,
    const int32_t* row_starts,
    const int32_t* row_ends,
    int32_t* output,
    int64_t row_stride,
    int32_t num_rows,
    int32_t num_columns,
    int32_t grid_size,
    cudaStream_t stream);

}  // namespace flash_mla_csa::learned_topk
