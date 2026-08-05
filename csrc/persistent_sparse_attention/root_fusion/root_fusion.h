#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

struct RootFusionParams {
  const void* hidden;
  void* hidden_fp8;
  uint32_t* hidden_scale;
  uint32_t hidden_scale_stride_k;
  const void* fused_wqa_wkv;
  const uint32_t* fused_wqa_wkv_scale;
  uint32_t fused_wqa_wkv_scale_stride_k;
  void* qkv;
  const void* q_norm_weight;
  const void* kv_norm_weight;
  void* q_norm;
  void* kv_norm;
  void* q_norm_fp8;
  uint32_t* q_norm_scale;
  uint32_t q_norm_scale_stride_k;
  uint32_t* projection_barrier;
  const void* index_compressor_weight;
  float* index_kv_score;
  const void* index_weight_projection;
  void* index_weights;
  uint32_t* index_barrier;
  uint64_t* row_ready;
  uint32_t num_state_slots;
  float* c4_workspace;
  float* state_cache;
  uint32_t state_stride0;
  uint32_t state_stride1;
  const int32_t* token_to_request;
  const int64_t* positions;
  const int64_t* state_slot_mapping;
  const int32_t* state_block_table;
  uint32_t state_block_table_stride;
  const float* rms_weight;
  const float* cos_sin_cache;
  uint32_t cos_sin_stride;
  uint8_t* kv_cache;
  const int64_t* kv_slot_mapping;
  uint32_t kv_cache_stride0;
  const float* ape;
};

cudaError_t launch_root_fusion(
    const RootFusionParams& params,
    uint32_t grid_sms,
    uint32_t projection_sms,
    cudaStream_t stream);
