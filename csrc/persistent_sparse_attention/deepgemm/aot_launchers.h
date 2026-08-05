#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace flash_mla::csa::deepgemm_adapters {

struct FrontendLaunchParams {
  uint32_t shape_m;
  CUtensorMap p1_a, p1_sfa, p1_b, p1_sfb, p1_cd;
  CUtensorMap p4_a, p4_sfa, p4_b, p4_sfb, p4_cd;
  const void* hidden;
  uint32_t hidden_stride;
  void* hidden_fp8;
  uint32_t hidden_fp8_stride;
  void* hidden_sf;
  uint32_t hidden_sf_stride_k;
  const void* qkv;
  uint32_t qkv_stride;
  const void* q_weight;
  const void* kv_weight;
  void* q_norm_fp8;
  uint32_t q_norm_fp8_stride;
  void* q_norm_sf;
  uint32_t q_norm_sf_stride_k;
  void* kv_out;
  uint32_t kv_out_stride;
  const int64_t* positions;
  const float* cos_sin_cache;
  uint32_t cos_sin_stride;
  float rms_eps;
  uint32_t* grid_sync_counters;
  uint32_t debug_m_tile_begin;
  uint32_t debug_m_tile_end;
  uint32_t debug_skip_mask;
};

cudaError_t launch_persistent_frontend_aot(
    const FrontendLaunchParams& params,
    uint32_t num_sms,
    cudaStream_t stream);

struct LogitsLaunchParams {
  uint32_t num_q_tokens;
  uint32_t num_kv_tokens;
  uint32_t logits_stride;
  const uint32_t* row_starts;
  const uint32_t* row_ends;
  float* logits;
  CUtensorMap q, sf_q, kv, sf_kv, weights;
};

cudaError_t launch_index_logits_aot(
    const LogitsLaunchParams& params,
    uint32_t num_sms,
    cudaStream_t stream);

struct WqBLaunchParams {
  uint32_t shape_m;
  uint32_t shape_n;
  uint32_t shape_k;
  CUtensorMap a, b, sfa, sfb, d;
  uint32_t* chunk_counters;
  uint32_t* chunk_ready;
  uint64_t* chunk_clocks;
  uint32_t chunk_size;
  uint32_t num_chunks;
  uint32_t steady_sms;
  uint32_t head_tiles;
};

cudaError_t launch_wqb_ready_aot(
    const WqBLaunchParams& params,
    bool fp4_weight,
    uint32_t num_sms,
    cudaStream_t stream);

}  // namespace flash_mla::csa::deepgemm_adapters
