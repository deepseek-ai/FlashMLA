#pragma once

#include "internal/projection_fusion.cuh"
#include "internal/index_compress_fusion.cuh"

namespace deep_gemm {

// One cluster-2 launch replaces the disjoint 108/40 Green Context pair.
// Logical clusters [0, kLeftSMs/2) run the projection prefix, while the
// remaining clusters run C0 -> C4 -> IW0.  Each lane keeps its own barriers;
// there is intentionally no cross-lane synchronization.
template <
    uint32_t kLeftSMs, uint32_t kRightSMs, uint32_t kPQuantILP,
    uint32_t kPBlockM, uint32_t kPBlockN, uint32_t kPBlockK,
    uint32_t kPSwizzleA, uint32_t kPSwizzleB, uint32_t kPSwizzleCD,
    uint32_t kPNumStages,
    uint32_t kPNumNonEpilogueThreads, uint32_t kPNumEpilogueThreads,
    uint32_t kPNumMulticast, bool kPIsMulticastOnA, bool kPSwapAB,
    uint32_t kC4WarpConfig, bool kDenseCurrentC4, bool kStoreDenseC0,
    uint32_t kC0BlockM, uint32_t kC0BlockN, uint32_t kC0BlockK,
    uint32_t kC0SwizzleA, uint32_t kC0SwizzleB, uint32_t kC0SwizzleCD,
    uint32_t kC0NumStages,
    uint32_t kC0NumNonEpilogueThreads, uint32_t kC0NumEpilogueThreads,
    uint32_t kC0NumMulticast, bool kC0IsMulticastOnA,
    uint32_t kC0KAlignment, bool kC0SwapAB,
    uint32_t kIW0BlockM, uint32_t kIW0BlockN, uint32_t kIW0BlockK,
    uint32_t kIW0SwizzleA, uint32_t kIW0SwizzleB, uint32_t kIW0SwizzleCD,
    uint32_t kIW0NumStages,
    uint32_t kIW0NumNonEpilogueThreads, uint32_t kIW0NumEpilogueThreads,
    uint32_t kIW0NumMulticast, bool kIW0IsMulticastOnA,
    uint32_t kIW0KAlignment, bool kIW0SwapAB>
CUTLASS_GLOBAL void __launch_bounds__(512, 1)
root_fusion_kernel(
        const __grid_constant__ cute::TmaDescriptor p_tensor_map_a,
        const __grid_constant__ cute::TmaDescriptor p_tensor_map_b,
        const __grid_constant__ cute::TmaDescriptor p_tensor_map_sfa,
        const __grid_constant__ cute::TmaDescriptor p_tensor_map_sfb,
        const __grid_constant__ cute::TmaDescriptor p_tensor_map_d,
        const __nv_bfloat16* hidden,
        uint8_t* hidden_fp8,
        uint32_t* hidden_scale,
        uint32_t hidden_scale_stride_k,
        const __nv_bfloat16* q_norm_weight,
        const __nv_bfloat16* kv_norm_weight,
        const __nv_bfloat16* qkv,
        __nv_bfloat16* q_norm,
        __nv_bfloat16* kv_norm,
        uint8_t* q_norm_fp8,
        uint32_t* q_norm_scale,
        uint32_t q_norm_scale_stride_k,
        uint32_t* p_stage_barrier,
        uint32_t p_profile_mode,
        const __grid_constant__ cute::TmaDescriptor c0_tensor_map_a,
        const __grid_constant__ cute::TmaDescriptor c0_tensor_map_b,
        const __grid_constant__ cute::TmaDescriptor c0_tensor_map_d,
        const __grid_constant__ cute::TmaDescriptor iw0_tensor_map_a,
        const __grid_constant__ cute::TmaDescriptor iw0_tensor_map_b,
        const __grid_constant__ cute::TmaDescriptor iw0_tensor_map_d,
        uint32_t* right_lane_barrier,
        uint64_t* row_ready,
        uint32_t num_state_slots,
        float* c4_workspace,
        uint32_t right_profile_mode,
        float* current_kv_score,
        float* state_cache,
        uint32_t state_stride0,
        uint32_t state_stride1,
        const int32_t* token_to_req,
        const int64_t* positions,
        const int64_t* state_slot_mapping,
        const int32_t* state_block_table,
        uint32_t state_block_table_stride,
        const float* rms_weight,
        const float* cos_sin_cache,
        uint32_t cos_sin_stride,
        uint8_t* kv_cache,
        const int64_t* kv_slot_mapping,
        uint32_t kv_cache_stride0,
        const float* ape) {
    constexpr uint32_t kClusterSize = 2;
    static_assert(kLeftSMs % kClusterSize == 0);
    static_assert(kRightSMs % kClusterSize == 0);
    const auto block_idx = static_cast<uint32_t>(blockIdx.x);
    const auto cluster_idx = block_idx / kClusterSize;
    if (cluster_idx < kLeftSMs / kClusterSize) {
        projection_fusion_body<
            kLeftSMs, kPQuantILP,
            kPBlockM, kPBlockN, kPBlockK,
            kPSwizzleA, kPSwizzleB, kPSwizzleCD, kPNumStages,
            kPNumNonEpilogueThreads, kPNumEpilogueThreads,
            kPNumMulticast, kPIsMulticastOnA, kPSwapAB>(
                p_tensor_map_a, p_tensor_map_b,
                p_tensor_map_sfa, p_tensor_map_sfb, p_tensor_map_d,
                hidden, hidden_fp8, hidden_scale, hidden_scale_stride_k,
                q_norm_weight, kv_norm_weight, qkv, q_norm, kv_norm,
                q_norm_fp8, q_norm_scale, q_norm_scale_stride_k,
                p_stage_barrier, p_profile_mode, block_idx);
        return;
    }

    index_compress_fusion_body<
        kLeftSMs, kRightSMs, kC4WarpConfig,
        kDenseCurrentC4, kStoreDenseC0,
        kC0BlockM, kC0BlockN, kC0BlockK,
        kC0SwizzleA, kC0SwizzleB, kC0SwizzleCD, kC0NumStages,
        kC0NumNonEpilogueThreads, kC0NumEpilogueThreads,
        kC0NumMulticast, kC0IsMulticastOnA,
        kC0KAlignment, kC0SwapAB,
        kIW0BlockM, kIW0BlockN, kIW0BlockK,
        kIW0SwizzleA, kIW0SwizzleB, kIW0SwizzleCD, kIW0NumStages,
        kIW0NumNonEpilogueThreads, kIW0NumEpilogueThreads,
        kIW0NumMulticast, kIW0IsMulticastOnA,
        kIW0KAlignment, kIW0SwapAB>(
            c0_tensor_map_a, c0_tensor_map_b, c0_tensor_map_d,
            iw0_tensor_map_a, iw0_tensor_map_b, iw0_tensor_map_d,
            right_lane_barrier, row_ready, num_state_slots, c4_workspace,
            right_profile_mode, current_kv_score, state_cache,
            state_stride0, state_stride1, token_to_req, positions,
            state_slot_mapping, state_block_table,
            state_block_table_stride, rms_weight,
            cos_sin_cache, cos_sin_stride, kv_cache,
            kv_slot_mapping, kv_cache_stride0, ape, block_idx - kLeftSMs);
}

}  // namespace deep_gemm
