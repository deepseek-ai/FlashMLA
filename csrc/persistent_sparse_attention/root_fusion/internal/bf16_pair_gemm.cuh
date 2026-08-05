#pragma once

#include "bf16_gemm_adapter.cuh"
#include <deep_gemm/ptx/ld_st.cuh>

namespace deep_gemm {

template <uint32_t kNumCTAs>
CUTLASS_DEVICE void csa_root_lane_barrier(
        uint32_t* counter, const uint32_t cta_idx) {
    static constexpr uint32_t kFinishSumTag = 0x80000000u;
    __syncthreads();
    if (threadIdx.x == 0) {
        const auto old_value = ptx::atomic_add_rel(
            counter, cta_idx == 0 ? (kFinishSumTag - (kNumCTAs - 1)) : 1);
        uint32_t new_value;
        do {
            new_value = ptx::ld_acq(counter);
        } while (((new_value ^ old_value) & kFinishSumTag) == 0);
    }
    __syncthreads();
}

template <
    uint32_t kLeftSMs, uint32_t kRightSMs,
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
bf16_pair_gemm_kernel(
        const __grid_constant__ cute::TmaDescriptor c0_tensor_map_a,
        const __grid_constant__ cute::TmaDescriptor c0_tensor_map_b,
        const __grid_constant__ cute::TmaDescriptor c0_tensor_map_d,
        const __grid_constant__ cute::TmaDescriptor iw0_tensor_map_a,
        const __grid_constant__ cute::TmaDescriptor iw0_tensor_map_b,
        const __grid_constant__ cute::TmaDescriptor iw0_tensor_map_d,
        uint32_t* lane_barrier) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    constexpr uint32_t kClusterSize = 2;
    static_assert(kLeftSMs % kClusterSize == 0);
    static_assert(kRightSMs % kClusterSize == 0);

    // Route whole clusters, never individual CTAs.  Splitting the two CTAs in
    // a cluster would invalidate cluster barriers, TMEM allocation, and TMA
    // multicast.  108/40 maps to 54 left clusters and 20 right clusters.
    const auto cluster_idx =
        static_cast<uint32_t>(blockIdx.x) / kClusterSize;
    if (cluster_idx < kLeftSMs / kClusterSize)
        return;

    const auto cta_idx = static_cast<uint32_t>(blockIdx.x) - kLeftSMs;
    DG_DEVICE_ASSERT(cta_idx < kRightSMs);

    sm100_bf16_gemm_body<
        cute::UMMA::Major::K, cute::UMMA::Major::K,
        8192, 512, 7168,
        kC0BlockM, kC0BlockN, kC0BlockK,
        1,
        kC0SwizzleA, kC0SwizzleB, kC0SwizzleCD,
        kC0NumStages,
        kC0NumNonEpilogueThreads, kC0NumEpilogueThreads,
        kC0NumMulticast, kC0IsMulticastOnA,
        kRightSMs,
        kC0KAlignment,
        kC0SwapAB, true,
        GemmType::Normal, false, float,
        100>(
            nullptr, 8192, 512, 7168,
            c0_tensor_map_a, c0_tensor_map_b, c0_tensor_map_d, cta_idx);

    // The ordinary standalone kernel relies on the kernel epilogue for this
    // wait.  Physical stage fusion needs the store complete before publishing
    // the release to the next device body.
    cute::tma_store_wait<0>();
    csa_root_lane_barrier<kRightSMs>(lane_barrier, cta_idx);

    sm100_bf16_gemm_body<
        cute::UMMA::Major::K, cute::UMMA::Major::K,
        8192, 64, 7168,
        kIW0BlockM, kIW0BlockN, kIW0BlockK,
        1,
        kIW0SwizzleA, kIW0SwizzleB, kIW0SwizzleCD,
        kIW0NumStages,
        kIW0NumNonEpilogueThreads, kIW0NumEpilogueThreads,
        kIW0NumMulticast, kIW0IsMulticastOnA,
        kRightSMs,
        kIW0KAlignment,
        kIW0SwapAB, true,
        GemmType::Normal, false, cutlass::bfloat16_t,
        100>(
            nullptr, 8192, 64, 7168,
            iw0_tensor_map_a, iw0_tensor_map_b, iw0_tensor_map_d, cta_idx);
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports SM100");
#endif
}

}  // namespace deep_gemm
