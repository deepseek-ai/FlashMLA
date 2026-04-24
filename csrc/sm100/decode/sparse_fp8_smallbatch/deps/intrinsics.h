#pragma once

// Minimal device intrinsics needed by the smallbatch kernel that are NOT
// provided by kerutils/device/sm100/intrinsics.cuh. Everything in kerutils
// (tcgen05_before/after_thread_sync, tmem_ld/st_32dp32bNx, float2_add/mul/fma,
// umma_arrive_noelect, tma_gather4, elect_one_sync, etc.) is consumed directly
// from kerutils by the kernel TU and is NOT re-declared here.

#include <cute/tensor.hpp>
#include <cute/arch/simd_sm100.hpp>

#include "defines.h"    // main's csrc/defines.h: transac_bar_t, bf16, fp8, etc.

namespace nv_smallbatch::sm100 {

using namespace cute;

// DSMEM write: put 16 bytes into peer CTA's SMEM while also signaling a
// transaction barrier. Not provided by kerutils because it's specific to the
// 2-CTA-cluster DSMEM K-dequant sharing pattern in this kernel.
template<typename T>
CUTE_DEVICE
static void st_async_128b(void* dst_ptr, const T& data, const transac_bar_t* mbar_ptr) {
    static_assert(sizeof(T) == 16, "Data type must be 16 bytes (128 bits) for st_async_128b.");
    long2 data_long2 = *reinterpret_cast<const long2*>(&data);
    uint32_t dst_addr = cute::cast_smem_ptr_to_uint(dst_ptr);
    uint32_t mbar_addr = cute::cast_smem_ptr_to_uint(mbar_ptr);
    asm volatile (
        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.s64 [%0], {%1, %2}, [%3]; \n"
        :
        : "r"(dst_addr), "l"(data_long2.x), "l"(data_long2.y), "r"(mbar_addr)
    );
}

// DSMEM peer addressing: translate a local SMEM pointer into the peer CTA's
// SMEM address within the same cluster. Needed for st_async_128b targets and
// for cluster-visible barriers.
static constexpr int PEER_ADDR_MASK = 16777216; // peer_addr = my_addr ^ PEER_ADDR_MASK (sm_100 2-CTA cluster)
template<typename T>
CUTE_DEVICE
T* get_peer_addr(const T* p) {
    return (T*)((int64_t)(p) ^ PEER_ADDR_MASK);
}

// Template variant of tma_gather4 that supports USE_CTA0_MBAR: when true, the
// mbarrier address is rewritten to target CTA 0's copy so both CTAs in a
// cluster signal the same barrier. Kerutils' tma_gather4 doesn't take this
// template parameter.
template<bool USE_CTA0_MBAR = false>
CUTE_DEVICE void tma_gather4(const void* desc_ptr, transac_bar_t* mbar_ptr, void* smem_ptr, int col_idx, int4 row_idxs, TMA::CacheHintSm90 cache_hint) {
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    uint32_t mbar_addr = cute::cast_smem_ptr_to_uint(mbar_ptr);
    if constexpr (USE_CTA0_MBAR) {
        mbar_addr &= Sm100MmaPeerBitMask;
    }
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;\n"
        :
        : "r"(smem_addr), "l"(desc_ptr), "r"(col_idx),
          "r"(row_idxs.x), "r"(row_idxs.y), "r"(row_idxs.z), "r"(row_idxs.w),
          "r"(mbar_addr), "l"(uint64_t(cache_hint))
        : "memory"
    );
}

}  // namespace nv_smallbatch::sm100
