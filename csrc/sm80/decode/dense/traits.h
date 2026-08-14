#pragma once

namespace sm80 {

namespace cfg {

// Q tile / KV page geometry.
//
// BLOCK_SIZE_M = 16: one warp covers the M dim exactly via mma.m16n8k16.
// This is the smallest tile that doesn't waste mma rows.
//
// PAGE_BLOCK_SIZE is the KV cache page size and is fixed at 64 by the API. KV_TILE is the
// SMEM staging granularity and is deliberately smaller: staging a whole page twice costs
// 162 KB, which allows only one CTA per SM and leaves the kernel latency bound at 6.25%
// occupancy. Each page is walked as PAGE_BLOCK_SIZE / KV_TILE sub-tiles instead.
constexpr int BLOCK_SIZE_M    = 16;
constexpr int PAGE_BLOCK_SIZE = 64;
constexpr int KV_TILE         = 16;
constexpr int TILES_PER_PAGE  = PAGE_BLOCK_SIZE / KV_TILE;
static_assert(PAGE_BLOCK_SIZE % KV_TILE == 0, "KV_TILE must divide the KV page");
constexpr int HEAD_DIM_K      = 576;
constexpr int HEAD_DIM_V      = 512;

// Threading.
//   4 warpgroups x 1 warp/wg x 32 threads = 128 threads / CTA.
//   Each wg owns the full M dim (one warp suffices) and a quarter of V.
//   All wgs compute QK^T independently (4x duplicated, ~1.4 KFLOPS/CTA);
//   compute is not the bottleneck so this redundancy is acceptable.
//
// Tried 8-wg V-eighth (256 thread, spill 320B -> 80B, per-warp rO 32 fp32):
// measured 30-40% regression vs 4-wg, attributed to extra __syncthreads +
// duplicate QK^T cycles outweighing the spill saving (which was small to
// begin with: 320B/iter << 73KB/iter HBM K traffic).
constexpr int NUM_THREADS         = 128;
constexpr int NUM_WARPS           = NUM_THREADS / 32;          // 4
constexpr int NUM_WARPGROUPS      = 4;
constexpr int WARPS_PER_WG        = NUM_WARPS / NUM_WARPGROUPS;// 1
constexpr int ROWS_PER_WARP       = BLOCK_SIZE_M / WARPS_PER_WG;// 16
constexpr int HEAD_DIM_V_PER_WG   = HEAD_DIM_V / NUM_WARPGROUPS;// 128

// SMEM row stride. Swizzle alone (no padding) gives 0 bank conflict for the
// "all-lanes-same-column-different-row" pattern that dominates QK^T / PV.
// Combining padding with swizzle re-creates conflicts -- DON'T.
constexpr int SMEM_PAD_K      = 0;
constexpr int SMEM_STRIDE_K   = HEAD_DIM_K + SMEM_PAD_K;

// Number of sK buffers in SMEM. Three 16-token stages cost the same as one 48-token
// buffer and give a deeper prefetch than the previous two 64-token stages did.
//   sQ 18 KB + 3 x sK 18 KB = 72 KB <= 81.5 KB, so two CTAs fit per SM.
constexpr int SK_STAGES       = 3;

// Opt-in shared memory a single CTA may claim on SM80, which is the 164 KB per-SM pool
// less the kilobyte the driver reserves.
constexpr int MAX_SMEM_PER_BLOCK_BYTES = 163 * 1024;

// SMEM region byte sizes.
template<typename T>
constexpr int smem_q_bytes() { return BLOCK_SIZE_M * SMEM_STRIDE_K * sizeof(T); }
template<typename T>
constexpr int smem_k_bytes() { return KV_TILE * SMEM_STRIDE_K * sizeof(T); }
template<typename T>
constexpr int smem_total_bytes() { return smem_q_bytes<T>() + SK_STAGES * smem_k_bytes<T>(); }

// How many CTAs of this kernel share one SM. Both supported element types are two bytes
// wide, so the footprint does not depend on which one is in use.
constexpr int ctas_per_sm() {
    constexpr int bytes = smem_total_bytes<unsigned short>();
    return MAX_SMEM_PER_BLOCK_BYTES / bytes > 0 ? MAX_SMEM_PER_BLOCK_BYTES / bytes : 1;
}

}

}
