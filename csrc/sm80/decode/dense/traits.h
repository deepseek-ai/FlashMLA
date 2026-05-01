#pragma once

namespace sm80 {

namespace cfg {

// Q tile / KV page geometry.
//
// BLOCK_SIZE_M = 16: one warp covers the M dim exactly via mma.m16n8k16.
// This is the smallest tile that doesn't waste mma rows; it lets us fit a
// double sK buffer in SMEM (sQ 18 KB + 2 x sK 144 KB = 162 KB <= 164 KB cap),
// which is the key to overlapping K loading with PV compute.
constexpr int BLOCK_SIZE_M    = 16;
constexpr int PAGE_BLOCK_SIZE = 64;
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

// Number of sK buffers in SMEM (double-buffered for cross-block prefetch).
constexpr int SK_STAGES       = 2;

// SMEM region byte sizes.
template<typename T>
constexpr int smem_q_bytes() { return BLOCK_SIZE_M * SMEM_STRIDE_K * sizeof(T); }
template<typename T>
constexpr int smem_k_bytes() { return PAGE_BLOCK_SIZE * SMEM_STRIDE_K * sizeof(T); }
template<typename T>
constexpr int smem_total_bytes() { return smem_q_bytes<T>() + SK_STAGES * smem_k_bytes<T>(); }

}

}
