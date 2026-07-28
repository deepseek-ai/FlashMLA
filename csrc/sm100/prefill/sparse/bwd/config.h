#pragma once
//
// Sparse MLA bwd kernel config.
//
// This is the PIVOT path: start from dense MLA bwd kernel
// (csrc/sm100/prefill/dense/kernel/sm100_fmha_bwd_mla_kernel_tma_warpspecialized.hpp,
// 1829 lines, working/verified) and adapt to sparse by replacing the K/V TMA
// load with `ku::tma_gather4_cta_group_1` and adding the fused kl_target store.
//
// The dense kernel works for D_QK=192, D_V=128 with TileShape <_64,_128,_192,_128>.
// We target DSA absorbed-MLA: D_QK=576, D_V=512. Two viable TileShape options:
//   (a) <_64, _64, _576, _512>  -- single CTA processes full D in one MMA call.
//                                  TMEM budget tight (dQ accum = 64*576/32 = 1152
//                                  cols, exceeds the 512 limit). NOT viable.
//   (b) <_64, _64, _192, _128>  -- split D_QK across 3 cluster CTAs (192*3=576)
//                                  and D_V across 4 chunks (128*4=512).
//
// We pick (b). Dense MLA bwd uses cluster <_1,_1,_1> (single CTA). With 3-CTA
// cluster, each CTA computes a 192-col slice of dQ + a 128-col chunk of dV/dK.
// The gather4 PTX must use `cta_group::1` (added in our kerutils/intrinsics.cuh)
// since cta_group::2 requires pair-clusters.
//
// Sparse-specific deltas from dense MLA bwd:
//   1. Load warp K/V load: cute::copy(tma_load_k, ...) -> ku::tma_gather4_cta_group_1
//      with row_idxs from gather4-aligned indices[k_tile*B_TOPK:].
//   2. New `tensor_map_kv` descriptor (single CUtensorMap for K+V since DSA's
//      kv tensor is one buffer with V = kv[:, :, :D_V]).
//   3. Fused kl_target: Compute warp computes kl_partial = sum_h(P) and
//      Reduce warp TMA-stores it to kl_target[s_q, topk] alongside dQ.
//   4. dKV scatter: SM90_TMA_REDUCE_ADD_2D::copy with crd1=indices[k_pos]
//      replaces dense's sequential dKV TMA store.
//   5. Validity mask for indices < 0 or >= s_kv: P=0 in softmax block.
//
// Out of scope (sparse-fwd-output reuse):
//   - Splitting iter_index based on causal block structure (dense MLA bwd does
//     this for masking optimization). For sparse, all K positions are already
//     causal-valid via gIndices, so no mask block needed.
//
// File layout (parallels dense MLA bwd):
//   config.h                           -- this file (shape constants)
//   sparse_bwd_kernel.hpp              -- adapted from dense MLA bwd kernel
//   sparse_bwd_host.cuh                -- run_sparse_bwd_kernel host wrapper
//   sum_OdO.cuh                        -- D-row (sum(O*dO)) precompute
//   instantiations/sparse_bwd_k576.cu  -- instantiation TU
//

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"

namespace sm100::sparse_bwd {

using namespace cute;

using Element       = cutlass::bfloat16_t;
using ElementAcc    = float;

// DSA absorbed MLA shape constants.
static constexpr int D_QK         = 576;
static constexpr int D_V          = 512;
static constexpr int B_H          = 64;
// B_TOPK = 128 (was 64) so TileShape K matches dense MLA bwd's hardcoded
// assumption. Dense compute() at line ~1385 uses `make_shape(_64, _32, _2, _2)`
// = 8192 element layout which only composes cleanly with TileShapeK=128
// (4096 vs 4096*2). With B_TOPK=64 the layout composition produced a 0-sized
// slice -> static_assert "Ambiguous race-condition detected" in cute::copy.
// For a 32K-context topk=2048 workload this means 16 K-tiles per token (was 32 at B_TOPK=64).
static constexpr int B_TOPK       = 128;
static constexpr int kStages      = 2;

// Cluster: dense MLA bwd uses <_1,_1,_1> -- match that for pipeline / UMMA
// compatibility. The D_QK split (each CTA owns a 192-col slice) is handled
// at the grid level by spawning 3 independent CTAs per Q-token, but each is
// its own cluster of size 1.
using ClusterShape = Shape<_1, _1, _1>;
static constexpr int D_QK_PARTIAL = D_QK / 3;          // 192 (per CTA)
static constexpr int D_V_CHUNK    = 128;               // 4 chunks for full D_V
static_assert(D_QK % 3 == 0 && D_QK_PARTIAL % 64 == 0, "D_QK_PARTIAL must align to 64-col swizzle");

// MMA tile shape: matches dense MLA bwd's <_64, _128, _192, _128> pattern.
//   Q dim = B_H = 64    (token batch per CTA)
//   K dim = B_TOPK = 64 (topk batch per CTA -- smaller than dense's 128 because
//                        DSA's per-CTA topk slice fits B_TOPK=64 cleanly)
//   D_QK dim = D_QK_PARTIAL = 192 (split across 3-CTA cluster)
//   D_V dim  = D_V_CHUNK = 128 (4 chunks across iterations, NOT cluster)
using TileShape = Shape<Int<B_H>, Int<B_TOPK>, Int<D_QK_PARTIAL>, Int<D_V_CHUNK>>;

// Pipeline depths (mirrors dense MLA bwd).
static constexpr int kStagesComputeSmem      = 1;
static constexpr int kStagesReduceTmaStore   = 2;
static constexpr int kNumComputeWarps        = 8;
static constexpr int kNumReduceWarps         = 4;
static constexpr int kNumLoadWarps           = 1;
static constexpr int kNumMmaWarps            = 1;
static constexpr int kNumWarps               = kNumComputeWarps + kNumReduceWarps
                                             + kNumLoadWarps + kNumMmaWarps
                                             + 2; // 2 empty warps for round-up to 16
static constexpr int NumThreadsPerWarp       = 32;
static constexpr int NUM_THREADS             = kNumWarps * NumThreadsPerWarp;

// Sparse-specific: number of K-tiles per token = ceil_div(topk, B_TOPK).
// For a 32K-context topk=2048 workload: 32 K-tiles. For T=2K topk=1024: 16 K-tiles.
static constexpr int kMaxTopK    = 2048;
static constexpr int kMaxKTiles  = (kMaxTopK + B_TOPK - 1) / B_TOPK;   // 32

} // namespace sm100::sparse_bwd
