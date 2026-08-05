#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/epilogue/sm100_persistent_frontend_elementwise.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/epilogue/sm100_store_cd.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

// P0 + P1 + F1' + P4 as one persistent kernel.
//
// Structure follows `the public fused-GEMM implementation`: one grid, the phases
// separated by a **grid-wide barrier**, and the intermediates handed over
// through global memory. That is the whole trick -- what four launches cost is
// four trips through the driver and four ramp-ups, not the memory traffic, so
// keeping `hidden_fp8` / `qkv` / `q_norm_fp8` in global (exactly as the
// reference implementation does) costs nothing and leaves every tensor in the
// layout its consumer already expects.
//
//   P0   hidden [T, 7168] BF16            -> hidden_fp8 + packed UE8M0
//   P1   hidden_fp8 x Wqa/Wkv             -> qkv [T, 2048] BF16
//   F1'  rmsnorm(qkv[:, :1536] / [1536:]) -> q_norm_fp8 (+sf), kv RoPE-d
//   P4   q_norm_fp8 x Wq_b                -> q_2d [T, 65536] BF16
//
// Inside a phase there is no synchronisation at all: the two GEMM phases are
// the stock persistent warp-specialised loop over a linear tile index, so the
// TMA warp still runs ahead of the epilogue across tiles -- a CTA issues the
// loads for tile i+1 while its epilogue is still writing back tile i, which is
// the "start the next load during the last write-back" overlap, kept within
// each phase.
//
// The two GEMMs share one tile config. Both are FP8 x FP8 -> BF16, K-major and
// swap-AB, so the shared-memory and TMEM layouts are identical and only the
// descriptors and the K extent change; the pipeline state (`stage_idx`,
// `phase`, `tma_stage_idx`, `accum_iter`) therefore carries across the phase
// boundary untouched, the same way it carries across tiles.
//
// Warp roles: 0 TMA, 1 UMMA, 2 UTCCP, 3.. epilogue. Every warp of every CTA
// runs the two elementwise phases, since no GEMM is in flight then.

// Backoff ceiling for the grid-sync spin, in nanoseconds. Swept by hand; see
// the note in `grid_sync`.
#ifndef FLASH_MLA_PERSISTENT_FRONTEND_SPIN_MAX_NS
#define FLASH_MLA_PERSISTENT_FRONTEND_SPIN_MAX_NS 32
#endif

// 0 = the real grid sync. 1 = keep only the two CTA-wide named barriers and
// drop the global counter, which prices the barrier's two halves separately
// (WRONG RESULTS -- measurement only).
#ifndef FLASH_MLA_PERSISTENT_FRONTEND_SYNC_MODE
#define FLASH_MLA_PERSISTENT_FRONTEND_SYNC_MODE 0
#endif

// How many counters one grid barrier is spread over. Must be <= 32 so that the
// polling warp can combine every group with a single `__all_sync`.
#ifndef FLASH_MLA_PERSISTENT_FRONTEND_SYNC_GROUPS
#define FLASH_MLA_PERSISTENT_FRONTEND_SYNC_GROUPS 16
#endif

// 1 = spin on a relaxed load and pay the acquire fence once, after the wait
// succeeds, instead of on every poll. Measured: no difference (12.52 vs 12.48
// for the three barriers), so the acquire ordering is not the cost either.
// Left at 0 -- the acquire load is the better-tested path.
#ifndef FLASH_MLA_PERSISTENT_FRONTEND_RELAXED_SPIN
#define FLASH_MLA_PERSISTENT_FRONTEND_RELAXED_SPIN 0
#endif

namespace persistent_frontend {

CUTLASS_DEVICE uint32_t ld_relaxed_gpu(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(ret) : "l"(ptr) : "memory");
    return ret;
}

// `cooperative_groups::this_grid().sync()` in the form DeepGEMM already uses in
// `comm/barrier.cuh`, but over a plain counter so this kernel does not need a
// `layout::Workspace`. One counter per phase boundary.
// NOTE the CTA-level sync is a *named* barrier, not `__syncthreads()`.
// `__syncthreads()` is hardware barrier 0 with a count of blockDim, and
// `sm100_store_cd_swap_ab` uses barrier 0 with a count of
// `kNumUMMAStoreThreads`. The TMA/UMMA/UTCCP warps finish their tiles well
// before the epilogue drains its stores, so they arrive at barrier 0 with a
// conflicting count while the store loop is still using it -- and the store
// loop's "this smem stage is free" barrier then releases early, publishing a
// stale staging buffer. That is invisible with a single tile per phase and
// consistent with more.
// Measurement: with every phase skipped the three barriers cost 5.97 us, and
// replacing the counter with nothing (keeping both named barriers) gives back
// all 5.97. The CTA-wide named barriers are free; the whole cost is 152 CTAs
// serialising on one address. Raising the spin's backoff ceiling changes
// nothing, so it is arrival throughput and not detection latency.
//
// So shard the counter `kNumGroups` ways. CTA b arrives on counter `b %
// kNumGroups`, which cuts same-address contention by that factor, and then one
// warp polls all the counters at once -- `kNumGroups <= 32` keeps that a single
// `__all_sync` over one warp, i.e. one L2 round trip for the whole grid rather
// than one per group.
//
// Every counter takes exactly one `kFinishSumTag` per round (its leader adds
// the tag minus its group's population, everyone else adds one), so all
// counters carry the same tag bit at any consistent point and a CTA can use
// the flip of *its own* counter to know which tag to demand of the others.
// That is what keeps the barrier self-resetting across launches, which is why
// the harness does not have to re-zero the workspace inside the timed loop.
template <uint32_t kNumSMs, uint32_t kNumThreads,
          uint32_t kNumGroups = FLASH_MLA_PERSISTENT_FRONTEND_SYNC_GROUPS, uint32_t kBarrierIdx = 2>
CUTLASS_DEVICE void grid_sync(uint32_t* count_ptr) {
    static constexpr uint32_t kFinishSumTag = 0x80000000u;
    static constexpr uint32_t kSpinBackoffMaxNs = FLASH_MLA_PERSISTENT_FRONTEND_SPIN_MAX_NS;
    static_assert(kNumGroups <= 32 and kNumGroups <= kNumSMs);
    // A named barrier, not `__syncthreads()`: barrier 0 belongs to
    // `sm100_store_cd_swap_ab`'s store pipeline, with a count of
    // `kNumUMMAStoreThreads` rather than blockDim.
    cutlass::arch::NamedBarrier::sync(kNumThreads, kBarrierIdx);
    if (FLASH_MLA_PERSISTENT_FRONTEND_SYNC_MODE == 0 and threadIdx.x < 32) {
        const auto group = blockIdx.x % kNumGroups;
        // |group g| = #{ b < kNumSMs : b % kNumGroups == g }, and its leader is
        // the CTA with blockIdx.x == g.
        const auto group_size =
            (kNumSMs - group + kNumGroups - 1) / kNumGroups;
        uint32_t old_value = 0;
        if (threadIdx.x == 0)
            old_value = ptx::atomic_add_rel(
                count_ptr + group,
                blockIdx.x < kNumGroups ? (kFinishSumTag - (group_size - 1)) : 1u);
        const auto want_tag =
            (__shfl_sync(0xffffffffu, old_value, 0) & kFinishSumTag) ^ kFinishSumTag;
        bool arrived = threadIdx.x >= kNumGroups;
        const auto start_clock = clock64();
        uint32_t backoff = 32;
        while (not __all_sync(0xffffffffu, arrived)) {
            // `kSpinBackoffMaxNs == 0` is a pure spin. `__nanosleep`'s real
            // granularity is a hardware quantum, not the requested value, which
            // is why 2048 / 256 / 64 all measured the same.
            if (kSpinBackoffMaxNs != 0) {
                __nanosleep(backoff);
                backoff = backoff < kSpinBackoffMaxNs ? backoff * 2 : kSpinBackoffMaxNs;
            }
            if (not arrived) {
                const auto seen = FLASH_MLA_PERSISTENT_FRONTEND_RELAXED_SPIN
                                      ? ld_relaxed_gpu(count_ptr + threadIdx.x)
                                      : ptx::ld_acq(count_ptr + threadIdx.x);
                arrived = (seen & kFinishSumTag) == want_tag;
            }
            DG_TRAP_ONLY_DEVICE_ASSERT(clock64() - start_clock < 2000000000ll and
                                       "persistent attention front-end grid sync timeout");
        }
        // The relaxed spin only established that the counters flipped; this is
        // the acquire half of the pair, paid once instead of per poll.
        if (FLASH_MLA_PERSISTENT_FRONTEND_RELAXED_SPIN)
            __threadfence();
    }
    cutlass::arch::NamedBarrier::sync(kNumThreads, kBarrierIdx);
}

}  // namespace persistent_frontend

template <uint32_t SHAPE_HIDDEN,     // 7168, P1's K
          uint32_t SHAPE_QKV,        // 2048, P1's N
          uint32_t SHAPE_Q_RANK,     // 1536, P4's K
          uint32_t SHAPE_KV,         // 512
          uint32_t SHAPE_NOPE,       // 448
          uint32_t SHAPE_ROPE,       // 64
          uint32_t SHAPE_QUERY,      // 65536, P4's N
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_N_P1, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode,
          uint32_t kNumStages,
          uint32_t kNumNonEpilogueThreads, uint32_t kNumEpilogueThreads,
          uint32_t kNumSMs,
          typename a_dtype_t, typename b_dtype_t, typename cd_dtype_t>
CUTLASS_GLOBAL void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_persistent_frontend_fused_impl(uint32_t shape_m,
                              const __grid_constant__ cute::TmaDescriptor tensor_map_p1_a,
                              const __grid_constant__ cute::TmaDescriptor tensor_map_p1_sfa,
                              const __grid_constant__ cute::TmaDescriptor tensor_map_p1_b,
                              const __grid_constant__ cute::TmaDescriptor tensor_map_p1_sfb,
                              const __grid_constant__ cute::TmaDescriptor tensor_map_p1_cd,
                              const __grid_constant__ cute::TmaDescriptor tensor_map_p4_a,
                              const __grid_constant__ cute::TmaDescriptor tensor_map_p4_sfa,
                              const __grid_constant__ cute::TmaDescriptor tensor_map_p4_b,
                              const __grid_constant__ cute::TmaDescriptor tensor_map_p4_sfb,
                              const __grid_constant__ cute::TmaDescriptor tensor_map_p4_cd,
                              // P0 in/out
                              const nv_bfloat16* hidden, uint32_t hidden_stride,
                              uint8_t* hidden_fp8, uint32_t hidden_fp8_stride,
                              uint32_t* hidden_sf, uint32_t hidden_sf_stride_k,
                              // F1' in/out (`qkv` is P1's output, read back here)
                              const nv_bfloat16* qkv, uint32_t qkv_stride,
                              const nv_bfloat16* q_weight, const nv_bfloat16* kv_weight,
                              uint8_t* q_norm_fp8, uint32_t q_norm_fp8_stride,
                              uint32_t* q_norm_sf, uint32_t q_norm_sf_stride_k,
                              nv_bfloat16* kv_out, uint32_t kv_out_stride,
                              const int64_t* positions,
                              const float* cos_sin_cache, uint32_t cos_sin_stride,
                              float rms_eps,
                              uint32_t* grid_sync_counters,
                              uint32_t dbg_m_tile_begin, uint32_t dbg_m_tile_end,
                              uint32_t dbg_skip_mask) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    // 2-CTA cluster, non-swap. CTAs 2k and 2k+1 take adjacent m_blocks of the
    // same n_block, so UMMA_M = 128*2 is the pair jointly forming the MMA's M
    // dimension, and they split B (each loads LOAD_BLOCK_N = BLOCK_N/2) so the
    // weight tile is fetched once per cluster instead of once per CTA.
    constexpr uint32_t kNumMulticast = 2;
    using Allocator = cute::TMEM::Allocator2Sm;

    DG_STATIC_ASSERT(BLOCK_K == 128, "Invalid block K");

    // Swap-AB. Measured both ways at BLOCK_N=128: swap-AB 120.6 us end to end
    // against 129.9 for non-swap. The tuner's own non-swap pick for Wq_b
    // (BLOCK_N=224, 51.2 us standalone) is faster still, but 224 divides
    // neither 2048 nor 65536, so it cannot be shared by both GEMM phases
    // without a partial-n-tile store path.
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * kNumMulticast;
    constexpr uint32_t UMMA_N = BLOCK_N;
    constexpr uint32_t UMMA_K = 32;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M;
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / kNumMulticast;
    DG_STATIC_ASSERT(BLOCK_M == 32 or BLOCK_M == 64 or BLOCK_M == LAYOUT_AD_M, "Invalid block M");

    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    constexpr uint32_t SF_BLOCK_M = math::constexpr_align(BLOCK_M, kNumUTCCPAlignedElems);
    constexpr uint32_t SF_BLOCK_N = math::constexpr_align(BLOCK_N, kNumUTCCPAlignedElems);
    constexpr uint32_t kNumSFStagesPerLoad = 4;

    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumTMAStoreStages = 2;
    constexpr uint32_t STORE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N = kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumUMMAStoreThreads = STORE_BLOCK_M;

    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cd_dtype_t);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t);
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t);
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0 and SMEM_A_SIZE_PER_STAGE % 1024 == 0 and
                     SMEM_B_SIZE_PER_STAGE % 1024 == 0, "A/B shared memory must be 1024B aligned");

    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<
        kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // `ceil_div`, not exact division: the CD/B/SFB tensor maps carry the true
    // N extent, so the TMA clips a partial last n-tile on both the load and
    // the store. That is how the stock kernel handles non-divisible N, and it
    // is what lets both phases share the tuner's preferred BLOCK_N=224 even
    // though 224 divides neither 2048 nor 65536.
    constexpr uint32_t kNumP1NTiles = math::constexpr_ceil_div(SHAPE_QKV, BLOCK_N_P1);
    constexpr uint32_t kNumP4NTiles = math::constexpr_ceil_div(SHAPE_QUERY, BLOCK_N);
    constexpr uint32_t kNumP1KBlocks = SHAPE_HIDDEN / BLOCK_K;
    constexpr uint32_t kNumP4KBlocks = SHAPE_Q_RANK / BLOCK_K;
    DG_STATIC_ASSERT(SHAPE_HIDDEN % BLOCK_K == 0 and SHAPE_Q_RANK % BLOCK_K == 0, "Invalid K");

    constexpr uint32_t kNumThreads = kNumNonEpilogueThreads + kNumEpilogueThreads;
    constexpr uint32_t kNumWarps = kNumThreads / 32;

    comm::cluster_sync_with_relaxed_arrive();
    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const auto cta_rank = cute::block_rank_in_cluster();
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();
    constexpr uint32_t kFirstEpilogueWarpIdx = kNumNonEpilogueThreads / 32;

    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_p1_a);
        cute::prefetch_tma_descriptor(&tensor_map_p1_b);
        cute::prefetch_tma_descriptor(&tensor_map_p1_cd);
        cute::prefetch_tma_descriptor(&tensor_map_p4_a);
        cute::prefetch_tma_descriptor(&tensor_map_p4_b);
        cute::prefetch_tma_descriptor(&tensor_map_p4_cd);
    }

    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    auto smem_cd = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE);
    });
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<a_dtype_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<b_dtype_t*>(smem_buffer + SMEM_CD_SIZE +
                                            kNumStages * SMEM_A_SIZE_PER_STAGE +
                                            i * SMEM_B_SIZE_PER_STAGE);
    });
    auto sf_start_ptr = reinterpret_cast<uint8_t*>(smem_b[kNumStages]);
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE +
                                           i * SMEM_SFB_SIZE_PER_STAGE);
    });

    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_sfb[kNumStages]);
    auto full_barriers         = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers        = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });
    auto with_sf_full_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 2 + i; });
    auto tmem_full_barriers    = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 3 + i; });
    auto tmem_empty_barriers   = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages + i; });
    auto tmem_ptr_in_smem      = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2);

    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(1);
            with_sf_full_barriers[i]->init(kNumMulticast * 32);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            tmem_full_barriers[i]->init(1);
            tmem_empty_barriers[i]->init(kNumMulticast * kNumUMMAStoreThreads);
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    comm::cluster_sync_with_relaxed_arrive();

    cudaGridDependencySynchronize();

    // Pipeline state. This carries across *both* GEMM phases exactly as it
    // carries across tiles: every warp of a CTA walks the same tile list, so
    // the k-stage parity stays consistent, and the grid barrier between the
    // phases guarantees the previous phase is fully drained first.
    uint32_t stage_idx = 0, phase = 0;
    uint32_t tma_stage_idx = 0;
    uint32_t accum_iter = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    const auto all_m_tiles = math::ceil_div(shape_m, BLOCK_M);
    // Debug: run only m-tiles [begin, end). `end == 0` means "all of them".
    const auto m_tile_begin = dbg_m_tile_begin;
    const auto m_tile_end = dbg_m_tile_end == 0 ? all_m_tiles : dbg_m_tile_end;
    const auto num_m_tiles = m_tile_end - m_tile_begin;

    // One GEMM phase: the stock persistent warp-specialised loop over a linear
    // tile index. `kIsP1` picks the descriptors and the K extent; nothing else
    // differs between the two.
    auto run_gemm = [&]<bool kIsP1>(const cute::TmaDescriptor& map_a,
                                    const cute::TmaDescriptor& map_sfa,
                                    const cute::TmaDescriptor& map_b,
                                    const cute::TmaDescriptor& map_sfb,
                                    const cute::TmaDescriptor& map_cd) {
        constexpr uint32_t kNumNTiles = kIsP1 ? kNumP1NTiles : kNumP4NTiles;
        constexpr uint32_t kNumKBlocks = kIsP1 ? kNumP1KBlocks : kNumP4KBlocks;
        // The two phases share every shared-memory and TMEM *offset* -- those
        // stay sized for the wider BLOCK_N -- but not the tile width. P1's N is
        // only 2048, so at BLOCK_N=224 it has 8 x 10 = 80 tiles to hand out to
        // 152 CTAs and 72 of them get nothing; the loop below is
        // `tile = blockIdx.x; tile < num_tiles; tile += kNumSMs`, so a CTA past
        // `num_tiles` simply never runs. A narrower tile is what puts them to
        // work. P4 keeps 224, where its UMMA amortizes best.
        constexpr uint32_t PH_BLOCK_N = kIsP1 ? BLOCK_N_P1 : BLOCK_N;
        constexpr uint32_t PH_UMMA_N = PH_BLOCK_N;
        constexpr uint32_t PH_LOAD_BLOCK_N = PH_BLOCK_N / kNumMulticast;
        constexpr uint32_t PH_SF_BLOCK_N =
            math::constexpr_align(PH_BLOCK_N, kNumUTCCPAlignedElems);
        constexpr uint32_t PH_SMEM_B_SIZE_PER_STAGE =
            PH_LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
        DG_STATIC_ASSERT(PH_BLOCK_N <= BLOCK_N,
                         "per-phase BLOCK_N must fit the shared smem/TMEM layout");
        DG_STATIC_ASSERT(PH_BLOCK_N % STORE_BLOCK_N == 0, "Invalid per-phase BLOCK_N");
        const auto num_tiles = num_m_tiles * kNumNTiles;
        // Iterate m fastest, the way `sched::Scheduler::get_swizzled_block_idx`
        // groups tiles for L2. With n fastest (the obvious mapping) 152
        // concurrent CTAs pull 152 *different* B tiles; Wq_b's B is 100 MB, so
        // that is the worst case for weight reuse. m-fastest makes each group
        // of `num_m_tiles` CTAs share one B tile.
        auto tile_to_mn = [&](const uint32_t& t, uint32_t& m_idx, uint32_t& n_idx) {
            m_idx = (m_tile_begin + t % num_m_tiles) * BLOCK_M;
            n_idx = (t / num_m_tiles) * PH_BLOCK_N;
        };

        if (warp_idx == 0 and cute::elect_one_sync()) {
            // TMA load warp. No barrier between tiles, so the loads for tile
            // i+1 are issued while the epilogue is still writing back tile i.
            for (uint32_t tile = blockIdx.x; tile < num_tiles; tile += kNumSMs) {
                uint32_t m_idx, n_idx;
                tile_to_mn(tile, m_idx, n_idx);
                for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocks;
                     advance_pipeline(k_block_idx)) {
                    empty_barriers[stage_idx]->wait(phase ^ 1);
                    const auto k_idx = k_block_idx * BLOCK_K;
                    tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode, a_dtype_t, false>(
                        &map_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, m_idx, 1, 0);
                    tma::copy<BLOCK_K, PH_LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t, false>(
                        &map_b, full_barriers[stage_idx], smem_b[stage_idx], k_idx,
                        n_idx + cta_rank * PH_LOAD_BLOCK_N, 1, 0);
                    auto num_arrival_bytes = SMEM_A_SIZE_PER_STAGE + PH_SMEM_B_SIZE_PER_STAGE;
                    if (k_block_idx % kNumSFStagesPerLoad == 0) {
                        const auto sf_k_idx = k_block_idx / kNumSFStagesPerLoad;
                        tma::copy<BLOCK_M, 1, 0>(&map_sfa, full_barriers[stage_idx],
                                                 smem_sfa[stage_idx], m_idx, sf_k_idx);
                        tma::copy<PH_BLOCK_N, 1, 0>(&map_sfb, full_barriers[stage_idx],
                                                    smem_sfb[stage_idx], n_idx, sf_k_idx);
                        num_arrival_bytes += (BLOCK_M + PH_BLOCK_N) * sizeof(uint32_t);
                    }
                    full_barriers[stage_idx]->arrive_and_expect_tx(num_arrival_bytes);
                }
            }
        } else if (warp_idx == 1 and is_leader_cta) {
            // UMMA issue warp (leader CTA only)
            auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
                a_dtype_t, b_dtype_t, float, cutlass::float_ue8m0_t,
                UMMA_M, PH_UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
            auto sf_desc = mma::sm100::make_sf_desc(nullptr);
            auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a[0], 0, 0);
            auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, PH_LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b[0], 0, 0);
            uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
            uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

            for (uint32_t tile = blockIdx.x; tile < num_tiles; tile += kNumSMs) {
                const auto accum_stage_idx = accum_iter % kNumEpilogueStages;
                const auto accum_phase_idx = (accum_iter / kNumEpilogueStages) & 1;
                ++ accum_iter;
                tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
                ptx::tcgen05_after_thread_sync();

                for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocks;
                     advance_pipeline(k_block_idx)) {
                    with_sf_full_barriers[stage_idx]->wait(phase);
                    ptx::tcgen05_after_thread_sync();

                    const auto a_desc_base_lo = ptx::exchange(a_desc_lo, stage_idx);
                    const auto b_desc_base_lo = ptx::exchange(b_desc_lo, stage_idx);
                    if (cute::elect_one_sync()) {
                        // Must match the allocator's CTA group or ptxas rejects the kernel.
                        using cute_utccp_t = cute::SM100_UTCCP_4x32dp128bit_2cta;
                        if (k_block_idx % kNumSFStagesPerLoad == 0) {
                            #pragma unroll
                            for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                                mma::sm100::replace_smem_desc_addr(sf_desc, smem_sfa[stage_idx] + i * kNumUTCCPAlignedElems);
                                cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
                            }
                            #pragma unroll
                            for (uint32_t i = 0; i < PH_SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                                mma::sm100::replace_smem_desc_addr(sf_desc, smem_sfb[stage_idx] + i * kNumUTCCPAlignedElems);
                                cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
                            }
                        }
                        const auto sf_id = k_block_idx % kNumSFStagesPerLoad;
                        utils::for_each_static_until<BLOCK_K / UMMA_K>(
                            std::make_integer_sequence<uint32_t, BLOCK_K / UMMA_K>(),
                            [&]<uint32_t kUMMAKIdx>() {
                                constexpr uint32_t kOffset = kUMMAKIdx * UMMA_K;
                                const auto runtime_instr_desc =
                                    mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, sf_id, sf_id);
                                a_desc.lo = mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, BLOCK_M, kSwizzleAMode, a_dtype_t>(a_desc_base_lo, 0, kOffset);
                                b_desc.lo = mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, PH_LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(b_desc_base_lo, 0, kOffset);
                                ptx::SM100_MMA_MXF8F6F4_2x1SM_SS::fma(
                                    a_desc, b_desc, accum_stage_idx * PH_UMMA_N,
                                    kUMMAKIdx > 0 or k_block_idx > 0, runtime_instr_desc,
                                    kTmemStartColOfSFA, kTmemStartColOfSFB);
                            });
                    }
                    __syncwarp();
                    constexpr uint16_t kCTAMask = (1 << kNumMulticast) - 1;
                    cutlass::arch::umma_arrive_multicast_2x1SM(
                        reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]), kCTAMask);
                    if (k_block_idx == kNumKBlocks - 1)
                        cutlass::arch::umma_arrive_multicast_2x1SM(
                            reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]), kCTAMask);
                    __syncwarp();
                }
            }
        } else if (warp_idx == 2) {
            // UTCCP transposer
            auto transpose = [&](uint32_t* smem_ptr) {
                uint32_t values[4];
                #pragma unroll
                for (uint32_t i = 0; i < 4; ++ i)
                    values[i] = ptx::ld_shared(smem_ptr + i * 32 + lane_idx);
                __syncwarp();
                ptx::st_shared(smem_ptr + lane_idx * 4, values[0], values[1], values[2], values[3]);
            };
            for (uint32_t tile = blockIdx.x; tile < num_tiles; tile += kNumSMs) {
                for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocks;
                     advance_pipeline(k_block_idx)) {
                    full_barriers[stage_idx]->wait(phase);
                    if (k_block_idx % kNumSFStagesPerLoad == 0) {
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i)
                            transpose(smem_sfa[stage_idx] + i * kNumUTCCPAlignedElems);
                        #pragma unroll
                        for (uint32_t i = 0; i < PH_SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i)
                            transpose(smem_sfb[stage_idx] + i * kNumUTCCPAlignedElems);
                        cutlass::arch::fence_view_async_shared();
                    }
                    with_sf_full_barriers[stage_idx]->arrive(0u);
                }
            }
        } else if (warp_idx >= kFirstEpilogueWarpIdx and
                   warp_idx < (kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {
            // Epilogue warpgroup
            const auto epilogue_warp_idx = warp_idx - kFirstEpilogueWarpIdx;
            // `tmem_base_addr` below is just `accum_stage_idx * UMMA_N`, which
            // assumes this CTA's tensor-memory allocation starts at column 0.
            // The stock GEMM asserts that; dropping the assert turns a bad
            // allocation into silently wrong output instead of a trap.
            DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);
            for (uint32_t tile = blockIdx.x; tile < num_tiles; tile += kNumSMs) {
                const auto accum_stage_idx = accum_iter % kNumEpilogueStages;
                const auto accum_phase_idx = (accum_iter / kNumEpilogueStages) & 1;
                ++ accum_iter;
                tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
                ptx::tcgen05_after_thread_sync();

                uint32_t store_m_idx, store_n_idx;
                tile_to_mn(tile, store_m_idx, store_n_idx);
                epilogue::sm100_store_cd<
                    BLOCK_M, PH_BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                    kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                    GemmType::Normal, false, cd_dtype_t,
                    epilogue::transform::EpilogueIdentity>
                (smem_cd, tma_stage_idx, accum_stage_idx * PH_UMMA_N,
                 store_m_idx, store_n_idx, 0,
                 epilogue_warp_idx, lane_idx,
                 tmem_empty_barriers[accum_stage_idx], map_cd);
            }
            // The stores are async; drain before the grid barrier so the next
            // phase's ordinary loads see this phase's output.
            if (epilogue_warp_idx == 0 and lane_idx == 0) {
                cute::tma_store_wait<0>();
                ptx::fence_proxy_async_global();
            }
        }
    };

    // ---- P0: BF16 hidden -> FP8 + packed UE8M0 -------------------------------
    // `dbg_skip_elementwise` produces wrong output on purpose: it prices the
    // two elementwise phases inside the merged kernel, which is where the
    // remaining gap to the four-kernel chain is expected to be.
    if ((dbg_skip_mask & 1u) == 0)
    epilogue::hca_fe::p0_quant<SHAPE_HIDDEN>(
        hidden, hidden_stride, hidden_fp8, hidden_fp8_stride,
        hidden_sf, hidden_sf_stride_k, 0, shape_m,
        blockIdx.x * kNumWarps + warp_idx, kNumSMs * kNumWarps, lane_idx);
    // The consumer is a TMA, which lives in the async proxy, so every writing
    // thread needs the proxy fence as well as the memory fence.
    __threadfence();
    ptx::fence_proxy_async_global();
    // Skip-mask bits 4/5/6 drop barrier 0/1/2 individually, which is what
    // separates per-barrier latency from a one-off "force full residency"
    // effect: if the cost were residency, dropping two of the three would
    // give back nothing.
    if ((dbg_skip_mask & 16u) == 0)
    persistent_frontend::grid_sync<kNumSMs, kNumThreads>(grid_sync_counters + 0 * FLASH_MLA_PERSISTENT_FRONTEND_SYNC_GROUPS);

    // ---- P1: hidden_fp8 x Wqa/Wkv -> qkv -------------------------------------
    if ((dbg_skip_mask & 4u) == 0)
    run_gemm.template operator()<true>(tensor_map_p1_a, tensor_map_p1_sfa,
                                     tensor_map_p1_b, tensor_map_p1_sfb,
                                     tensor_map_p1_cd);
    // Skip-mask bits 4/5/6 drop barrier 0/1/2 individually, which is what
    // separates per-barrier latency from a one-off "force full residency"
    // effect: if the cost were residency, dropping two of the three would
    // give back nothing.
    if ((dbg_skip_mask & 32u) == 0)
    persistent_frontend::grid_sync<kNumSMs, kNumThreads>(grid_sync_counters + 1 * FLASH_MLA_PERSISTENT_FRONTEND_SYNC_GROUPS);

    // ---- F1': RMSNorm(q, kv) + q quantization + kv RoPE ----------------------
    if ((dbg_skip_mask & 2u) == 0)
    epilogue::hca_fe::f1_norm_quant_rope<
        SHAPE_Q_RANK, SHAPE_KV, SHAPE_NOPE, SHAPE_ROPE>(
        qkv, qkv_stride, q_weight, kv_weight,
        q_norm_fp8, q_norm_fp8_stride, q_norm_sf, q_norm_sf_stride_k,
        kv_out, kv_out_stride, positions, cos_sin_cache, cos_sin_stride,
        rms_eps, 0, shape_m,
        blockIdx.x * kNumWarps + warp_idx, kNumSMs * kNumWarps, lane_idx);
    __threadfence();
    ptx::fence_proxy_async_global();
    // Skip-mask bits 4/5/6 drop barrier 0/1/2 individually, which is what
    // separates per-barrier latency from a one-off "force full residency"
    // effect: if the cost were residency, dropping two of the three would
    // give back nothing.
    if ((dbg_skip_mask & 64u) == 0)
    persistent_frontend::grid_sync<kNumSMs, kNumThreads>(grid_sync_counters + 2 * FLASH_MLA_PERSISTENT_FRONTEND_SYNC_GROUPS);

    // ---- P4: q_norm_fp8 x Wq_b -> q_2d ---------------------------------------
    if ((dbg_skip_mask & 8u) == 0)
    run_gemm.template operator()<false>(tensor_map_p4_a, tensor_map_p4_sfa,
                                      tensor_map_p4_b, tensor_map_p4_sfb,
                                      tensor_map_p4_cd);

    // Not `__syncthreads()`: that is hardware barrier 0 with a count of
    // blockDim, and `sm100_store_cd_swap_ab` uses barrier 0 with a count of
    // `kNumUMMAStoreThreads`. The non-epilogue warps leave the tile loop first,
    // so they would arrive with a conflicting count while the epilogue is
    // still draining stores.
    cutlass::arch::NamedBarrier::sync(kNumThreads, 1);
    comm::cluster_sync_with_relaxed_arrive();
    if (warp_idx == 0)
        Allocator().free(0, kNumTmemCols);

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
