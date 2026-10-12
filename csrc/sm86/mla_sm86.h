// Host-side interface of the sm_86 MLA decode kernels (plain C++, safe to include from flash_api.cpp).
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "../flash_mla.h"

namespace mla_sm86 {

struct Plan {
    int bm, nt, ns, minb;   // rows per CTA, tokens per KV tile, smem ring stages, CTAs per SM
    bool spec;              // false: v2a (all warps cp.async + CTA barrier); true: v2b (producer warp + mbarriers)
};

// Compile-time defaults (override with -DFLASH_MLA_SM86_<NAME>=..., e.g. through the FLASH_MLA_DEFINES env var of setup.py)
// and run-time overrides (environment variable of the same name, read on every call; used by tuning sweeps).
#ifndef FLASH_MLA_SM86_BM_SMALL      // rows per CTA when s_q * h_q / h_k <= 16
#define FLASH_MLA_SM86_BM_SMALL 16
#endif
#ifndef FLASH_MLA_SM86_BM_LARGE      // rows per CTA otherwise
#define FLASH_MLA_SM86_BM_LARGE 32
#endif
#ifndef FLASH_MLA_SM86_NT            // tokens per KV tile (16 or 32; clamped to the page size)
#define FLASH_MLA_SM86_NT 16
#endif
#ifndef FLASH_MLA_SM86_NS            // smem ring stages
#define FLASH_MLA_SM86_NS 3
#endif
#ifndef FLASH_MLA_SM86_MINB          // CTAs per SM
#define FLASH_MLA_SM86_MINB 1
#endif
#ifndef FLASH_MLA_SM86_SPEC          // 1 = v2b (producer warp + mbarriers)
#define FLASH_MLA_SM86_SPEC 0
#endif
#ifndef FLASH_MLA_SM86_FUSE          // max splits merged by the fused last-arriver combine (0 = always use the combine kernel)
#define FLASH_MLA_SM86_FUSE 0
#endif
#ifndef FLASH_MLA_SM86_OV            // scheduler: fixed per-segment overhead, in 32-token blocks
#define FLASH_MLA_SM86_OV 2
#endif
#ifndef FLASH_MLA_SM86_MINBLK        // scheduler: minimum 32-token blocks per active part (0 = off)
#define FLASH_MLA_SM86_MINBLK 0
#endif

inline int env_int(const char *name, int dflt) {
    const char *e = std::getenv(name);
    return e ? std::atoi(e) : dflt;
}

// Config selection by the number of rows M = s_q * h_q / h_k (shared by the kernel launcher and the metadata builder).
// FLASH_MLA_SM86_CFG="bm,nt,ns[,minb[,spec]]" overrides the whole plan at run time.
inline Plan pick_plan(int M, int page_size) {
    Plan p;
    p.bm = (M <= 16) ? FLASH_MLA_SM86_BM_SMALL : FLASH_MLA_SM86_BM_LARGE;
    p.nt = FLASH_MLA_SM86_NT;
    p.ns = FLASH_MLA_SM86_NS;
    p.minb = FLASH_MLA_SM86_MINB;
    p.spec = FLASH_MLA_SM86_SPEC != 0;
    if (const char *e = std::getenv("FLASH_MLA_SM86_CFG")) {
        int bm, nt, ns, mb = 1, sp = 0;
        if (std::sscanf(e, "%d,%d,%d,%d,%d", &bm, &nt, &ns, &mb, &sp) >= 3) { p.bm = bm; p.nt = nt; p.ns = ns; p.minb = mb; p.spec = sp != 0; }
    }
    if (p.nt > page_size) p.nt = page_size;
    return p;
}

inline int num_sm_parts_for(int M, int h_k, int sm_count) {
    const Plan p = pick_plan(M, 64);
    const int num_m_blocks = (M + p.bm - 1) / p.bm;
    const int parts = sm_count * p.minb / h_k / num_m_blocks;
    return parts < 1 ? 1 : parts;
}

// Granularity (tokens) of the tile scheduler: split boundaries are multiples of it, so it must be a multiple of the
// kernel's Nt (a tile never straddles a page and never starts mid-tile).
// Fixed at 32 so that impl="legacy" (BlockN = 32) can consume the same metadata; every Nt in use (16/32) divides it.
inline int sched_granule(int /*M*/) { return 32; }
// Max splits per batch element merged by the fused (last-arriver) combine; larger counts use the combine kernel.
inline int fuse_max_splits_default() { return env_int("FLASH_MLA_SM86_FUSE", FLASH_MLA_SM86_FUSE); }
// Tunables of the metadata builder (units = scheduler granules).  Overridable with FLASH_MLA_SM86_OV / _MINBLK.
inline int sched_overhead_blocks() { return env_int("FLASH_MLA_SM86_OV", FLASH_MLA_SM86_OV); }
inline int sched_min_blocks_per_part() { return env_int("FLASH_MLA_SM86_MINBLK", FLASH_MLA_SM86_MINBLK); }
constexpr int kMaxSplitsPerBatch = 256;

// Defined in mla_decode_sm86_{bf16,fp16}.cu.  T = __nv_bfloat16 / __half.
template <typename T>
void mla_sm86_run(Flash_fwd_mla_params &params, cudaStream_t stream);

// Workspace layout (bytes) used by the split-KV partials.
struct WorkspaceLayout {
    size_t oaccum_bytes, lseaccum_off, lseaccum_bytes, lsetmp_off, lsetmp_bytes, counters_off, counters_bytes, total;
};
inline WorkspaceLayout workspace_layout(int b, int M, int h_k, int num_sm_parts) {
    auto up = [](size_t x) { return (x + 255) & ~size_t(255); };
    WorkspaceLayout w;
    const size_t slots = size_t(b) + size_t(num_sm_parts);
    w.oaccum_bytes = slots * h_k * M * 512 * sizeof(float);
    w.lseaccum_off = up(w.oaccum_bytes);
    w.lseaccum_bytes = slots * h_k * M * sizeof(float);
    w.lsetmp_off = w.lseaccum_off + up(w.lseaccum_bytes);
    w.lsetmp_bytes = size_t(b) * h_k * M * sizeof(float);
    w.counters_off = w.lsetmp_off + up(w.lsetmp_bytes);
    w.counters_bytes = size_t(b) * h_k * ((M + 15) / 16) * sizeof(int);
    w.total = w.counters_off + up(w.counters_bytes);
    return w;
}

}  // namespace mla_sm86
