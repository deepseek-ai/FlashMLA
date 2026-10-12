// Host-side interface of the sm_86 MLA decode kernels (plain C++, safe to include from flash_api.cpp).
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "../flash_mla.h"

namespace mla_sm86 {

struct Plan {
    int bm, nt, ns, minb;   // rows per CTA, tokens per KV tile, smem ring stages, CTAs per SM
    bool spec;              // v2b: producer warp + mbarriers
    bool pipe;              // v2c: software-pipelined tile loop
};

// Compile-time defaults (override with -DFLASH_MLA_SM86_<NAME>=..., e.g. through the FLASH_MLA_DEFINES env var of setup.py)
// and run-time overrides (environment variable of the same name, read on every call; used by tuning sweeps).
#ifndef FLASH_MLA_SM86_BM_SMALL      // rows per CTA when s_q * h_q / h_k <= 16
#define FLASH_MLA_SM86_BM_SMALL 16
#endif
#ifndef FLASH_MLA_SM86_BM_LARGE      // rows per CTA otherwise
#define FLASH_MLA_SM86_BM_LARGE 32
#endif
#ifndef FLASH_MLA_SM86_NT_SMALL       // tokens per KV tile when BM = 16 (16 or 32; clamped to the page size)
#define FLASH_MLA_SM86_NT_SMALL 16
#endif
#ifndef FLASH_MLA_SM86_NS_SMALL       // smem ring stages when BM = 16
#define FLASH_MLA_SM86_NS_SMALL 3
#endif
#ifndef FLASH_MLA_SM86_NT_LARGE       // tokens per KV tile when BM = 32
#define FLASH_MLA_SM86_NT_LARGE 32
#endif
#ifndef FLASH_MLA_SM86_NS_LARGE       // smem ring stages when BM = 32
#define FLASH_MLA_SM86_NS_LARGE 2
#endif
#ifndef FLASH_MLA_SM86_MINB          // CTAs per SM
#define FLASH_MLA_SM86_MINB 1
#endif
#ifndef FLASH_MLA_SM86_SPEC          // 1 = v2b (producer warp + mbarriers)
#define FLASH_MLA_SM86_SPEC 0
#endif
#ifndef FLASH_MLA_SM86_PIPE          // 1 = v2c (software-pipelined tile loop)
#define FLASH_MLA_SM86_PIPE 0
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
// FLASH_MLA_SM86_CFG="bm,nt,ns[,minb[,spec[,pipe]]]" overrides the whole plan at run time.
inline Plan pick_plan(int M, int page_size) {
    Plan p;
    p.bm = (M <= 16) ? FLASH_MLA_SM86_BM_SMALL : FLASH_MLA_SM86_BM_LARGE;
    p.nt = (p.bm == 16) ? FLASH_MLA_SM86_NT_SMALL : FLASH_MLA_SM86_NT_LARGE;
    p.ns = (p.bm == 16) ? FLASH_MLA_SM86_NS_SMALL : FLASH_MLA_SM86_NS_LARGE;
    p.minb = FLASH_MLA_SM86_MINB;
    p.spec = FLASH_MLA_SM86_SPEC != 0;
    p.pipe = FLASH_MLA_SM86_PIPE != 0;
    if (const char *e = std::getenv("FLASH_MLA_SM86_CFG")) {
        int bm, nt, ns, mb = 1, sp = 0, pi = 0;
        if (std::sscanf(e, "%d,%d,%d,%d,%d,%d", &bm, &nt, &ns, &mb, &sp, &pi) >= 3) {
            p.bm = bm; p.nt = nt; p.ns = ns; p.minb = mb; p.spec = sp != 0; p.pipe = pi != 0;
        }
    }
    if (p.nt > page_size) {   // a tile must lie within one page: shrink it and deepen the ring to keep the bytes in flight
        p.nt = page_size;
        if (p.ns < 3) p.ns = 3;
    }
    return p;
}

// Quantized-KV kernels (fp8 e4m3 / int8): rows per CTA, ring stages, CTAs per SM.
struct PlanQ {
    int bm, nt, ns, minb;
};
#ifndef FLASH_MLA_SM86_QNT            // tokens per tile of the quantized kernels (16 or 32; clamped to the page size)
#define FLASH_MLA_SM86_QNT 32
#endif
#ifndef FLASH_MLA_SM86_QNS            // ring stages of the quantized kernels
#define FLASH_MLA_SM86_QNS 2
#endif
#ifndef FLASH_MLA_SM86_QMINB          // CTAs per SM of the quantized kernels with BM=16
#define FLASH_MLA_SM86_QMINB 1
#endif
// FLASH_MLA_SM86_QCFG="bm,ns[,minb[,nt]]" overrides at run time.
inline PlanQ pick_plan_q(int M) {
    PlanQ p;
    p.bm = (M <= 16) ? 16 : 32;
    p.nt = FLASH_MLA_SM86_QNT;
    p.ns = FLASH_MLA_SM86_QNS;   // 2 stages of 32 tokens (BM 16 and 32): measured best on sm_86
    p.minb = p.bm == 16 ? FLASH_MLA_SM86_QMINB : 1;
    if (const char *e = std::getenv("FLASH_MLA_SM86_QCFG")) {
        int bm, ns, mb = 1, nt = p.nt;
        if (std::sscanf(e, "%d,%d,%d,%d", &bm, &ns, &mb, &nt) >= 2) { p.bm = bm; p.ns = ns; p.minb = mb; p.nt = nt; }
    }
    return p;
}

inline int num_sm_parts_for_q(int M, int h_k, int sm_count) {
    const PlanQ p = pick_plan_q(M);
    const int num_m_blocks = (M + p.bm - 1) / p.bm;
    const int parts = sm_count * p.minb / h_k / num_m_blocks;
    return parts < 1 ? 1 : parts;
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

// Defined in mla_decode_sm86_{bf16,fp16}.cu / mla_decode_sm86_q_{bf16,fp16}.cu.  T = __nv_bfloat16 / __half.
template <typename T>
void mla_sm86_run(Flash_fwd_mla_params &params, cudaStream_t stream);
// quantized KV cache (params.kv_dtype = kKVFp8E4m3 / kKVInt8, params.kv_scale_ptr)
template <typename T>
void mla_sm86_run_q(Flash_fwd_mla_params &params, cudaStream_t stream);

// Workspace layout (bytes).  The self-resetting combine counters sit at a FIXED offset 0 (kCounterBytes, independent of the
// problem size) so that one zero-initialised workspace can be shared by calls with different batch sizes: nothing but the
// counters ever writes there.  The fp32 partial O / LSE buffers follow.
constexpr size_t kCounterBytes = 64 * 1024;
constexpr int kMaxCounters = int(kCounterBytes / sizeof(int));
struct WorkspaceLayout {
    size_t counters_off, counters_bytes, oaccum_off, oaccum_bytes, lseaccum_off, lseaccum_bytes, lsetmp_off, lsetmp_bytes, total;
};
inline int num_counters(int b, int M, int h_k) { return b * h_k * ((M + 15) / 16); }
inline WorkspaceLayout workspace_layout(int b, int M, int h_k, int num_sm_parts) {
    auto up = [](size_t x) { return (x + 255) & ~size_t(255); };
    WorkspaceLayout w;
    const size_t slots = size_t(b) + size_t(num_sm_parts);
    w.counters_off = 0;
    w.counters_bytes = kCounterBytes;
    w.oaccum_off = w.counters_off + w.counters_bytes;
    w.oaccum_bytes = slots * h_k * M * 512 * sizeof(float);
    w.lseaccum_off = w.oaccum_off + up(w.oaccum_bytes);
    w.lseaccum_bytes = slots * h_k * M * sizeof(float);
    w.lsetmp_off = w.lseaccum_off + up(w.lseaccum_bytes);
    w.lsetmp_bytes = size_t(b) * h_k * M * sizeof(float);
    w.total = w.lsetmp_off + up(w.lsetmp_bytes);
    return w;
}

}  // namespace mla_sm86
