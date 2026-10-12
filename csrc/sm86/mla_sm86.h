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

// Config selection by the number of rows M = s_q * h_q / h_k (shared by the kernel launcher and the metadata builder).
inline Plan pick_plan(int M, int page_size) {
    Plan p;
    p.bm = (M <= 16) ? 16 : 32;
    p.nt = 16;
    p.ns = 4;
    p.minb = 1;
    p.spec = false;
    if (const char *e = std::getenv("FLASH_MLA_SM86_CFG")) {   // "bm,nt,ns[,minb[,spec]]" for tuning sweeps
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

// Granularity (tokens) of the tile scheduler; must be a multiple of every Nt used.
constexpr int kSchedGranule = 32;
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
