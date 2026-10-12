// Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/flash_api.cpp
/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/fast_math.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "flash_mla.h"
#include "static_switch.h"
#include "sm86/mla_sm86.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Tile-scheduler metadata.  `num_q_tokens_per_head_k` = s_q * h_q / h_k (the number of query rows sharing one KV head).
// Fully device-side (no host sync): safe to call inside CUDA-Graph capture.
std::vector<at::Tensor>
get_mla_metadata(
    at::Tensor &seqlens_k,
    const int num_q_tokens_per_head_k,
    const int num_heads_k
) {
    CHECK_DEVICE(seqlens_k);
    TORCH_CHECK(seqlens_k.is_contiguous());
    TORCH_CHECK(seqlens_k.dtype() == torch::kInt32);

    auto dprops = at::cuda::getCurrentDeviceProperties();
    const bool is_sm90 = dprops->major >= 9;
    int batch_size = seqlens_k.size(0);
    int *seqlens_k_ptr = seqlens_k.data_ptr<int>();
    auto options = seqlens_k.options();

    int sm_count = dprops->multiProcessorCount;
    int block_size_n, fixed_overhead_num_blocks, num_sm_parts, min_blocks_per_part = 0;
    if (is_sm90) {   // original sm90 scheduling constants
        block_size_n = 64;
        fixed_overhead_num_blocks = 5;
        num_sm_parts = sm_count / num_heads_k / cutlass::ceil_div(num_q_tokens_per_head_k, 32);
    } else {
        block_size_n = mla_sm86::sched_granule(num_q_tokens_per_head_k);
        fixed_overhead_num_blocks = mla_sm86::sched_overhead_blocks();
        min_blocks_per_part = mla_sm86::sched_min_blocks_per_part();
        num_sm_parts = mla_sm86::num_sm_parts_for(num_q_tokens_per_head_k, num_heads_k, sm_count);
    }
    num_sm_parts = std::max(num_sm_parts, 1);

    auto tile_scheduler_metadata = torch::empty({num_sm_parts, TileSchedulerMetaDataSize}, options);
    auto num_splits = torch::empty({batch_size + 1}, options);
    int *tile_scheduler_metadata_ptr = tile_scheduler_metadata.data_ptr<int>();
    int *num_splits_ptr = num_splits.data_ptr<int>();

    at::cuda::CUDAGuard device_guard{(char)seqlens_k.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    Mla_metadata_params params = {};
    params.seqlens_k_ptr = seqlens_k_ptr;
    params.tile_scheduler_metadata_ptr = tile_scheduler_metadata_ptr;
    params.num_splits_ptr = num_splits_ptr;
    params.batch_size = batch_size;
    params.block_size_n = block_size_n;
    params.fixed_overhead_num_blocks = fixed_overhead_num_blocks;
    params.num_sm_parts = num_sm_parts;
    params.min_blocks_per_part = min_blocks_per_part;
    get_mla_metadata_func(params, stream);

    return {tile_scheduler_metadata, num_splits};
}

// Bytes of workspace needed by impl="v2" for the given problem (see mla_sm86::workspace_layout).
int64_t
get_mla_workspace_bytes(const int batch_size, const int num_q_tokens_per_head_k, const int num_heads_k, const int num_sm_parts) {
    return (int64_t)mla_sm86::workspace_layout(batch_size, num_q_tokens_per_head_k, num_heads_k, num_sm_parts).total;
}

std::vector<at::Tensor>
mha_fwd_kvcache_mla(
    at::Tensor &q,                               // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor &kcache,                    // num_blocks x page_block_size x num_heads_k x head_size
    std::optional<const at::Tensor> &vcache_,    // num_blocks x page_block_size x num_heads_k x head_size_v
    const int head_size_v,
    const at::Tensor &seqlens_k,                 // batch_size
    const at::Tensor &block_table,               // batch_size x max_num_blocks_per_seq
    const float softmax_scale,
    bool is_causal,
    const at::Tensor &tile_scheduler_metadata,   // num_sm_parts x TileSchedulerMetaDataSize
    const at::Tensor &num_splits,                // batch_size + 1
    const std::string &impl,                     // "v2" (sm86 kernels) | "legacy" | "legacy_ws"
    std::optional<at::Tensor> out_,              // optional preallocated output (b, s_q, h_q, head_size_v)
    std::optional<at::Tensor> lse_,              // optional preallocated lse (b, h_q, s_q), fp32
    std::optional<at::Tensor> workspace_,        // optional preallocated workspace (bytes >= get_mla_workspace_bytes)
    std::optional<at::Tensor> kv_scale_          // per-token fp32 dequant scale for quantized KV (unused for native KV)
) {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    TORCH_CHECK(is_sm90 || is_sm8x, "Only sm80 to sm90 (inclusive) are supported");
    TORCH_CHECK(impl == "v2" || impl == "legacy" || impl == "legacy_ws", "impl must be one of 'v2', 'legacy', 'legacy_ws'");
    const bool use_v2 = impl == "v2";

    at::Tensor vcache = vcache_.has_value() ? vcache_.value() : kcache;

    auto q_dtype = q.dtype();
    TORCH_CHECK(kcache.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(!kv_scale_.has_value(), "quantized KV cache is not supported yet");

    CHECK_DEVICE(q); CHECK_DEVICE(kcache); CHECK_DEVICE(vcache);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(kcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(vcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    CHECK_DEVICE(block_table);
    TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
    TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");

    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    const int seqlen_q_ori = sizes[1];
    const int num_heads_ori = sizes[2];
    const int head_size = sizes[3];
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size_v % 32 == 0, "head_size_v should be a multiple of 32");

    const int max_num_blocks_per_seq = block_table.size(1);
    const int num_blocks = kcache.size(0);
    const int page_block_size = kcache.size(1);
    const int num_heads_k = kcache.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be postive");
    TORCH_CHECK(num_heads_ori % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (seqlen_q_ori == 1) { is_causal = false; }

    const int ngroups = num_heads_ori / num_heads_k;
    const int seqlen_q = seqlen_q_ori * ngroups;
    const int num_heads = num_heads_k;
    q = q.view({batch_size, seqlen_q_ori, num_heads_k, ngroups, head_size}).transpose(2, 3)
            .reshape({batch_size, seqlen_q, num_heads, head_size});

    int head_size_k = head_size;
    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(kcache, num_blocks, page_block_size, num_heads_k, head_size_k);
    if (vcache_.has_value()) { CHECK_SHAPE(vcache, num_blocks, page_block_size, num_heads_k, head_size_v); }
    CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);


    TORCH_CHECK(seqlens_k.dtype() == torch::kInt32, "seqlens_k must have dtype int32");
    CHECK_DEVICE(seqlens_k);
    CHECK_CONTIGUOUS(seqlens_k);
    CHECK_SHAPE(seqlens_k, batch_size);

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();

    TORCH_CHECK(tile_scheduler_metadata.dtype() == torch::kInt32, "tile_scheduler_metadata must have dtype int32");
    TORCH_CHECK(tile_scheduler_metadata.size(1) == TileSchedulerMetaDataSize);
    CHECK_DEVICE(tile_scheduler_metadata);
    CHECK_CONTIGUOUS(tile_scheduler_metadata);
    TORCH_CHECK(num_splits.dtype() == torch::kInt32, "num_splits must have dtype int32");
    CHECK_DEVICE(num_splits);
    CHECK_CONTIGUOUS(num_splits);
    const int num_sm_parts = tile_scheduler_metadata.size(0);

    // ---- output / lse (kernel layout: out (b, M, h_k, dv), lse (b, h_k, M), M = s_q * ngroups) ----
    at::Tensor out, softmax_lse;
    bool out_direct = false, lse_direct = false;
    if (out_.has_value()) {
        const at::Tensor &o = out_.value();
        CHECK_DEVICE(o);
        TORCH_CHECK(o.dtype() == q_dtype, "out must have the same dtype as q");
        CHECK_SHAPE(o, batch_size, seqlen_q_ori, num_heads_ori, head_size_v);
        out_direct = (num_heads_k == 1) && o.is_contiguous();
    }
    if (lse_.has_value()) {
        const at::Tensor &l = lse_.value();
        CHECK_DEVICE(l);
        TORCH_CHECK(l.dtype() == torch::kFloat, "lse must be float32");
        CHECK_SHAPE(l, batch_size, num_heads_ori, seqlen_q_ori);
        lse_direct = (seqlen_q_ori == 1) && l.is_contiguous();   // kernel layout == final layout only for s_q == 1
    }
    out = out_direct ? out_.value().view({batch_size, seqlen_q, num_heads, head_size_v})
                     : torch::empty({batch_size, seqlen_q, num_heads, head_size_v}, opts);
    softmax_lse = lse_direct ? lse_.value().view({batch_size, num_heads, seqlen_q})
                             : torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));

    Flash_fwd_mla_params params = {};
    // Set the sizes.
    params.b = batch_size;
    params.seqlen_q = seqlen_q;
    params.cu_seqlens_k = seqlens_k.data_ptr<int>();
    params.h = num_heads;
    params.h_h_k_ratio = num_heads / num_heads_k;
    params.ngroups = ngroups;
    params.is_causal = is_causal;
    params.d = head_size;
    params.d_v = head_size_v;
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = float(softmax_scale * M_LOG2E);
    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = kcache.data_ptr();
    params.v_ptr = vcache.data_ptr();
    params.o_ptr = out.data_ptr();
    params.softmax_lse_ptr = softmax_lse.data_ptr();
    // All stride are in elements, not bytes.
    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = kcache.stride(0);
    params.v_batch_stride = vcache.stride(0);
    params.o_batch_stride = out.stride(0);
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = kcache.stride(-3);
    params.v_row_stride = vcache.stride(-3);
    params.o_row_stride = out.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = kcache.stride(-2);
    params.v_head_stride = vcache.stride(-2);
    params.o_head_stride = out.stride(-2);

    params.block_table = block_table.data_ptr<int>();
    params.block_table_batch_stride = block_table.stride(0);
    params.page_block_size = page_block_size;

    params.tile_scheduler_metadata_ptr = tile_scheduler_metadata.data_ptr<int>();
    params.num_sm_parts = num_sm_parts;
    params.num_splits_ptr = num_splits.data_ptr<int>();

    // ---- split-KV partial buffers (workspace or per-call allocation) ----
    at::Tensor softmax_lse_accum, out_accum, lse_tmp;
    if (workspace_.has_value()) {
        const at::Tensor &ws = workspace_.value();
        CHECK_DEVICE(ws);
        CHECK_CONTIGUOUS(ws);
        const auto lay = mla_sm86::workspace_layout(batch_size, seqlen_q, num_heads, num_sm_parts);
        TORCH_CHECK((size_t)(ws.numel() * ws.element_size()) >= lay.total, "workspace too small: need ", lay.total, " bytes");
        char *base = reinterpret_cast<char *>(ws.data_ptr());
        TORCH_CHECK(reinterpret_cast<uintptr_t>(base) % 256 == 0, "workspace must be 256-byte aligned");
        params.oaccum_ptr = base;
        params.softmax_lseaccum_ptr = base + lay.lseaccum_off;
        params.workspace = base;
        params.combine_counters = reinterpret_cast<int *>(base + lay.counters_off);
        params.fuse_max_splits = mla_sm86::fuse_max_splits_default();
        if (!lse_direct && use_v2 && lse_.has_value()) params.softmax_lse_ptr = base + lay.lsetmp_off;   // staged, then copied into lse
    } else {
        softmax_lse_accum = torch::empty({batch_size + num_sm_parts, num_heads, seqlen_q}, opts.dtype(at::kFloat));
        out_accum = torch::empty({batch_size + num_sm_parts, num_heads, seqlen_q, head_size_v}, opts.dtype(at::kFloat));
        params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
        params.oaccum_ptr = out_accum.data_ptr();
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(head_size == 576);

    if (use_v2) {
        TORCH_CHECK(is_sm8x, "impl='v2' targets sm80/sm86");
        TORCH_CHECK(head_size_v == 512, "impl='v2' requires head_size_v == 512");
        TORCH_CHECK(!vcache_.has_value(), "impl='v2' requires K and V to share the cache");
        TORCH_CHECK(page_block_size == 16 || page_block_size == 32 || page_block_size == 64,
                    "impl='v2' supports page_block_size 16/32/64");
        TORCH_CHECK(q.stride(-3) % 2 == 0 && reinterpret_cast<uintptr_t>(q.data_ptr()) % 4 == 0, "q rows must be 4-byte aligned");
        TORCH_CHECK(kcache.stride(-3) % 8 == 0 && kcache.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(kcache.data_ptr()) % 16 == 0,
                    "k_cache rows must be 16-byte aligned");
        TORCH_CHECK(out.stride(-3) % 2 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 4 == 0, "out rows must be 4-byte aligned");
        if (q_dtype == torch::kBFloat16) {
            mla_sm86::mla_sm86_run<__nv_bfloat16>(params, stream);
        } else if (q_dtype == torch::kHalf) {
            mla_sm86::mla_sm86_run<__half>(params, stream);
        } else {
            TORCH_CHECK(false, "Unsupported tensor dtype for query");
        }
    } else if (q_dtype == torch::kBFloat16) {
        if (is_sm90) {
#ifndef FLASH_MLA_NO_SM90
            mha_fwd_splitkv_mla<cutlass::bfloat16_t, 576, true>::run(params, stream);
#else
            TORCH_CHECK(false, "this build has no sm90 kernels (FLASH_MLA_ARCH=80/86)");
#endif
        } else if (impl == "legacy_ws") {
            mha_fwd_splitkv_mla_ws<cutlass::bfloat16_t, 576>::run(params, stream);
        } else {
            mha_fwd_splitkv_mla<cutlass::bfloat16_t, 576, false>::run(params, stream);
        }
    }
    #ifndef FLASH_MLA_DISABLE_FP16
    else if (q_dtype == torch::kHalf) {
        if (is_sm90) {
#ifndef FLASH_MLA_NO_SM90
            mha_fwd_splitkv_mla<cutlass::half_t, 576, true>::run(params, stream);
#else
            TORCH_CHECK(false, "this build has no sm90 kernels (FLASH_MLA_ARCH=80/86)");
#endif
        } else {
            TORCH_CHECK(false, "legacy sm80 kernels only support bfloat16");
        }
    }
    #endif
    else {
        TORCH_CHECK(false, "Unsupported tensor dtype for query");
    }

    // ---- back to the user-visible layouts ----
    if (!out_direct) {
        at::Tensor out_final = out.view({batch_size, seqlen_q_ori, ngroups, num_heads_k, head_size_v}).transpose(2, 3)
                .reshape({batch_size, seqlen_q_ori, num_heads_ori, head_size_v});
        if (out_.has_value()) { out_.value().copy_(out_final); out_final = out_.value(); }
        out = out_final;
    } else {
        out = out_.value();
    }
    if (!lse_direct) {
        at::Tensor lse_view = softmax_lse;
        if (params.softmax_lse_ptr != softmax_lse.data_ptr()) {   // lse kernel output lives in the workspace
            lse_view = torch::from_blob(params.softmax_lse_ptr, {batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
        }
        at::Tensor lse_final = lse_view.view({batch_size, num_heads_k, seqlen_q_ori, ngroups}).transpose(2, 3)
                .reshape({batch_size, num_heads_ori, seqlen_q_ori});
        if (lse_.has_value()) { lse_.value().copy_(lse_final); lse_final = lse_.value(); }
        softmax_lse = lse_final;
    } else {
        softmax_lse = lse_.value();
    }

    return {out, softmax_lse};
}

// Human-readable description of the sm86 kernel configuration chosen for a problem (compile-time defaults + env overrides).
std::string sm86_config(const int num_q_tokens_per_head_k, const int page_block_size) {
    const auto p = mla_sm86::pick_plan(num_q_tokens_per_head_k, page_block_size);
    char buf[256];
    std::snprintf(buf, sizeof(buf), "bm=%d nt=%d stages=%d ctas_per_sm=%d spec=%d fuse_max_splits=%d sched_ov=%d sched_min_blocks=%d",
                  p.bm, p.nt, p.ns, p.minb, (int)p.spec, mla_sm86::fuse_max_splits_default(), mla_sm86::sched_overhead_blocks(),
                  mla_sm86::sched_min_blocks_per_part());
    return std::string(buf);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA";
    m.def("get_mla_metadata", &get_mla_metadata);
    m.def("get_mla_workspace_bytes", &get_mla_workspace_bytes);
    m.def("sm86_config", &sm86_config);
    m.def("fwd_kvcache_mla", &mha_fwd_kvcache_mla);
}
