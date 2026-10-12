// Torch glue + pybind module `flash_mla_fused_cuda` (separate from flash_mla_cuda so csrc/flash_api.cpp stays untouched).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "mla_decode_prologue.h"

#define CHECK_CUDA_T(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")

static void check_vec16(const at::Tensor &t, const char *name, std::initializer_list<int> strided_dims) {
    TORCH_CHECK(t.stride(-1) == 1, name, ": last dimension must be contiguous");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, ": data pointer must be 16-byte aligned");
    for (int d : strided_dims)
        TORCH_CHECK((t.stride(d) * t.element_size()) % 16 == 0, name, ": stride(", d, ") must be a multiple of 16 bytes");
}

// Fused decode prologue. All outputs are written in place; no allocation, no host sync (CUDA-graph safe).
//   q_nope [T,H,512], q_pe [T,H,64] (any token/head strides), kv_c [T,512], k_pe [T,64]
//   cos_sin [max_pos,64] fp32 (cos | sin), block_table [B,max_blocks] int32, cache_seqlens [B] int32 (length incl. new tokens)
//   q_out [T,H,576] contiguous; kv_cache: native [nblk,page,(1,)576] or quantized uint8 [nblk,page,(1,)640] (+ kv_scale [nblk,page] fp32)
void mla_decode_prologue(const at::Tensor &q_nope, const at::Tensor &q_pe, const at::Tensor &kv_c, const at::Tensor &k_pe,
                         const at::Tensor &cos_sin, const at::Tensor &block_table, const at::Tensor &cache_seqlens,
                         int64_t s_q, at::Tensor &q_out, at::Tensor &kv_cache, c10::optional<at::Tensor> kv_scale,
                         c10::optional<at::Tensor> norm_w, double eps, bool interleaved, int64_t kv_dtype) {
    for (const at::Tensor *t : std::initializer_list<const at::Tensor *>{&q_nope, &q_pe, &kv_c, &k_pe, &cos_sin, &block_table,
                                                                          &cache_seqlens, &q_out, &kv_cache})
        CHECK_CUDA_T(*t);
    TORCH_CHECK(q_nope.dim() == 3 && q_nope.size(2) == 512, "q_nope must be [T,H,512]");
    const int64_t T = q_nope.size(0), H = q_nope.size(1);
    TORCH_CHECK(q_pe.dim() == 3 && q_pe.size(0) == T && q_pe.size(1) == H && q_pe.size(2) == 64, "q_pe must be [T,H,64]");
    TORCH_CHECK(kv_c.dim() == 2 && kv_c.size(0) == T && kv_c.size(1) == 512, "kv_c must be [T,512]");
    TORCH_CHECK(k_pe.dim() == 2 && k_pe.size(0) == T && k_pe.size(1) == 64, "k_pe must be [T,64]");
    const auto dt = q_nope.scalar_type();
    TORCH_CHECK(dt == at::kBFloat16 || dt == at::kHalf, "q_nope must be bf16 or fp16");
    TORCH_CHECK(q_pe.scalar_type() == dt && kv_c.scalar_type() == dt && k_pe.scalar_type() == dt && q_out.scalar_type() == dt,
                "q_nope, q_pe, kv_c, k_pe, q_out must share one dtype");
    check_vec16(q_nope, "q_nope", {0, 1});
    check_vec16(kv_c, "kv_c", {0});
    TORCH_CHECK(q_pe.stride(-1) == 1 && k_pe.stride(-1) == 1, "q_pe / k_pe: last dimension must be contiguous");
    TORCH_CHECK(q_out.is_contiguous() && q_out.dim() == 3 && q_out.size(0) == T && q_out.size(1) == H && q_out.size(2) == 576,
                "q_out must be contiguous [T,H,576]");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(q_out.data_ptr()) % 16 == 0, "q_out must be 16-byte aligned");
    TORCH_CHECK(cos_sin.scalar_type() == at::kFloat && cos_sin.dim() == 2 && cos_sin.size(1) == 64 && cos_sin.is_contiguous(),
                "cos_sin must be contiguous fp32 [max_pos,64]");
    TORCH_CHECK(block_table.scalar_type() == at::kInt && block_table.dim() == 2 && block_table.stride(-1) == 1,
                "block_table must be int32 [B,max_blocks]");
    TORCH_CHECK(cache_seqlens.scalar_type() == at::kInt && cache_seqlens.is_contiguous(), "cache_seqlens must be contiguous int32 [B]");
    TORCH_CHECK(s_q >= 1 && T % s_q == 0, "T must be a multiple of s_q");
    const int64_t B = T / s_q;
    TORCH_CHECK(block_table.size(0) == B && cache_seqlens.size(0) == B, "block_table / cache_seqlens batch must be T / s_q");
    TORCH_CHECK(kv_cache.is_contiguous(), "kv_cache must be contiguous");
    TORCH_CHECK(kv_cache.dim() >= 3, "kv_cache must be [nblk,page,(1,)row]");
    TORCH_CHECK(kv_dtype >= 0 && kv_dtype <= 2, "kv_dtype must be 0 (native), 1 (fp8 e4m3) or 2 (int8)");
    const int64_t nblk = kv_cache.size(0), page = kv_cache.size(1);
    MlaPrologueParams p{};
    if (kv_dtype == 0) {
        TORCH_CHECK(kv_cache.scalar_type() == dt && kv_cache.size(-1) == 576, "native kv_cache must have the q dtype and 576 columns");
    } else {
        TORCH_CHECK(dt == at::kBFloat16, "quantized caches need bf16 inputs (docs/kv_quant_format.md)");
        TORCH_CHECK(kv_cache.scalar_type() == at::kByte && kv_cache.size(-1) == 640, "quantized kv_cache must be uint8 [..,640]");
        TORCH_CHECK(kv_scale.has_value() && kv_scale->scalar_type() == at::kFloat && kv_scale->is_contiguous() &&
                    kv_scale->numel() == nblk * page, "quantized caches need a contiguous fp32 kv_scale [nblk,page]");
        CHECK_CUDA_T(*kv_scale);
        p.kv_scale = kv_scale->data_ptr<float>();
    }
    TORCH_CHECK(reinterpret_cast<uintptr_t>(kv_cache.data_ptr()) % 16 == 0, "kv_cache must be 16-byte aligned");
    if (norm_w.has_value()) {
        CHECK_CUDA_T(*norm_w);
        TORCH_CHECK(norm_w->scalar_type() == dt && norm_w->numel() == 512 && norm_w->is_contiguous(), "norm_w must be [512] with the q dtype");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(norm_w->data_ptr()) % 16 == 0, "norm_w must be 16-byte aligned");
        p.norm_w = norm_w->data_ptr();
    }

    c10::cuda::CUDAGuard guard(q_nope.device());
    p.q_nope = q_nope.data_ptr();
    p.q_pe = q_pe.data_ptr();
    p.kv_c = kv_c.data_ptr();
    p.k_pe = k_pe.data_ptr();
    p.q_nope_stride_t = q_nope.stride(0);
    p.q_nope_stride_h = q_nope.stride(1);
    p.q_pe_stride_t = q_pe.stride(0);
    p.q_pe_stride_h = q_pe.stride(1);
    p.kv_c_stride_t = kv_c.stride(0);
    p.k_pe_stride_t = k_pe.stride(0);
    p.cos_sin = cos_sin.data_ptr<float>();
    p.max_pos = static_cast<int>(cos_sin.size(0));
    p.block_table = block_table.data_ptr<int>();
    p.block_table_stride = block_table.stride(0);
    p.seqlens = cache_seqlens.data_ptr<int>();
    p.s_q = static_cast<int>(s_q);
    p.T = static_cast<int>(T);
    p.H = static_cast<int>(H);
    p.page = static_cast<int>(page);
    p.q_out = q_out.data_ptr();
    p.kv_cache = kv_cache.data_ptr();
    p.eps = static_cast<float>(eps);
    p.kv_dtype = static_cast<int>(kv_dtype);
    p.interleaved = interleaved ? 1 : 0;
    p.is_bf16 = dt == at::kBFloat16 ? 1 : 0;
    TORCH_CHECK(T <= 65535, "T (grid.y) must be <= 65535");
    launch_mla_decode_prologue(p, at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA fused decode prologue (RoPE + q concat + paged KV write, optional FP8/INT8 quantize-on-write)";
    m.def("mla_decode_prologue", &mla_decode_prologue, "fused MLA decode prologue (in place)");
}
