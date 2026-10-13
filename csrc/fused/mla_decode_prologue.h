#pragma once
// Fused MLA decode prologue (one launch): RoPE(q_pe), RoPE(k_pe), q = [q_nope_absorbed | q_pe_rot],
// optional RMSNorm of kv_c, append [kv_c | k_pe_rot] to the paged KV cache (bf16/fp16 or FP8/INT8 + per-token scale).
// Layout of the quantized cache: docs/kv_quant_format.md.
#include <cstdint>
#include <cuda_runtime.h>

struct MlaPrologueParams {
    // inputs; strides in elements, last dim contiguous, 16-byte aligned rows (checked by the host glue)
    const void *q_nope;      // [T, H, 512]  (q_nope after the W_UK absorption bmm)
    const void *q_pe;        // [T, H, 64]   (not yet rotated)
    const void *kv_c;        // [T, 512]     (output of kv_a_proj, before kv_a_layernorm unless norm_w == nullptr)
    const void *k_pe;        // [T, 64]      (not yet rotated)
    int64_t q_nope_stride_t, q_nope_stride_h;
    int64_t q_pe_stride_t, q_pe_stride_h;
    int64_t kv_c_stride_t, k_pe_stride_t;
    const float *cos_sin;    // [max_pos, 64]: cos[0..31] | sin[0..31]
    int max_pos;
    const int *block_table;  // [B, max_blocks]
    int64_t block_table_stride;
    const int *seqlens;      // [B], cache length INCLUDING the s_q new tokens; token (b, j) is written at seqlens[b] - s_q + j
    int s_q, T, H, page;
    // outputs
    void *q_out;             // [T, H, 576]
    void *kv_cache;          // bf16/fp16 [nblk, page, 576]  or  uint8 [nblk, page, 640]
    float *kv_scale;         // [nblk, page] fp32, quantized caches only
    // options
    const void *norm_w;      // [512] RMSNorm weight (same dtype as kv_c) or nullptr
    float eps;
    int kv_dtype;            // 0 native, 1 fp8 e4m3, 2 int8 (flash_mla.h: kKVNative / kKVFp8E4m3 / kKVInt8)
    int interleaved;         // 0: neox/half-split rotation (i, i+32); 1: GPT-J/interleaved rotation (2i, 2i+1)
    int is_bf16;             // 1: bf16, 0: fp16
};

void launch_mla_decode_prologue(const MlaPrologueParams &p, cudaStream_t stream);
