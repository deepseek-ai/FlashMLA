#pragma once

#include "kernel.h"

#include <numeric>

#include "acl/acl.h"
#include "kernel_operator.h"
#include "c_api/asc_simd.h"

#include "kerutils/kerutils_for_ascend_npu.h"

#include "config.h"

// int32 * int32 -> int64
[[maybe_unused]] __simd_callee__ inline void asc_mul(vector_int64_t &dst0, vector_int64_t &dst1, vector_int32_t src0, vector_int32_t src1, vector_bool mask) {
    vector_int32_t dst_lo32bit, dst_hi32bit;
    asc_mull(dst_lo32bit, dst_hi32bit, src0, src1, mask);
    asc_intlv(*(vector_int32_t*)&dst0, *(vector_int32_t*)&dst1, dst_lo32bit, dst_hi32bit);
}

[[maybe_unused]] __simd_callee__ inline void asc_duplicate_scalar(vector_int64_t &dst, int32_t src_lo32bit, int32_t src_hi32bit) {
    vector_int32_t src_lo32bit_vec, src_hi32bit_vec;
    asc_duplicate_scalar(src_lo32bit_vec, src_lo32bit);
    asc_duplicate_scalar(src_hi32bit_vec, src_hi32bit);
    vector_int64_t tmp;
    asc_intlv(*(vector_int32_t*)&dst, *(vector_int32_t*)&tmp, src_lo32bit_vec, src_hi32bit_vec);
}

// int32 * int32 + int64 (bias_lo, bias_hi) -> int64
// Return the lo & hi 32bit of the result
[[maybe_unused]] __simd_callee__ inline void asc_mul_and_add_wo_intlv(vector_int32_t &dst_lo32bit, vector_int32_t &dst_hi32bit, vector_int32_t src0, vector_int32_t src1, vector_int32_t bias_lo, vector_int32_t bias_hi, vector_bool mask) {
    asc_mull(dst_lo32bit, dst_hi32bit, src0, src1, mask);
    vector_bool carry;
    asc_add(carry, dst_lo32bit, dst_lo32bit, bias_lo, mask);
    asc_addc(carry, dst_hi32bit, dst_hi32bit, bias_hi, carry, mask);
}

// int32 * int32 + int64 (bias_lo, bias_hi) -> int64
// dst0 is the first half part of the result, and dst1 is the second half part of the result
[[maybe_unused]] __simd_callee__ inline void asc_mul_and_add(vector_int64_t &dst0, vector_int64_t &dst1, vector_int32_t src0, vector_int32_t src1, vector_int32_t bias_lo, vector_int32_t bias_hi, vector_bool mask) {
    vector_int32_t dst_lo32bit, dst_hi32bit;
    asc_mul_and_add_wo_intlv(dst_lo32bit, dst_hi32bit, src0, src1, bias_lo, bias_hi, mask);
    asc_intlv(*(vector_int32_t*)&dst0, *(vector_int32_t*)&dst1, dst_lo32bit, dst_hi32bit);
}
    
[[maybe_unused]] __simd_callee__ inline void asc_storealign_postupdate_i64x2(__ubuf__ int64_t* &dst_ptr, const vector_int64_t &src0, const vector_int64_t &src1, vector_bool full_mask) {
    asc_storealign_postupdate(
        (__ubuf__ int32_t*&)dst_ptr,
        *(vector_int32_t*)&src0,
        NUM_UINT64_IN_VEC * 2,
        full_mask
    );
    asc_storealign_postupdate(
        (__ubuf__ int32_t*&)dst_ptr,
        *(vector_int32_t*)&src1,
        NUM_UINT64_IN_VEC * 2,
        full_mask
    );
}

// Return (t+1) >= MOD ? t+1-MOD : t+1
template<uint32_t MOD>
__aicore__ inline void plus_one_and_mod(uint32_t &t) {
    t += 1;
    t = (t >= MOD ? t - MOD : t);
}

template<pipe_t PIPE>
__aicore__ inline void cross_core_set_flag_2aiv(uint32_t flag_id) {
    if ASC_IS_AIC {
        AscendC::CrossCoreSetFlag<4, PIPE>(flag_id);
        AscendC::CrossCoreSetFlag<4, PIPE>(flag_id + 16);
    }
}

template<pipe_t PIPE>
__aicore__ inline void cross_core_wait_flag_2aiv(uint32_t flag_id) {
    if ASC_IS_AIC {
        AscendC::CrossCoreWaitFlag<4, PIPE>(flag_id);
        AscendC::CrossCoreWaitFlag<4, PIPE>(flag_id + 16);
    }
}

namespace ascend::prefill::sparse_fwd {

template<Config CONFIG>
template<ModelType CACHE_MODEL_TYPE>
__simd_vf__ void Kernel<CONFIG>::process_kv_index_vf(
    __ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t index_buf_idx,
    uint32_t s_kv, uint32_t stride_kv_block, int32_t valid_len, // Use `int32_t` dtype for `valid_len` because we're going to subtract its
    uint32_t page_block_size, FastDivMod page_block_size_fast_div_mod
) {
    using CacheFormat = KVCacheFormat<CACHE_MODEL_TYPE>;
    vector_bool full_mask = asc_create_mask_b32(PAT_ALL);
    vector_int32_t full_stride_vec;
    [[maybe_unused]] vector_int32_t full_stride_kv_block_vec, full_zeros_vec, full_0xffffffff_vec;
    [[maybe_unused]] vector_uint32_t full_block_size_fastdivmod_multiplier_vec;
    if constexpr (IS_DECODE) {
        asc_duplicate_scalar(full_stride_kv_block_vec, stride_kv_block);
        asc_duplicate_scalar(full_block_size_fastdivmod_multiplier_vec, page_block_size_fast_div_mod.multiplier);
        asc_duplicate_scalar(full_zeros_vec, 0);
        asc_duplicate_scalar(full_0xffffffff_vec, 0xffffffff);
    } else {
        asc_duplicate_scalar(
            full_stride_vec,
            D_QK * sizeof(bf16)     // We've asserted about `params.stride_kv_s_kv`
        );
    }
    if constexpr (IS_DECODE) {
        // Override s_kv for decode to let compiler optimize more aggressively
        s_kv = 0x80000000u;
    }

    vector_int32_t rel_offset_even, rel_offset_odd;
    asc_arange(rel_offset_even, 0);
    asc_arange(rel_offset_odd, NUM_UINT32_IN_VEC);
    asc_deintlv(rel_offset_even, rel_offset_odd, rel_offset_even, rel_offset_odd);

    __ubuf__ int32_t* cur_indices_ptr = ub_buf.indices;
    __ubuf__ uint32_t* cur_smaller_indices_ptr = ub_buf.smaller_indices[index_buf_idx];
    __ubuf__ uint32_t* cur_larger_indices_ptr = ub_buf.larger_indices[index_buf_idx];
    __ubuf__ int64_t* cur_global_kv_offset_ptr = ub_buf.global_kv_ptr_offset;
    __ubuf__ int64_t* cur_src_stride_in_gather2_ptr = ub_buf.src_stride_in_gather2;

    static_assert(INDEX_BLOCK_SIZE % (NUM_UINT32_IN_VEC*2) == 0);
    uint16_t num_index_vecs = INDEX_BLOCK_SIZE / NUM_UINT32_IN_VEC;
    if constexpr (IS_DECODE) {
        // Process only the index vectors consumed by this cache segment.
        // Keep one pair for an empty segment.
        uint32_t num_indices = valid_len > 0 ? (uint32_t)valid_len : 1u;
        num_indices = num_indices < INDEX_BLOCK_SIZE ? num_indices : INDEX_BLOCK_SIZE;
        num_index_vecs = ((num_indices + 2*NUM_UINT32_IN_VEC - 1) / (2*NUM_UINT32_IN_VEC)) * 2;
    }
    #pragma unroll 1
    for (uint16_t i = 0; i < num_index_vecs; i += 2) {
        // Load indices from UB
        vector_int32_t cur_indices_lo, cur_indices_hi;
        asc_loadalign_postupdate(
            cur_indices_lo,
            cur_indices_ptr,
            NUM_UINT32_IN_VEC
        );
        asc_loadalign_postupdate(
            cur_indices_hi,
            cur_indices_ptr,
            NUM_UINT32_IN_VEC
        );

        // Deinterleave indices. Group them by odd/even
        vector_uint32_t cur_indices_odd, cur_indices_even;
        asc_deintlv(cur_indices_even, cur_indices_odd, *(vector_uint32_t*)&cur_indices_lo, *(vector_uint32_t*)&cur_indices_hi);

        // Mask by `valid_len`
        if (valid_len < (int32_t)NUM_UINT32_IN_VEC*2) {
            vector_bool is_invalid_even, is_invalid_odd;
            asc_ge_scalar(is_invalid_even, *(vector_int32_t*)&rel_offset_even, valid_len, full_mask);
            asc_ge_scalar(is_invalid_odd, *(vector_int32_t*)&rel_offset_odd, valid_len, full_mask);
            vdup(cur_indices_even, s_kv, is_invalid_even, MODE_MERGING);
            vdup(cur_indices_odd, s_kv, is_invalid_odd, MODE_MERGING);
        }
        valid_len -= (int32_t)NUM_UINT32_IN_VEC*2;

        // Resolve smaller & larger indices from odd & even indices
        vector_uint32_t cur_indices_smaller, cur_indices_larger;
        asc_min(cur_indices_smaller, cur_indices_even, cur_indices_odd, full_mask);
        asc_max(cur_indices_larger, cur_indices_even, cur_indices_odd, full_mask);

        // Get validness mask
        vector_bool is_valid_larger;
        asc_lt_scalar(is_valid_larger, cur_indices_larger, s_kv, full_mask);    // [INT_MIN, -1] will be wrapped to [INT_MAX+1, UINT_MAX] after converison, so we only have to compare the uint32_t-interpreted `cur_indices` with `s_kv`

        // Calculate global ptr offset
        vector_int64_t global_ptr_offset_lo, global_ptr_offset_hi;
        vector_int64_t src_stride_lo, src_stride_hi;
        if constexpr (!IS_DECODE) {
            // Prefill
            // global_ptr_offset = indices_smaller * (D_QK * sizeof(bf16))
            asc_min_scalar(cur_indices_smaller, cur_indices_smaller, s_kv, full_mask);  // The `global_ptr_offset` of positions where both smaller & larger indices are invalid will be s_kv * (D_QK * sizeof(bf16))
            asc_mul(global_ptr_offset_lo, global_ptr_offset_hi, *(vector_int32_t*)&cur_indices_smaller, full_stride_vec, full_mask);
            vector_int32_t cur_indices_absdiff;
            asc_sub(cur_indices_absdiff, *(vector_int32_t*)&cur_indices_larger, *(vector_int32_t*)&cur_indices_smaller, is_valid_larger);
            asc_mul(src_stride_lo, src_stride_hi, cur_indices_absdiff, full_stride_vec, full_mask);
        } else {
            // Decode
            vector_bool is_valid_smaller;
            asc_lt_scalar(is_valid_smaller, cur_indices_smaller, s_kv, full_mask);
            
            // Define `GET_BLOCK_IDX_AND_OFFSET_IN_BLOCK` as a macro because lambda with `__simd_callee__` makes bisheng crash
            #define GET_BLOCK_IDX_AND_OFFSET_IN_BLOCK(block_idx, offset_in_block, indices) \
                { \
                    vector_uint32_t tmp; \
                    asc_mull(tmp, (block_idx), (indices), full_block_size_fastdivmod_multiplier_vec, full_mask); \
                    asc_shiftright_scalar((block_idx), (block_idx), page_block_size_fast_div_mod.right_shift_amount, full_mask); \
                    asc_mul_scalar(tmp, (block_idx), page_block_size, full_mask); \
                    asc_sub((offset_in_block), (indices), tmp, full_mask); \
                }
            vector_uint32_t block_idx_smaller, offset_in_block_smaller;
            vector_uint32_t block_idx_larger, offset_in_block_larger;
            GET_BLOCK_IDX_AND_OFFSET_IN_BLOCK(block_idx_smaller, offset_in_block_smaller, cur_indices_smaller);
            GET_BLOCK_IDX_AND_OFFSET_IN_BLOCK(block_idx_larger, offset_in_block_larger, cur_indices_larger);
            #undef GET_BLOCK_IDX_AND_OFFSET_IN_BLOCK

            // global_kv_ptr_offset = block_idx_smaller * stride_kv_block + offset_in_block_smaller * CacheFormat::BYTES_PER_TOKEN
            vector_uint32_t offset_in_block_smaller_bytes;
            asc_mul_scalar(offset_in_block_smaller_bytes, offset_in_block_smaller, CacheFormat::BYTES_PER_TOKEN, full_mask);
            vector_int32_t global_kv_ptr_offset_lo32bit, global_kv_ptr_offset_hi32bit;
            asc_mul_and_add_wo_intlv(
                global_kv_ptr_offset_lo32bit, global_kv_ptr_offset_hi32bit,
                *(vector_int32_t*)&block_idx_smaller, full_stride_kv_block_vec, *(vector_int32_t*)&offset_in_block_smaller_bytes,
                full_zeros_vec, full_mask
            );
            asc_select(global_kv_ptr_offset_lo32bit, global_kv_ptr_offset_lo32bit, full_0xffffffff_vec, is_valid_smaller);
            asc_select(global_kv_ptr_offset_hi32bit, global_kv_ptr_offset_hi32bit, full_0xffffffff_vec, is_valid_smaller);
            asc_intlv(
                *(vector_int32_t*)&global_ptr_offset_lo, *(vector_int32_t*)&global_ptr_offset_hi,
                global_kv_ptr_offset_lo32bit, global_kv_ptr_offset_hi32bit
            );

            // src_stride = block_idx_delta * stride_kv_block + offset_in_block_delta * CacheFormat::BYTES_PER_TOKEN
            // For positions where the larger index is invalid, block_idx_delta = offset_in_block_delta = 0
            vector_int32_t block_idx_delta, offset_in_block_delta;
            asc_sub(block_idx_delta, *(vector_int32_t*)&block_idx_larger, *(vector_int32_t*)&block_idx_smaller, is_valid_larger);
            asc_sub(offset_in_block_delta, *(vector_int32_t*)&offset_in_block_larger, *(vector_int32_t*)&offset_in_block_smaller, is_valid_larger);

            vector_int32_t offset_in_block_delta_bytes;
            asc_mul_scalar(offset_in_block_delta_bytes, offset_in_block_delta, CacheFormat::BYTES_PER_TOKEN, full_mask);
            vector_int32_t offset_in_block_delta_bytes_hi;
            asc_shiftright_scalar(offset_in_block_delta_bytes_hi, offset_in_block_delta_bytes, 31, full_mask);  // Sign-extend the bias to 64 bits
            asc_mul_and_add(
                src_stride_lo, src_stride_hi,
                block_idx_delta, full_stride_kv_block_vec, offset_in_block_delta_bytes,
                offset_in_block_delta_bytes_hi, full_mask
            );
        }

        asc_storealign_postupdate(
            cur_smaller_indices_ptr,
            cur_indices_smaller,
            NUM_UINT32_IN_VEC,
            full_mask
        );
        asc_storealign_postupdate(
            cur_larger_indices_ptr,
            cur_indices_larger,
            NUM_UINT32_IN_VEC,
            full_mask
        );
        asc_storealign_postupdate_i64x2(cur_global_kv_offset_ptr, global_ptr_offset_lo, global_ptr_offset_hi, full_mask);
        asc_storealign_postupdate_i64x2(cur_src_stride_in_gather2_ptr, src_stride_lo, src_stride_hi, full_mask);

    }
}


template<Config CONFIG>
__simd_vf__ void Kernel<CONFIG>::clear_kv_buf_vf(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t kv_buf_idx) {
    auto full_mask = asc_create_mask_b16(PAT_ALL);
    // No matter what `gathered_kv_t` is, we view the array as `bfloat16` to simplify code
    vector_bf16 zeros_vec;
    asc_duplicate_scalar(zeros_vec, bfloat16_t(0.0f), full_mask);
    __ubuf__ bf16* cur_ptr = (__ubuf__ bf16*)ub_buf.gathered_kv[kv_buf_idx][0];
    #pragma unroll 1
    for (uint16_t i = 0; i < sizeof(ub_buf.gathered_kv[kv_buf_idx]) / 256; ++i) {
        asc_storealign_postupdate(cur_ptr, zeros_vec, NUM_BF16_IN_VEC, full_mask);
    }
    if constexpr (HAVE_FP4) {
        __ubuf__ bf16* fp4_ptr = (__ubuf__ bf16*)ub_buf.gathered_kv_fp4[kv_buf_idx][0];
        #pragma unroll 1
        for (uint16_t i = 0; i < sizeof(ub_buf.gathered_kv_fp4[kv_buf_idx]) / 256; ++i)
            asc_storealign_postupdate(fp4_ptr, zeros_vec, NUM_BF16_IN_VEC, full_mask);
    }

}


template<Config CONFIG>
__simd_vf__ void Kernel<CONFIG>::kv_nd2nz_for_prefill_vf(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t kv_buf_idx) {
    auto full_mask = asc_create_mask_b16(PAT_ALL);
    __ubuf__ bf16* cur_gathered_kv_ptr = ub_buf.gathered_kv[kv_buf_idx][0];
    __ubuf__ bf16* cur_gathered_kv_in_nz_ptr_0 = ub_buf.gathered_kv_in_nz;
    __ubuf__ bf16* cur_gathered_kv_in_nz_ptr_1 = cur_gathered_kv_in_nz_ptr_0 + (B_TOPK_PER_V+1)*NUM_BF16_IN_VEC;

    // asc_storealign_postupdate is buggy, it won't update the pointer, since CANN forgets to pass the pointer as reference
    // As a result we have to use `vsstb`
    asc_store_align_config_post config;
    config.block_stride = B_TOPK_PER_V+1;
    config.repeat_stride = FRACTAL_W*sizeof(bf16) / 32;
    asc_store_align_config_post config_last_row;
    config_last_row.block_stride = B_TOPK_PER_V+1;
    config_last_row.repeat_stride = ((B_TOPK_PER_V+1)*NUM_BF16_IN_VEC*2 - (B_TOPK_PER_V-1)*FRACTAL_W)*sizeof(bf16) / 32;

    /*
    Explanation about the loop structure below:
    - `vsstb` only accepts positive `offset` for "post_update", so we must find an order which only increases (instead of decreases) `cur_gathered_kv_in_nz_ptr`. This is why we use a column (head dim) major loop layout.
    - To balance traffic between two bank groups, we process two adjacent `NUM_BF16_IN_VEC` elements together so that two 256B bank groups are interleaved
    */
    #pragma unroll 1
    for (uint16_t j = 0; j < D_VO / NUM_BF16_IN_VEC; j += 2) {
        #pragma unroll
        for (uint16_t i = 0; i < B_TOPK_PER_V; ++i) {
            vector_bfloat16_t cur_data[2];
            
            asc_loadalign_postupdate(
                cur_data[0],
                cur_gathered_kv_ptr,
                NUM_BF16_IN_VEC
            );
            asc_loadalign_postupdate(
                cur_data[1],
                cur_gathered_kv_ptr,
                i+1 == B_TOPK_PER_V ? -(int)((B_TOPK_PER_V-1)*D_QK) + (int)NUM_BF16_IN_VEC : (int)(D_QK - NUM_BF16_IN_VEC)
            );
            int32_t cur_config = i+1 == B_TOPK_PER_V ? config_last_row.config : config.config;
            vsstb(
                cur_data[0],
                cur_gathered_kv_in_nz_ptr_0,
                cur_config,
                full_mask,
                POST_UPDATE
            );
            vsstb(
                cur_data[1],
                cur_gathered_kv_in_nz_ptr_1,
                cur_config,
                full_mask,
                POST_UPDATE
            );
        }
    }
}


template<Config CONFIG>
template<ModelType CACHE_MODEL_TYPE>
__simd_vf__ void Kernel<CONFIG>::kv_dequant_and_nd2nz_for_decode_vf(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t kv_buf_idx) {
    if constexpr (CACHE_MODEL_TYPE == ModelType::V41_FP4) {
        static_assert(HAVE_FP4 && FP4Format::D_QK == D_QK && FP4Format::QUANT_TILE_SIZE == 16);
        // V4.1 FP4: all 512 dimensions use E2M1 (even dimension in the low nibble),
        // with one E4M3 scale per 16 dimensions, including RoPE.
        auto full_mask = asc_create_mask_b16(PAT_ALL);
        __ubuf__ fp8_e4m3fn_t* raw_scales = (__ubuf__ fp8_e4m3fn_t*)ub_buf.gathered_kv_fp4[kv_buf_idx][0] + FP4Format::QUANT_BYTES;
        __ubuf__ bf16* dst_scales = ub_buf.fp4_scales_bf16[0];
        // Gather four rows of inline scale bytes and zero-extend each byte to b16.
        // Index i selects row i/32, scale i%32, without reading adjacent raw bytes.
        vector_uint16_t scale_indices, row_indices;
        asc_arange((vector_int16_t&)scale_indices, int16_t(0));
        asc_shiftright_scalar(row_indices, scale_indices, 5, full_mask);
        asc_shiftleft_scalar(row_indices, row_indices, 8, full_mask);
        asc_add(scale_indices, row_indices, scale_indices, full_mask);
        #pragma unroll 1
        for (uint16_t row = 0; row < B_TOPK_PER_V; row += 4) {
            vector_fp8_e4m3fn_t packed;
            asc_gather((vector_uint16_t&)packed, (__ubuf__ uint8_t*)raw_scales, scale_indices, full_mask);
            raw_scales += 4 * FP4_UB_TOKEN_BYTES;
            vector_float even, odd;
            asc_e4m32float(even, packed, full_mask, ASC_DISPERSE_FIRST_QUARTER);
            asc_e4m32float(odd, packed, full_mask, ASC_DISPERSE_THIRD_QUARTER);
            vector_uint32_t even_bits;
            vector_bf16 scales_bf16;
            asc_shiftright_scalar(even_bits, (vector_uint32_t&)even, 16, full_mask);
            asc_or((vector_uint32_t&)scales_bf16, even_bits, (vector_uint32_t&)odd, full_mask);
            asc_storealign_postupdate(dst_scales, scales_bf16, NUM_BF16_IN_VEC, full_mask);
        }
        asc_mem_bar(VST_VLD);

        asc_store_align_config_post config;
        config.block_stride = B_TOPK_PER_V + 1;
        config.repeat_stride = (B_TOPK_PER_V + 1) * (256 / 32);
        __ubuf__ fp4x2_e2m1_t* payload = (__ubuf__ fp4x2_e2m1_t*)ub_buf.gathered_kv_fp4[kv_buf_idx][0];
        __ubuf__ bf16* scales = ub_buf.fp4_scales_bf16[0];
        __ubuf__ bf16* dst_row = ub_buf.gathered_kv_in_nz;
        #pragma unroll 1
        for (uint16_t row = 0; row < B_TOPK_PER_V; ++row) {
            __ubuf__ bf16* dst = dst_row;
            dst_row += FRACTAL_W;
            #pragma unroll
            for (uint16_t chunk = 0; chunk < D_QK / NUM_BF16_IN_VEC; ++chunk) {
                vector_bf16 scale, values;
                asc_loadalign_brc_elem2datablock_postupdate(scale, scales, 8);
                vector_fp4x2_e2m1_t packed;
                // UNPACK4 keeps each low/high nibble pair together: one conversion yields 128 ordered BF16 values.
                asc_loadalign_unpack4_postupdate(packed, payload, 64);
                asc_e2m1x22bfloat16(values, packed, full_mask, ASC_DISPERSE_FIRST_QUARTER);
                asc_mul(values, values, scale, full_mask);
                vsstb(values, dst, config.config, full_mask, POST_UPDATE);
            }
            payload += FP4_UB_TOKEN_BYTES - FP4Format::QUANT_BYTES;
        }
    } else {
        static_assert(D_QK == 512 && SCALE_GRAN == 32);
        auto full_mask = asc_create_mask_b8(PAT_ALL);
        auto scale_mask = asc_create_mask_b16(PAT_ALL);

        // Gather eight inline scale rows at once, then convert UE8M0 to BF16.
        // Index i selects row i/16, scale i%16; raw records remain read-only.
        vector_uint16_t scale_indices, row_indices;
        asc_arange((vector_int16_t&)scale_indices, int16_t(0));
        asc_shiftright_scalar(row_indices, scale_indices, 4, scale_mask);
        asc_mul_scalar(row_indices, row_indices, FP8_UB_TOKEN_BYTES - NUM_SCALES_PER_TOKEN, scale_mask);
        asc_add(scale_indices, row_indices, scale_indices, scale_mask);
        __ubuf__ uint8_t* raw_scales = (__ubuf__ uint8_t*)ub_buf.gathered_kv[kv_buf_idx][0] + FP8Format::QUANT_BYTES;
        __ubuf__ uint16_t* dst_scales = (__ubuf__ uint16_t*)ub_buf.fp8_scales_bf16[0];
        #pragma unroll 1
        for (uint16_t row = 0; row < B_TOPK_PER_V; row += 8) {
            vector_uint16_t scales_bf16;
            asc_gather(scales_bf16, raw_scales, scale_indices, scale_mask);
            raw_scales += 8 * FP8_UB_TOKEN_BYTES;
            // Preserve the existing clamp for invalid/NaN scale encodings.
            asc_min_scalar(scales_bf16, scales_bf16, 0xE0, scale_mask);
            asc_shiftleft_scalar(scales_bf16, scales_bf16, 7, scale_mask);
            asc_storealign_postupdate(dst_scales, scales_bf16, NUM_BF16_IN_VEC, scale_mask);
        }

        // Now scales are converted to BF16
        asc_mem_bar(VST_VLD);
        asc_store_align_config_post config;
        config.block_stride = B_TOPK_PER_V+1;
        config.repeat_stride = (B_TOPK_PER_V+1) * (256/32);

        __ubuf__ gathered_kv_t* cur_gathered_kv_ptr = ub_buf.gathered_kv[kv_buf_idx][0];
        __ubuf__ bf16* cur_gathered_scale_factors_ptr = ub_buf.fp8_scales_bf16[0];
        __ubuf__ bf16* cur_dst_base_ptr = ub_buf.gathered_kv_in_nz;

        #pragma unroll 1
        for (uint16_t i = 0; i < B_TOPK_PER_V; ++i) {
            vector_bf16 scale_factors[4];
            asc_loadalign_brc_elem2datablock_postupdate(scale_factors[0], cur_gathered_scale_factors_ptr, 8);        // s0x16, s1x16, s2x16, ..., s7x16
            asc_loadalign_brc_elem2datablock_postupdate(scale_factors[2], cur_gathered_scale_factors_ptr, 8);    // s8x16, s9x16, s10x16, ..., s15x16
            asc_intlv(scale_factors[0], scale_factors[1], scale_factors[0], scale_factors[0]);  // scale_factors[0]: s0x32, s1x32, s2x32, s3x32; scale_factors[1]: s4x32, s5x32, s6x32, s7x32
            asc_intlv(scale_factors[2], scale_factors[3], scale_factors[2], scale_factors[2]);

            __ubuf__ bf16* cur_dst_ptr = cur_dst_base_ptr;
            cur_dst_base_ptr += FRACTAL_W;

            // Finite FP8 values are exactly representable in BF16; their FP32 low halves are zero.
            #pragma unroll
            for (uint16_t chunk_idx = 0; chunk_idx < D_QK / NUM_BF16_IN_VEC; ++chunk_idx) {
                vector_fp8_e4m3fn_t data_fp8;   // 0 x 1 x 2 x 3 x ...
                asc_loadalign_unpack_postupdate(data_fp8, cur_gathered_kv_ptr, 128);
                vector_float data_fp32_even, data_fp32_odd;
                asc_e4m32float(data_fp32_even, data_fp8, full_mask);
                asc_e4m32float_v3(data_fp32_odd, data_fp8, full_mask);
                vector_bf16 data_bf16;
                vector_uint32_t even_bf16_bits;
                asc_shiftright_scalar(even_bf16_bits, (vector_uint32_t&)data_fp32_even, 16, full_mask);
                // Valid KV payloads are finite; masked gathers duplicate a valid token or retain zero/finite data.
                asc_or((vector_uint32_t&)data_bf16, even_bf16_bits, (vector_uint32_t&)data_fp32_odd, full_mask);
                asc_mul(data_bf16, data_bf16, scale_factors[chunk_idx], full_mask);
                vsstb(
                    data_bf16,
                    cur_dst_ptr,
                    config.config,
                    full_mask,
                    POST_UPDATE
                );
            }
            cur_gathered_kv_ptr += FP8_UB_TOKEN_BYTES - FP8Format::QUANT_BYTES;
        }
    }
}


template<Config CONFIG>
template<bool IS_BLOCK0>
__simd_vf__ void Kernel<CONFIG>::softmax_vf(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, float sm_scale, float rescale_threshold_div_sm_scale, uint32_t job_idx, uint32_t kv_block_idx, uint32_t index_buf_idx, uint32_t ub_indices_arr_offset, uint32_t s_kv, uint32_t subblock_idx) {
    // p and s are laid out in NZ
    auto full_mask = asc_create_mask_b16(PAT_ALL);
    auto vl8_mask = asc_create_mask_b32(PAT_VL8);

    vector_float old_rows_max_scaled, old_rows_sum, old_row_max_for_o;
    if constexpr (not IS_BLOCK0) {
        asc_loadalign(old_rows_max_scaled, ub_buf.row_max[job_idx&1]);  // Backup the old row-max
        asc_mem_bar(VLD_VST);
        asc_loadalign(old_rows_sum, ub_buf.row_sum[job_idx&1]);         // Backup the old row-sum. Put after mem_bar
        asc_mul_scalar(old_rows_max_scaled, old_rows_max_scaled, sm_scale, full_mask);
    }

    // Get `col_masks` from `smaller_indices` and `larger_indices`
    // Each "col_mask" controls 16 columns, and the masks for those 16 columns are repeated for four times, so that it can be applied directly onto P loaded from SMEM
    vector_bool col_valid_masks[B_TOPK / 16];
    __ubuf__ uint32_t* cur_smaller_indices_ptr = ub_buf.smaller_indices[index_buf_idx] + ub_indices_arr_offset;
    __ubuf__ uint32_t* cur_larger_indices_ptr = ub_buf.larger_indices[index_buf_idx] + ub_indices_arr_offset;
    #pragma unroll
    for (uint32_t col_group = 0; col_group < B_TOPK / 16; ++col_group) {
        // Load 8 smaller indices and 8 larger indices ...
        vector_uint32_t smaller_indices, larger_indices;
        asc_loadalign_brc_datablock_postupdate(smaller_indices, cur_smaller_indices_ptr, 8);
        asc_loadalign_brc_datablock_postupdate(larger_indices, cur_larger_indices_ptr, 8);
        // ... and then use asc_intlv to interleave them
        vector_uint32_t interleaved_indices, t;
        asc_intlv(interleaved_indices, t, smaller_indices, larger_indices);
        // For each pair of indices, there are three cases:
        //  1) Both indices are valid. Token with smaller index will be placed at position 0 and token with larger index will be placed at position 1
        //  2) One index is valid and the other one is invalid. The valid token will be placed at position 0
        //  3) Both indices are invalid
        // In any case above, we can use `smaller_indices` and `larger_indices` to distinguish which token is valid and which one is invalid
        asc_lt_scalar(col_valid_masks[col_group], interleaved_indices, s_kv, full_mask);
    }

    // Create a mask whose first "num valid heads in this subblock" * 2 rows are set to 1
    vector_bool valid_rows_mask_for_row_max;
    if constexpr (H_Q == 64) {
        // MMA_M is 64. Each AIV is responsible for 32 heads, occupying all 32 * 2 lanes in SIMD reg (*2 since every row-max is repeated twice)
        valid_rows_mask_for_row_max = full_mask;
    } else if constexpr (H_Q == 32) {
        // MMA_M is 32. Each AIV is responsible for 16 heads, occupying the first 16 * 2 = 32 lanes in SIMD reg
        valid_rows_mask_for_row_max = asc_create_mask_b32(PAT_VL32);
    } else {
        uint32_t my_subblock_start_head_idx = subblock_idx * MMA_M_PER_V;
        uint32_t num_valid_heads = H_Q >= my_subblock_start_head_idx ? H_Q - my_subblock_start_head_idx : 0u;
        num_valid_heads = min(num_valid_heads, MMA_M_PER_V);
        vector_int32_t arange_vec;
        asc_arange(arange_vec, 0);
        asc_lt_scalar(valid_rows_mask_for_row_max, arange_vec, 2 * num_valid_heads, full_mask);
    }

    __ubuf__ float* cur_row_max_ptr = ub_buf.row_max[job_idx&1];
    static_assert(MMA_M_PER_V % 4 == 0);
    #pragma unroll 1
    for (uint16_t start_row = 0; start_row < MMA_M_PER_V / 4; start_row += 1) {
        vector_float cur_rows_max;
        if constexpr (IS_BLOCK0) {
            asc_duplicate_scalar(cur_rows_max, std::numeric_limits<float>::lowest());
        } else {
            vlds(cur_rows_max, ub_buf.row_max[job_idx&1], start_row*8, E2B_B32);
        }

        __ubuf__ float* cur_p_ptr = ub_buf.p + start_row*4*FRACTAL_W;
        #pragma unroll
        for (uint16_t col = 0; col < B_TOPK / 16; col += 1) {
            vector_float p;
            asc_loadalign_postupdate(
                p,
                cur_p_ptr,
                MMA_M_PER_V * FRACTAL_W
            );
            vmax(cur_rows_max, cur_rows_max, p, col_valid_masks[col], MODE_MERGING);    // vmax with MODE_MERGING retains the original value of `dst` if one position is being masked-off
        }

        // Use `asc_reduce_max_datablock` to reduce among every 8 elements, and then use `deintlv` and another `asc_max` to reduce adjacent 2 elements
        asc_reduce_max_datablock(cur_rows_max, cur_rows_max, full_mask);
        vector_float datablockwise_max_lo8, datablockwise_max_hi8;
        asc_deintlv(datablockwise_max_lo8, datablockwise_max_hi8, cur_rows_max, cur_rows_max);
        asc_max(cur_rows_max, datablockwise_max_lo8, datablockwise_max_hi8, full_mask);
        // NOTE: dst0 and dst1 of `asc_intlv` must NOT be the same register; otherwise the writeback of dst1 clobbers dst0
        vector_float intlv_lo, intlv_hi;
        asc_intlv(intlv_lo, intlv_hi, cur_rows_max, cur_rows_max);

        // Now intlv_lo[0~1] is the maximum of the 0th row, intlv_lo[2~3] is the maximum of the 1th row ...
        asc_storealign_postupdate(
            cur_row_max_ptr,
            intlv_lo,
            4 * 2,   // 4 rows, 2 replicas per row
            vl8_mask
        );
    }

    asc_mem_bar(VST_VLD);

    vector_float cur_scales;
    if constexpr (not IS_BLOCK0) {
        vector_float new_rows_max;
        asc_loadalign(new_rows_max, ub_buf.row_max[job_idx&1]);
        asc_loadalign(old_row_max_for_o, ub_buf.row_max_for_o[job_idx&1]);
        vector_float row_max_max_delta;
        asc_sub(row_max_max_delta, new_rows_max, old_row_max_for_o, full_mask);
        asc_reduce_max(row_max_max_delta, row_max_max_delta, valid_rows_mask_for_row_max);
        asc_duplicate(row_max_max_delta, row_max_max_delta, full_mask);    // The maximum of {new_rows_max[i] - old_row_max_for_o[i] | i = 0...MMA_M_PER_V-1} is broadcasted to every number in `row_max_max_delta`

        vector_bool should_rescale;
        asc_gt_scalar(should_rescale, row_max_max_delta, rescale_threshold_div_sm_scale, full_mask);
        vector_float new_row_max_for_o;
        asc_select(new_row_max_for_o, new_rows_max, old_row_max_for_o, should_rescale);
        asc_storealign(ub_buf.row_max_for_o[job_idx&1], new_row_max_for_o, full_mask);

        asc_mem_bar(VST_VLD);

        vector_float new_row_max_for_o_scaled, old_row_max_for_o_scaled;
        asc_mul_scalar(new_row_max_for_o_scaled, new_row_max_for_o, sm_scale, full_mask);
        asc_mul_scalar(old_row_max_for_o_scaled, old_row_max_for_o, sm_scale, full_mask);
        asc_exp_sub(cur_scales, old_row_max_for_o_scaled, new_row_max_for_o_scaled, full_mask);
        asc_storealign(ub_buf.row_scales[job_idx&1][kv_block_idx&1], cur_scales, full_mask);
        asc_storealign_1st(&(ub_buf.row_max_max_delta[job_idx&1][kv_block_idx&1]), row_max_max_delta);
    } else {
        vector_float row_max;
        asc_loadalign(row_max, ub_buf.row_max[job_idx&1]);
        asc_storealign(ub_buf.row_max_for_o[job_idx&1], row_max, full_mask);
        asc_mem_bar(VST_VLD);
    }

    __ubuf__ float* cur_p_ptr = ub_buf.p;
    __ubuf__ float* cur_row_sum_ptr = ub_buf.row_sum[job_idx&1];
    static_assert(MMA_M_PER_V % 8 == 0);
    #pragma unroll 1    // Don't unroll to reduce code size, which prevents VF's ICache miss
    for (uint32_t start_row = 0; start_row < MMA_M_PER_V; start_row += 8) {
        vector_float cur_rows_max_for_o[2], cur_rows_sum[2];
        #pragma unroll
        for (uint32_t i = 0; i < 2; ++i) {
            vlds(cur_rows_max_for_o[i], ub_buf.row_max_for_o[job_idx&1], (start_row + i*4)*2, E2B_B32);
            asc_mul_scalar(cur_rows_max_for_o[i], cur_rows_max_for_o[i], sm_scale, full_mask);
        }

        __ubuf__ bf16* cur_s_ptr = (__ubuf__ bf16*)ub_buf.p + start_row*FRACTAL_W;
        #pragma unroll
        for (uint32_t start_col = 0; start_col < B_TOPK / 16; start_col += 2) {
            vector_float p[2][2];
            vector_bf16 s[2][2];
            #pragma unroll
            for (uint32_t local_col = 0; local_col < 2; ++local_col) {
                #pragma unroll
                for (uint32_t local_row = 0; local_row < 2; ++local_row) {
                    // The last load of each `start_row` iteration also rewinds the pointer to the first 4-row block of the next iteration
                    asc_loadalign_postupdate(
                        p[local_row][local_col],
                        cur_p_ptr,
                        local_row == 0
                            ? 4*FRACTAL_W
                            : MMA_M_PER_V*FRACTAL_W - 4*FRACTAL_W
                                + (start_col+2 == B_TOPK/16 && local_col+1 == 2
                                    ? 8*FRACTAL_W - (B_TOPK/16)*(MMA_M_PER_V*FRACTAL_W)
                                    : 0)
                    );
                    asc_mul_scalar(p[local_row][local_col], p[local_row][local_col], sm_scale, full_mask);
                    asc_exp_sub(p[local_row][local_col], p[local_row][local_col], cur_rows_max_for_o[local_row], col_valid_masks[start_col+local_col]);
                    if (start_col == 0 && local_col == 0) {
                        cur_rows_sum[local_row] = p[local_row][local_col];
                    } else {
                        asc_add(cur_rows_sum[local_row], cur_rows_sum[local_row], p[local_row][local_col], full_mask);
                    }

                    if (local_col == 0) {
                        asc_float2bfloat16_rn(s[local_row][local_col], p[local_row][local_col], full_mask);
                    } else {
                        asc_float2bfloat16_rn_v2_impl(s[local_row][local_col], p[local_row][local_col], full_mask);
                    }
                }
            }
            asc_add(s[0][0], s[0][0], s[0][1], full_mask);
            asc_add(s[1][0], s[1][0], s[1][1], full_mask);
            asc_deintlv(s[0][0], s[1][0], s[0][0], s[1][0]);
            asc_storealign_postupdate(
                cur_s_ptr,
                s[0][0],
                2*MMA_M_PER_V*FRACTAL_W,
                full_mask
            );
            asc_storealign_postupdate(
                cur_s_ptr,
                s[1][0],
                2*MMA_M_PER_V*FRACTAL_W,
                full_mask
            );
        }

        #pragma unroll
        for (uint32_t i = 0; i < 2; ++i) {
            asc_reduce_sum_datablock(cur_rows_sum[i], cur_rows_sum[i], full_mask);
            asc_pair_reduce_sum(cur_rows_sum[i], cur_rows_sum[i], full_mask);
            // NOTE: dst0 and dst1 of `asc_intlv` must NOT be the same register; otherwise the writeback of dst1 clobbers dst0
            vector_float intlv_lo, intlv_hi;
            asc_intlv(intlv_lo, intlv_hi, cur_rows_sum[i], cur_rows_sum[i]);
            
            asc_storealign_postupdate(
                cur_row_sum_ptr,
                intlv_lo,
                4 * 2,   // 4 rows, 2 replicas per row
                vl8_mask
            );
        }
    }
    
    if constexpr (not IS_BLOCK0) {
        asc_mem_bar(VST_VLD);
        vector_float new_rows_sum;
        asc_loadalign(new_rows_sum, ub_buf.row_sum[job_idx&1]);
        asc_madd(old_rows_sum, cur_scales, new_rows_sum, full_mask);
        asc_storealign(ub_buf.row_sum[job_idx&1], old_rows_sum, full_mask);
    }
}


/*
output_accum := (cur_output_frag [+ output_accum]) [* row_scales]
*/
template<Config CONFIG>
template<bool PERFORM_SCALE, bool PERFORM_ADD>
__simd_vf__ void Kernel<CONFIG>::add_and_scale_o_vf(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t output_frag_idx, uint32_t cur_o_frag_buf_idx, uint32_t job_idx, uint32_t kv_block_idx) {
    auto full_mask = asc_create_mask_b32(PAT_ALL);
    __ubuf__ float* cur_output_accum_ptr = ub_buf.output_accum[0] + output_frag_idx * OUTPUT_FRAG_D_VO;
    __ubuf__ float* cur_cur_output_frag_ptr = ub_buf.cur_output_frag[cur_o_frag_buf_idx][0];
    
    #pragma unroll 1
    for (uint16_t row = 0; row < MMA_M_PER_V; row += 1) {
        vector_float cur_scale;
        if constexpr (PERFORM_SCALE) {
            asc_loadalign_brc(cur_scale, ub_buf.row_scales[job_idx&1][kv_block_idx&1], 2*row);
        }
        
        static constexpr uint32_t NUM_FLOAT_IN_VEC = 256 / sizeof(float);
        #pragma unroll
        for (uint16_t col = 0; col < OUTPUT_FRAG_D_VO / NUM_FLOAT_IN_VEC; ++col) {
            vector_float new_output;
            asc_loadalign_postupdate(
                new_output,
                cur_cur_output_frag_ptr,
                NUM_FLOAT_IN_VEC
            );
            if constexpr (PERFORM_ADD) {
                vector_float old_output;
                asc_loadalign(old_output, cur_output_accum_ptr);
                asc_add(new_output, old_output, new_output, full_mask);
            }
            if constexpr (PERFORM_SCALE) {
                asc_mul(new_output, new_output, cur_scale, full_mask);
            }
            asc_storealign_postupdate(
                cur_output_accum_ptr,
                new_output,
                NUM_FLOAT_IN_VEC + (col+1 == OUTPUT_FRAG_D_VO / NUM_FLOAT_IN_VEC ? D_VO - OUTPUT_FRAG_D_VO : 0),
                full_mask
            );
        }
    }
}


template<Config CONFIG>
__simd_vf__ void Kernel<CONFIG>::get_final_lse_and_denorm_vf(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t job_idx, float sm_scale) {
    auto full_mask = asc_create_mask_b32(PAT_ALL);

    vector_float row_max, row_max_for_o, row_sum, t;
    
    asc_loadalign(row_max, ub_buf.row_max[job_idx&1]);
    asc_loadalign(row_max_for_o, ub_buf.row_max_for_o[job_idx&1]);
    asc_loadalign(row_sum, ub_buf.row_sum[job_idx&1]);
    
    asc_deintlv(row_max, t, row_max, row_max);
    asc_deintlv(row_max_for_o, t, row_max_for_o, row_max_for_o);
    asc_deintlv(row_sum, t, row_sum, row_sum);
    
    vector_bool is_row_all_masked;
    asc_eq_scalar(is_row_all_masked, row_max, std::numeric_limits<float>::lowest(), full_mask);

    asc_mul_scalar(row_max, row_max, sm_scale, full_mask);
    asc_mul_scalar(row_max_for_o, row_max_for_o, sm_scale, full_mask);

    vdup(row_max, -std::numeric_limits<float>::infinity(),is_row_all_masked, MODE_MERGING);
    asc_storealign(ub_buf.row_max[job_idx&1], row_max, full_mask);
    
    vector_float lse;
    asc_ln(lse, row_sum, full_mask);
    asc_add(lse, lse, row_max_for_o, full_mask);
    vdup(lse, +std::numeric_limits<float>::infinity(),is_row_all_masked, MODE_MERGING);
    asc_storealign(ub_buf.row_sum[job_idx&1], lse, full_mask);

    vector_float row_sum_for_denorm = row_sum;
    if constexpr (HAVE_ATTN_SINK) {
        vector_float attn_sink;
        asc_loadalign(attn_sink, ub_buf.attn_sink);
        asc_exp_sub(attn_sink, attn_sink, row_max_for_o, full_mask);
        asc_add(row_sum_for_denorm, row_sum_for_denorm, attn_sink, full_mask);
    }
    vector_float denorm;
    asc_duplicate_scalar(denorm, 1.0f);
    asc_div(denorm, denorm, row_sum_for_denorm, full_mask);
    vdup(denorm, 0, is_row_all_masked, MODE_MERGING);

    asc_storealign(ub_buf.row_denorm[job_idx&1], denorm, full_mask);
}


/*
output = float2bfloat16((cur_output_frag [+ output_accum]) * denorm)
*/
template<Config CONFIG>
template<bool ADD_OUTPUT_ACCUM>
__simd_vf__ void Kernel<CONFIG>::get_final_output_vf(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t output_frag_idx, uint32_t cur_o_frag_buf_idx, uint32_t job_idx, uint32_t kv_block_idx, bool is_warmup_mode) {
    auto full_mask = asc_create_mask_b16(PAT_ALL);

    __ubuf__ float* cur_output_accum_ptr = ub_buf.output_accum[0] + output_frag_idx * OUTPUT_FRAG_D_VO;
    __ubuf__ float* cur_cur_output_frag_ptr = ub_buf.cur_output_frag[cur_o_frag_buf_idx][0];
    __ubuf__ bf16* cur_final_output_ptr = (__ubuf__ bf16*)(ub_buf.output_accum[0]) + output_frag_idx * (OUTPUT_FRAG_D_VO*2);
    
    static_assert(OUTPUT_FRAG_D_VO / NUM_FLOAT_IN_VEC == 2);
    #pragma unroll 1
    for (uint16_t row = 0; row < MMA_M_PER_V / 2; row += 1) {
        vector_float cur_denorm[2];
        #pragma unroll
        for (uint32_t i = 0; i < 2; ++i) {
            asc_loadalign_brc(cur_denorm[i], ub_buf.row_denorm[job_idx&1], row*2+i);
        }

        vector_bf16 output_bf16[2][2];
        #pragma unroll
        for (uint32_t i = 0; i < 2; ++i) {
            vector_float old_output[2];
            if constexpr (ADD_OUTPUT_ACCUM) {
                asc_loadalign_postupdate(
                    old_output[0],
                    cur_output_accum_ptr,
                    NUM_FLOAT_IN_VEC
                );
                asc_loadalign_postupdate(
                    old_output[1],
                    cur_output_accum_ptr,
                    NUM_FLOAT_IN_VEC + D_VO - OUTPUT_FRAG_D_VO
                );
            }
            #pragma unroll
            for (uint32_t j = 0; j < 2; ++j) {
                vector_float output_f32;
                asc_loadalign_postupdate(
                    output_f32,
                    cur_cur_output_frag_ptr,
                    NUM_FLOAT_IN_VEC
                );
                if constexpr (ADD_OUTPUT_ACCUM) {
                    asc_add(output_f32, output_f32, old_output[j], full_mask);
                }
                asc_mul(output_f32, output_f32, cur_denorm[i], full_mask);
                if (i == 0) {
                    asc_float2bfloat16_rn(output_bf16[i][j], output_f32, full_mask);
                } else {
                    asc_float2bfloat16_rn_v2_impl(output_bf16[i][j], output_f32, full_mask);    // Use `_impl` since `_v2` is marked as depreciated and I cannot find a alternative
                }
            }
        }

        asc_add(output_bf16[0][0], output_bf16[0][0], output_bf16[1][0], full_mask);
        asc_add(output_bf16[0][1], output_bf16[0][1], output_bf16[1][1], full_mask);
        asc_deintlv(output_bf16[0][0], output_bf16[0][1], output_bf16[0][0], output_bf16[0][1]);
        asc_storealign_postupdate(
            cur_final_output_ptr,
            output_bf16[0][0],
            2*D_VO,
            full_mask
        );
        asc_storealign_postupdate(
            cur_final_output_ptr,
            output_bf16[0][1],
            2*D_VO,
            full_mask
        );
    }
}


template<Config CONFIG>
__aicore__ void Kernel<CONFIG>::sparse_attn_fwd_kernel_devfunc(const Params params, const AuxParams aux_params) {
    AscendC::InitSocState();

    struct OuterLoopArgs {
        uint32_t job_idx;
        uint32_t batch_or_s_q_idx;  // For prefill, it's s_q_idx; for decoding, it's batch_idx (since params.s_q must be 1)
        uint32_t topk_length, extra_topk_length;    // Only decoding have `extra_topk_length`
        uint32_t num_orig_topk_blocks, num_topk_blocks;    // Only decoding have `num_orig_topk_blocks`.
        uint32_t sum_prev_num_topk_blocks;

        __aicore__ bool is_valid() {
            return batch_or_s_q_idx != 0xFFFFFFFFu;
        }
    };

    auto _get_outer_loop_args = [&](uint32_t job_idx, uint32_t s_q_idx, uint32_t sum_prev_num_topk_blocks) __aicore__ -> OuterLoopArgs {
        uint32_t topk_length = params.topk_length != nullptr ? (uint32_t)params.topk_length[s_q_idx] : params.topk;
        uint32_t num_topk_blocks = max((topk_length + B_TOPK - 1) / B_TOPK, 1u);
        uint32_t extra_topk_length, num_orig_topk_blocks;
        if constexpr (IS_DECODE) {
            extra_topk_length = params.extra_topk_length != nullptr ? (uint32_t)params.extra_topk_length[s_q_idx] : params.extra_topk;
            uint32_t num_extra_topk_blocks = (extra_topk_length + B_TOPK - 1) / B_TOPK;
            num_orig_topk_blocks = num_topk_blocks;
            num_topk_blocks += num_extra_topk_blocks;
        }
        return {job_idx, s_q_idx, topk_length, extra_topk_length, num_orig_topk_blocks, num_topk_blocks, sum_prev_num_topk_blocks};
    };

    auto get_first_job = [&]() __aicore__ -> OuterLoopArgs {
        return _get_outer_loop_args(0u, (uint32_t)get_block_idx(), 0u);
    };

    auto get_next_job = [&](const OuterLoopArgs &cur_job) __aicore__ -> OuterLoopArgs {
        uint32_t total_q;
        if constexpr (IS_DECODE) {
            total_q = (uint32_t)params.b;
        } else {
            total_q = (uint32_t)params.s_q;
        }
        uint32_t nxt_s_q_idx = cur_job.batch_or_s_q_idx + GRID_SHAPE;
        if (nxt_s_q_idx >= total_q) {
            return OuterLoopArgs {
                cur_job.job_idx + 1,
                0xFFFFFFFFu
            };
        } else {
            return _get_outer_loop_args(
                cur_job.job_idx + 1,
                nxt_s_q_idx,
                cur_job.num_topk_blocks + cur_job.sum_prev_num_topk_blocks
            );
        }
    };

    __ubuf__ UnifiedBufferMemoryPlan* ub_buf_ptr = (__ubuf__ UnifiedBufferMemoryPlan*)0;
    __ubuf__ UnifiedBufferMemoryPlan& ub_buf = *ub_buf_ptr;

    // Since L1 buffer is splitted into two bank groups (the first one spans address 0 ~ 256K while the second one spans address 256K ~ 512K), we split every buffer into two parts to balance load between those two bank groups
    __cbuf__ bf16* l1_q_buf_lo = (__cbuf__ bf16*)(0*1024);
    __cbuf__ bf16* l1_gathered_kv_buf_lo = (__cbuf__ bf16*)(32*2*1024);    // NUM_L1_KV_BUFS*B_TOPK_PER_V*D_QK
    __cbuf__ bf16* l1_s_buf = (__cbuf__ bf16*)(256*1024 - 64*64*sizeof(bf16)/2);
    __cbuf__ bf16* l1_gathered_kv_buf_hi = (__cbuf__ bf16*)(288*1024);    // NUM_L1_KV_BUFS*B_TOPK_PER_V*D_QK
    __cbuf__ bf16* l1_q_buf_hi = (__cbuf__ bf16*)((512-32*2)*1024);

    volatile __ssbuf__ SSBufferMemoryPlan& ss_buf = *(volatile __ssbuf__ SSBufferMemoryPlan*)(AscendC::GetSsbufBaseAddr());
    
    if ASC_IS_AIV {
        // VECTOR Core
        uint32_t subblock_idx = (uint32_t)get_subblockid();
        uint32_t this_subblock_start_head_idx = subblock_idx * MMA_M_PER_V;
        uint32_t num_this_subblock_valid_heads = H_Q >= this_subblock_start_head_idx ? min(H_Q - this_subblock_start_head_idx, MMA_M_PER_V) : 0;   // AIV0 processes head [0, MMA_M_PER_V) and AIV1 processes head [MMA_M_PER_V, MMA_M)
        asc_set_gm2ub_loop_size(1ul, 1ul);
        asc_set_ub2gm_loop_size(1ul, 1ul);
        asc_set_ctrl(0ull); // Don't let the scalar core to use dcache to load from UB since 1) it's not faster 2) cache refreshing leads to some overhead or even hazards if not carefully analysed

        if constexpr (HAVE_ATTN_SINK) {
            asc_copy_gm2ub_align(ub_buf.attn_sink, params.attn_sink + this_subblock_start_head_idx, num_this_subblock_valid_heads * sizeof(float));
            set_flag(PIPE_MTE2, PIPE_V, event_t(VectorCoreFlags::ATTN_SINK_FULL));
        }

        AscendC::CrossCoreSetFlag<4, PIPE_S>(CrossCoreFlags::UBUF_P_EMPTY);
        #pragma unroll
        for (uint32_t i = 0; i < NUM_OUTPUT_FRAG_BUFS; ++i) 
            AscendC::CrossCoreSetFlag<4, PIPE_S>(CrossCoreFlags::UBUF_O_FRAG_EMPTY+i);

        auto dispatch_cache_format = [&](bool is_extra, auto&& action) __aicore__ {
            if constexpr (MODEL_TYPE == EXTRA_MODEL_TYPE) {
                action.template operator()<MODEL_TYPE>();
            } else if (is_extra) {
                action.template operator()<EXTRA_MODEL_TYPE>();
            } else {
                action.template operator()<MODEL_TYPE>();
            }
        };

        auto clear_all_kv_bufs = [&]() __aicore__ {
            #pragma unroll 1
            for (uint32_t i = 0; i < NUM_UB_KV_BUFS; ++i) {
                get_buf(PIPE_V, VectorCoreBufIDs::GATHERED_KV + i, false);
                clear_kv_buf_vf(ub_buf, i);
                rls_buf(PIPE_V, VectorCoreBufIDs::GATHERED_KV + i, false);
            }
        };

        auto run_issue_index_fetch = [&](const OuterLoopArgs &args, uint32_t kv_block_idx) __aicore__ {
            // Fetch the indices of the KV block group that `kv_block_idx` belongs to
            // (`INDEX_BLOCK_SIZE / B_TOPK` KV blocks per group)
            bool is_extra_kv = kv_block_idx >= args.num_orig_topk_blocks;
            uint32_t partial_kv_block_idx = IS_DECODE && is_extra_kv ? kv_block_idx - args.num_orig_topk_blocks : kv_block_idx;
            if (partial_kv_block_idx % (INDEX_BLOCK_SIZE / B_TOPK) != 0)
                return;

            uint32_t topk_length = IS_DECODE && is_extra_kv ? args.extra_topk_length : args.topk_length;
            uint32_t index_copy_len = min(INDEX_BLOCK_SIZE, topk_length - partial_kv_block_idx * B_TOPK);
            __gm__ int* index_base;
            if constexpr (IS_DECODE) {
                index_base =
                    (is_extra_kv ? params.extra_indices : params.indices) + 
                    args.batch_or_s_q_idx * (is_extra_kv ? params.stride_extra_indices_b : params.stride_indices_b) +
                    partial_kv_block_idx * B_TOPK;
            } else {
                index_base = params.indices + args.batch_or_s_q_idx * params.stride_indices_s_q + kv_block_idx * B_TOPK;
            }
            get_buf(PIPE_MTE2, VectorCoreBufIDs::RAW_INDICES, false);
            asc_copy_gm2ub(
                ub_buf.indices,
                index_base,
                index_copy_len * sizeof(int32_t)
            );
            rls_buf(PIPE_MTE2, VectorCoreBufIDs::RAW_INDICES, false);
        };

        uint32_t cur_index_buf_for_run_process_index = 0;
        auto run_process_index = [&](const OuterLoopArgs &args, uint32_t kv_block_idx) __aicore__ {
            bool is_extra_kv = kv_block_idx >= args.num_orig_topk_blocks;
            uint32_t partial_kv_block_idx = IS_DECODE && is_extra_kv ? kv_block_idx - args.num_orig_topk_blocks : kv_block_idx;
            if (partial_kv_block_idx % (INDEX_BLOCK_SIZE / B_TOPK) != 0)
                return;

            uint32_t topk_length = IS_DECODE && is_extra_kv ? args.extra_topk_length : args.topk_length;
            uint32_t valid_len = topk_length - partial_kv_block_idx * B_TOPK;
            get_buf(PIPE_V, VectorCoreBufIDs::RAW_INDICES, false);
            get_buf(PIPE_V, VectorCoreBufIDs::PROCESSED_INDICES, false);
            if constexpr (IS_DECODE) {
                dispatch_cache_format(is_extra_kv, [&]<ModelType CACHE_MODEL_TYPE>() __aicore__ {
                    process_kv_index_vf<CACHE_MODEL_TYPE>(
                        ub_buf,
                        cur_index_buf_for_run_process_index,
                        0x7FFFFFFFu,
                        is_extra_kv ? params.stride_extra_kv_block : params.stride_kv_block,
                        valid_len,
                        is_extra_kv ? params.extra_page_block_size : params.page_block_size,
                        is_extra_kv ? aux_params.extra_page_block_size_fast_div_mod : aux_params.page_block_size_fast_div_mod
                    );
                });
            } else {
                process_kv_index_vf<MODEL_TYPE>(
                    ub_buf, cur_index_buf_for_run_process_index, params.s_kv, 0, valid_len, 0, aux_params.page_block_size_fast_div_mod);
            }
            rls_buf(PIPE_V, VectorCoreBufIDs::PROCESSED_INDICES, false);
            rls_buf(PIPE_V, VectorCoreBufIDs::RAW_INDICES, false);
            if (args.job_idx == 0 && kv_block_idx == 0) {
                // kv_nd2nz_vf(ub_buf, 0);  // VF Warmup
            }
            plus_one_and_mod<NUM_INDEX_BUFS>(cur_index_buf_for_run_process_index);
        };

        uint32_t cur_run_issue_kv_gather_kv_buf_idx = 0u;   // Self-incremental
        auto run_issue_kv_gather = [&](const OuterLoopArgs &args, uint32_t kv_block_idx) __aicore__ {
            // Gather two complete tokens (raw + scales for decode) with one MTE2 instruction.
            bool is_extra_kv = kv_block_idx >= args.num_orig_topk_blocks;
            uint32_t partial_kv_block_idx = IS_DECODE && is_extra_kv ? kv_block_idx - args.num_orig_topk_blocks : kv_block_idx;
            uint32_t kv_block_idx_in_group = partial_kv_block_idx % (INDEX_BLOCK_SIZE / B_TOPK);

            uint32_t kv_buf_idx = cur_run_issue_kv_gather_kv_buf_idx;
            plus_one_and_mod<NUM_UB_KV_BUFS>(cur_run_issue_kv_gather_kv_buf_idx);
            get_buf(PIPE_MTE2, VectorCoreBufIDs::GATHERED_KV + kv_buf_idx, false);

            __ubuf__ int64_t* cur_ub_global_kv_ptr_offset = ub_buf.global_kv_ptr_offset + (kv_block_idx_in_group*B_TOPK + subblock_idx*B_TOPK_PER_V)/2;
            __ubuf__ int64_t* cur_ub_src_stride_in_gather2_ptr = ub_buf.src_stride_in_gather2 + (kv_block_idx_in_group*B_TOPK + subblock_idx*B_TOPK_PER_V)/2;
            get_buf(PIPE_S, VectorCoreBufIDs::PROCESSED_INDICES, false);

            if constexpr (IS_DECODE) {
                dispatch_cache_format(is_extra_kv, [&]<ModelType CACHE_MODEL_TYPE>() __aicore__ {
                    using F = KVCacheFormat<CACHE_MODEL_TYPE>;
                    static constexpr uint32_t UB_TOKEN_BYTES = F::IS_FP4 ? FP4_UB_TOKEN_BYTES : FP8_UB_TOKEN_BYTES;
                    __ubuf__ uint8_t* dst;
                    if constexpr (F::IS_FP4) dst = ub_buf.gathered_kv_fp4[kv_buf_idx][0];
                    else dst = (__ubuf__ uint8_t*)ub_buf.gathered_kv[kv_buf_idx][0];
                    __gm__ uint8_t* kv_base = (__gm__ uint8_t*)(is_extra_kv ? params.extra_kv : params.kv);
                    #pragma unroll
                    for (uint32_t i = 0; i < B_TOPK_PER_V / 2; ++i) {
                        uint64_t offset = cur_ub_global_kv_ptr_offset[i];
                        uint32_t n_burst = offset != 0xFFFFFFFFFFFFFFFFul ? 2 : 0;
                        asc_copy_gm2ub_align(dst, kv_base + offset, n_burst,
                            F::BYTES_PER_TOKEN, 0, 0, false,
                            static_cast<asc_load_l2_cache_mode>(aux_params.kv_cache_hint_for_decoding),
                            cur_ub_src_stride_in_gather2_ptr[i], UB_TOKEN_BYTES);
                        dst += 2 * UB_TOKEN_BYTES;
                    }
                });
            } else {
                __ubuf__ gathered_kv_t* cur_ub_gathered_kv_ptr = ub_buf.gathered_kv[kv_buf_idx][0];
                uint64_t invalid_global_kv_ptr_offset = (uint64_t)params.s_kv * D_QK * sizeof(bf16);
                #pragma unroll
                for (uint32_t i = 0; i < B_TOPK_PER_V / 2; i += 1) {
                    uint64_t cur_global_kv_ptr_offset = cur_ub_global_kv_ptr_offset[i];
                    __gm__ gathered_kv_t* cur_base_ptr = (__gm__ gathered_kv_t*)((__gm__ char*)params.kv + cur_global_kv_ptr_offset);
                    uint32_t n_burst = cur_global_kv_ptr_offset != invalid_global_kv_ptr_offset ? 2 : 0;
                    asc_copy_gm2ub_align(
                        (__ubuf__ gathered_kv_t*)cur_ub_gathered_kv_ptr,
                        cur_base_ptr,
                        n_burst,   // Skip the load if both indices are invalid to avoid one L2 slice being congested
                        D_QK * sizeof(gathered_kv_t),
                        0, 0, false,
                        asc_load_l2_cache_mode::NORMAL_LAST_VICTIM,
                        *(cur_ub_src_stride_in_gather2_ptr+i),
                        D_QK * sizeof(gathered_kv_t)
                    );
                    cur_ub_gathered_kv_ptr += 2 * D_VO;
                }
            }
            
            rls_buf(PIPE_S, VectorCoreBufIDs::PROCESSED_INDICES, false);
            rls_buf(PIPE_MTE2, VectorCoreBufIDs::GATHERED_KV + kv_buf_idx, false);
        };

        uint32_t run_kv_nd2nz_and_push_to_cube_kv_buf_idx = 0u;   // Self-incremental
        auto run_kv_nd2nz_and_push_to_cube = [&](const OuterLoopArgs &args, uint32_t kv_block_idx) __aicore__ {
            uint32_t kv_buf_idx = run_kv_nd2nz_and_push_to_cube_kv_buf_idx;
            plus_one_and_mod<NUM_UB_KV_BUFS>(run_kv_nd2nz_and_push_to_cube_kv_buf_idx);
            get_buf(PIPE_V, VectorCoreBufIDs::GATHERED_KV + kv_buf_idx, false);
            get_buf(PIPE_V, VectorCoreBufIDs::KV_IN_NZ, false);
            if constexpr (!IS_DECODE) {
                kv_nd2nz_for_prefill_vf(ub_buf, kv_buf_idx);
            } else {
                dispatch_cache_format(kv_block_idx >= args.num_orig_topk_blocks, [&]<ModelType CACHE_MODEL_TYPE>() __aicore__ {
                    kv_dequant_and_nd2nz_for_decode_vf<CACHE_MODEL_TYPE>(ub_buf, kv_buf_idx);
                });
            }
            rls_buf(PIPE_V, VectorCoreBufIDs::KV_IN_NZ, false);
            rls_buf(PIPE_V, VectorCoreBufIDs::GATHERED_KV + kv_buf_idx, false);

            uint32_t l1_kv_buf_idx = (args.sum_prev_num_topk_blocks + kv_block_idx) % NUM_L1_KV_BUFS;
            get_buf(PIPE_MTE3, VectorCoreBufIDs::KV_IN_NZ, false);
            AscendC::CrossCoreWaitFlag<4, PIPE_MTE3>(CrossCoreFlags::L1_KV_BUF_EMPTY+l1_kv_buf_idx);
            asc_copy_ub2l1(
                (subblock_idx?l1_gathered_kv_buf_hi:l1_gathered_kv_buf_lo) + l1_kv_buf_idx*(B_TOPK_PER_V*D_QK),
                ub_buf.gathered_kv_in_nz,
                D_QK / FRACTAL_W,
                B_TOPK_PER_V * FRACTAL_W * sizeof(bf16) / 32,
                /*src_gap=*/ 1,
                /*dst_gap=*/ 0
            );
            AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(CrossCoreFlags::L1_KV_BUF_FULL+l1_kv_buf_idx);
            rls_buf(PIPE_MTE3, VectorCoreBufIDs::KV_IN_NZ, false);

            if (args.job_idx == 0 && kv_block_idx == 2) {
                // VF warmup - prevent the cache-miss-like behaviour when we call one VF for the first time
                // get_final_output_vf<false>(ub_buf, 0, 0, 0, 0, true);
            }
        };

        uint32_t cur_index_buf_idx_for_softmax = NUM_INDEX_BUFS - 1;
        auto run_softmax = [&](const OuterLoopArgs &args, uint32_t kv_block_idx) __aicore__ {
            bool is_extra_kv = kv_block_idx >= args.num_orig_topk_blocks;
            uint32_t partial_kv_block_idx = IS_DECODE && is_extra_kv ? kv_block_idx - args.num_orig_topk_blocks : kv_block_idx;
            if (partial_kv_block_idx % (INDEX_BLOCK_SIZE / B_TOPK) == 0)
                plus_one_and_mod<NUM_INDEX_BUFS>(cur_index_buf_idx_for_softmax);

            AscendC::CrossCoreWaitFlag<4, PIPE_V>(CrossCoreFlags::UBUF_P_FULL);
            get_buf(PIPE_V, VectorCoreBufIDs::FINAL_LOGITS_LSE + (args.job_idx&1), false);
            uint32_t s_kv;
            if constexpr (IS_DECODE) {
                s_kv = 0x7FFFFFFF;
            } else {
                s_kv = params.s_kv; // TODO Comment what is "invalid" for smaller / larger indices
            }
            uint32_t ub_indices_arr_offset = (partial_kv_block_idx % (INDEX_BLOCK_SIZE / B_TOPK)) * (B_TOPK / 2);
            if (kv_block_idx == 0) {
                softmax_vf<true>(ub_buf, params.sm_scale, aux_params.rescale_threshold_div_sm_scale, args.job_idx, kv_block_idx, cur_index_buf_idx_for_softmax, ub_indices_arr_offset, s_kv, subblock_idx);
            } else {
                softmax_vf<false>(ub_buf, params.sm_scale, aux_params.rescale_threshold_div_sm_scale, args.job_idx, kv_block_idx, cur_index_buf_idx_for_softmax, ub_indices_arr_offset, s_kv, subblock_idx);
            }
            rls_buf(PIPE_V, VectorCoreBufIDs::FINAL_LOGITS_LSE + (args.job_idx&1), false);
            set_flag(PIPE_V, PIPE_MTE3, event_t(VectorCoreFlags::S_FULL));
            if (kv_block_idx != 0) {
                set_flag(PIPE_V, PIPE_S, event_t(VectorCoreFlags::ROW_MAX_MAX_DELTA_FULL));
            }

            wait_flag(PIPE_V, PIPE_MTE3, event_t(VectorCoreFlags::S_FULL));
            AscendC::CrossCoreWaitFlag<4, PIPE_MTE3>(CrossCoreFlags::L1_S_EMPTY);
            asc_copy_ub2l1(
                l1_s_buf + subblock_idx * MMA_M_PER_V * FRACTAL_W,
                (__ubuf__ bf16*)ub_buf.p,
                B_TOPK / FRACTAL_W,
                MMA_M_PER_V * FRACTAL_W * sizeof(bf16) / 32,
                MMA_M_PER_V * FRACTAL_W * sizeof(bf16) / 32,
                (MMA_M - MMA_M_PER_V) * FRACTAL_W * sizeof(bf16) / 32
            );
            AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(CrossCoreFlags::L1_S_FULL);
            AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(CrossCoreFlags::UBUF_P_EMPTY);

            if (kv_block_idx+1 == args.num_topk_blocks) {
                if constexpr (HAVE_ATTN_SINK) {
                    if (args.job_idx == 0)
                        wait_flag(PIPE_MTE2, PIPE_V, event_t(VectorCoreFlags::ATTN_SINK_FULL));
                }
                get_buf(PIPE_V, VectorCoreBufIDs::FINAL_LOGITS_LSE + (args.job_idx&1), false);
                get_final_lse_and_denorm_vf(ub_buf, args.job_idx, params.sm_scale);
                rls_buf(PIPE_V, VectorCoreBufIDs::FINAL_LOGITS_LSE + (args.job_idx&1), false);

                get_buf(PIPE_MTE3, VectorCoreBufIDs::FINAL_LOGITS_LSE + (args.job_idx&1), false);
                if constexpr (!IS_DECODE) {
                    // Decoding doesn't have `max_logits`
                    asc_copy_ub2gm_align(
                        params.max_logits + args.batch_or_s_q_idx*params.h_q + this_subblock_start_head_idx,
                        ub_buf.row_max[args.job_idx&1],
                        num_this_subblock_valid_heads * sizeof(float)
                    );
                }
                __gm__ float* lse_base;
                if constexpr (IS_DECODE) {
                    lse_base = params.lse + args.batch_or_s_q_idx * params.stride_lse_s_q + this_subblock_start_head_idx;
                } else {
                    lse_base = params.lse + args.batch_or_s_q_idx * params.h_q + this_subblock_start_head_idx;
                }
                asc_copy_ub2gm_align(
                    lse_base,
                    ub_buf.row_sum[args.job_idx&1],
                    num_this_subblock_valid_heads * sizeof(float)
                );
                rls_buf(PIPE_MTE3, VectorCoreBufIDs::FINAL_LOGITS_LSE + (args.job_idx&1), false);
            }
        };

        auto run_set_is_rescale_triggered_flag = [&](const OuterLoopArgs &args, uint32_t kv_block_idx) __aicore__ {
            if (kv_block_idx > 0) {
                wait_flag(PIPE_V, PIPE_S, event_t(VectorCoreFlags::ROW_MAX_MAX_DELTA_FULL));
                ss_buf.is_rescale_triggered[args.job_idx%8][kv_block_idx%8][subblock_idx] = ub_buf.row_max_max_delta[args.job_idx&1][kv_block_idx&1] > aux_params.rescale_threshold_div_sm_scale;
                AscendC::CrossCoreSetFlag<4, PIPE_S>(CrossCoreFlags::SS_BUFFER_FULL_V2C);
            }
        };

        uint32_t o_block_idx;   // Private ro `run_scale_o`, marks whether this O block is the first one sent from FixPipe, and is resetted upon every request
        auto run_scale_o = [&](const OuterLoopArgs &args, uint32_t kv_block_idx) __aicore__ {
            if (kv_block_idx == 0)
                o_block_idx = 0;

            bool is_last_block = kv_block_idx+1 == args.num_topk_blocks;
            bool is_rescale_triggered;
            if (kv_block_idx == 0) {
                is_rescale_triggered = false;
            } else {
                AscendC::CrossCoreWaitFlag<4, PIPE_S>(CrossCoreFlags::SS_BUFFER_FULL_C2V);
                uint64_t t = *(volatile __ssbuf__ uint64_t*)ss_buf.is_rescale_triggered[args.job_idx%8][kv_block_idx%8];
                // uint64_t t = 0;
                is_rescale_triggered = t > 0;   // Trigger o copy out when at least one VECTOR Core requires rescaling
            }

            if (is_rescale_triggered) {
                static_assert(D_VO % OUTPUT_FRAG_D_VO == 0);
                #pragma unroll 1
                for (uint32_t i = 0; i < D_VO / OUTPUT_FRAG_D_VO; ++i) {
                    uint32_t cur_o_frag_buf_idx = i % NUM_OUTPUT_FRAG_BUFS;
                    get_buf(PIPE_V, VectorCoreBufIDs::O_ACCUM + i, false);
                    AscendC::CrossCoreWaitFlag<4, PIPE_V>(CrossCoreFlags::UBUF_O_FRAG_FULL+cur_o_frag_buf_idx);
                    if (o_block_idx == 0) {
                        add_and_scale_o_vf<true, false>(ub_buf, i, cur_o_frag_buf_idx, args.job_idx, kv_block_idx);
                    } else {
                        add_and_scale_o_vf<true, true>(ub_buf, i, cur_o_frag_buf_idx, args.job_idx, kv_block_idx);
                    }
                    AscendC::CrossCoreSetFlag<4, PIPE_V>(CrossCoreFlags::UBUF_O_FRAG_EMPTY+cur_o_frag_buf_idx);
                    rls_buf(PIPE_V, VectorCoreBufIDs::O_ACCUM + i, false);
                }
                o_block_idx += 1;
            }

            if (is_last_block) {
                // We're at the last block. FixPipe must copy the last few tiles to us. We need to
                //   1) Add it to the current `ub_buf.output_accum`, if there is any previous O
                //   2) Multiply it by `denorm` and cast to bf16
                static_assert(D_VO % OUTPUT_FRAG_D_VO == 0);
                #pragma unroll
                for (uint32_t i = 0; i < D_VO / OUTPUT_FRAG_D_VO; ++i) {
                    uint32_t cur_o_frag_buf_idx = i % NUM_OUTPUT_FRAG_BUFS;
                    get_buf(PIPE_V, VectorCoreBufIDs::O_ACCUM + i, false);
                    AscendC::CrossCoreWaitFlag<4, PIPE_V>(CrossCoreFlags::UBUF_O_FRAG_FULL+cur_o_frag_buf_idx);
                    if (o_block_idx == 0) {
                        get_final_output_vf<false>(ub_buf, i, cur_o_frag_buf_idx, args.job_idx, kv_block_idx, false);
                    } else {
                        get_final_output_vf<true>(ub_buf, i, cur_o_frag_buf_idx, args.job_idx, kv_block_idx, false);
                    }
                    AscendC::CrossCoreSetFlag<4, PIPE_V>(CrossCoreFlags::UBUF_O_FRAG_EMPTY+cur_o_frag_buf_idx);
                    rls_buf(PIPE_V, VectorCoreBufIDs::O_ACCUM + i, false);

                    uint32_t stride_o_batch_or_s_q;
                    if constexpr (IS_DECODE) {
                        stride_o_batch_or_s_q = params.stride_o_b;
                    } else {
                        stride_o_batch_or_s_q = params.h_q*D_VO;
                    }
                    get_buf(PIPE_MTE3, VectorCoreBufIDs::O_ACCUM + i, false);
                    asc_copy_ub2gm_align(
                        params.out + (uint64_t)args.batch_or_s_q_idx * stride_o_batch_or_s_q + subblock_idx * (MMA_M_PER_V*D_VO) + i*OUTPUT_FRAG_D_VO,
                        (__ubuf__ bf16*)ub_buf.output_accum + i*(OUTPUT_FRAG_D_VO*2),
                        num_this_subblock_valid_heads,
                        OUTPUT_FRAG_D_VO * sizeof(bf16),
                        asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                        D_VO * sizeof(bf16),
                        D_VO * sizeof(float)
                    );
                    rls_buf(PIPE_MTE3, VectorCoreBufIDs::O_ACCUM + i, false);
                }
            }
        };

        static constexpr uint32_t INVALID_ITEM_KV_BLOCK_IDX = 0xffffffffu;
        struct SystolicArrayItem {
            uint32_t kv_block_idx;
            OuterLoopArgs args;
        };
        
        static constexpr uint32_t SYSTOLIC_ARR_DEPTH = 6;
        SystolicArrayItem systolic_arr_items[SYSTOLIC_ARR_DEPTH];
        uint32_t systolic_array_head = 0u;
        auto systolic_arr_init = [&]() __aicore__ {
            #pragma unroll
            for (uint32_t i = 0; i < SYSTOLIC_ARR_DEPTH; ++i)
                systolic_arr_items[i].kv_block_idx = INVALID_ITEM_KV_BLOCK_IDX;
        };

        auto systolic_arr_shift_and_push = [&](const SystolicArrayItem &new_item) __aicore__ {
            systolic_arr_items[systolic_array_head] = new_item;
            systolic_array_head += 1;
            if (systolic_array_head == SYSTOLIC_ARR_DEPTH)
                systolic_array_head = 0;
        };

        auto systolic_arr_activate = [&]() __aicore__ {
            #define GET_SYSTOLIC_ARR_ITEM_WITH_OFFSET(offset) (systolic_arr_items[((offset)+systolic_array_head)%SYSTOLIC_ARR_DEPTH])
            #define ACTIVATE(offset, func) \
                if (GET_SYSTOLIC_ARR_ITEM_WITH_OFFSET(offset).kv_block_idx != INVALID_ITEM_KV_BLOCK_IDX) {\
                    func(GET_SYSTOLIC_ARR_ITEM_WITH_OFFSET(offset).args, GET_SYSTOLIC_ARR_ITEM_WITH_OFFSET(offset).kv_block_idx); \
                }

            if constexpr (!IS_DECODE) {
                ACTIVATE(5, run_issue_index_fetch);
            }
            ACTIVATE(3, run_kv_nd2nz_and_push_to_cube);
            ACTIVATE(1, run_softmax);
            if constexpr (IS_DECODE) {
                ACTIVATE(5, run_issue_index_fetch);
            }
            ACTIVATE(5, run_process_index);
            ACTIVATE(0, run_scale_o);

            // If we've just launched `run_process_index`, then we launch `issue_kv_gather` after `set_is_rescale_triggered_flag` because
            //  1) `run_process_index` is using VF, issuing `set_is_rescale_triggered_flag` now won't cause VF to be idle
            //  2) `run_issue_kv_gather` must wait for `run_process_index`. If we launch `set_is_rescale_triggered_flag` after `issue_kv_gather`, the scalar core must wait for `process_index` to complete which cause cube core to be idle
            bool issue_kv_gather_later = GET_SYSTOLIC_ARR_ITEM_WITH_OFFSET(5).kv_block_idx != INVALID_ITEM_KV_BLOCK_IDX && GET_SYSTOLIC_ARR_ITEM_WITH_OFFSET(5).kv_block_idx % (INDEX_BLOCK_SIZE / B_TOPK) == 0;
            if (issue_kv_gather_later) {
                ACTIVATE(1, run_set_is_rescale_triggered_flag);
            }
            ACTIVATE(5, run_issue_kv_gather);
            if (!issue_kv_gather_later) {
                ACTIVATE(1, run_set_is_rescale_triggered_flag);
            }
            #undef ACTIVATE
        };

        systolic_arr_init();

        OuterLoopArgs cur_job = get_first_job();

        // Clear the K/V gather buffers: gathers that are skipped (both indices of a pair
        // invalid) must leave zeros behind, otherwise the SV MMA may compute 0 * stale
        // (possibly NaN/Inf) values, which poison the O accumulator
        clear_all_kv_bufs();

        bool is_draining = false;
        // We fuse the main loop and the draining loop together to reduce code size and avoid icache miss
        while (true) {
            #pragma unroll 1
            for (uint32_t k = 0; k < (is_draining ? SYSTOLIC_ARR_DEPTH : cur_job.num_topk_blocks); ++k) {
                systolic_arr_shift_and_push(SystolicArrayItem{is_draining ? INVALID_ITEM_KV_BLOCK_IDX : k, cur_job});
                systolic_arr_activate();
            }
            if (is_draining)
                break;
            cur_job = get_next_job(cur_job);
            if (!cur_job.is_valid()) {
                is_draining = true;
            }
        };
    } else {
        // CUBE Core
        __ca__ bf16* l0a_buf = (__ca__ bf16*)0;
        __cb__ bf16* l0b_buf = (__cb__ bf16*)0;
        __cc__ float* l0c_out_buf = (__cc__ float*)0;
        __cc__ float* l0c_p_buf = l0c_out_buf + MMA_M*D_VO;

        uint64_t q_nz_params = uint64_t(1)
                             | (uint64_t(1) << 16)
                             | (uint64_t(MMA_M_PER_V) << 32);
        asc_set_gm2l1_nz_para(q_nz_params); // TODO 还有其他 param 需要设置吗
        asc_set_l0c2gm_nz2nd(1u, 0u, 0u);
        asc_set_l3d_rpt_b(1<<16);

        cross_core_set_flag_2aiv<PIPE_S>(CrossCoreFlags::L1_S_EMPTY);
        #pragma unroll
        for (uint32_t i = 0; i < NUM_L1_KV_BUFS; ++i)
            cross_core_set_flag_2aiv<PIPE_S>(CrossCoreFlags::L1_KV_BUF_EMPTY+i);
        #pragma unroll
        for (uint32_t i = 0; i < NUM_L1_Q_BUFS; ++i)
            set_flag(PIPE_MTE1, PIPE_MTE2, event_t(CubeCoreFlags::L1_Q_EMPTY+i));

        auto run_load_q = [&](const OuterLoopArgs &args) __aicore__ {
            // Load Q
            uint32_t q_l1_buf_idx = args.job_idx % NUM_L1_Q_BUFS;
            wait_flag(PIPE_MTE1, PIPE_MTE2, event_t(CubeCoreFlags::L1_Q_EMPTY+q_l1_buf_idx));
            __gm__ bf16* q_base;
            if constexpr (IS_DECODE) {
                q_base = params.q + (uint64_t)args.batch_or_s_q_idx * params.stride_q_b;
            } else {
                q_base = params.q + (uint64_t)args.batch_or_s_q_idx * params.stride_q_s_q;
            }
            uint32_t num_valid_q_heads_part0 = min(H_Q, MMA_M_PER_V);
            uint32_t num_valid_q_heads_part1 = max((int)H_Q-(int)MMA_M_PER_V, 0);
            asc_copy_gm2l1_nd2nz(
                l1_q_buf_lo + q_l1_buf_idx * (MMA_M_PER_V * D_QK),
                q_base,
                params.stride_q_h_q * sizeof(bf16),
                (uint32_t)LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                num_valid_q_heads_part0,
                D_QK,
                0u,
                false
            );
            asc_copy_gm2l1_nd2nz(
                l1_q_buf_hi + q_l1_buf_idx * (MMA_M_PER_V * D_QK),
                q_base + (uint64_t)(MMA_M_PER_V) * params.stride_q_h_q,
                params.stride_q_h_q * sizeof(bf16),
                (uint32_t)LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                num_valid_q_heads_part1,
                D_QK,
                0u,
                false
            );
            set_flag(PIPE_MTE2, PIPE_MTE1, event_t(CubeCoreFlags::L1_Q_FULL+q_l1_buf_idx));
        };

        uint32_t cur_l0a_buf_idx = 0u; // self-incremental
        auto run_qk_gemm = [&](const OuterLoopArgs &args, uint32_t kv_block_idx) __aicore__ {
            // Pipelined: Load Q to L0A, load K to L0B, issue MMA
            asc_set_mmad_direction_m();
            uint32_t kv_l1_buf_idx = (args.sum_prev_num_topk_blocks + kv_block_idx) % NUM_L1_KV_BUFS;
            #pragma unroll
            for (uint32_t tile_idx = 0; tile_idx < D_QK / L12L0_TILE_SIZE; ++tile_idx) {
                bool is_last_tile = tile_idx+1 == D_QK / L12L0_TILE_SIZE;
                uint32_t q_l1_buf_idx = args.job_idx % NUM_L1_Q_BUFS;
                uint32_t q_l0a_buf_idx = cur_l0a_buf_idx;
                plus_one_and_mod<NUM_L0A_BUFS>(cur_l0a_buf_idx);
                if (tile_idx == 0 && kv_block_idx == 0) {
                    wait_flag(PIPE_MTE2, PIPE_MTE1, event_t(CubeCoreFlags::L1_Q_FULL+q_l1_buf_idx));
                }
                get_buf(PIPE_MTE1, CubeCoreBufIDs::L0A_BUF+q_l0a_buf_idx, false);
                asc_copy_l12l0a(
                    l0a_buf + q_l0a_buf_idx * (MMA_M * L12L0_TILE_SIZE),
                    l1_q_buf_lo + q_l1_buf_idx * (MMA_M_PER_V * D_QK),
                    0, tile_idx * (L12L0_TILE_SIZE/FRACTAL_W),
                    MMA_M_PER_V / FRACTAL_H, L12L0_TILE_SIZE / FRACTAL_W,
                    MMA_M_PER_V / FRACTAL_H, MMA_M / FRACTAL_H
                );
                asc_copy_l12l0a(
                    l0a_buf + q_l0a_buf_idx * (MMA_M * L12L0_TILE_SIZE) + MMA_M_PER_V * FRACTAL_W,
                    l1_q_buf_hi + q_l1_buf_idx * (MMA_M_PER_V * D_QK),
                    0, tile_idx * (L12L0_TILE_SIZE/FRACTAL_W),
                    MMA_M_PER_V / FRACTAL_H, L12L0_TILE_SIZE / FRACTAL_W,
                    MMA_M_PER_V / FRACTAL_H, MMA_M / FRACTAL_H
                );
                rls_buf(PIPE_MTE1, CubeCoreBufIDs::L0A_BUF+q_l0a_buf_idx, false);

                if (is_last_tile && kv_block_idx+1 == args.num_topk_blocks) {
                    set_flag(PIPE_MTE1, PIPE_MTE2, event_t(CubeCoreFlags::L1_Q_EMPTY+q_l1_buf_idx));
                }

                // Load K to L0B
                uint32_t kv_l0b_buf_idx = tile_idx;
                if (tile_idx == 0) {
                    cross_core_wait_flag_2aiv<PIPE_MTE1>(CrossCoreFlags::L1_KV_BUF_FULL+kv_l1_buf_idx);
                }
                get_buf(PIPE_MTE1, CubeCoreBufIDs::L0B_BUF+kv_l0b_buf_idx, false);
                asc_copy_l12l0b(
                    l0b_buf + kv_l0b_buf_idx*(B_TOPK*L12L0_TILE_SIZE),
                    l1_gathered_kv_buf_lo + kv_l1_buf_idx*(B_TOPK_PER_V*D_QK),
                    0, tile_idx*(L12L0_TILE_SIZE/FRACTAL_W),
                    B_TOPK_PER_V / FRACTAL_H, L12L0_TILE_SIZE / FRACTAL_W,
                    B_TOPK_PER_V / FRACTAL_H, B_TOPK / FRACTAL_H
                );
                asc_copy_l12l0b(
                    l0b_buf + kv_l0b_buf_idx*(B_TOPK*L12L0_TILE_SIZE) + B_TOPK_PER_V*FRACTAL_H,
                    l1_gathered_kv_buf_hi + kv_l1_buf_idx*(B_TOPK_PER_V*D_QK),
                    0, tile_idx*(L12L0_TILE_SIZE/FRACTAL_W),
                    B_TOPK_PER_V / FRACTAL_H, L12L0_TILE_SIZE / FRACTAL_W,
                    B_TOPK_PER_V / FRACTAL_H, B_TOPK / FRACTAL_H
                );
                rls_buf(PIPE_MTE1, CubeCoreBufIDs::L0B_BUF+kv_l0b_buf_idx, false);
                
                // Issue QK^T MMA
                get_buf(PIPE_M, CubeCoreBufIDs::L0A_BUF+q_l0a_buf_idx, false);
                get_buf(PIPE_M, CubeCoreBufIDs::L0B_BUF+kv_l0b_buf_idx, false);
                asc_mmad(
                    l0c_p_buf,
                    l0a_buf + q_l0a_buf_idx * (MMA_M * L12L0_TILE_SIZE),
                    l0b_buf + kv_l0b_buf_idx * (B_TOPK * L12L0_TILE_SIZE),
                    MMA_M, L12L0_TILE_SIZE, B_TOPK,
                    is_last_tile ? 3 : 2,
                    true, false, tile_idx == 0
                );
                rls_buf(PIPE_M, CubeCoreBufIDs::L0A_BUF+q_l0a_buf_idx, false);
                rls_buf(PIPE_M, CubeCoreBufIDs::L0B_BUF+kv_l0b_buf_idx, false);
            }

            cross_core_wait_flag_2aiv<PIPE_FIX>(CrossCoreFlags::UBUF_P_EMPTY);
            asc_copy_l0c2ub(
                ub_buf.p,
                l0c_p_buf,
                B_TOPK,
                MMA_M,
                MMA_M_PER_V * 16,
                MMA_M,
                /*dual_dst_ctl=*/ 1,
                false,
                0,
                /*unit_flag_ctl=*/ 3,
                0, 0, false,
                /*NZ2ND_en=*/ false,
                0, 0, 0, 0, 0, 0, 0, 0
            );
            cross_core_set_flag_2aiv<PIPE_FIX>(CrossCoreFlags::UBUF_P_FULL);
        };

        auto run_sv_gemm = [&](const OuterLoopArgs &args, uint32_t kv_block_idx) __aicore__ {
            bool is_last_block = kv_block_idx+1 == args.num_topk_blocks;
            
            auto trigger_o_copy_out = [&]<bool IS_LAST_BLOCK>() __aicore__ {
                /*
                Synchronization:
                - last SV in the current req -> out copy-out -> the first SV in the next req: synchronize by unit_flag
                - others: by L0C_O_FULL / L0C_O_EMPTY
                */
                uint32_t unit_flag = IS_LAST_BLOCK ? 3 : 0;
                if constexpr (not IS_LAST_BLOCK) {
                    set_flag(PIPE_M, PIPE_FIX, event_t(CubeCoreFlags::L0C_O_FULL));
                    wait_flag(PIPE_M, PIPE_FIX, event_t(CubeCoreFlags::L0C_O_FULL));
                }
                static_assert(D_VO % OUTPUT_FRAG_D_VO == 0);
                #pragma unroll
                for (uint32_t i = 0; i < D_VO / OUTPUT_FRAG_D_VO; ++i) {
                    uint32_t cur_o_frag_buf_idx = i % NUM_OUTPUT_FRAG_BUFS;
                    cross_core_wait_flag_2aiv<PIPE_FIX>(CrossCoreFlags::UBUF_O_FRAG_EMPTY+cur_o_frag_buf_idx);
                    asc_copy_l0c2ub(
                        ub_buf.cur_output_frag[cur_o_frag_buf_idx][0],
                        l0c_out_buf + i * (MMA_M * OUTPUT_FRAG_D_VO),
                        OUTPUT_FRAG_D_VO,
                        MMA_M,
                        /*loop_dst_stride=*/ OUTPUT_FRAG_D_VO,
                        /*loop_src_stride=*/ MMA_M,
                        /*dual_dst_ctl=*/ 1,
                        false, 0,
                        /*unit_flag_ctl=*/ unit_flag,
                        0, 0, false,
                        /*NZ2ND_en=*/ true,
                        0, 0, 0, 0, 0, 0, 0, 0
                    );
                    cross_core_set_flag_2aiv<PIPE_FIX>(CrossCoreFlags::UBUF_O_FRAG_FULL+cur_o_frag_buf_idx);
                }
                if constexpr (not IS_LAST_BLOCK) {
                    set_flag(PIPE_FIX, PIPE_M, event_t(CubeCoreFlags::L0C_O_EMPTY));
                    wait_flag(PIPE_FIX, PIPE_M, event_t(CubeCoreFlags::L0C_O_EMPTY));
                }
            };

            // Pipelined: Load V to L0B (w/ transpose), and issue MMA
            asc_set_mmad_direction_n();
            uint32_t kv_l1_buf_idx = (args.sum_prev_num_topk_blocks + kv_block_idx) % NUM_L1_KV_BUFS;
            uint32_t s_l0a_buf_idx = cur_l0a_buf_idx;
            __ca__ bf16* s_l0a_buf_ptr = l0a_buf + s_l0a_buf_idx * (MMA_M * L12L0_TILE_SIZE);
            bool should_trigger_o_copy_out_before_mma;  // Will be postponed until we've issued copy for KV[0] and S, since it relies on SS_BUFFER_FULL_V2C and blocks scalar code
            
            #pragma unroll
            for (uint32_t tile_idx = 0; tile_idx < D_QK / L12L0_TILE_SIZE; ++tile_idx) {
                // Load V to L0B
                uint32_t kv_l0b_buf_idx = tile_idx;
                bool is_last_tile = tile_idx+1 == D_QK / L12L0_TILE_SIZE;
                get_buf(PIPE_MTE1, CubeCoreBufIDs::L0B_BUF+kv_l0b_buf_idx, false);
                asc_copy_l12l0b_transpose(
                    l0b_buf + kv_l0b_buf_idx * (B_TOPK * L12L0_TILE_SIZE),
                    l1_gathered_kv_buf_lo + kv_l1_buf_idx*(B_TOPK_PER_V*D_QK),
                    0, tile_idx*(L12L0_TILE_SIZE/FRACTAL_W),
                    B_TOPK_PER_V / FRACTAL_H, L12L0_TILE_SIZE / FRACTAL_W,
                    B_TOPK_PER_V / FRACTAL_H, L12L0_TILE_SIZE / FRACTAL_H
                );
                asc_copy_l12l0b_transpose(
                    l0b_buf + kv_l0b_buf_idx * (B_TOPK * L12L0_TILE_SIZE) + B_TOPK_PER_V * L12L0_TILE_SIZE,
                    l1_gathered_kv_buf_hi + kv_l1_buf_idx*(B_TOPK_PER_V*D_QK),
                    0, tile_idx*(L12L0_TILE_SIZE/FRACTAL_W),
                    B_TOPK_PER_V / FRACTAL_H, L12L0_TILE_SIZE / FRACTAL_W,
                    B_TOPK_PER_V / FRACTAL_H, L12L0_TILE_SIZE / FRACTAL_H
                );
                rls_buf(PIPE_MTE1, CubeCoreBufIDs::L0B_BUF+kv_l0b_buf_idx, false);
                if (is_last_tile) {
                    cross_core_set_flag_2aiv<PIPE_MTE1>(CrossCoreFlags::L1_KV_BUF_EMPTY+kv_l1_buf_idx);
                }

                if (tile_idx == 0) {
                    // Load S to L0A
                    cross_core_wait_flag_2aiv<PIPE_MTE1>(CrossCoreFlags::L1_S_FULL);
                    plus_one_and_mod<NUM_L0A_BUFS>(cur_l0a_buf_idx);
                    get_buf(PIPE_MTE1, CubeCoreBufIDs::L0A_BUF+s_l0a_buf_idx, false);
                    asc_copy_l12l0a(
                        s_l0a_buf_ptr,
                        l1_s_buf,
                        0, 0,
                        MMA_M / FRACTAL_H, B_TOPK / FRACTAL_W,
                        MMA_M / FRACTAL_H, MMA_M / FRACTAL_H
                    );
                    rls_buf(PIPE_MTE1, CubeCoreBufIDs::L0A_BUF+s_l0a_buf_idx, false);
                    cross_core_set_flag_2aiv<PIPE_MTE1>(CrossCoreFlags::L1_S_EMPTY);
                    get_buf(PIPE_M, CubeCoreBufIDs::L0A_BUF+s_l0a_buf_idx, false);

                    // Read ss buf, and decide whether to copy back the current o_accumulator before MMA
                    if (kv_block_idx > 0) {
                        auto ssbuf_ptr = ss_buf.is_rescale_triggered[args.job_idx%8][kv_block_idx%8];
                        cross_core_wait_flag_2aiv<PIPE_S>(CrossCoreFlags::SS_BUFFER_FULL_V2C);
                        cross_core_set_flag_2aiv<PIPE_S>(CrossCoreFlags::SS_BUFFER_FULL_C2V);
                        uint64_t is_rescale_triggered = *(volatile __ssbuf__ uint64_t*)ssbuf_ptr;
                        should_trigger_o_copy_out_before_mma = is_rescale_triggered > 0;   // Trigger o copy out when at least one VECTOR Core requires rescaling
                        if (should_trigger_o_copy_out_before_mma) {
                            trigger_o_copy_out.template operator()<false>();
                        }
                    } else {
                        should_trigger_o_copy_out_before_mma = false;
                    }
                }

                // Issue SV MMA
                uint32_t unit_flag = is_last_block ? 3 : (kv_block_idx == 0 ? 2 : 0);
                get_buf(PIPE_M, CubeCoreBufIDs::L0B_BUF+kv_l0b_buf_idx, false);
                asc_mmad(
                    l0c_out_buf + tile_idx * (MMA_M * L12L0_TILE_SIZE),
                    s_l0a_buf_ptr,
                    l0b_buf + kv_l0b_buf_idx * (B_TOPK * L12L0_TILE_SIZE),
                    MMA_M, B_TOPK, L12L0_TILE_SIZE,
                    unit_flag,
                    true, false,
                    kv_block_idx == 0 || should_trigger_o_copy_out_before_mma  // Drop the last value when this block is the first block or the previous block is copied out
                );
                rls_buf(PIPE_M, CubeCoreBufIDs::L0B_BUF+kv_l0b_buf_idx, false);
            }
            rls_buf(PIPE_M, CubeCoreBufIDs::L0A_BUF+s_l0a_buf_idx, false);

            if (is_last_block) {
                trigger_o_copy_out.template operator()<true>();
            }
        };

        OuterLoopArgs cur_job = get_first_job();
        run_load_q(cur_job);
        run_qk_gemm(cur_job, 0);
        do {
            OuterLoopArgs nxt_job = get_next_job(cur_job);
            if (nxt_job.is_valid()) {
                run_load_q(nxt_job);
            }
            for (uint32_t k = 0; k+1 < cur_job.num_topk_blocks; ++k) {
                run_qk_gemm(cur_job, k+1);
                run_sv_gemm(cur_job, k);
            }
            if (nxt_job.is_valid()) {
                run_qk_gemm(nxt_job, 0);
            }
            run_sv_gemm(cur_job, cur_job.num_topk_blocks-1);
            cur_job = nxt_job;
        } while (cur_job.is_valid());
    }
}


template<typename Kernel>
__global__ __mix__(1, 2) void sparse_attn_fwd_kernel(const typename Kernel::Params params, const typename Kernel::AuxParams aux_params) {
    Kernel::sparse_attn_fwd_kernel_devfunc(params, aux_params);
}

inline uint32_t min(uint32_t x, uint32_t y) {
    return x < y ? x : y;
}

template<Config CONFIG>
void Kernel<CONFIG>::run(const Params& params) {
    if constexpr (IS_DECODE) {
        KU_ASSERT_NPU(params.s_q == 1, "s_q must be 1");    // Since we have dynamic DFlash prediction length, s_q is always 1
    }
    KU_ASSERT_NPU(params.h_q == H_Q, "dispatch failure");
    KU_ASSERT_NPU(params.topk % B_TOPK == 0, "topk must be a multiple of B_TOPK (%d)", B_TOPK);
    KU_ASSERT_NPU(params.num_sm == GRID_SHAPE, "The number of AI Cores does not match expected: %d", params.num_sm);
    if constexpr (!IS_DECODE) { 
        KU_ASSERT_NPU(params.stride_kv_s_kv == D_QK, "kv must be contiguous");
    }
    AuxParams aux_params = {
        RESCALE_THRESHOLD / params.sm_scale
    };
    if constexpr (IS_DECODE) {
        KU_ASSERT_NPU(params.page_block_size > 1, "page_block_size must > 1");
        aux_params.page_block_size_fast_div_mod = FastDivMod(params.page_block_size);
        if (params.extra_page_block_size > 0) {
            KU_ASSERT_NPU(params.extra_page_block_size > 1, "extra_page_block_size must > 1");
            aux_params.extra_page_block_size_fast_div_mod = FastDivMod(params.extra_page_block_size);
        }
        // Allocate KV in L2 only when a small shared cache is read repeatedly.
        // Independent requests benefit from bypassing allocation instead.
        static constexpr int64_t SMALL_CACHE_BYTES = 8 * 1024 * 1024;
        static constexpr int64_t MIN_CACHE_REUSE = 4;
        int64_t kv_bytes = int64_t(params.num_blocks) * params.stride_kv_block
            + int64_t(params.extra_num_blocks) * params.stride_extra_kv_block;
        int64_t kv_tokens = int64_t(params.num_blocks) * params.page_block_size
            + int64_t(params.extra_num_blocks) * params.extra_page_block_size;
        int64_t token_reads = int64_t(params.b) * (int64_t(params.topk) + params.extra_topk);
        bool reuse_small_cache = kv_bytes <= SMALL_CACHE_BYTES && token_reads >= MIN_CACHE_REUSE * kv_tokens
            && params.topk_length == nullptr && params.extra_topk_length == nullptr;
        aux_params.kv_cache_hint_for_decoding = static_cast<uint32_t>(reuse_small_cache
            ? asc_load_l2_cache_mode::NORMAL_LAST_VICTIM : asc_load_l2_cache_mode::NOTALLOC_CLEAN);
    }
    uint32_t total_q;
    if constexpr (IS_DECODE) {
        total_q = (uint32_t)params.b;
    } else {
        total_q = (uint32_t)params.s_q;
    }
    uint32_t num_ctas = min(total_q, GRID_SHAPE);
    sparse_attn_fwd_kernel<Kernel<CONFIG>><<<num_ctas, 0, params.stream>>>(params, aux_params);
    KU_ACLRT_CHECK_NPU(aclrtGetLastError(aclrtLastErrLevel::ACL_RT_THREAD_LEVEL));
}


template<Config CONFIG>
void run_sparse_fwd_kernel(const ParamT<CONFIG.FWD_MODE>& params) {
    Kernel<CONFIG>::run(params);
}

}
