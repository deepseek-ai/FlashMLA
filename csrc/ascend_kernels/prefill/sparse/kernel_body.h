#pragma once

#include "kernel.h"

#include <numeric>

#include <acl/acl.h>
#include <kernel_operator.h>
#include <c_api/asc_simd.h>

#include <kerutils/kerutils_for_ascend_npu.h>

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
        vector_bool is_invalid_even, is_invalid_odd;
        asc_ge_scalar(is_invalid_even, *(vector_int32_t*)&rel_offset_even, valid_len, full_mask);
        asc_ge_scalar(is_invalid_odd, *(vector_int32_t*)&rel_offset_odd, valid_len, full_mask);
        vdup(cur_indices_even, s_kv, is_invalid_even, MODE_MERGING);
        vdup(cur_indices_odd, s_kv, is_invalid_odd, MODE_MERGING);
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
        auto full_mask = asc_create_mask_b8(PAT_ALL);
        __ubuf__ fp8_e4m3fn_t* cur_raw_scales_ptr = (__ubuf__ fp8_e4m3fn_t*)ub_buf.gathered_kv_fp4[kv_buf_idx][0] + FP4Format::QUANT_BYTES;
        __ubuf__ bf16* cur_dst_scales_ptr = (__ubuf__ bf16*)ub_buf.vf_scratchpad;
        constexpr uint32_t SCALE_LOAD_BLOCK_SIZE = FP4_UB_TOKEN_BYTES / 32;
        constexpr uint32_t SCALE_LOAD_REPEAT_SIZE = 8 * (FP4_UB_TOKEN_BYTES / 32);
        constexpr uint32_t SCALE_LOAD_CTRL = (SCALE_LOAD_BLOCK_SIZE << 16u | SCALE_LOAD_REPEAT_SIZE & 0xFFFFu);
        #pragma unroll
        for (uint16_t row = 0; row < B_TOPK_PER_V; row += 8) {
            vector_fp8_e4m3fn_t packed;
            vsldb(packed, cur_raw_scales_ptr, SCALE_LOAD_CTRL, full_mask, POST_UPDATE);
            vector_float p0, p1, p2, p3;
            asc_e4m32float(p0, packed, full_mask, ASC_DISPERSE_FIRST_QUARTER);
            asc_e4m32float(p1, packed, full_mask, ASC_DISPERSE_SECOND_QUARTER);
            asc_e4m32float(p2, packed, full_mask, ASC_DISPERSE_THIRD_QUARTER);
            asc_e4m32float(p3, packed, full_mask, ASC_DISPERSE_FOURTH_QUARTER);
            vector_uint32_t b0, b2;
            asc_shiftright_scalar(b0, (vector_uint32_t&)p0, 16, full_mask);
            asc_or(b0, b0, (vector_uint32_t&)p1, full_mask);
            asc_shiftright_scalar(b2, (vector_uint32_t&)p2, 16, full_mask);
            asc_or(b2, b2, (vector_uint32_t&)p3, full_mask);
            asc_storealign_intlv((__ubuf__ uint32_t*)(cur_dst_scales_ptr + row * FP4Format::NUM_SCALES_EACH_TOKEN), b0, b2);
        }
        asc_mem_bar(VST_VLD);

        auto none_mask = asc_create_mask_b16(PAT_ALLF);
        asc_store_align_config_post config;
        config.block_stride = B_TOPK_PER_V + 1;
        config.repeat_stride = (B_TOPK_PER_V + 1) * (256 / 32);
        __ubuf__ fp4x2_e2m1_t* cur_payload_ptr = (__ubuf__ fp4x2_e2m1_t*)ub_buf.gathered_kv_fp4[kv_buf_idx][0];
        __ubuf__ bf16* cur_scales_ptr = (__ubuf__ bf16*)ub_buf.vf_scratchpad;
        __ubuf__ bf16* cur_dst_ptr = ub_buf.gathered_kv_in_nz;
        #pragma unroll 1
        for (uint16_t row = 0; row < B_TOPK_PER_V; ++row) {
            #pragma unroll
            for (uint16_t chunk = 0; chunk < D_QK / NUM_BF16_IN_VEC; ++chunk) {
                vector_bf16 scale, values;
                asc_loadalign_brc_elem2datablock_postupdate(scale, cur_scales_ptr, 8);
                vector_fp4x2_e2m1_t packed;
                // UNPACK4 keeps each low/high nibble pair together: one conversion yields 128 ordered BF16 values.
                asc_loadalign_unpack4_postupdate(
                    packed,
                    cur_payload_ptr,
                    64 + (chunk+1 == D_QK/NUM_BF16_IN_VEC ? FP4_UB_TOKEN_BYTES - FP4Format::QUANT_BYTES : 0)
                );
                asc_e2m1x22bfloat16(values, packed, full_mask, ASC_DISPERSE_FIRST_QUARTER);
                asc_mul(values, values, scale, full_mask);
                vsstb(values, cur_dst_ptr, config.config, full_mask, POST_UPDATE);
            }

            vector_bf16 tmp;
            asc_storealign_postupdate(cur_dst_ptr, tmp, -(D_QK/NUM_BF16_IN_VEC)*(B_TOPK_PER_V+1)*256/sizeof(bf16) + FRACTAL_W, none_mask);  // Use this to update `cur_dst_ptr`
        }
    } else {
        static_assert(D_QK == 512 && SCALE_GRAN == 32);
        auto full_mask = asc_create_mask_b8(PAT_ALL);
        auto none_mask = asc_create_mask_b16(PAT_ALLF);

        // Gather eight inline scale rows at once, then convert UE8M0 to BF16.
        // Index i selects row i/16, scale i%16; raw records remain read-only.
        vector_uint16_t scale_indices, row_indices;
        asc_arange((vector_int16_t&)scale_indices, int16_t(0));
        asc_shiftright_scalar(row_indices, scale_indices, 4, full_mask);
        asc_mul_scalar(row_indices, row_indices, FP8_UB_TOKEN_BYTES - NUM_SCALES_PER_TOKEN, full_mask);
        asc_add(scale_indices, row_indices, scale_indices, full_mask);
        __ubuf__ uint8_t* raw_scales = (__ubuf__ uint8_t*)ub_buf.gathered_kv[kv_buf_idx][0] + FP8Format::QUANT_BYTES;
        __ubuf__ uint16_t* dst_scales = (__ubuf__ uint16_t*)ub_buf.vf_scratchpad;
        #pragma unroll 1
        for (uint16_t row = 0; row < B_TOPK_PER_V; row += 8) {
            vector_uint16_t scales_bf16;
            asc_gather(scales_bf16, raw_scales, scale_indices, full_mask);
            raw_scales += 8 * FP8_UB_TOKEN_BYTES;
            // Preserve the existing clamp for invalid/NaN scale encodings.
            asc_min_scalar(scales_bf16, scales_bf16, 0xE0, full_mask);
            asc_shiftleft_scalar(scales_bf16, scales_bf16, 7, full_mask);
            asc_storealign_postupdate(dst_scales, scales_bf16, NUM_BF16_IN_VEC, full_mask);
        }

        // Now scales are converted to BF16
        asc_mem_bar(VST_VLD);
        asc_store_align_config_post config;
        config.block_stride = B_TOPK_PER_V+1;
        config.repeat_stride = (B_TOPK_PER_V+1) * (256/32);

        __ubuf__ gathered_kv_t* cur_gathered_kv_ptr = ub_buf.gathered_kv[kv_buf_idx][0];
        __ubuf__ bf16* cur_gathered_scale_factors_ptr = (__ubuf__ bf16*)ub_buf.vf_scratchpad;
        __ubuf__ bf16* cur_dst_ptr = ub_buf.gathered_kv_in_nz;

        #pragma unroll 1
        for (uint16_t i = 0; i < B_TOPK_PER_V; ++i) {
            vector_bf16 scale_factors[4];
            asc_loadalign_brc_elem2datablock_postupdate(scale_factors[0], cur_gathered_scale_factors_ptr, 8);        // s0x16, s1x16, s2x16, ..., s7x16
            asc_loadalign_brc_elem2datablock_postupdate(scale_factors[2], cur_gathered_scale_factors_ptr, 8);    // s8x16, s9x16, s10x16, ..., s15x16
            asc_intlv(scale_factors[0], scale_factors[1], scale_factors[0], scale_factors[0]);  // scale_factors[0]: s0x32, s1x32, s2x32, s3x32; scale_factors[1]: s4x32, s5x32, s6x32, s7x32
            asc_intlv(scale_factors[2], scale_factors[3], scale_factors[2], scale_factors[2]);

            // Finite FP8 values are exactly representable in BF16; their FP32 low halves are zero.
            #pragma unroll
            for (uint16_t chunk_idx = 0; chunk_idx < D_QK / NUM_BF16_IN_VEC; ++chunk_idx) {
                vector_fp8_e4m3fn_t data_fp8;   // 0 x 1 x 2 x 3 x ...
                asc_loadalign_unpack_postupdate(
                    data_fp8,
                    cur_gathered_kv_ptr,
                    128 + (chunk_idx+1 == D_QK / NUM_BF16_IN_VEC ? FP8_UB_TOKEN_BYTES - FP8Format::QUANT_BYTES : 0)
                );
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

            vector_bf16 tmp;
            asc_storealign_postupdate(cur_dst_ptr, tmp, -(D_QK/NUM_BF16_IN_VEC)*(B_TOPK_PER_V+1)*256/sizeof(bf16) + FRACTAL_W, none_mask);  // Use this to update `cur_dst_ptr`
        }
    }
}


template<bool POST_UPDATE>
__simd_callee__ void load_and_accumulate_rms(__ubuf__ bf16* &cur_q_frag_ptr, vector_float& row_accum0, vector_float& row_accum1, vector_bool full_mask) {
    vector_float input0, input1;
    if constexpr (POST_UPDATE) {
        asc_loadalign_unpack_postupdate((vector_bf16&)input0, cur_q_frag_ptr, NUM_FLOAT_IN_VEC);
        asc_loadalign_unpack_postupdate((vector_bf16&)input1, cur_q_frag_ptr, NUM_FLOAT_IN_VEC);
    } else {
        asc_loadalign_unpack((vector_bf16&)input0, cur_q_frag_ptr, 0);
        asc_loadalign_unpack((vector_bf16&)input1, cur_q_frag_ptr, NUM_FLOAT_IN_VEC);
    }
    asc_shiftleft_scalar((vector_uint32_t&)input0, (vector_uint32_t&)input0, 16, full_mask);
    asc_shiftleft_scalar((vector_uint32_t&)input1, (vector_uint32_t&)input1, 16, full_mask);
    row_accum0 = asc_fma(input0, input0, row_accum0, full_mask);
    row_accum1 = asc_fma(input1, input1, row_accum1, full_mask);
}

template<Config CONFIG>
__simd_vf__ void Kernel<CONFIG>::q_norm_vf(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t job_idx, uint32_t q_frag_idx, float sm_scale, float rms_norm_eps) {
    bool is_first_q_frag = q_frag_idx == NUM_Q_FRAGS-1;
    bool is_last_q_frag = q_frag_idx == 0;  // We loop in reversed order
    vector_bool full_mask = asc_create_mask_b16(PAT_ALL);
    vector_bool b32_vl8_mask = asc_create_mask_b32(PAT_VL8);

    static constexpr uint32_t NUM_ROWS = MMA_M_PER_V / 4;   // 4: NUM_FLOAT_IN_VEC / FRACTAL_W
    vector_float row_accum[NUM_ROWS];
    #pragma unroll
    for (uint32_t i = 0; i < NUM_ROWS; ++i)
        asc_loadalign(row_accum[i], ub_buf.q_norm_accum_wksp, i*NUM_FLOAT_IN_VEC);

    __ubuf__ bfloat16_t* cur_q_frag_ptr = ub_buf.q_frag[q_frag_idx % NUM_UB_Q_FRAG_BUFS];
    #pragma unroll 1
    for (uint32_t col = 0; col < Q_FRAG_D_QK / FRACTAL_W; ++col) {
        #pragma unroll
        for (uint32_t row = 0; row < NUM_ROWS; row += 2) {
            load_and_accumulate_rms<true>(cur_q_frag_ptr, row_accum[row+0], row_accum[row+1], full_mask);
        }
    }

    if (is_last_q_frag) {
        vector_float full_rms_norm_eps_vec, full_1_div_d_qk_vec, full_sm_scale_vec;
        asc_duplicate_scalar(full_rms_norm_eps_vec, rms_norm_eps);
        asc_duplicate_scalar(full_1_div_d_qk_vec, 1.0f/D_QK);
        asc_duplicate_scalar(full_sm_scale_vec, sm_scale);

        #pragma unroll
        for (uint32_t i = 0; i < NUM_ROWS; i += 2) {
            vector_float cur_row_accum;
            asc_deintlv(row_accum[i], row_accum[i+1], row_accum[i], row_accum[i+1]);
            asc_add(cur_row_accum, row_accum[i], row_accum[i+1], full_mask);
            asc_reduce_sum_datablock(cur_row_accum, cur_row_accum, full_mask);

            asc_madd(cur_row_accum, full_1_div_d_qk_vec, full_rms_norm_eps_vec, full_mask);
            asc_sqrt(cur_row_accum, cur_row_accum, full_mask);
            asc_div(cur_row_accum, full_sm_scale_vec, cur_row_accum, full_mask);

            asc_storealign(ub_buf.sm_scale_div_rms[job_idx % NUM_SM_SCALE_DIV_RMS_BUFS], cur_row_accum, i*4, b32_vl8_mask);
        }
    } else {
        #pragma unroll
        for (uint32_t i = 0; i < NUM_ROWS; ++i)
            asc_storealign(ub_buf.q_norm_accum_wksp, row_accum[i], i*NUM_FLOAT_IN_VEC, full_mask);
    }
}


// Perform RoPE on the last Q frag
template<Config CONFIG>
__simd_vf__ void Kernel<CONFIG>::q_rope_vf(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t rope_cos_sin_buf_idx) {
    static_assert(D_ROPE == 64);
    auto full_mask = asc_create_mask_b16(PAT_ALL);
    __ubuf__ float* cur_rope_cos_buf_ptr = ub_buf.rope_cos_sin_buf[rope_cos_sin_buf_idx];
    __ubuf__ float* cur_rope_sin_buf_ptr = ub_buf.rope_cos_sin_buf[rope_cos_sin_buf_idx] + (D_ROPE / 2);
    __ubuf__ bf16* cur_q_frag_ptr = ub_buf.q_frag[(NUM_Q_FRAGS-1) % NUM_UB_Q_FRAG_BUFS];

    static constexpr uint32_t NUM_ROWS = MMA_M_PER_V / 4;   // 4: NUM_FLOAT_IN_VEC / FRACTAL_W
    [[maybe_unused]] vector_float row_accum[NUM_ROWS];
    if constexpr (HAVE_Q_NORM) {
        #pragma unroll
        for (uint32_t i = 0; i < NUM_ROWS; ++i)
            asc_duplicate_scalar(row_accum[i], 0, full_mask);
    }

    if constexpr (HAVE_Q_NORM) {
        #pragma unroll 1
        for (uint32_t i = 0; i < (Q_FRAG_D_QK - D_ROPE) / FRACTAL_W; ++i) {
            #pragma unroll
            for (uint32_t j = 0; j < MMA_M_PER_V / (NUM_BF16_IN_VEC / FRACTAL_W); ++j) {
                load_and_accumulate_rms<true>(cur_q_frag_ptr, row_accum[2*j+0], row_accum[2*j+1], full_mask);
            }
        }
    }

    cur_q_frag_ptr = ub_buf.q_frag[(NUM_Q_FRAGS-1) % NUM_UB_Q_FRAG_BUFS] + (Q_FRAG_D_QK - D_ROPE) * MMA_M_PER_V;
    #pragma unroll 1
    for (uint32_t i = 0; i < D_ROPE / FRACTAL_W; ++i) {
        vector_float rope_cos, rope_sin, neg_sin, tmp;
        asc_loadalign_brc_datablock_postupdate(rope_cos, cur_rope_cos_buf_ptr, 8);
        asc_loadalign_brc_datablock_postupdate(rope_sin, cur_rope_sin_buf_ptr, 8);
        asc_neg(neg_sin, rope_sin, full_mask);

        static_assert(MMA_M_PER_V % (NUM_BF16_IN_VEC / FRACTAL_W) == 0);
        #pragma unroll
        for (uint32_t j = 0; j < MMA_M_PER_V / (NUM_BF16_IN_VEC / FRACTAL_W); ++j) {
            vector_bf16 data;
            asc_loadalign(data, cur_q_frag_ptr);
            if constexpr (HAVE_Q_NORM) {
                load_and_accumulate_rms<false>(cur_q_frag_ptr, row_accum[2*j+0], row_accum[2*j+1], full_mask);
            }
            vector_float data_even, data_odd;
            asc_bfloat162float(data_even, data, full_mask, ASC_POSITION_EVEN);
            asc_bfloat162float(data_odd, data, full_mask, ASC_POSITION_ODD);
            vector_float data_even_mul_sin, data_odd_mul_neg_sin;
            asc_mul(data_even_mul_sin, data_even, rope_sin, full_mask);
            asc_mul(data_odd_mul_neg_sin, data_odd, neg_sin, full_mask);
            asc_madd(data_even, rope_cos, data_odd_mul_neg_sin, full_mask);
            asc_madd(data_odd, rope_cos, data_even_mul_sin, full_mask);
            vector_bf16 data_even_bf16, data_odd_bf16;
            asc_float2bfloat16_rn(data_even_bf16, data_even, full_mask);
            asc_float2bfloat16_rn_v2(data_odd_bf16, data_odd, full_mask);
            asc_or((vector_uint16_t&)data, (vector_uint16_t&)data_even_bf16, (vector_uint16_t&)data_odd_bf16, full_mask);
            asc_storealign_postupdate(cur_q_frag_ptr, data, NUM_BF16_IN_VEC, full_mask);
        }
    }

    if constexpr (HAVE_Q_NORM) {
        #pragma unroll
        for (uint32_t i = 0; i < NUM_ROWS; ++i)
            asc_storealign(ub_buf.q_norm_accum_wksp, row_accum[i], i*NUM_FLOAT_IN_VEC, full_mask);
    }
}


template<Config CONFIG>
template<bool IS_BLOCK0>
__simd_vf__ void Kernel<CONFIG>::softmax_vf(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, float sm_scale, uint32_t job_idx, uint32_t kv_block_idx, uint32_t index_buf_idx, uint32_t ub_indices_arr_offset, uint32_t s_kv, uint32_t subblock_idx) {
    // p and s are laid out in NZ
    auto full_mask = asc_create_mask_b16(PAT_ALL);
    auto vl8_mask = asc_create_mask_b32(PAT_VL8);


    /*
    Every `asc_loadalign_deintlv_postupdate` below reads 8 rows x 16 columns of P (2 UB vectors) at once and
    de-interleaves them into 8 rows x 8 even columns + 8 rows x 8 odd columns. The register layout is
    therefore [row][pair], where a `pair` is a pair of adjacent KV tokens (`2 * pair` and `2 * pair + 1`).

    The column-valid masks are derived from `smaller_indices` / `larger_indices`: `asc_loadalign_brc_datablock`
    repeats the 8 (pair) elements of a 16-column group 8 times, which matches the [row][pair] layout exactly.
    They only depend on the column group, so both loops that consume P iterate the column groups in the outer
    loop and compute the two masks once per group; each pair dies at the end of its group instead of staying
    alive per row group (2 masks * 4 groups would overflow the 8 predicate registers).
    */
    static_assert(MMA_M_PER_V % 8 == 0);
    static_assert(B_TOPK % FRACTAL_W == 0);
    static constexpr uint32_t NUM_ROW_GROUPS = MMA_M_PER_V / 8;
    static constexpr uint32_t NUM_COL_GROUPS = B_TOPK / FRACTAL_W;

    __ubuf__ uint32_t* cur_smaller_indices_ptr = ub_buf.smaller_indices[index_buf_idx] + ub_indices_arr_offset;
    __ubuf__ uint32_t* cur_larger_indices_ptr = ub_buf.larger_indices[index_buf_idx] + ub_indices_arr_offset;

    __ubuf__ float* cur_row_max_ptr = ub_buf.row_max[job_idx&1];

    // Merge the row max of this KV block with the row max of the previous KV blocks
    vector_float cur_rows_max[NUM_ROW_GROUPS];
    #pragma unroll
    for (uint32_t rg = 0; rg < NUM_ROW_GROUPS; ++rg) {
        if constexpr (IS_BLOCK0) {
            asc_duplicate_scalar(cur_rows_max[rg], std::numeric_limits<float>::lowest());
        } else {
            asc_loadalign_brc_elem2datablock(cur_rows_max[rg], ub_buf.row_max[job_idx&1], rg*8);
        }
    }

    // Iterate the column groups in the outer loop so that the two column-valid masks of a group are computed only once per group.
    // Every chunk of P is visited exactly once and in ascending address order (chunk (col_group, rg) resides at 128 * (4 * col_group + rg))
    #pragma unroll
    for (uint32_t col_group = 0; col_group < NUM_COL_GROUPS; ++col_group) {
        __ubuf__ float* cur_p_ptr = ub_buf.p + 2*NUM_FLOAT_IN_VEC*NUM_ROW_GROUPS*col_group;
        vector_bool col_valid_even, col_valid_odd;
        {
            vector_uint32_t smaller_indices, larger_indices;
            asc_loadalign_brc_datablock(smaller_indices, cur_smaller_indices_ptr, col_group*8);
            asc_loadalign_brc_datablock(larger_indices, cur_larger_indices_ptr, col_group*8);
            // For each pair of indices, there are three cases:
            //  1) Both indices are valid. Token with smaller index will be placed at position 0 and token with larger index will be placed at position 1
            //  2) One index is valid and the other one is invalid. The valid token will be placed at position 0
            //  3) Both indices are invalid
            // In any case above, we can use `smaller_indices` and `larger_indices` to distinguish which token is valid and which one is invalid
            asc_lt_scalar(col_valid_even, smaller_indices, s_kv, full_mask);
            asc_lt_scalar(col_valid_odd, larger_indices, s_kv, full_mask);
        }
        #pragma unroll
        for (uint32_t rg = 0; rg < NUM_ROW_GROUPS; ++rg) {
            vector_float p_even, p_odd;
            asc_loadalign_deintlv(p_even, p_odd, cur_p_ptr, 2*NUM_FLOAT_IN_VEC*rg);
            // vmax with MODE_MERGING retains the original value of `dst` if one position is being masked-off
            vmax(cur_rows_max[rg], cur_rows_max[rg], p_even, col_valid_even, MODE_MERGING);
            vmax(cur_rows_max[rg], cur_rows_max[rg], p_odd, col_valid_odd, MODE_MERGING);
        }
    }

    #pragma unroll
    for (uint32_t rg = 0; rg < NUM_ROW_GROUPS; ++rg) {
        asc_reduce_max_datablock(cur_rows_max[rg], cur_rows_max[rg], full_mask);
        asc_storealign(cur_row_max_ptr, cur_rows_max[rg], rg*8, vl8_mask);
    }

    asc_mem_bar(VST_VLD);

    if constexpr (not IS_BLOCK0) {
        [[maybe_unused]] vector_float sm_scale_div_rms;
        if constexpr (HAVE_Q_NORM) {
            asc_loadalign(sm_scale_div_rms, ub_buf.sm_scale_div_rms[job_idx % NUM_SM_SCALE_DIV_RMS_BUFS]);
        }
        vector_float new_rows_max, old_row_max_for_o;
        asc_loadalign(new_rows_max, ub_buf.row_max[job_idx&1]);
        asc_loadalign(old_row_max_for_o, ub_buf.row_max_for_o[job_idx&1]);
        vector_float row_max_max_delta;
        asc_sub(row_max_max_delta, new_rows_max, old_row_max_for_o, full_mask);

        if constexpr (HAVE_Q_NORM) {
            asc_mul(row_max_max_delta, row_max_max_delta, sm_scale_div_rms, full_mask);
        } else {
            asc_mul_scalar(row_max_max_delta, row_max_max_delta, sm_scale, full_mask);
        }
        // Create a mask whose first "num valid heads in this subblock" lanes are set to 1
        vector_bool valid_rows_mask_for_row_max;
        if constexpr (H_Q_PER_AI_CORE == 64) {
            // MMA_M is 64. Each AIV is responsible for 32 heads, occupying the first 32 lanes in SIMD reg
            valid_rows_mask_for_row_max = asc_create_mask_b32(PAT_VL32);
        } else if constexpr (H_Q_PER_AI_CORE == 32) {
            // MMA_M is 32. Each AIV is responsible for 16 heads, occupying the first 16 lanes in SIMD reg
            valid_rows_mask_for_row_max = asc_create_mask_b32(PAT_VL16);
        } else {
            uint32_t my_subblock_start_head_idx = subblock_idx * MMA_M_PER_V;
            uint32_t num_valid_heads = H_Q_PER_AI_CORE >= my_subblock_start_head_idx ? H_Q_PER_AI_CORE - my_subblock_start_head_idx : 0u;
            num_valid_heads = min(num_valid_heads, MMA_M_PER_V);
            vector_int32_t arange_vec;
            asc_arange(arange_vec, 0);
            asc_lt_scalar(valid_rows_mask_for_row_max, arange_vec, num_valid_heads, full_mask);
        }
        asc_reduce_max(row_max_max_delta, row_max_max_delta, valid_rows_mask_for_row_max);
        asc_duplicate(row_max_max_delta, row_max_max_delta, full_mask);    // The maximum of {new_rows_max[i] - old_row_max_for_o[i] | i = 0...MMA_M_PER_V-1} is broadcasted to every number in `row_max_max_delta`

        vector_bool should_rescale;
        asc_gt_scalar(should_rescale, row_max_max_delta, RESCALE_THRESHOLD, full_mask);
        vector_float new_row_max_for_o;
        asc_select(new_row_max_for_o, new_rows_max, old_row_max_for_o, should_rescale);
        asc_storealign(ub_buf.row_max_for_o[job_idx&1], new_row_max_for_o, full_mask);

        vector_float row_rescale_factor;
        vector_float old_new_row_max_for_o_diff;    // old_row_max_for_o - new_row_max_for_o
        asc_sub(old_new_row_max_for_o_diff, old_row_max_for_o, new_row_max_for_o, full_mask);
        if constexpr (HAVE_Q_NORM) {
            asc_mul(row_rescale_factor, old_new_row_max_for_o_diff, sm_scale_div_rms, full_mask);
        } else {
            asc_mul_scalar(row_rescale_factor, old_new_row_max_for_o_diff, sm_scale, full_mask);
        }
        asc_exp(row_rescale_factor, row_rescale_factor, full_mask);
        asc_storealign(ub_buf.row_rescale_factor[job_idx&1][kv_block_idx&1], row_rescale_factor, full_mask);

        asc_mem_bar(VST_VLD);

        asc_storealign_1st(&(ub_buf.row_max_max_delta[job_idx&1][kv_block_idx&1]), row_max_max_delta);
    } else {
        vector_float row_max;
        asc_loadalign(row_max, ub_buf.row_max[job_idx&1]);
        asc_storealign(ub_buf.row_max_for_o[job_idx&1], row_max, full_mask);
        asc_mem_bar(VST_VLD);
    }

    __ubuf__ float* cur_row_sum_ptr = ub_buf.row_sum[job_idx&1];
    __ubuf__ float* cur_row_max_for_o_ptr = ub_buf.row_max_for_o[job_idx&1];
    __ubuf__ float* cur_row_sum_rescale_ptr = ub_buf.row_rescale_factor[job_idx&1][kv_block_idx&1];
    [[maybe_unused]] __ubuf__ float* cur_sm_scale_div_rms_ptr = ub_buf.sm_scale_div_rms[job_idx % NUM_SM_SCALE_DIV_RMS_BUFS];
    static constexpr uint32_t S_GROUP_STRIDE = 2*MMA_M_PER_V*FRACTAL_W;
    __ubuf__ bf16* cur_s_base_ptr = (__ubuf__ bf16*)ub_buf.p;
    // The per-row values (row max for O, old row sum) are the same for all column groups, and the column-valid
    // masks only depend on the column group. So instead of `rg` outer / `col_group` inner we iterate
    // `col_group` outer / `rg` inner: each pair of column-valid masks is then computed once per column group
    // instead of once per (row group, column group). The per-row values therefore have to stay live in
    // registers across the column groups, and the inner `rg` loop must be fully unrolled because the arrays
    // can only be indexed by a compile-time constant.
    vector_float cur_rows_max_for_o[NUM_ROW_GROUPS];
    vector_float cur_rows_sum[NUM_ROW_GROUPS];
    [[maybe_unused]] vector_float cur_sm_scale_div_rms[NUM_ROW_GROUPS];
    #pragma unroll
    for (uint32_t rg = 0; rg < NUM_ROW_GROUPS; ++rg) {
        uint32_t start_row = rg * 8;

        // All per-row values hold one float per head: element-to-datablock gives the 8 rows x 8 lanes layout of P directly
        asc_loadalign_brc_elem2datablock(cur_rows_max_for_o[rg], cur_row_max_for_o_ptr, start_row);
        if constexpr (HAVE_Q_NORM) {
            asc_loadalign_brc_elem2datablock(cur_sm_scale_div_rms[rg], cur_sm_scale_div_rms_ptr, start_row);
            asc_mul(cur_rows_max_for_o[rg], cur_rows_max_for_o[rg], cur_sm_scale_div_rms[rg], full_mask);
        } else {
            asc_mul_scalar(cur_rows_max_for_o[rg], cur_rows_max_for_o[rg], sm_scale, full_mask);
        }

        if constexpr (IS_BLOCK0) {
            asc_duplicate_scalar(cur_rows_sum[rg], 0);
        } else {
            asc_loadalign(cur_rows_sum[rg], cur_row_sum_ptr, rg*NUM_FLOAT_IN_VEC);
            vector_float row_sum_rescale;
            asc_loadalign_brc_elem2datablock(row_sum_rescale, cur_row_sum_rescale_ptr, start_row);
            asc_mul(cur_rows_sum[rg], cur_rows_sum[rg], row_sum_rescale, full_mask);
        }
    }

    #pragma unroll 1    // Don't unroll to reduce code size, which prevents VF's ICache miss
    for (uint32_t col_group = 0; col_group < NUM_COL_GROUPS; ++col_group) {
        vector_bool col_valid_even, col_valid_odd;
        {
            vector_uint32_t smaller_indices, larger_indices;
            asc_loadalign_brc_datablock(smaller_indices, cur_smaller_indices_ptr, col_group*8);
            asc_loadalign_brc_datablock(larger_indices, cur_larger_indices_ptr, col_group*8);
            asc_lt_scalar(col_valid_even, smaller_indices, s_kv, full_mask);
            asc_lt_scalar(col_valid_odd, larger_indices, s_kv, full_mask);
        }

        __ubuf__ float* cur_p_col_ptr = ub_buf.p + col_group*(MMA_M_PER_V*FRACTAL_W);
        __ubuf__ bf16* cur_s_col_ptr = cur_s_base_ptr + col_group*S_GROUP_STRIDE;
        #pragma unroll
        for (uint32_t rg = 0; rg < NUM_ROW_GROUPS; ++rg) {
            uint32_t start_row = rg * 8;

            __ubuf__ float* cur_p_chunk_ptr = cur_p_col_ptr + start_row*FRACTAL_W;
            __ubuf__ float* cur_p_load_ptr = cur_p_chunk_ptr;
            vector_float p_even, p_odd;
            asc_loadalign_deintlv_postupdate(p_even, p_odd, cur_p_load_ptr, 2*NUM_FLOAT_IN_VEC);
            if constexpr (HAVE_Q_NORM) {
                asc_mul(p_even, p_even, cur_sm_scale_div_rms[rg], full_mask);
                asc_mul(p_odd, p_odd, cur_sm_scale_div_rms[rg], full_mask);
            } else {
                asc_mul_scalar(p_even, p_even, sm_scale, full_mask);
                asc_mul_scalar(p_odd, p_odd, sm_scale, full_mask);
            }


            vector_float p_exp_even, p_exp_odd;
            asc_exp_sub(p_exp_even, p_even, cur_rows_max_for_o[rg], col_valid_even);
            asc_exp_sub(p_exp_odd, p_odd, cur_rows_max_for_o[rg], col_valid_odd);
            // Add the two tokens of every pair first, so swapping the two tokens of a pair does not change `row_sum`
            vector_float pair_sum;
            asc_add(pair_sum, p_exp_even, p_exp_odd, full_mask);
            asc_add(cur_rows_sum[rg], cur_rows_sum[rg], pair_sum, full_mask);

            vector_bf16 s_even, s_odd, s_bf16;
            asc_float2bfloat16_rn(s_even, p_exp_even, full_mask);
            asc_float2bfloat16_rn_v2(s_odd, p_exp_odd, full_mask);
            asc_or((vector_uint16_t&)s_bf16, (vector_uint16_t&)s_even, (vector_uint16_t&)s_odd, full_mask);
            asc_storealign(cur_s_col_ptr + start_row*FRACTAL_W, s_bf16, full_mask);
        }
    }

    #pragma unroll
    for (uint32_t rg = 0; rg < NUM_ROW_GROUPS; ++rg) {
        asc_storealign(cur_row_sum_ptr, cur_rows_sum[rg], rg*NUM_FLOAT_IN_VEC, full_mask);
    }
}


/*
output_accum := (cur_output_frag [+ output_accum]) * row_rescale_factor
*/
template<Config CONFIG>
template<bool PERFORM_ADD>
__simd_vf__ void Kernel<CONFIG>::add_and_scale_o_vf(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t output_frag_idx, uint32_t cur_o_frag_buf_idx, float sm_scale, uint32_t job_idx, uint32_t kv_block_idx) {
    auto full_mask = asc_create_mask_b32(PAT_ALL);
    // `output_accum` holds one [MMA_M_PER_V][OUTPUT_FRAG_D_VO] block per fragment, so this fragment's rows
    // are contiguous (row stride = OUTPUT_FRAG_D_VO floats).
    __ubuf__ float* cur_output_accum_ptr = ub_buf.output_accum[output_frag_idx];
    __ubuf__ float* cur_cur_output_frag_ptr = ub_buf.cur_output_frag[cur_o_frag_buf_idx][0];
    
    #pragma unroll 1
    for (uint16_t row = 0; row < MMA_M_PER_V; row += 1) {
        vector_float cur_scale;
        asc_loadalign_brc(cur_scale, ub_buf.row_rescale_factor[job_idx&1][kv_block_idx&1], row);

        static constexpr uint32_t NUM_FLOAT_IN_VEC = 256 / sizeof(float);
        #pragma unroll
        for (uint16_t col = 0; col < OUTPUT_FRAG_D_VO / NUM_FLOAT_IN_VEC; ++col) {
            uint32_t cur_offset = row * (uint32_t)OUTPUT_FRAG_D_VO + col * NUM_FLOAT_IN_VEC;
            vector_float new_output;
            asc_loadalign(new_output, cur_cur_output_frag_ptr, cur_offset);
            if constexpr (PERFORM_ADD) {
                vector_float old_output;
                asc_loadalign(old_output, cur_output_accum_ptr, cur_offset);
                asc_add(new_output, old_output, new_output, full_mask);
            }
            asc_mul(new_output, new_output, cur_scale, full_mask);
            asc_storealign(cur_output_accum_ptr, new_output, cur_offset, full_mask);
        }
    }
}


template<Config CONFIG>
__simd_vf__ void Kernel<CONFIG>::get_final_lse_and_denorm_vf(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t job_idx, float sm_scale) {
    auto full_mask = asc_create_mask_b32(PAT_ALL);
    auto b32_vl8_mask = asc_create_mask_b32(PAT_VL8);
    // Only the first MMA_M_PER_V lanes hold one value per head
    vector_bool row_lane_mask;
    if constexpr (MMA_M_PER_V == 32) {
        row_lane_mask = asc_create_mask_b32(PAT_VL32);
    } else {
        static_assert(MMA_M_PER_V == 16);
        row_lane_mask = asc_create_mask_b32(PAT_VL16);
    }

    __ubuf__ float* cur_row_sum_ptr = ub_buf.row_sum[job_idx&1];
    __ubuf__ float* cur_row_sum_write_back_ptr = ub_buf.row_sum[job_idx&1];
    static_assert(MMA_M_PER_V % 8 == 0);
    #pragma unroll 1
    for (uint32_t i = 0; i < MMA_M_PER_V/8; i += 1) {
        vector_float row_sum;
        asc_loadalign(row_sum, cur_row_sum_ptr, i*NUM_FLOAT_IN_VEC);
        asc_reduce_sum_datablock(row_sum, row_sum, full_mask);
        asc_storealign_postupdate(cur_row_sum_write_back_ptr, row_sum, 8, b32_vl8_mask);
    }
    
    vector_float row_max, row_max_for_o;
    asc_loadalign(row_max, ub_buf.row_max[job_idx&1]);
    asc_loadalign(row_max_for_o, ub_buf.row_max_for_o[job_idx&1]);

    vector_bool is_row_all_masked;
    asc_eq_scalar(is_row_all_masked, row_max, std::numeric_limits<float>::lowest(), row_lane_mask);

    if constexpr (HAVE_Q_NORM) {
        vector_float sm_scale_div_rms;
        asc_loadalign(sm_scale_div_rms, ub_buf.sm_scale_div_rms[job_idx % NUM_SM_SCALE_DIV_RMS_BUFS]);
        asc_mul(row_max, row_max, sm_scale_div_rms, full_mask);
        asc_mul(row_max_for_o, row_max_for_o, sm_scale_div_rms, full_mask);
    } else {
        asc_mul_scalar(row_max, row_max, sm_scale, full_mask);
        asc_mul_scalar(row_max_for_o, row_max_for_o, sm_scale, full_mask);
    }
    
    vector_float row_sum;
    asc_mem_bar(VST_VLD);   // Defer the loading of row_sum until now to workaround the overhead of the mem bar
    asc_loadalign(row_sum, ub_buf.row_sum[job_idx&1]);

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

    if constexpr (HAVE_Q_O_ROPE_AND_O_CAST) {
        // Repeat each number in denorm for 4 times, for `get_final_output_vf_with_cast_and_rope_stage2()`
        vector_float denorm0, denorm1, denorm2, denorm3;
        asc_intlv(denorm0, denorm2, denorm, denorm);
        asc_intlv(denorm0, denorm1, denorm0, denorm0);
        asc_intlv(denorm2, denorm3, denorm2, denorm2);
        __ubuf__ float* cur_row_denorm_ptr = ub_buf.row_denorm[job_idx % NUM_ROW_DENORM_BUFS];
        asc_storealign_postupdate(cur_row_denorm_ptr, denorm0, NUM_FLOAT_IN_VEC, full_mask);
        asc_storealign_postupdate(cur_row_denorm_ptr, denorm1, NUM_FLOAT_IN_VEC, full_mask);
        asc_storealign_postupdate(cur_row_denorm_ptr, denorm2, NUM_FLOAT_IN_VEC, full_mask);
        asc_storealign_postupdate(cur_row_denorm_ptr, denorm3, NUM_FLOAT_IN_VEC, full_mask);
    } else {
        asc_storealign(ub_buf.row_denorm[job_idx % NUM_ROW_DENORM_BUFS], denorm, full_mask);
    }
}


/*
output = float2bfloat16((cur_output_frag [+ output_accum]) * denorm)
or, output = per_token_cast(rope((cur_output_frag [+ output_accum]) * denorm))
*/
template<Config CONFIG>
template<bool HAVE_PREVIOUS_OUTPUT_ACCUM>
__simd_vf__ void Kernel<CONFIG>::get_final_output_vf(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t output_frag_idx, uint32_t cur_o_frag_buf_idx, uint32_t job_idx) {
    static_assert(OUTPUT_FRAG_D_VO / NUM_FLOAT_IN_VEC == 2);

    __ubuf__ float* cur_output_accum_ptr = ub_buf.output_accum[output_frag_idx];
    __ubuf__ float* cur_cur_output_frag_ptr = ub_buf.cur_output_frag[cur_o_frag_buf_idx][0];
    __ubuf__ bf16* cur_final_output_ptr = (__ubuf__ bf16*)(ub_buf.output_accum[output_frag_idx]);
    bool const is_last_frag = output_frag_idx + 1 == D_VO / OUTPUT_FRAG_D_VO;

    auto full_mask = asc_create_mask_b8(PAT_ALL);
    #pragma unroll 1
    for (uint16_t row = 0; row < MMA_M_PER_V / 2; row += 1) {
        vector_float cur_denorm[2];
        #pragma unroll
        for (uint32_t i = 0; i < 2; ++i) {
            asc_loadalign_brc(cur_denorm[i], ub_buf.row_denorm[job_idx % NUM_ROW_DENORM_BUFS], row*2+i);
        }

        vector_bf16 output_bf16[2][2];
        #pragma unroll
        for (uint32_t i = 0; i < 2; ++i) {
            #pragma unroll
            for (uint32_t j = 0; j < 2; ++j) {
                // Direct addressing is faster than postupdate
                uint32_t cur_read_offset = row * (2*OUTPUT_FRAG_D_VO) + (i*2 + j) * NUM_FLOAT_IN_VEC;
                vector_float output_f32;
                asc_loadalign(output_f32, cur_output_accum_ptr, cur_read_offset);
                if constexpr (HAVE_PREVIOUS_OUTPUT_ACCUM) {
                    vector_float old_output;
                    asc_loadalign(old_output, cur_cur_output_frag_ptr, cur_read_offset);
                    asc_add(output_f32, output_f32, old_output, full_mask);
                }
                asc_mul(output_f32, output_f32, cur_denorm[i], full_mask);
                if (i == 0) {
                    asc_float2bfloat16_rn(output_bf16[i][j], output_f32, full_mask);
                } else {
                    asc_float2bfloat16_rn_v2(output_bf16[i][j], output_f32, full_mask);
                }
            }
        }

        asc_or((vector_uint16_t&)output_bf16[0][0], (vector_uint16_t&)output_bf16[0][0], (vector_uint16_t&)output_bf16[1][0], full_mask);   // "asc_or" is one cycle less than "asc_add"
        asc_or((vector_uint16_t&)output_bf16[0][1], (vector_uint16_t&)output_bf16[0][1], (vector_uint16_t&)output_bf16[1][1], full_mask);
        asc_deintlv(output_bf16[0][0], output_bf16[0][1], output_bf16[0][0], output_bf16[0][1]);
        uint32_t cur_write_offset = row * (2*OUTPUT_FRAG_D_VO);
        asc_storealign(cur_final_output_ptr, output_bf16[0][0], cur_write_offset, full_mask);
        asc_storealign(cur_final_output_ptr, output_bf16[0][1], cur_write_offset + NUM_BF16_IN_VEC, full_mask);
    }
}

/*
Subroutine for the fused mode (HAVE_Q_O_ROPE_AND_O_CAST = true)'s partial O copy out stage 0
Read one O fragment, get the maximum of every 8 numbers, and save that to `ub.o_block_max`

If `HAVE_PREVIOUS_OUTPUT_ACCUM` is `true`, the previously accumulated O resides in `ub_buf.output_accum`, so
this VF adds the two together and saves the sum to `ub_buf.output_accum`.

If `HAVE_PREVIOUS_OUTPUT_ACCUM` is `false`, the newly arrived O *is* the accumulator.
*/
template<Config CONFIG>
template<bool HAVE_PREVIOUS_OUTPUT_ACCUM>
__simd_vf__ void Kernel<CONFIG>::get_final_output_vf_with_cast_and_rope_stage0(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t output_frag_idx, uint32_t cur_o_frag_buf_idx, uint32_t job_idx) {
    static_assert(OUTPUT_FRAG_D_VO == 128 && SCALE_GRAN == 32);

    auto full_mask = asc_create_mask_b8(PAT_ALL);
    auto b32_vl8_mask = asc_create_mask_b32(PAT_VL8);

    __ubuf__ float* cur_cur_output_frag_ptr = ub_buf.cur_output_frag[cur_o_frag_buf_idx][0];
    __ubuf__ float* cur_output_accum_ptr = ub_buf.output_accum[output_frag_idx];
    __ubuf__ float* cur_block_max_ptr = ub_buf.o_block_max + output_frag_idx * (MMA_M_PER_V * OUTPUT_FRAG_D_VO / SCALE_GRAN * 4);
    #pragma unroll 1
    for (uint32_t row = 0; row < MMA_M_PER_V; row += 1) {
        #pragma unroll
        for (uint32_t j = 0; j < OUTPUT_FRAG_D_VO / NUM_FLOAT_IN_VEC; ++j) {
            vector_float input;
            if constexpr (HAVE_PREVIOUS_OUTPUT_ACCUM) {
                vector_float t;
                asc_loadalign(input, cur_output_accum_ptr);
                asc_loadalign_postupdate(t, cur_cur_output_frag_ptr, NUM_FLOAT_IN_VEC);
                asc_add(input, input, t, full_mask);
                asc_storealign_postupdate(cur_output_accum_ptr, input, NUM_FLOAT_IN_VEC, full_mask);
            } else {
                // no previous accumulator: the CUBE Core has already written the O into `output_accum`
                asc_loadalign_postupdate(input, cur_output_accum_ptr, NUM_FLOAT_IN_VEC);
            }
            asc_abs(input, input, full_mask);
            asc_reduce_max_datablock((vector_uint32_t&)input, (vector_uint32_t&)input, full_mask);
            asc_storealign_postupdate(cur_block_max_ptr, input, 8, b32_vl8_mask);
        }
    }
}

/*
Subroutine for the fused mode (HAVE_Q_O_ROPE_AND_O_CAST = true)'s partial O copy out stage 1
Similar to stage0, but performs RoPE, and reduce among every 2 numbers in `ub.o_block_max`
*/
template<Config CONFIG>
template<bool HAVE_PREVIOUS_OUTPUT_ACCUM>
__simd_vf__ void Kernel<CONFIG>::get_final_output_vf_with_cast_and_rope_stage1(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t job_idx, uint32_t rope_cos_sin_buf_idx) {
    auto full_mask = asc_create_mask_b8(PAT_ALL);
    auto b32_vl8_mask = asc_create_mask_b32(PAT_VL8);   // 8 x b32 = 32 B = the 8 live maxima lanes

    vector_float rope_a, rope_b;    // [cos0, sin0, cos1, sin1, ...], [-sin0, cos0, -sin1, cos1, ...]
    {
        vector_float rope_cos, rope_sin, neg_sin, tmp;
        asc_loadalign(rope_cos, ub_buf.rope_cos_sin_buf[rope_cos_sin_buf_idx]);
        asc_loadalign(rope_sin, ub_buf.rope_cos_sin_buf[rope_cos_sin_buf_idx], D_ROPE/2);
        asc_neg(neg_sin, rope_sin, full_mask);
        asc_intlv(rope_a, tmp, rope_cos, rope_sin);
        asc_intlv(rope_b, tmp, neg_sin, rope_cos);
    }

    // Get the maximum of each 8 numbers for the last output fragment
    __ubuf__ float* cur_cur_output_frag_ptr = ub_buf.cur_output_frag[(NUM_OUTPUT_FRAGS-1)%NUM_OUTPUT_FRAG_BUFS][0];
    __ubuf__ float* cur_output_accum_ptr = ub_buf.output_accum[NUM_OUTPUT_FRAGS-1];
    __ubuf__ float* cur_block_max_ptr = ub_buf.o_block_max + (NUM_OUTPUT_FRAGS-1) * (MMA_M_PER_V * OUTPUT_FRAG_D_VO / SCALE_GRAN * 4);
    #pragma unroll 1
    for (uint32_t row = 0; row < MMA_M_PER_V; row += 1) {
        #pragma unroll
        for (uint32_t j = 0; j < OUTPUT_FRAG_D_VO / NUM_FLOAT_IN_VEC; ++j) {
            vector_float input;
            if constexpr (HAVE_PREVIOUS_OUTPUT_ACCUM) {
                vector_float t;
                asc_loadalign(input, cur_output_accum_ptr);
                asc_loadalign_postupdate(t, cur_cur_output_frag_ptr, NUM_FLOAT_IN_VEC);
                asc_add(input, input, t, full_mask);
                asc_storealign_postupdate(cur_output_accum_ptr, input, NUM_FLOAT_IN_VEC, full_mask);
            } else {
                // no previous accumulator: the CUBE Core has already written the O into `output_accum`
                asc_loadalign_postupdate(input, cur_output_accum_ptr, NUM_FLOAT_IN_VEC);
            }

            // RoPE
            if (j == OUTPUT_FRAG_D_VO / NUM_FLOAT_IN_VEC - 1) {
                vector_float x0, x1;
                asc_mul(x0, input, rope_a, full_mask);
                asc_mul(x1, input, rope_b, full_mask);
                asc_pair_reduce_sum(x0, x0, full_mask);
                asc_pair_reduce_sum(x1, x1, full_mask);
                asc_intlv(input, x0, x0, x1);
                asc_storealign(cur_output_accum_ptr, input, -NUM_FLOAT_IN_VEC, full_mask);
            }
            
            asc_abs(input, input, full_mask);
            asc_reduce_max_datablock((vector_uint32_t&)input, (vector_uint32_t&)input, full_mask);
            asc_storealign_postupdate(cur_block_max_ptr, input, 8, b32_vl8_mask);
        }
    }
    asc_mem_bar(VST_VLD);

    // Reduce-max every 2 adjacent numbers in `ub_buf.o_block_max`
    // i.e. every 2 adjacent 8-element datablocks -> one 16-dim maximum per lane.
    cur_block_max_ptr = ub_buf.o_block_max;
    __ubuf__ float* cur_block_max_ptr2 = ub_buf.o_block_max;
    #pragma unroll 1
    for (uint32_t i = 0; i < MMA_M_PER_V * (D_VO / SCALE_GRAN) * 4 / NUM_FLOAT_IN_VEC / 2; i += 1) {
        vector_float block_max, block_max0, block_max1;
        asc_loadalign_deintlv_postupdate(block_max0, block_max1, cur_block_max_ptr, 2*NUM_FLOAT_IN_VEC);
        asc_max(block_max, block_max0, block_max1, full_mask);
        asc_storealign_postupdate(cur_block_max_ptr2, block_max, NUM_FLOAT_IN_VEC, full_mask);
    }
}

/*
Subroutine for the fused mode (HAVE_Q_O_ROPE_AND_O_CAST = true)'s partial O copy out stage 2
- Process those amax (absolute of maximum), get scale and scale inv; the multipliers are staged over the amax data
- cast-to-fp8
*/
template<Config CONFIG>
__simd_vf__ void Kernel<CONFIG>::get_final_output_vf_with_cast_and_rope_stage2(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t job_idx) {
    __ubuf__ float* cur_block_max_ptr = ub_buf.o_block_max;
    auto full_mask = asc_create_mask_b8(PAT_ALL);
    vector_uint32_t full_254_vector;
    asc_duplicate_scalar(full_254_vector, 254);

    // Get the scale and scale_inv for each quantization chunk (i.e. 32 elements)
    cur_block_max_ptr = ub_buf.o_block_max;
    __ubuf__ uint8_t* cur_o_scale_factors_ptr = ub_buf.o_scale_factors;
    __ubuf__ float* cur_denorm_ptr = ub_buf.row_denorm[job_idx % NUM_ROW_DENORM_BUFS];
    #pragma unroll 1
    for (uint32_t o_frag_idx = 0; o_frag_idx < NUM_OUTPUT_FRAGS; ++o_frag_idx) {
        static constexpr uint32_t NUM_ITERS_PER_O_FRAG = constexpr_max(MMA_M_PER_V * (OUTPUT_FRAG_D_VO / SCALE_GRAN) / NUM_FLOAT_IN_VEC, 1);
        #pragma unroll
        for (uint32_t i = 0; i < NUM_ITERS_PER_O_FRAG; i += 1) {
            bool is_last_i_in_o_frag = i+1 == NUM_ITERS_PER_O_FRAG;
            vector_float block_max, block_max0, block_max1;
            asc_loadalign_deintlv_postupdate(block_max0, block_max1, cur_block_max_ptr, 2*NUM_FLOAT_IN_VEC);
            asc_max(block_max, block_max0, block_max1, full_mask);

            vector_float cur_denorm;
            asc_loadalign_postupdate(
                cur_denorm,
                cur_denorm_ptr,
                is_last_i_in_o_frag ? -(NUM_ITERS_PER_O_FRAG-1)*NUM_FLOAT_IN_VEC : NUM_FLOAT_IN_VEC
            );  // Each number in `cur_denorm` is repeated for 4 times
            asc_mul(block_max, block_max, cur_denorm, full_mask);
            asc_max_scalar(block_max, block_max, 1e-4f, full_mask);
            asc_mul_scalar(block_max, block_max, 1.0f/448.0f, full_mask);

            vector_uint32_t exponent;
            asc_add_scalar((vector_int32_t&)exponent, (vector_int32_t&)block_max, -1, full_mask);
            asc_shiftright_scalar(exponent, exponent, 23, full_mask);
            asc_add_scalar((vector_int32_t&)exponent, (vector_int32_t&)exponent, 1, full_mask);
            asc_storealign_pack_quarter_postupdate(
                (__ubuf__ uint32_t *&)cur_o_scale_factors_ptr,
                exponent,
                NUM_FLOAT_IN_VEC / 4,   // /4 since the unit of the offset is `sizeof(uint32_t)`
                full_mask
            );

            vector_uint32_t exponent_inv;
            vector_float chunk_mult, tmp;
            asc_sub(exponent_inv, full_254_vector, exponent, full_mask);
            asc_shiftleft_scalar(exponent_inv, exponent_inv, 23, full_mask);
            asc_mul(chunk_mult, (vector_float&)exponent_inv, cur_denorm, full_mask);

            // Interleave so that each scale appears in `block_max` for two times
            // Since CANN does not provide `asc_storealign_intlv_postupdate`, we `postupdate` the `cur_block_max_ptr` by `asc_loadalign_deintlv_postupdate` and give a `-2 * NUM_FLOAT_IN_VEC` offset here
            asc_storealign_intlv((__ubuf__ uint32_t*&)cur_block_max_ptr, (vector_uint32_t&)chunk_mult,
                                 (vector_uint32_t&)chunk_mult, -2 * NUM_FLOAT_IN_VEC);
        }
    }
    asc_mem_bar(VST_VLD);

    __ubuf__ float* cur_output_accum_ptr = ub_buf.output_accum[0];
    cur_block_max_ptr = ub_buf.o_block_max;
    __ubuf__ uint32_t* fp8_ptr = (__ubuf__ uint32_t*)(ub_buf.output_accum[0]);
    #pragma unroll 1
    for (uint32_t row = 0; row < MMA_M_PER_V * (D_VO / OUTPUT_FRAG_D_VO); row += 2) {
        vector_float a0, a1, scale_inv;
        vector_fp8_e4m3fn_t a_fp8, a0_fp8, a1_fp8;

        // Row `row+0`: Use 2 x storealign_pack_quarter
        // Row `row+1`: Use `or` + `storealign_pack`
        // Interleave those two ways to balance load between ALU (`or`) and store unit
        asc_loadalign_deintlv_postupdate(a0, a1, cur_output_accum_ptr, 2*NUM_FLOAT_IN_VEC);
        asc_loadalign_brc_elem2datablock_postupdate(scale_inv, cur_block_max_ptr, 8);
        asc_mul(a0, a0, scale_inv, full_mask);
        asc_mul(a1, a1, scale_inv, full_mask);

        asc_float2e4m3_rn_sat(a0_fp8, a0, full_mask, ASC_DISPERSE_FIRST_QUARTER);
        asc_float2e4m3_rn_sat(a1_fp8, a1, full_mask, ASC_DISPERSE_SECOND_QUARTER);
        asc_or((vector_uint8_t&)a_fp8, (vector_uint8_t&)a0_fp8, (vector_uint8_t&)a1_fp8, full_mask);
        asc_storealign_pack_postupdate(fp8_ptr, (vector_uint32_t&)a_fp8, 32, full_mask);

        asc_loadalign_deintlv_postupdate(a0, a1, cur_output_accum_ptr, 2*NUM_FLOAT_IN_VEC);
        asc_loadalign_brc_elem2datablock_postupdate(scale_inv, cur_block_max_ptr, 8);
        asc_mul(a0, a0, scale_inv, full_mask);
        asc_mul(a1, a1, scale_inv, full_mask);

        asc_float2e4m3_rn_sat(a0_fp8, a0, full_mask, ASC_DISPERSE_FIRST_QUARTER);
        asc_float2e4m3_rn_sat(a1_fp8, a1, full_mask, ASC_DISPERSE_SECOND_QUARTER);
        asc_or((vector_uint8_t&)a_fp8, (vector_uint8_t&)a0_fp8, (vector_uint8_t&)a1_fp8, full_mask);
        asc_storealign_pack_postupdate(fp8_ptr, (vector_uint32_t&)a_fp8, 32, full_mask);
    }
}


template<Config CONFIG>
__aicore__ void Kernel<CONFIG>::sparse_attn_fwd_kernel_devfunc(const Params params, const AuxParams aux_params) {
    AscendC::InitSocState();

    // An AI core processes head [this_ai_core_start_head_idx, this_ai_core_start_head_idx + H_Q_PER_AI_CORE)
    // Bisheng doesn't optimize get_block_idx()%1, hence the lazy initialization of the two offsets.
    uint32_t head_split_idx = 0;
    uint32_t this_ai_core_start_head_idx = 0;
    if constexpr (NUM_HEAD_SPLITS > 1) {
        head_split_idx = (uint32_t)get_block_idx() % NUM_HEAD_SPLITS;
        this_ai_core_start_head_idx = head_split_idx * H_Q_PER_AI_CORE;
    }

    struct OuterLoopArgs {
        uint32_t job_idx;
        uint32_t batch_or_s_q_idx;  // For prefill, it's s_q_idx; for decoding, it's batch_idx (since params.s_q must be 1)
        uint32_t num_topk_blocks, topk_length;
        uint32_t num_orig_topk_blocks, extra_topk_length;    // Only decoding have `num_orig_topk_blocks` and `extra_topk_length`

        __aicore__ bool is_valid() {
            return batch_or_s_q_idx != 0xFFFFFFFFu;
        }
    };

    auto _get_outer_loop_args = [&](uint32_t job_idx, uint32_t s_q_idx) __aicore__ -> OuterLoopArgs {
        uint32_t topk_length = params.topk_length != nullptr ? (uint32_t)params.topk_length[s_q_idx] : params.topk;
        uint32_t num_topk_blocks = max((topk_length + B_TOPK - 1) / B_TOPK, 2u);    // Each q token must have at least 2 blocks, since we may enable partial copy out
        uint32_t extra_topk_length, num_orig_topk_blocks;
        if constexpr (IS_DECODE) {
            extra_topk_length = params.extra_topk_length != nullptr ? (uint32_t)params.extra_topk_length[s_q_idx] : params.extra_topk;
            uint32_t num_extra_topk_blocks = (extra_topk_length + B_TOPK - 1) / B_TOPK;
            num_orig_topk_blocks = num_topk_blocks;
            num_topk_blocks += num_extra_topk_blocks;
        }
        return {job_idx, s_q_idx, num_topk_blocks, topk_length, num_orig_topk_blocks, extra_topk_length};
    };

    auto get_first_job = [&]() __aicore__ -> OuterLoopArgs {
        return _get_outer_loop_args(0u, (uint32_t)get_block_idx() / NUM_HEAD_SPLITS);
    };

    auto get_next_job = [&](const OuterLoopArgs &cur_job) __aicore__ -> OuterLoopArgs {
        uint32_t total_q;
        if constexpr (IS_DECODE) {
            total_q = (uint32_t)params.b;
        } else {
            total_q = (uint32_t)params.s_q;
        }
        uint32_t nxt_s_q_idx = cur_job.batch_or_s_q_idx + Q_TOKENS_PER_JOB_STEP;
        if (nxt_s_q_idx >= total_q) {
            return OuterLoopArgs {
                cur_job.job_idx + 1,
                0xFFFFFFFFu
            };
        } else {
            return _get_outer_loop_args(
                cur_job.job_idx + 1,
                nxt_s_q_idx
            );
        }
    };

    auto have_next_job = [&](const OuterLoopArgs &cur_job) __aicore__ -> bool {
        uint32_t total_q;
        if constexpr (IS_DECODE) {
            total_q = (uint32_t)params.b;
        } else {
            total_q = (uint32_t)params.s_q;
        }
        uint32_t nxt_s_q_idx = cur_job.batch_or_s_q_idx + Q_TOKENS_PER_JOB_STEP;
        return nxt_s_q_idx < total_q;
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
        if constexpr (HAVE_Q_O_ROPE_AND_O_CAST) {
            asc_datacache_preload((__gm__ uint64_t*)params.token_positions, 0);
        }
        int64_t pc = asc_get_program_counter() & 0xFFFFFFFFFFFF;
        asc_icache_preload(reinterpret_cast<void *>(pc), 8);    // 8 = 16K (AIV's icache size) / 2K

        uint32_t subblock_idx = (uint32_t)get_subblockid();
        uint32_t this_subblock_start_head_idx = this_ai_core_start_head_idx + subblock_idx * MMA_M_PER_V;
        uint32_t num_this_subblock_valid_heads = H_Q_PER_AI_CORE >= subblock_idx * MMA_M_PER_V ? min(H_Q_PER_AI_CORE - subblock_idx * MMA_M_PER_V, MMA_M_PER_V) : 0;   // AIV0 processes head [this_ai_core_start_head_idx, this_ai_core_start_head_idx + MMA_M_PER_V) and AIV1 processes the rest of this AI core's head half
        asc_set_gm2ub_loop_size(1ul, 1ul);
        asc_set_ub2gm_loop_size(1ul, 1ul);
        asc_set_ctrl(1ull<<49); // Let the scalar core use dcache as the cache when reading from UB

        if constexpr (HAVE_ATTN_SINK) {
            asc_copy_gm2ub_align(ub_buf.attn_sink, params.attn_sink + this_subblock_start_head_idx, num_this_subblock_valid_heads * sizeof(float));
            set_flag(PIPE_MTE2, PIPE_V, event_t(VectorCoreFlags::ATTN_SINK_FULL));
        }

        AscendC::CrossCoreSetFlag<4, PIPE_S>(CrossCoreFlags::UBUF_P_EMPTY);
        #pragma unroll
        for (uint32_t i = 0; i < NUM_OUTPUT_FRAG_BUFS; ++i) 
            AscendC::CrossCoreSetFlag<4, PIPE_S>(CrossCoreFlags::UBUF_O_FRAG_EMPTY+i);
        #pragma unroll
        for (uint32_t i = 0; i < NUM_OUTPUT_FRAGS; ++i) 
            AscendC::CrossCoreSetFlag<4, PIPE_S>(CrossCoreFlags::UBUF_O_ACCUM_EMPTY+i);
        if constexpr (HAVE_Q_O_ROPE_AND_O_CAST) {
            #pragma unroll
            for (uint32_t i = 0; i < NUM_ROPE_COS_SIN_BUFS; ++i)
                set_flag(PIPE_V, PIPE_MTE2, event_t(VectorCoreFlags::ROPE_COS_SIN_BUF_EMPTY + i));
            #pragma unroll
            for (uint32_t i = 0; i < NUM_UB_Q_FRAG_BUFS; ++i)
                AscendC::CrossCoreSetFlag<4, PIPE_S>(CrossCoreFlags::UBUF_Q_FRAG_EMPTY + i);
            #pragma unroll
            for (uint32_t i = 0; i < NUM_UB_Q_FRAG_BUFS; ++i)
                set_flag(PIPE_MTE3, PIPE_V, event_t(VectorCoreFlags::Q_FRAG_BUF_EMPTY + i));
        }

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

            asc_dcci_entire_ub();

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

        uint32_t cur_run_fetch_rope_cos_sin_buf_idx = 0u;   // Self-incremental
        auto run_fetch_rope_cos_and_sin = [&](const OuterLoopArgs &args, uint32_t kv_block_idx) __aicore__ {
            if (kv_block_idx != 0) return;
            if constexpr (HAVE_Q_O_ROPE_AND_O_CAST) {
                uint32_t token_position = params.token_positions[args.batch_or_s_q_idx];
                uint32_t rope_buf_idx = cur_run_fetch_rope_cos_sin_buf_idx;
                plus_one_and_mod<NUM_ROPE_COS_SIN_BUFS>(cur_run_fetch_rope_cos_sin_buf_idx);
                if (args.batch_or_s_q_idx + Q_TOKENS_PER_JOB_STEP < params.s_q)
                    asc_datacache_preload((__gm__ uint64_t *)(params.token_positions + args.batch_or_s_q_idx), Q_TOKENS_PER_JOB_STEP * sizeof(int32_t));

                wait_flag(PIPE_V, PIPE_MTE2, event_t(VectorCoreFlags::ROPE_COS_SIN_BUF_EMPTY + rope_buf_idx));
                asc_copy_gm2ub(
                    ub_buf.rope_cos_sin_buf[rope_buf_idx],
                    params.cos_sin_cache + token_position * D_ROPE,
                    D_ROPE * sizeof(float)
                );
                set_flag(PIPE_MTE2, PIPE_V, event_t(VectorCoreFlags::ROPE_COS_SIN_BUF_FULL + rope_buf_idx));
            }
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
                            NUM_HEAD_SPLITS > 1 ? asc_load_l2_cache_mode::NORMAL_LAST_VICTIM
                                               : asc_load_l2_cache_mode::NOTALLOC_KEEP, // When use the decoding kernel to perform prefill-like work, adjusting this may have benefit
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
        uint32_t run_kv_nd2nz_and_push_to_cube_l1_kv_buf_idx = 0u;
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

            uint32_t l1_kv_buf_idx = run_kv_nd2nz_and_push_to_cube_l1_kv_buf_idx;
            plus_one_and_mod<NUM_L1_KV_BUFS>(run_kv_nd2nz_and_push_to_cube_l1_kv_buf_idx);
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

        uint32_t cur_run_q_rope_rope_cos_sin_buf_idx = 0u;   // Self-incremental
        auto run_q_rope = [&](const OuterLoopArgs &args, uint32_t kv_block_idx) __aicore__ {
            if (kv_block_idx != 0) return;
            if constexpr (HAVE_Q_O_ROPE_AND_O_CAST) {
                uint32_t rope_buf_idx = cur_run_q_rope_rope_cos_sin_buf_idx;
                plus_one_and_mod<NUM_ROPE_COS_SIN_BUFS>(cur_run_q_rope_rope_cos_sin_buf_idx);
                uint32_t q_l1_buf_idx = args.job_idx % NUM_L1_Q_BUFS;   // The same L1 Q buffer the CUBE Core loaded

                wait_flag(PIPE_MTE2, PIPE_V, event_t(VectorCoreFlags::ROPE_COS_SIN_BUF_FULL + rope_buf_idx));

                uint32_t q_frag_idx = NUM_Q_FRAGS-1;
                uint32_t q_frag_buf_idx = q_frag_idx % NUM_UB_Q_FRAG_BUFS;

                wait_flag(PIPE_MTE3, PIPE_V, event_t(VectorCoreFlags::Q_FRAG_BUF_EMPTY + q_frag_buf_idx));
                AscendC::CrossCoreWaitFlag<4, PIPE_V>(CrossCoreFlags::UBUF_Q_FRAG_FULL + q_frag_buf_idx);
                q_rope_vf(ub_buf, rope_buf_idx);
                set_flag(PIPE_V, PIPE_MTE3, event_t(VectorCoreFlags::Q_FRAG_BUF_FULL + q_frag_buf_idx));

                wait_flag(PIPE_V, PIPE_MTE3, event_t(VectorCoreFlags::Q_FRAG_BUF_FULL + q_frag_buf_idx));
                asc_copy_ub2l1(
                    (subblock_idx ? l1_q_buf_hi : l1_q_buf_lo)
                        + q_l1_buf_idx * (MMA_M_PER_V * D_QK) + MMA_M_PER_V * (D_QK - D_ROPE),
                    ub_buf.q_frag[(NUM_Q_FRAGS-1) % NUM_UB_Q_FRAG_BUFS] + (Q_FRAG_D_QK - D_ROPE) * MMA_M_PER_V,
                    MMA_M_PER_V * D_ROPE * sizeof(bf16)
                );
                AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(CrossCoreFlags::L1_Q_ROPE_FRAG_FULL);
                AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(CrossCoreFlags::UBUF_Q_FRAG_EMPTY + q_frag_buf_idx);
                set_flag(PIPE_MTE3, PIPE_V, event_t(VectorCoreFlags::Q_FRAG_BUF_EMPTY + q_frag_buf_idx));
            }
        };

        auto run_q_norm = [&](const OuterLoopArgs &args, uint32_t kv_block_idx) __aicore__ {
            if (kv_block_idx != 0) return;
            if constexpr (HAVE_Q_NORM) {
                uint32_t q_l1_buf_idx = args.job_idx % NUM_L1_Q_BUFS;   // The same L1 Q buffer the CUBE Core loaded

                #pragma unroll 1
                for (uint32_t q_frag_idx = NUM_Q_FRAGS-2; q_frag_idx != 0xffffffffu; --q_frag_idx) {
                    uint32_t q_frag_buf_idx = q_frag_idx % NUM_UB_Q_FRAG_BUFS;

                    AscendC::CrossCoreWaitFlag<4, PIPE_V>(CrossCoreFlags::UBUF_Q_FRAG_FULL + q_frag_buf_idx);
                    q_norm_vf(ub_buf, args.job_idx, q_frag_idx, params.sm_scale, params.rms_norm_eps);
                    AscendC::CrossCoreSetFlag<4, PIPE_V>(CrossCoreFlags::UBUF_Q_FRAG_EMPTY + q_frag_buf_idx);
                }
            }
        };

        uint32_t cur_index_buf_idx_for_softmax = NUM_INDEX_BUFS - 1;
        auto run_softmax = [&](const OuterLoopArgs &args, uint32_t kv_block_idx) __aicore__ {
            bool is_last_block = kv_block_idx+1 == args.num_topk_blocks;
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
            asc_ub_dcci_single(ub_buf.row_max_max_delta);
            uint32_t ub_indices_arr_offset = (partial_kv_block_idx % (INDEX_BLOCK_SIZE / B_TOPK)) * (B_TOPK / 2);
            if (kv_block_idx == 0) {
                softmax_vf<true>(ub_buf, params.sm_scale, args.job_idx, kv_block_idx, cur_index_buf_idx_for_softmax, ub_indices_arr_offset, s_kv, subblock_idx);
            } else {
                softmax_vf<false>(ub_buf, params.sm_scale, args.job_idx, kv_block_idx, cur_index_buf_idx_for_softmax, ub_indices_arr_offset, s_kv, subblock_idx);
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
        };

        auto run_get_final_lse_and_denorm = [&](const OuterLoopArgs &args, uint32_t kv_block_idx) __aicore__ {
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
                        params.max_logits + args.batch_or_s_q_idx*H_Q + this_subblock_start_head_idx,
                        ub_buf.row_max[args.job_idx&1],
                        num_this_subblock_valid_heads * sizeof(float)
                    );
                }
                __gm__ float* lse_base;
                if constexpr (IS_DECODE) {
                    lse_base = params.lse
                        + args.batch_or_s_q_idx * params.stride_lse_s_q + this_subblock_start_head_idx;
                } else {
                    lse_base = params.lse
                        + args.batch_or_s_q_idx * H_Q + this_subblock_start_head_idx;
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
                ss_buf.is_rescale_triggered[args.job_idx%8][kv_block_idx%8][subblock_idx] = ub_buf.row_max_max_delta[args.job_idx&1][kv_block_idx&1] > RESCALE_THRESHOLD;
                AscendC::CrossCoreSetFlag<4, PIPE_S>(CrossCoreFlags::SS_BUFFER_FULL_V2C);
            }
        };

        uint32_t o_copy_idx = 0u;   // Private to `run_scale_o` and `run_o_copy_out`, marks whether this O block is the first one sent from FixPipe, and is resetted upon every request
        auto run_scale_o = [&](const OuterLoopArgs &args, uint32_t kv_block_idx) __aicore__ {
            if (kv_block_idx == 0) {
                o_copy_idx = 0;
                return;
            }

            AscendC::CrossCoreWaitFlag<4, PIPE_S>(CrossCoreFlags::SS_BUFFER_FULL_C2V);
            uint64_t t = *(volatile __ssbuf__ uint64_t*)ss_buf.is_rescale_triggered[args.job_idx%8][kv_block_idx%8];
            bool is_rescale_triggered = t > 0;   // Trigger o copy out when at least one VECTOR Core requires rescaling

            if (is_rescale_triggered) {
                static_assert(D_VO % OUTPUT_FRAG_D_VO == 0);
                #pragma unroll 1
                for (uint32_t i = 0; i < D_VO / OUTPUT_FRAG_D_VO; ++i) {
                    uint32_t cur_o_frag_buf_idx = i % NUM_OUTPUT_FRAG_BUFS;
                    get_buf(PIPE_V, VectorCoreBufIDs::O_ACCUM + i, false);
                    AscendC::CrossCoreWaitFlag<4, PIPE_V>(CrossCoreFlags::UBUF_O_FRAG_FULL+cur_o_frag_buf_idx);
                    if (o_copy_idx == 0) {
                        add_and_scale_o_vf<false>(ub_buf, i, cur_o_frag_buf_idx, params.sm_scale, args.job_idx, kv_block_idx);
                    } else {
                        add_and_scale_o_vf<true>(ub_buf, i, cur_o_frag_buf_idx, params.sm_scale, args.job_idx, kv_block_idx);
                    }
                    AscendC::CrossCoreSetFlag<4, PIPE_V>(CrossCoreFlags::UBUF_O_FRAG_EMPTY+cur_o_frag_buf_idx);
                    rls_buf(PIPE_V, VectorCoreBufIDs::O_ACCUM + i, false);
                }
                o_copy_idx += 1;
            }
        };

        static constexpr uint32_t NUM_PARTIAL_COPY_OUT_STAGES = HAVE_Q_O_ROPE_AND_O_CAST ? 3 : 2;
        uint32_t cur_run_o_copy_out_rope_cos_sin_buf_idx = 0u;   // Self-incremental
        uint32_t cur_partial_copy_out_stage_idx = 0u;    // We expect the systolic array to call run_o_copy_out for NUM_PARTIAL_COPY_OUT_STAGES consecutive times
        uint32_t backup_o_copy_idx = 0u;    // Copied from `o_copy_idx` during stage 0, since `o_copy_idx` may be overwritten by later blocks
        auto run_o_copy_out = [&](const OuterLoopArgs &args, uint32_t kv_block_idx) __aicore__ {
            bool is_last_block = kv_block_idx+1 == args.num_topk_blocks;
            if (!is_last_block)
                return;

            uint32_t partial_copy_out_stage_idx = cur_partial_copy_out_stage_idx;
            plus_one_and_mod<NUM_PARTIAL_COPY_OUT_STAGES>(cur_partial_copy_out_stage_idx);

            if (partial_copy_out_stage_idx == 0) {
                backup_o_copy_idx = o_copy_idx;
            }
            bool have_previous_output_accum = backup_o_copy_idx > 0;
            
            uint32_t stride_o_batch_or_s_q;
            if constexpr (IS_DECODE) {
                stride_o_batch_or_s_q = params.stride_o_b;
            } else {
                stride_o_batch_or_s_q = H_Q*D_VO;
            }

            /*
            Rules for using output frag / accumulators, as well as waiting & signalling the CrossCoreFlags:

            If the current q token never triggers O rescale, then:
            - FixPipe waits for `CrossCoreFlags::UBUF_O_ACCUM_EMPTY`, writes to `ub_buf.output_accum`, and signals `CrossCoreFlags::UBUF_O_ACCUM_FULL`
            - VF reads from `ub_buf.output_accum`, write result to `ub_buf.output_accum`
            - MTE3 reads from `ub_buf.output_accum`, and signals `CrossCoreFlags::UBUF_O_ACCUM_EMPTY` after the copy is finished

            If the current q token have triggered O rescale before, then:
            - FixPipe waits for `CrossCoreFlags::UBUF_O_ACCUM_EMPTY` (must consume this flag because VECTOR core always sets it)
            - FixPipe waits for `CrossCoreFlags::UBUF_O_FRAG_EMPTY`, writes to `ub_buf.cur_output_frag`, and signals `CrossCoreFlags::UBUF_O_FRAG_FULL`
            - VF reads from `ub_buf.cur_output_frag` and `ub_buf.output_accum`, write result to `ub_buf.output_accum`, and signals `CrossCoreFlags::UBUF_O_FRAG_EMPTY`
            - MTE3 reads from `ub_buf.output_accum`, and signals `CrossCoreFlags::UBUF_O_ACCUM_EMPTY` after the copy is finished
            */
            /*
            Explanation about partial copy-out:
            The O epilogue (cast-to-bf16 for non-fused cases, cast-to-fp8 for fused cases) is too long to be fit into one pipeline stage, so we break into multiple (`NUM_PARTIAL_COPY_OUT_STAGES`) pipeline stages.
            For non-fused cases (i.e. HAVE_Q_O_ROPE_AND_O_CAST == false), we process (multiply `1/lse` and cast-to-bf16) the first two O frags in the first partial copy-out stage, and process the remaining two O frags in the second partial copy-out stage.
            For fused cases, we get the datablock-wise (i.e. every 8 numbers) absolute maximum (amax) of the first three O frags in the first stage, perform RoPE and get datablock-wise amax of the last O frags in the second stage, and do the remaining (scale calculation & cast-to-fp8) in the third partial copy-out stage.
            We require each q token to have at least 2 KV blocks, to avoid the next O's partial copy-out stage interleaves with the current one
            */
            if constexpr (HAVE_Q_O_ROPE_AND_O_CAST) {
                if (partial_copy_out_stage_idx == 0) {
                    #pragma unroll 1
                    for (uint32_t o_frag_idx = 0; o_frag_idx < NUM_OUTPUT_FRAGS - 1; ++o_frag_idx) {
                        uint32_t cur_o_frag_buf_idx = o_frag_idx % NUM_OUTPUT_FRAG_BUFS;
                        get_buf(PIPE_V, VectorCoreBufIDs::O_ACCUM + o_frag_idx, false);
                        if (have_previous_output_accum) {
                            AscendC::CrossCoreWaitFlag<4, PIPE_V>(CrossCoreFlags::UBUF_O_FRAG_FULL+cur_o_frag_buf_idx);
                            get_final_output_vf_with_cast_and_rope_stage0<true>(ub_buf, o_frag_idx, cur_o_frag_buf_idx, args.job_idx);
                            AscendC::CrossCoreSetFlag<4, PIPE_V>(CrossCoreFlags::UBUF_O_FRAG_EMPTY+cur_o_frag_buf_idx);
                        } else {
                            AscendC::CrossCoreWaitFlag<4, PIPE_V>(CrossCoreFlags::UBUF_O_ACCUM_FULL+o_frag_idx);
                            get_final_output_vf_with_cast_and_rope_stage0<false>(ub_buf, o_frag_idx, cur_o_frag_buf_idx, args.job_idx);
                        }
                        rls_buf(PIPE_V, VectorCoreBufIDs::O_ACCUM + o_frag_idx, false);
                    }

                } else if (partial_copy_out_stage_idx == 1) {
                    uint32_t rope_buf_idx = cur_run_o_copy_out_rope_cos_sin_buf_idx;
                    plus_one_and_mod<NUM_ROPE_COS_SIN_BUFS>(cur_run_o_copy_out_rope_cos_sin_buf_idx);

                    get_buf(PIPE_V, VectorCoreBufIDs::O_ACCUM + 3, false);
                    if (have_previous_output_accum) {
                        AscendC::CrossCoreWaitFlag<4, PIPE_V>(CrossCoreFlags::UBUF_O_FRAG_FULL+(NUM_OUTPUT_FRAGS-1)%NUM_OUTPUT_FRAG_BUFS);
                        get_final_output_vf_with_cast_and_rope_stage1<true>(ub_buf, args.job_idx, rope_buf_idx);
                        AscendC::CrossCoreSetFlag<4, PIPE_V>(CrossCoreFlags::UBUF_O_FRAG_EMPTY+(NUM_OUTPUT_FRAGS-1)%NUM_OUTPUT_FRAG_BUFS);
                    } else {
                        AscendC::CrossCoreWaitFlag<4, PIPE_V>(CrossCoreFlags::UBUF_O_ACCUM_FULL+(NUM_OUTPUT_FRAGS-1));
                        get_final_output_vf_with_cast_and_rope_stage1<false>(ub_buf, args.job_idx, rope_buf_idx);
                    }
                    rls_buf(PIPE_V, VectorCoreBufIDs::O_ACCUM + 3, false);
                    set_flag(PIPE_V, PIPE_MTE2, event_t(VectorCoreFlags::ROPE_COS_SIN_BUF_EMPTY + rope_buf_idx));

                } else {
                    #pragma unroll
                    for (uint32_t i = 0; i < NUM_OUTPUT_FRAGS; ++i) {
                        get_buf(PIPE_V, VectorCoreBufIDs::O_ACCUM + i, false);
                    }

                    get_final_output_vf_with_cast_and_rope_stage2(ub_buf, args.job_idx);

                    #pragma unroll
                    for (uint32_t i = 0; i < NUM_OUTPUT_FRAGS; ++i) {
                        rls_buf(PIPE_V, VectorCoreBufIDs::O_ACCUM + i, false);
                    }

                    if (num_this_subblock_valid_heads != 0) {
                        #pragma unroll
                        for (uint32_t i = 0; i < NUM_OUTPUT_FRAGS; ++i) {
                            get_buf(PIPE_MTE3, VectorCoreBufIDs::O_ACCUM + i, false);
                        }

                        // Layout of params.out_sf_scratchpad: [s_q][D_VO/SCALE_GRAN/4][heads][4]
                        asc_copy_ub2gm_align(
                            (__gm__ uint8_t*)(params.out_sf_scratchpad)
                                + (uint64_t)args.batch_or_s_q_idx * (H_Q * D_VO / SCALE_GRAN)
                                + (uint64_t)this_subblock_start_head_idx * 4,
                            ub_buf.o_scale_factors,
                            NUM_OUTPUT_FRAGS,
                            num_this_subblock_valid_heads * (OUTPUT_FRAG_D_VO / SCALE_GRAN),
                            asc_store_l2_cache_mode::NORMAL_LAST_VICTIM,    // Use LV because we're going to transpose it later
                            H_Q * (OUTPUT_FRAG_D_VO / SCALE_GRAN),
                            MMA_M_PER_V * (OUTPUT_FRAG_D_VO / SCALE_GRAN)
                        );

                        if (!have_next_job(args)) {
                            asc_sync_inter_arrive(PIPE_MTE3, InterCoreFlags::SYNC_BEFORE_SF_TRANSPOSE);
                        }

                        asc_set_ub2gm_loop_size(NUM_OUTPUT_FRAGS, 1);
                        asc_set_ub2gm_loop1_stride(MMA_M_PER_V * OUTPUT_FRAG_D_VO, OUTPUT_FRAG_D_VO);
                        asc_copy_ub2gm_align(
                            (__gm__ uint8_t*)(params.out_fp8 + (uint64_t)args.batch_or_s_q_idx * (H_Q * D_VO)
                                + this_subblock_start_head_idx * D_VO),
                            (__ubuf__ uint8_t*)(ub_buf.output_accum[0]),
                            num_this_subblock_valid_heads,
                            OUTPUT_FRAG_D_VO,
                            asc_store_l2_cache_mode::NOTALLOC_CLEAN,
                            D_VO,
                            OUTPUT_FRAG_D_VO
                        );
                        asc_set_ub2gm_loop_size(1, 1);
                        
                        #pragma unroll
                        for (uint32_t i = 0; i < NUM_OUTPUT_FRAGS; ++i) {
                            rls_buf(PIPE_MTE3, VectorCoreBufIDs::O_ACCUM + i, false);
                        }
                        #pragma unroll
                        for (uint32_t i = 0; i < NUM_OUTPUT_FRAGS; ++i) {
                            // Must release CrossCoreFlags::UBUF_O_ACCUM_EMPTY + 0 after all MTE3 copies are finished (instead of the first MTE3) because all FP8 resides in `ub.output_accum[0]`
                            AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(CrossCoreFlags::UBUF_O_ACCUM_EMPTY + i);
                        }
                    } else {
                        if (!have_next_job(args)) {
                            // Otherwise the waiter waits infinitely
                            asc_sync_inter_arrive(PIPE_MTE3, InterCoreFlags::SYNC_BEFORE_SF_TRANSPOSE);
                        }
                        #pragma unroll
                        for (uint32_t i = 0; i < NUM_OUTPUT_FRAGS; ++i) {
                            AscendC::CrossCoreSetFlag<4, PIPE_V>(CrossCoreFlags::UBUF_O_ACCUM_EMPTY + i);
                        }
                    }
                }
            } else {
                static constexpr uint32_t NUM_O_FRAG_PER_PARTIAL_COPY_OUT = NUM_OUTPUT_FRAGS / NUM_PARTIAL_COPY_OUT_STAGES;
                static_assert(NUM_OUTPUT_FRAGS % NUM_PARTIAL_COPY_OUT_STAGES == 0);
                uint32_t num_o_frags_in_this_partial_copy_out;
                uint32_t start_o_frag_idx;
                if (have_previous_output_accum) {
                    // FixPipe is going to write to `ub_buf.cur_output_frag`, but we only have one `cur_output_frag`. To avoid FixPipe being stuck, we must leave at most one chunk unreceived
                    num_o_frags_in_this_partial_copy_out = partial_copy_out_stage_idx == 1 ? 1 : 3;
                    start_o_frag_idx = partial_copy_out_stage_idx == 1 ? 3 : 0;
                } else {
                    // Balance workload between two partial copy-out stages
                    num_o_frags_in_this_partial_copy_out = 2;
                    start_o_frag_idx = partial_copy_out_stage_idx * 2;
                }
                #pragma unroll 1
                for (uint32_t i = 0; i < num_o_frags_in_this_partial_copy_out; ++i) {
                    uint32_t o_frag_idx = i + start_o_frag_idx;
                    uint32_t cur_o_frag_buf_idx = o_frag_idx % NUM_OUTPUT_FRAG_BUFS;
                    get_buf(PIPE_V, VectorCoreBufIDs::O_ACCUM + o_frag_idx, false);
                    if (have_previous_output_accum) {
                        AscendC::CrossCoreWaitFlag<4, PIPE_V>(CrossCoreFlags::UBUF_O_FRAG_FULL+cur_o_frag_buf_idx);
                        get_final_output_vf<true>(ub_buf, o_frag_idx, cur_o_frag_buf_idx, args.job_idx);
                        AscendC::CrossCoreSetFlag<4, PIPE_V>(CrossCoreFlags::UBUF_O_FRAG_EMPTY+cur_o_frag_buf_idx);
                    } else {
                        AscendC::CrossCoreWaitFlag<4, PIPE_V>(CrossCoreFlags::UBUF_O_ACCUM_FULL+o_frag_idx);
                        get_final_output_vf<false>(ub_buf, o_frag_idx, cur_o_frag_buf_idx, args.job_idx);
                    }
                    rls_buf(PIPE_V, VectorCoreBufIDs::O_ACCUM + o_frag_idx, false);

                    get_buf(PIPE_MTE3, VectorCoreBufIDs::O_ACCUM + o_frag_idx, false);
                    if (num_this_subblock_valid_heads != 0) {
                        // bf16 epilogue: `get_final_output_vf<false/true>` staged the fragment as bf16 in
                        // `output_accum` ([row][D_VO] bf16, this fragment at column i*OUTPUT_FRAG_D_VO)
                        asc_copy_ub2gm_align(
                            params.out + (uint64_t)args.batch_or_s_q_idx * stride_o_batch_or_s_q
                                + (uint64_t)this_subblock_start_head_idx * D_VO + o_frag_idx * OUTPUT_FRAG_D_VO,
                            (__ubuf__ bf16*)(ub_buf.output_accum[o_frag_idx]),
                            num_this_subblock_valid_heads,
                            OUTPUT_FRAG_D_VO * sizeof(bf16),
                            asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                            D_VO * sizeof(bf16),
                            OUTPUT_FRAG_D_VO * sizeof(bf16)
                        );
                    }
                    rls_buf(PIPE_MTE3, VectorCoreBufIDs::O_ACCUM + o_frag_idx, false);
                    AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(CrossCoreFlags::UBUF_O_ACCUM_EMPTY+o_frag_idx);
                }
            }
        };

        static constexpr uint32_t INVALID_ITEM_KV_BLOCK_IDX = 0xffffffffu;
        struct SystolicArrayItem {
            uint32_t kv_block_idx;
            OuterLoopArgs args;
        };
        
        static constexpr uint32_t SYSTOLIC_ARR_DEPTH = 8;
        SystolicArrayItem systolic_arr_items[SYSTOLIC_ARR_DEPTH];
        uint32_t systolic_array_head = 0u;
        auto systolic_arr_init = [&]() __aicore__ {
            #pragma unroll
            for (uint32_t i = 0; i < SYSTOLIC_ARR_DEPTH; ++i)
                systolic_arr_items[i].kv_block_idx = INVALID_ITEM_KV_BLOCK_IDX;
        };

        auto systolic_arr_shift_and_push = [&](const SystolicArrayItem &new_item) __aicore__ {
            systolic_arr_items[systolic_array_head] = new_item;
            plus_one_and_mod<SYSTOLIC_ARR_DEPTH>(systolic_array_head);
        };

        auto systolic_arr_activate = [&]() __aicore__ {
            #define GET_SYSTOLIC_ARR_ITEM_WITH_OFFSET(offset) (systolic_arr_items[((offset)+systolic_array_head)%SYSTOLIC_ARR_DEPTH])
            #define ACTIVATE(offset, func) \
                if (GET_SYSTOLIC_ARR_ITEM_WITH_OFFSET(offset).kv_block_idx != INVALID_ITEM_KV_BLOCK_IDX) {\
                    /*asc_sync_pipe(PIPE_ALL); if (get_block_idx() == 0 && subblock_idx == 0) printf("%d %s (%d, %d)\n", offset, #func, GET_SYSTOLIC_ARR_ITEM_WITH_OFFSET(offset).args.batch_or_s_q_idx, GET_SYSTOLIC_ARR_ITEM_WITH_OFFSET(offset).kv_block_idx);*/ \
                    func(GET_SYSTOLIC_ARR_ITEM_WITH_OFFSET(offset).args, GET_SYSTOLIC_ARR_ITEM_WITH_OFFSET(offset).kv_block_idx); \
                }

            ACTIVATE(7, run_fetch_rope_cos_and_sin);
            if constexpr (!IS_DECODE) {
                ACTIVATE(7, run_issue_index_fetch);
            }
            ACTIVATE(5, run_kv_nd2nz_and_push_to_cube);
            ACTIVATE(3, run_softmax);

            // ACTIVATE(2, run_scale_o) must be put before ACTIVATE(4, run_q_norm_and_rope), otherwise:
            //  - CUBE core first tries to send O(i-3) in SV(i-2) and then issue QK(i)
            //   (`O(i-3)` means the O accumulator after SV(i-3))
            //  - VECTOR core first waits for Q(i) in ACTIVATE(4, run_q_norm_and_rope), and then waits for O(i-3) in ACTIVATE(2, run_scale_o) 
            // which creates a deadlock:
            //  - VECTOR waits for Q(i), and is unable to signal CrossCoreFlags::UBUF_O_FRAG_EMPTY (because `q_rope_vf` -> `add_and_scale_o_vf` -> `CrossCoreFlags::UBUF_O_FRAG_EMPTY`)
            //  - CUBE waits for `CrossCoreFlags::UBUF_O_FRAG_EMPTY`, and is unable to send Q(i) (because of MTE1 ordering and issue queue depth)

            // Besides, ACTIVATE(0, run_o_copy_out) must be put before ACTIVATE(2, run_scale_o). Otherwise the newly result O in FP8 (which lives in `ub_buf.output_accum[0]`) may be overwritten
            if constexpr (NUM_PARTIAL_COPY_OUT_STAGES >= 2) {
                ACTIVATE(1, run_o_copy_out);
            }
            if constexpr (NUM_PARTIAL_COPY_OUT_STAGES >= 3) {
                ACTIVATE(0, run_o_copy_out);
            }
            ACTIVATE(2, run_scale_o);
            ACTIVATE(2, run_o_copy_out);
            ACTIVATE(4, run_q_rope);
            ACTIVATE(4, run_q_norm);
            ACTIVATE(3, run_get_final_lse_and_denorm);

            if constexpr (IS_DECODE) {
                ACTIVATE(7, run_issue_index_fetch);
            }
            ACTIVATE(7, run_process_index);

            // If we've just launched `run_process_index`, then we launch `issue_kv_gather` after `set_is_rescale_triggered_flag` because
            //  1) `run_process_index` is using VF, issuing `set_is_rescale_triggered_flag` now won't cause VF to be idle
            //  2) `run_issue_kv_gather` must wait for `run_process_index`. If we launch `set_is_rescale_triggered_flag` after `issue_kv_gather`, the scalar core must wait for `process_index` to complete which cause cube core to be idle
            bool issue_kv_gather_later = GET_SYSTOLIC_ARR_ITEM_WITH_OFFSET(7).kv_block_idx != INVALID_ITEM_KV_BLOCK_IDX && GET_SYSTOLIC_ARR_ITEM_WITH_OFFSET(7).kv_block_idx % (INDEX_BLOCK_SIZE / B_TOPK) == 0;
            if (issue_kv_gather_later) {
                ACTIVATE(3, run_set_is_rescale_triggered_flag);
            }
            ACTIVATE(7, run_issue_kv_gather);
            if (!issue_kv_gather_later) {
                ACTIVATE(3, run_set_is_rescale_triggered_flag);
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
            for (uint32_t k = 0; k < (is_draining ? SYSTOLIC_ARR_DEPTH - 1 : cur_job.num_topk_blocks); ++k) {
                systolic_arr_shift_and_push(SystolicArrayItem{is_draining ? INVALID_ITEM_KV_BLOCK_IDX : k, cur_job});
                systolic_arr_activate();
            }
            if (is_draining)
                break;
            cur_job = get_next_job(cur_job);
            if (!cur_job.is_valid()) {
                is_draining = true;
            }
        }
        if constexpr (HAVE_Q_O_ROPE_AND_O_CAST) {
            static_assert(NUM_SF_TRANSPOSE_BUFS <= 8);
            static constexpr uint32_t SF_BUF_START_IDX = 8;    // VectorCoreBufIDs 8..11, free after the psum buffers were removed
            __ubuf__ UnifiedBufferMemoryPlanForSFTranspose* ub_buf_for_transpose_ptr = (__ubuf__ UnifiedBufferMemoryPlanForSFTranspose*)0;
            __ubuf__ UnifiedBufferMemoryPlanForSFTranspose& ub_buf_for_transpose = *ub_buf_for_transpose_ptr;

            asc_set_ndim_loop0_stride(NUM_Q_TOKENS_PER_SF_TRANSPOSE, 1);
            asc_set_ndim_loop1_stride(NUM_SCALES_PER_TOKEN/2 * NUM_Q_TOKENS_PER_SF_TRANSPOSE, 2);
            asc_set_ndim_loop2_stride(NUM_Q_TOKENS_PER_SF_TRANSPOSE * 2, H_Q * 2);
            asc_set_ndim_loop3_stride(1, NUM_SCALES_PER_TOKEN/4 * H_Q * 2);
            asc_set_ndim_loop4_stride(0, 0);

            asc_sync_inter_wait(PIPE_MTE2, InterCoreFlags::SYNC_BEFORE_SF_TRANSPOSE);

            uint32_t nxt_buf_idx = 0;
            uint32_t total_q;
            if constexpr (IS_DECODE) {
                total_q = (uint32_t)params.b;
            } else {
                total_q = (uint32_t)params.s_q;
            }
            #pragma unroll 1
            for (uint32_t s_q_idx = (get_block_idx()*2 + get_subblockid()) * NUM_Q_TOKENS_PER_SF_TRANSPOSE; s_q_idx < total_q; s_q_idx += params.num_sm * NUM_VEC_CORE_PER_AI_CORE * NUM_Q_TOKENS_PER_SF_TRANSPOSE) {
                uint32_t num_tokens = min(NUM_Q_TOKENS_PER_SF_TRANSPOSE, total_q - s_q_idx);

                uint32_t cur_buf_idx = nxt_buf_idx;
                plus_one_and_mod<NUM_SF_TRANSPOSE_BUFS>(nxt_buf_idx);

                get_buf(PIPE_MTE2, SF_BUF_START_IDX + cur_buf_idx, false);
                asc_ndim_copy_gm2ub(
                    (__ubuf__ uint16_t*)(ub_buf_for_transpose.buf[cur_buf_idx]),
                    (__gm__ uint16_t*)(params.out_sf_scratchpad + (uint64_t)s_q_idx * (H_Q * NUM_SCALES_PER_TOKEN)),
                    2, H_Q, NUM_SCALES_PER_TOKEN / 4, num_tokens, 1,
                    0, 0, true, asc_load_l2_cache_mode::NOTALLOC_CLEAN
                );
                rls_buf(PIPE_MTE2, SF_BUF_START_IDX + cur_buf_idx, false);

                get_buf(PIPE_MTE3, SF_BUF_START_IDX + cur_buf_idx, false);
                asc_copy_ub2gm_align(
                    (__gm__ uint8_t*)(params.out_sf) + s_q_idx * sizeof(uint16_t),
                    (__ubuf__ uint8_t*)(ub_buf_for_transpose.buf[cur_buf_idx]),
                    H_Q * NUM_SCALES_PER_TOKEN / 4 * 2,
                    num_tokens * sizeof(uint16_t),
                    asc_store_l2_cache_mode::NORMAL_LAST_VICTIM,
                    params.stride_out_sf_head_dim * sizeof(uint16_t),
                    NUM_Q_TOKENS_PER_SF_TRANSPOSE * sizeof(uint16_t)
                );
                rls_buf(PIPE_MTE3, SF_BUF_START_IDX + cur_buf_idx, false);
            }
        }
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
        int64_t pc = asc_get_program_counter() & 0xFFFFFFFFFFFF;
        asc_icache_preload(reinterpret_cast<void *>(pc), 16);    // 16 = 32K (AIC's icache size) / 2K

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
            q_base += (uint64_t)this_ai_core_start_head_idx * params.stride_q_h_q;
            uint32_t num_valid_q_heads_part0 = min(H_Q_PER_AI_CORE, MMA_M_PER_V);
            uint32_t num_valid_q_heads_part1 = max((int)H_Q_PER_AI_CORE-(int)MMA_M_PER_V, 0);
            asc_copy_gm2l1_nd2nz(
                l1_q_buf_lo + q_l1_buf_idx * (MMA_M_PER_V * D_QK),
                q_base,
                params.stride_q_h_q * sizeof(bf16),
                (uint32_t)LD_L2CacheType::L2_CACHE_HINT_NOTALLOC_CLEAN,
                num_valid_q_heads_part0,
                D_QK,
                0u,
                false
            );
            asc_copy_gm2l1_nd2nz(
                l1_q_buf_hi + q_l1_buf_idx * (MMA_M_PER_V * D_QK),
                q_base + (uint64_t)(MMA_M_PER_V) * params.stride_q_h_q,
                params.stride_q_h_q * sizeof(bf16),
                (uint32_t)LD_L2CacheType::L2_CACHE_HINT_NOTALLOC_CLEAN,
                num_valid_q_heads_part1,
                D_QK,
                0u,
                false
            );
            set_flag(PIPE_MTE2, PIPE_MTE1, event_t(CubeCoreFlags::L1_Q_FULL+q_l1_buf_idx));
        };

        uint32_t cur_qk_gemm_l0a_buf_idx = 0u; // self-incremental
        uint32_t cur_qk_gemm_kv_l1_buf_idx = 0u;
        auto run_qk_gemm = [&](const OuterLoopArgs &args, uint32_t kv_block_idx) __aicore__ {
            // asc_sync_pipe(PIPE_ALL); if (get_block_idx() == 0) printf("QK %d, %d\n", args.batch_or_s_q_idx, kv_block_idx);
            // Pipelined: Load Q to L0A, load K to L0B, issue MMA
            bool is_first_block = kv_block_idx == 0;
            asc_set_mmad_direction_m();
            uint32_t kv_l1_buf_idx = cur_qk_gemm_kv_l1_buf_idx;
            plus_one_and_mod<NUM_L1_KV_BUFS>(cur_qk_gemm_kv_l1_buf_idx);
            uint32_t q_l1_buf_idx = args.job_idx % NUM_L1_Q_BUFS;
            if (is_first_block) {
                wait_flag(PIPE_MTE2, PIPE_MTE1, event_t(CubeCoreFlags::L1_Q_FULL+q_l1_buf_idx));
            }
            auto send_one_q_frag_to_ub = [&](uint32_t frag_idx) __aicore__ {
                if (HAVE_Q_O_ROPE_AND_O_CAST && is_first_block) {
                    uint32_t frag_buf_idx = frag_idx % NUM_UB_Q_FRAG_BUFS;
                    cross_core_wait_flag_2aiv<PIPE_MTE1>(CrossCoreFlags::UBUF_Q_FRAG_EMPTY + frag_buf_idx);
                    asc_copy_l12ub(
                        ub_buf.q_frag[frag_buf_idx],
                        l1_q_buf_lo + q_l1_buf_idx * (MMA_M_PER_V * D_QK) + frag_idx * (MMA_M_PER_V * Q_FRAG_D_QK),
                        0, 1, MMA_M_PER_V * Q_FRAG_D_QK * sizeof(bf16) / 32, 0, 0
                    );
                    asc_copy_l12ub(
                        ub_buf.q_frag[frag_buf_idx],
                        l1_q_buf_hi + q_l1_buf_idx * (MMA_M_PER_V * D_QK) + frag_idx * (MMA_M_PER_V * Q_FRAG_D_QK),
                        1, 1, MMA_M_PER_V * Q_FRAG_D_QK * sizeof(bf16) / 32, 0, 0
                    );
                    cross_core_set_flag_2aiv<PIPE_MTE1>(CrossCoreFlags::UBUF_Q_FRAG_FULL + frag_buf_idx);
                }
            };
            send_one_q_frag_to_ub(NUM_Q_FRAGS-1);   // Send the RoPE tile first
            if constexpr (HAVE_Q_NORM) {
                static_assert(NUM_UB_Q_FRAG_BUFS == 2);
                send_one_q_frag_to_ub(NUM_Q_FRAGS-2);   // And then, send the second last tile. This won't block MTE1 since we have two q frag bufs on UB
            }
            #pragma unroll
            for (uint32_t tile_idx = 0; tile_idx < D_QK / L12L0_TILE_SIZE; ++tile_idx) {
                bool is_last_tile = tile_idx+1 == D_QK / L12L0_TILE_SIZE;

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

                // Load Q to L0B
                uint32_t q_l0a_buf_idx = cur_qk_gemm_l0a_buf_idx;
                plus_one_and_mod<NUM_L0A_BUFS>(cur_qk_gemm_l0a_buf_idx);
                if (HAVE_Q_O_ROPE_AND_O_CAST && is_first_block && is_last_tile) {
                    cross_core_wait_flag_2aiv<PIPE_MTE1>(CrossCoreFlags::L1_Q_ROPE_FRAG_FULL);  // Wait for VECTOR core's RoPE result
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

            if constexpr (HAVE_Q_NORM) {
                send_one_q_frag_to_ub(1);
                send_one_q_frag_to_ub(0);
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

        // Number of times this token's O has been copied out so far
        uint32_t cur_sv_gemm_o_copy_out_cnt = 0u;
        uint32_t cur_sv_gemm_kv_l1_buf_idx = 0u;
        auto run_sv_gemm = [&](const OuterLoopArgs &args, uint32_t kv_block_idx) __aicore__ {
            // asc_sync_pipe(PIPE_ALL); if (get_block_idx() == 0) printf("SV %d, %d\n", args.batch_or_s_q_idx, kv_block_idx);
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
                bool should_save_to_output_accum = IS_LAST_BLOCK && cur_sv_gemm_o_copy_out_cnt == 0;
                #pragma unroll
                for (uint32_t i = 0; i < D_VO / OUTPUT_FRAG_D_VO; ++i) {
                    uint32_t cur_o_frag_buf_idx = i % NUM_OUTPUT_FRAG_BUFS;
                    if constexpr (IS_LAST_BLOCK) {
                        cross_core_wait_flag_2aiv<PIPE_FIX>(CrossCoreFlags::UBUF_O_ACCUM_EMPTY+i);
                    }
                    if (!should_save_to_output_accum) {
                        cross_core_wait_flag_2aiv<PIPE_FIX>(CrossCoreFlags::UBUF_O_FRAG_EMPTY+cur_o_frag_buf_idx);
                    }
                    __ubuf__ float* cur_o_dst_ptr;
                    if (should_save_to_output_accum) {
                        cur_o_dst_ptr = ub_buf.output_accum[i];
                    } else {
                        cur_o_dst_ptr = ub_buf.cur_output_frag[cur_o_frag_buf_idx][0];
                    }
                    asc_copy_l0c2ub(
                        cur_o_dst_ptr,
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
                    cross_core_set_flag_2aiv<PIPE_FIX>(should_save_to_output_accum ? CrossCoreFlags::UBUF_O_ACCUM_FULL+i : CrossCoreFlags::UBUF_O_FRAG_FULL+cur_o_frag_buf_idx);
                }
                if constexpr (not IS_LAST_BLOCK) {
                    set_flag(PIPE_FIX, PIPE_M, event_t(CubeCoreFlags::L0C_O_EMPTY));
                    wait_flag(PIPE_FIX, PIPE_M, event_t(CubeCoreFlags::L0C_O_EMPTY));
                }
            };

            // Pipelined: Load V to L0B (w/ transpose), and issue MMA
            asc_set_mmad_direction_n();
            uint32_t kv_l1_buf_idx = cur_sv_gemm_kv_l1_buf_idx;
            plus_one_and_mod<NUM_L1_KV_BUFS>(cur_sv_gemm_kv_l1_buf_idx);
            uint32_t s_l0a_buf_idx = cur_qk_gemm_l0a_buf_idx;
            __ca__ bf16* s_l0a_buf_ptr = l0a_buf + s_l0a_buf_idx * (MMA_M * L12L0_TILE_SIZE);
            bool should_trigger_o_copy_out_before_mma;  // Will be postponed until we've issued copy for KV[0] and S, since it relies on SS_BUFFER_FULL_V2C and blocks scalar core
            
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
                    plus_one_and_mod<NUM_L0A_BUFS>(cur_qk_gemm_l0a_buf_idx);
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
                            cur_sv_gemm_o_copy_out_cnt += 1;
                        }
                    } else {
                        should_trigger_o_copy_out_before_mma = false;
                        cur_sv_gemm_o_copy_out_cnt = 0;   // New token: nothing has been flushed yet
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
    if constexpr (HAVE_Q_O_ROPE_AND_O_CAST) {
        KU_ASSERT_NPU(params.enable_q_norm == HAVE_Q_NORM);
        KU_ASSERT_NPU(params.is_rope_neox_style == false && params.rope_dim == 64);
        KU_ASSERT_NPU(params.num_per_channels == O_QUANT_TILE_SIZE);
        KU_ASSERT_NPU(params.use_tma_aligned_col_major_sf == true && params.round_sf == true && params.use_packed_ue8m0 == true);
    }

    AuxParams aux_params = {};
    if constexpr (IS_DECODE) {
        KU_ASSERT_NPU(params.page_block_size > 1, "page_block_size must > 1");
        aux_params.page_block_size_fast_div_mod = FastDivMod(params.page_block_size);
        if (params.extra_page_block_size > 0) {
            KU_ASSERT_NPU(params.extra_page_block_size > 1, "extra_page_block_size must > 1");
            aux_params.extra_page_block_size_fast_div_mod = FastDivMod(params.extra_page_block_size);
            KU_ASSERT_NPU(params.extra_topk % B_TOPK == 0, "extra_topk must be a multiple of B_TOPK (%d)", B_TOPK);
        }
    }

    uint32_t total_q;
    if constexpr (IS_DECODE) {
        total_q = (uint32_t)params.b;
    } else {
        total_q = (uint32_t)params.s_q;
    }

    uint32_t num_ctas = min(total_q * NUM_HEAD_SPLITS, GRID_SHAPE);
    sparse_attn_fwd_kernel<Kernel<CONFIG>><<<num_ctas, 0, params.stream>>>(params, aux_params);
    KU_ACLRT_CHECK_NPU(aclrtGetLastError(aclrtLastErrLevel::ACL_RT_THREAD_LEVEL));
}


template<Config CONFIG>
void run_sparse_fwd_kernel(const ParamT<CONFIG.FWD_MODE, CONFIG.HAVE_Q_O_ROPE_AND_O_CAST>& params) {
    Kernel<CONFIG>::run(params);
}

}
