#pragma once

#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "kernels/kv_cache_format.h"
#include "kernels/sm100/helpers.h"

// Shared pieces of the sm100 decoding kernels for reading paged quantized KV caches (`kv` of format Orig plus an optional
// `extra_kv` of format Extra, see KVCacheFormat): iterating the KV blocks of a request, dequantizing a block, and the TMA tensor
// maps of the caches
namespace sm100 {

// Runs f.template operator()<F>(block_idx, is_extra_block) for the KV blocks [begin, end) of a request, F being KVCacheFormat of the
// cache the block comes from: blocks [0, num_orig_blocks) are from `kv`, the others from `extra_kv`. A block never mixes the two --
// the caller keeps the orig/extra boundary on a block boundary (head64 asserts topk % B_TOPK == 0; phase1 rounds each cache's
// block count up and masks the tail slots). When the formats differ, the blocks of each cache get their own loop, so that the code
// of the two formats is never interleaved (ptxas would otherwise merge the live ranges of both)
template<typename Orig, typename Extra, typename Fn>
CUTE_DEVICE void for_each_kv_block(int begin, int end, int num_orig_blocks, Fn &&f) {
    if constexpr (std::is_same_v<Orig, Extra>) {
        CUTE_NO_UNROLL
        for (int block_idx = begin; block_idx < end; ++block_idx) {
            f.template operator()<Orig>(block_idx, block_idx >= num_orig_blocks);
        }
    } else {
        CUTE_NO_UNROLL
        for (int block_idx = begin; block_idx < min(num_orig_blocks, end); ++block_idx) {
            f.template operator()<Orig>(block_idx, false);
        }
        CUTE_NO_UNROLL
        for (int block_idx = max(begin, num_orig_blocks); block_idx < end; ++block_idx) {
            f.template operator()<Extra>(block_idx, true);
        }
    }
}

// NUM_BYTES (4 / 8 / 16 / 32) from global memory with one load (two for 32), or zeros if !valid
template<int NUM_BYTES>
CUTE_DEVICE void ldg_or_zero(uint8_t *dst, const uint8_t *src, bool valid) {
    static_assert(NUM_BYTES == 4 || NUM_BYTES == 8 || NUM_BYTES == 16 || NUM_BYTES == 32);
    if constexpr (NUM_BYTES == 4) {
        *(uint32_t*)dst = valid ? __ldg((const uint32_t*)src) : 0u;
    } else if constexpr (NUM_BYTES == 8) {
        *(uint64_t*)dst = valid ? __ldg((const uint64_t*)src) : 0ull;
    } else {
        CUTE_UNROLL
        for (int i = 0; i < NUM_BYTES / 16; ++i) {
            *((int4*)dst + i) = valid ? __ldg((const int4*)src + i) : int4{0, 0, 0, 0};
        }
    }
}

// NUM_BYTES (4 / 8 / 16 / 32 / 64) between registers and shared memory with the widest accesses
template<int NUM_BYTES>
CUTE_DEVICE void copy_bytes(uint8_t *dst, const uint8_t *src) {
    static_assert(NUM_BYTES == 4 || NUM_BYTES == 8 || NUM_BYTES % 16 == 0);
    if constexpr (NUM_BYTES == 4) {
        *(uint32_t*)dst = *(const uint32_t*)src;
    } else if constexpr (NUM_BYTES == 8) {
        *(uint64_t*)dst = *(const uint64_t*)src;
    } else {
        CUTE_UNROLL
        for (int i = 0; i < NUM_BYTES / 16; ++i) {
            *((__int128_t*)dst + i) = *((const __int128_t*)src + i);
        }
    }
}

// Dequantizes one KV block of B_TOPK tokens (this CTA's D dims, stored in format F) with one warpgroup: reads the raw rows
// gathered by TMA and the 1 B scales from shared memory, converts in registers, and stores the bf16 result into a tile in the
// canonical SW128 K-major layout. fp8 and fp4 share the code below and differ only in the bytes of raw data per step
// (RawWord), the loading of scales, and the conversion instructions. A block is entirely fp8 or entirely fp4, so the format
// is a compile-time parameter (see for_each_kv_block).
//
// The unit of work is a step: a thread turns one RawWord (8 B of fp8 / 4 B of fp4) into one 16 B chunk of 8 bf16. 8 threads
// per token cover one swizzle-atom column (ELEMS_PER_STEP = 64 elements) per step and write the 8 chunks of one 128 B
// swizzle-atom row, so the STS.128 of a wavefront (8 lanes) is conflict-free; 16 tokens per pass, B_TOPK / 16 passes.
//  - Raw rows are RAW_TOKEN_SMEM_STRIDE apart. fp4 rows are read with LDS.32 and are padded to 8 mod 32 words (256 + 32, or
//    128 + 32 per CTA), so the 4 rows of a gather4 group start in 4 different quarters of the 32 banks and the LDS is
//    conflict-free. fp8 rows are read with LDS.64 and have no padding (a 2-way bank conflict).
//  - Scales are 1 B each (ue8m0 for fp8, e4m3 for fp4; V3.2's fp32 scales are converted by the index warp), SCALE_SMEM_STRIDE
//    apart, this CTA's NUM_SCALES first.
template<typename F, int D, int B_TOPK, int RAW_TOKEN_SMEM_STRIDE, int SCALE_SMEM_STRIDE>
struct KVBlockDequantizer {
    static constexpr int GROUP_SIZE = 8, NUM_GROUPS = 128 / GROUP_SIZE, ROWS_PER_GROUP = B_TOPK / NUM_GROUPS;
    static constexpr int ELEMS_PER_STEP = GROUP_SIZE * 8;   // One swizzle-atom column
    static constexpr int NUM_STEPS = D / ELEMS_PER_STEP;
    static constexpr int RAW_BYTES_PER_STEP = F::IS_FP4 ? ELEMS_PER_STEP / 2 : ELEMS_PER_STEP;
    static constexpr int RAW_BYTES_PER_THREAD_STEP = RAW_BYTES_PER_STEP / GROUP_SIZE;
    static constexpr int NUM_SCALES = D / F::QUANT_TILE_SIZE;                   // 7 for the 448 fp8 dims of V4, 3 for CTA1's 192 of them
    static constexpr int SCALE_LOAD_BYTES = (NUM_SCALES + 3) / 4 * 4;           // Whole words (the scale rows are padded, see KVCacheFormat)
    using RawWord = std::conditional_t<F::IS_FP4, uint32_t, uint64_t>;
    static_assert(D % ELEMS_PER_STEP == 0 && B_TOPK % NUM_GROUPS == 0 && SCALE_LOAD_BYTES <= SCALE_SMEM_STRIDE);
    static_assert(RAW_TOKEN_SMEM_STRIDE >= NUM_STEPS * RAW_BYTES_PER_STEP);   // RAW_TOKEN_SMEM_STRIDE must hold a row's data
    static_assert(!F::IS_FP4 || RAW_TOKEN_SMEM_STRIDE / 4 % 16 == 8);         // fp4 rows start in 4 different bank quarters, see above
    // The scale index of a step is a compile-time base plus this thread's part. fp8: the part is 0 or 1 (QUANT_TILE_SIZE >= 32,
    // written as a comparison so that the byte array stays in registers as a select); fp4: the part is idx_in_group / 2, and the
    // byte is extracted from a (compile-time) word of the row with one PRMT, replicated into bytes 0 and 1
    static_assert(F::IS_FP4 ? ELEMS_PER_STEP == 4 * F::QUANT_TILE_SIZE : F::QUANT_TILE_SIZE >= 32);

    int group_idx, idx_in_group;
    uint32_t raw_offset;    // Of this thread's first raw word within a block
    uint32_t dst_offset;    // Of this thread's first bf16 chunk within a tile (swizzle included: row % 8 is fixed for a thread since NUM_GROUPS % 8 == 0)
    uint32_t scale_prmt_sel11;   // fp4 only: the PRMT selector extracting this thread's scale of a step, see above

    CUTE_DEVICE explicit KVBlockDequantizer(int idx_in_warpgroup):
        group_idx(idx_in_warpgroup / GROUP_SIZE), idx_in_group(idx_in_warpgroup % GROUP_SIZE) {
        raw_offset = group_idx * RAW_TOKEN_SMEM_STRIDE + idx_in_group * RAW_BYTES_PER_THREAD_STEP;
        // The thread's first chunk in the SW128 K-major tile: 8-row swizzle atoms are stacked along the rows first, and within an
        // atom row the 16 B lane index is XORed with the row's position in the atom (the swizzle acts on byte addresses)
        const int row_in_atom = group_idx % 8;
        dst_offset = group_idx / 8 * (8 * 128) + row_in_atom * 128 + (idx_in_group ^ row_in_atom) * 16;
        scale_prmt_sel11 = (uint32_t)(idx_in_group / 2) * 0x11;
    }

    // raw / scales: the block's raw rows and scale rows in shared memory; dst: the shared memory address (cvta) of the bf16 tile.
    // before_first_store() runs once, right before the first STS, so that waiting for the tile to be free overlaps with the first
    // loads and conversions
    template<typename Fn>
    CUTE_DEVICE void run(const uint8_t *raw, const uint8_t *scales, uint32_t dst, Fn &&before_first_store) const {
        CUTE_UNROLL
        for (int local_row_idx = 0; local_row_idx < ROWS_PER_GROUP; ++local_row_idx) {
            const int row_idx = local_row_idx * NUM_GROUPS + group_idx;
            alignas(16) uint8_t scales_row[SCALE_LOAD_BYTES];
            copy_bytes<SCALE_LOAD_BYTES>(scales_row, scales + row_idx * SCALE_SMEM_STRIDE);
            const uint8_t *raw_row = raw + raw_offset + local_row_idx * NUM_GROUPS * RAW_TOKEN_SMEM_STRIDE;
            RawWord cur_data = *(const RawWord*)raw_row;
            CUTE_UNROLL
            for (int local_col_idx = 0; local_col_idx < NUM_STEPS; ++local_col_idx) {
                RawWord data = cur_data;
                if (local_col_idx + 1 < NUM_STEPS)
                    cur_data = *(const RawWord*)(raw_row + (local_col_idx + 1) * RAW_BYTES_PER_STEP);
                // Elements [local_col_idx*64 + idx_in_group*8, +8) of the row lie in this quant tile / word (see the ctor)
                const int scale_idx_base = local_col_idx * ELEMS_PER_STEP / F::QUANT_TILE_SIZE;
                ku::nvbf16x2 data_bf16[4];
                if constexpr (F::IS_FP4) {
                    fp4x8_to_bf16x2x4(data, data_bf16);
                    const uint32_t scale_word = *(const uint32_t*)(scales_row + scale_idx_base);   // Constant offset: the array stays in registers
                    ku::nvbf16x2 scale = e4m3x2_to_bf16x2(__byte_perm(scale_word, 0, scale_prmt_sel11));   // (s, s)
                    CUTE_UNROLL
                    for (int i = 0; i < 4; ++i)
                        data_bf16[i] = __hmul2(data_bf16[i], scale);   // Exact: e2m1 x e4m3 has at most 2 + 4 significant bits
                } else {
                    const int scale_idx = scale_idx_base + (F::QUANT_TILE_SIZE == 32 ? idx_in_group >= GROUP_SIZE / 2 : 0);
                    CUTE_UNROLL
                    for (int i = 0; i < 4; ++i) {
                        data_bf16[i] = fp8x2_to_bf16x2_with_scale(((ku::nve4m3x2*)&data)[i], ((__nv_fp8_e8m0*)scales_row)[scale_idx]);
                    }
                }
                if (local_row_idx == 0 && local_col_idx == 0) {
                    before_first_store();
                }
                asm volatile ("st.weak.shared::cta.b128 [%0], %1;\n"
                    :
                    : "r"(dst + dst_offset + local_row_idx * NUM_GROUPS * 128 + local_col_idx * B_TOPK * 128), "q"(*(__int128_t*)data_bf16)
                );
            }
        }
    }
};

// Host: TMA tensor map of one CTA's share of the quantized part of a paged KV cache, for gather4. dim0 = DIM0_BYTES of a token
// starting at byte `cta_byte_offset` (as uint32), dim1 = tokens at F::TMA_K_STRIDE; box = {BOX_BYTES, 1 row}. BOX_BYTES may exceed
// DIM0_BYTES, the rest of the box is out of bounds and zero-filled by TMA (the padding of the fp4 raw rows, see
// KVBlockDequantizer) -- or be smaller, reading a box out of a wider view (head128). Each gather4 writes its 4 rows BOX_BYTES
// apart in shared memory
template<typename F, int DIM0_BYTES, int BOX_BYTES>
static CUtensorMap make_kv_quant_part_tensor_map(const char *name, void *kv, int num_blocks, int64_t block_stride_bytes, int row_stride_bytes, int cta_byte_offset) {
    static_assert(DIM0_BYTES % 4 == 0 && BOX_BYTES % 16 == 0 && BOX_BYTES / 4 <= 256);
    KU_ASSERT((int64_t)kv % 16 == 0, "The base address of %s (%p) must be 16B aligned", name, kv);
    KU_ASSERT(row_stride_bytes == F::BYTES_PER_TOKEN, "%s.stride(-2) (%d) must be %d, i.e. each page block in the KV cache must be contiguous", name, row_stride_bytes, F::BYTES_PER_TOKEN);
    KU_ASSERT(block_stride_bytes % F::TMA_K_STRIDE == 0, "%s.stride(0) (%ld) must be a multiple of %d. Padding might be necessary", name, block_stride_bytes, F::TMA_K_STRIDE);
    KU_ASSERT((uint64_t)num_blocks * (uint64_t)(block_stride_bytes / F::TMA_K_STRIDE) <= INT32_MAX, "%s: too many rows for the int32 TMA coordinates", name);
    return ku::make_tensor_map(
        {(uint64_t)DIM0_BYTES / 4, (uint64_t)num_blocks * (uint64_t)(block_stride_bytes / F::TMA_K_STRIDE)},
        {(uint64_t)F::TMA_K_STRIDE},
        {BOX_BYTES / 4, 1},
        (uint8_t*)kv + cta_byte_offset,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B
    );
}

// Host: the TMA tensor map of the bf16 (RoPE) part of a paged KV cache (F::D_BF16 > 0), for gather4 into a SWIZZLE-byte swizzled tile; box = {BOX_ELEMS, 1 row}
template<typename F, int BOX_ELEMS, int SWIZZLE>
static CUtensorMap make_kv_bf16_part_tensor_map(void *kv, int num_blocks, int64_t block_stride_bytes) {
    static_assert(F::D_BF16 > 0 && (SWIZZLE == 64 || SWIZZLE == 128) && BOX_ELEMS * sizeof(bf16) <= SWIZZLE);
    return ku::make_tensor_map(
        {(uint64_t)F::D_BF16, (uint64_t)num_blocks * (uint64_t)(block_stride_bytes / F::TMA_K_STRIDE)},
        {(uint64_t)F::TMA_K_STRIDE},
        {BOX_ELEMS, 1},
        (uint8_t*)kv + (F::TMA_K_STRIDE - 2 * F::D_BF16),   // The bf16 part is the tail of the token's data
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        SWIZZLE == 64 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B : CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B
    );
}

}
