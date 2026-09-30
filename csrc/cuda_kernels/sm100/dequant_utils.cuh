#pragma once

#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "cuda_kernels/kv_cache_format.h"
#include "cuda_kernels/sm100/helpers.h"

namespace sm100 {

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

// One warpgroup dequantizes B_TOPK rows into a separate SW128 K-major bf16 tile.
// Eight lanes per token convert 64 elements per step; each STS.128 wavefront writes one row.
// FP4 LDS.32 uses 32 mod 64 B row strides. FP8 LDS.64 uses 64 mod 128 B, or 32 mod 128 B
// with row order 0,2,1,3 in each warp. Both mappings avoid raw-load bank conflicts.
// Scale offsets/strides are resolved in the constructor, allowing both CTAs to share one body.
template<typename F, int D, int B_TOPK, int RAW_TOKEN_SMEM_STRIDE, int SCALE_SMEM_STRIDE>
struct KVBlockDequantizer {
    static constexpr int GROUP_SIZE = 8, NUM_GROUPS = 128 / GROUP_SIZE, ROWS_PER_GROUP = B_TOPK / NUM_GROUPS;
    static constexpr int ELEMS_PER_STEP = GROUP_SIZE * 8;   // One swizzle-atom column
    static constexpr int NUM_STEPS = D / ELEMS_PER_STEP;
    static constexpr int RAW_BYTES_PER_STEP = F::IS_FP4 ? ELEMS_PER_STEP / 2 : ELEMS_PER_STEP;
    static constexpr int RAW_BYTES_PER_THREAD_STEP = RAW_BYTES_PER_STEP / GROUP_SIZE;
    static constexpr int NUM_SCALES = D / F::QUANT_TILE_SIZE;                   // 16 for the 512 fp8 dims of V4.1, 32 for its 512 fp4 dims
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
    uint32_t scale_offset, scale_row_step;
    uint32_t scale_prmt_sel11;   // fp4 only: the PRMT selector extracting this thread's scale of a step, see above

    CUTE_DEVICE explicit KVBlockDequantizer(int idx_in_warpgroup, int scale_stride = SCALE_SMEM_STRIDE, int scale_base_offset = 0):
        group_idx(idx_in_warpgroup / GROUP_SIZE), idx_in_group(idx_in_warpgroup % GROUP_SIZE) {
        if constexpr (!F::IS_FP4 && RAW_TOKEN_SMEM_STRIDE % 128 == 32) {
            group_idx = (group_idx & ~3) | ((group_idx & 1) << 1) | ((group_idx & 2) >> 1);
        }
        scale_offset = scale_base_offset + group_idx * scale_stride;
        scale_row_step = NUM_GROUPS * scale_stride;
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
            alignas(16) uint8_t scales_row[SCALE_LOAD_BYTES];
            copy_bytes<SCALE_LOAD_BYTES>(scales_row, scales + scale_offset + local_row_idx * scale_row_step);
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

}
