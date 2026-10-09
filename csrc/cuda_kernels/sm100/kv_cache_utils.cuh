#pragma once

#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "cuda_kernels/kv_cache_format.h"

namespace sm100 {

// Each block belongs to one cache. Separate format-specialized loops avoid overlapping live ranges.
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

// `scales` points to the scale subset in token 0; coord uses the tensor map's flattened token rows.
template<int BYTES>
CUTE_DEVICE void copy_kv_scales_async(uint8_t *dst, const uint8_t *scales, int coord, int token_stride) {
    static_assert(BYTES == 8 || BYTES == 16);
    using CopyT = std::conditional_t<BYTES == 16, uint4, uint64_t>;
    const uint8_t *src = scales + (uint64_t)(coord >= 0 ? coord : 0) * token_stride;
    cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<CopyT>::copy(*(const CopyT*)src, *(CopyT*)dst, coord >= 0);
}

// UINT32 keeps the TMA box dimension <= 256. Each gather4 writes four BOX_BYTES rows;
// bytes beyond Part's view are zero-filled. The caller aligns each gather4 destination to 128 B.
template<typename Part, int BOX_BYTES>
static CUtensorMap make_kv_quant_part_tensor_map(
    const char *name, void *kv, int num_blocks, int64_t block_stride_bytes, int row_stride_bytes,
    CUtensorMapL2promotion promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B
) {
    static constexpr int TOKEN_STRIDE = Part::BYTES_PER_TOKEN;
    static_assert(Part::RAW_TOKEN_DATA_BYTES % 4 == 0 && BOX_BYTES % 16 == 0 && BOX_BYTES / 4 <= 256);
    static_assert(TOKEN_STRIDE % 16 == 0 && Part::RAW_TOKEN_OFFSET % 16 == 0);
    KU_ASSERT((int64_t)kv % 16 == 0, "The base address of %s (%p) must be 16B aligned", name, kv);
    KU_ASSERT(row_stride_bytes == TOKEN_STRIDE, "%s.stride(-2) (%d) must be %d", name, row_stride_bytes, TOKEN_STRIDE);
    KU_ASSERT(block_stride_bytes % TOKEN_STRIDE == 0, "%s.stride(0) (%ld) must be a multiple of %d", name, block_stride_bytes, TOKEN_STRIDE);
    KU_ASSERT((uint64_t)num_blocks * (uint64_t)(block_stride_bytes / TOKEN_STRIDE) <= INT32_MAX, "%s: too many rows for the int32 TMA coordinates", name);
    return ku::make_tensor_map(
        {(uint64_t)Part::RAW_TOKEN_DATA_BYTES / 4, (uint64_t)num_blocks * (uint64_t)(block_stride_bytes / TOKEN_STRIDE)},
        {(uint64_t)TOKEN_STRIDE},
        {BOX_BYTES / 4, 1},
        (uint8_t*)kv + Part::RAW_TOKEN_OFFSET,
        CU_TENSOR_MAP_DATA_TYPE_UINT32,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        promotion
    );
}

}
