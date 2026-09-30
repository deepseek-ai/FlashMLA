#pragma once

#include <cuda_fp8.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cluster_launch.hpp>
#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "cuda_kernels/utils.h"
#include "cuda_kernels/sm100/helpers.h"
#include "cuda_kernels/sm100/common_subroutine.h"

#include "config.h"

namespace sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn {

static constexpr float MAX_INIT_VAL = -1e30;
static constexpr float O_QUANT_CLAMP_MIN_VALUE = 1e-4;

#ifdef KERUTILS_ENABLE_SM103A
static constexpr bool IS_TMEM_LD_WITH_RED_AVAILABLE = true;
#else
static constexpr bool IS_TMEM_LD_WITH_RED_AVAILABLE = false;
#endif

__device__ __forceinline__
float2 apply_rope(const float2 &x, const float &cur_cos, const float &cur_sin) {
    float2 a = {x.x, x.x};
    float2 b = {cur_cos, cur_sin};
    float2 c = {-x.y * cur_sin, +x.y * cur_cos};
    float2 y = ku::float2_fma(a, b, c);
    return y;
}

// To achieve prefill - decoding alignment while using block_size = 96, decoding must reproduce
// prefill's "natural" blocking of the concatenated indices array ([topk orig slots; extra_topk
// extra slots], tiled by B_TOPK). Since topk (e.g. 128) may not be a multiple of B_TOPK (e.g. 96),
// one KV block may straddle the orig/extra boundary, i.e. the last (partial) block of the orig KV
// and the first tokens of the extra KV are "stitched" into one block. Each KV block is thus
// classified into one of the following three categories:
enum class KVLocation {
    ORIG,           // All slots of this block come from `kv`/`indices`
    ORIG_AND_EXTRA, // This block straddles the boundary: slots < num_orig_slots come from `kv`/`indices`, the rest from `extra_kv`/`extra_indices`
    EXTRA           // All slots of this block come from `extra_kv`/`extra_indices`
};

// Block dequant of the KV producer: the lane-to-(row, chunk) mapping, the STS.128 offsets, and the per-block
// body that reads a block's scales and raw rows from its KV slot and writes the bf16 rows back into the slot.
// fp8 tokens (from `kv`) and fp4 tokens (from `extra_kv`) differ only in the number of elements per 16 B of
// raw data (CHUNK_ELEMS), the scale layout, and the conversion instructions.
//
// The unit of work is a chunk: 16 B of a raw row (one LDS.128). A quantized chunk = CHUNK_ELEMS elements =
// CHUNK_ELEMS * 2 B of the bf16 128 B swizzle-atom row of the KV slot (CHUNK_ELEMS / 8 STS.128).
// Lane-to-token mapping: 4 lanes per row, lane idx_in_row owning the idx_in_row-th quarter of the row's
// quantized chunks, and 8 consecutive rows (one "token" of the thread) per 8 consecutive lanes,
// NUM_TOKENS_PER_THREAD tokens per thread. With C = quantized chunks per lane, rows relative to the token:
//   lane          0  1  2  3 |  4  5  6  7 |  8 .. 11 | 12 .. 15 | 16 .. 19 | 20 .. 23 | 24 .. 27 | 28 .. 31
//   row           0  1  2  3 |  4  5  6  7 |  0 ..  3 |  4 ..  7 |  0 ..  3 |  4 ..  7 |  0 ..  3 |  4 ..  7
//   chunks         [0, C)    |  (rotated)  |  [C, 2C) | (rotated)| [2C, 3C) | (rotated)| [3C, 4C) | (rotated)
// The 4 lanes of one row are spread over the 4 wavefronts of an LDS/STS.128 (8 consecutive lanes each), so one
// wavefront covers 8 consecutive rows at one chunk position, which is bank-conflict-free on both sides:
//  - Raw LDS.128: the 8 rows are two 4-row gather4 groups whose rows hit two complementary sets of 4 bank
//    groups (KVFormatCta): the lanes of rows 4..7 process their chunks rotated by 4 (= 64 B,
//    see DequantRows::get_chunk_base).
//  - STS.128 into the SW128 K-major layout: a 16 B part of chunk c is written to the 16 B bank group
//    ((c % CHUNKS_PER_ATOM_ROW) * NUM_STS_PER_CHUNK + j) ^ (row % 8) of its swizzle-atom row; the 8 rows of a
//    wavefront have 8 distinct row % 8 and the same c % CHUNKS_PER_ATOM_ROW (a rotation by 4 is a multiple of it).
template<typename K, uint32_t CTA>
struct DequantRows {
    using OrigFormat = typename K::template KVFormatCta<K::MODEL_TYPE, CTA>;
    using ExtraFormat = typename K::template KVFormatCta<K::EXTRA_MODEL_TYPE, CTA>;
    static constexpr uint32_t B_TOPK = K::B_TOPK;
    static constexpr uint32_t NUM_DEQUANT_WARPS = 4, NUM_LANES_PER_ROW = 4;
    static constexpr uint32_t NUM_ROWS_PER_WAVEFRONT = 32 / NUM_LANES_PER_ROW;
    static constexpr uint32_t NUM_TOKENS_PER_THREAD = B_TOPK / (NUM_DEQUANT_WARPS * NUM_ROWS_PER_WAVEFRONT);
    static_assert(B_TOPK % (NUM_DEQUANT_WARPS * NUM_ROWS_PER_WAVEFRONT) == 0);
    static constexpr uint32_t NUM_ROWS_PER_WARP = NUM_TOKENS_PER_THREAD * NUM_ROWS_PER_WAVEFRONT;
    static constexpr uint32_t MAX_QUANT_CHUNKS_PER_LANE = std::max(OrigFormat::NUM_QUANT_CHUNKS_PER_LANE, ExtraFormat::NUM_QUANT_CHUNKS_PER_LANE);
    // Both caches hold the same dims (a V4.1 fp8 cache with a V4.1 fp4 one, or two of one format), so the
    // placement is the same for both formats
    static_assert(OrigFormat::QUANT_DIMS == ExtraFormat::QUANT_DIMS);
    static constexpr uint32_t QUANT_DIMS = OrigFormat::QUANT_DIMS;
    static_assert(MAX_QUANT_CHUNKS_PER_LANE <= 8);

    // fp8 scales: one ue8m0 per QUANT_TILE_SIZE elements, i.e. per CHUNKS_PER_TILE chunks. A lane's chunks are whole
    // tiles, so its scale bytes are the NUM_QUANT_CHUNKS_PER_LANE / CHUNKS_PER_TILE consecutive bytes at that offset
    // of the token's scale row, arranged so that chunk g uses byte g / CHUNKS_PER_TILE (a rotated lane reads the
    // rotated position or swaps the two halves of its word, see get_chunk_base).
    // fp4 scales: two e4m3 per chunk, so a lane's scales are the 2 * NUM_QUANT_CHUNKS_PER_LANE bytes at that offset of
    // the token's scale row
    template<typename F>
    static constexpr uint32_t chunks_per_tile() { return F::QUANT_TILE_SIZE / F::CHUNK_ELEMS; }
    static_assert(OrigFormat::IS_FP4 || OrigFormat::NUM_QUANT_CHUNKS_PER_LANE % chunks_per_tile<OrigFormat>() == 0);
    static_assert(ExtraFormat::IS_FP4 || ExtraFormat::NUM_QUANT_CHUNKS_PER_LANE % chunks_per_tile<ExtraFormat>() == 0);
    template<typename F>
    static constexpr uint32_t scale_bytes_per_lane() {
        return F::IS_FP4 ? F::NUM_QUANT_CHUNKS_PER_LANE * 2 : F::NUM_QUANT_CHUNKS_PER_LANE / chunks_per_tile<F>();
    }
    static constexpr uint32_t SCALE_WORDS_PER_LANE = ku::ceil_div(std::max(scale_bytes_per_lane<OrigFormat>(), scale_bytes_per_lane<ExtraFormat>()), 4u);
    static_assert(SCALE_WORDS_PER_LANE <= 2);

    // STS.128 offsets. Quantized chunk g of the lane is chunk c = get_chunk_base(g / 4) + g % 4 of the row, i.e. parts
    // [(c % CHUNKS_PER_ATOM_ROW) * NUM_STS_PER_CHUNK, +NUM_STS_PER_CHUNK) of atom row c / CHUNKS_PER_ATOM_ROW (an atom
    // row holds 64 dims: 4 fp8 chunks or 2 fp4 chunks). A lane owns QUANT_DIMS / 4 dims, which are whole atom rows
    // (QUANT_DIMS % 256 == 0): the part index of (g, j) is the same for every lane, so sts_offsets[p] is the swizzled
    // offset of part p (0..7) of the first atom row of the lane's first token and the chunk's atom column is added;
    // both formats share the table since it is indexed by part. The other tokens are multiples of 8 rows further
    template<typename F>
    static constexpr uint32_t chunks_per_atom_row() { return 128 / (F::CHUNK_ELEMS * 2); }
    template<typename F>
    static constexpr uint32_t num_sts_per_chunk() { return F::CHUNK_ELEMS * 2 / 16; }
    static_assert(QUANT_DIMS % 256 == 0);
    static constexpr uint32_t NUM_STS_OFFSETS = 8;
    static constexpr uint32_t STS_ATOM_COL_STRIDE_BYTES = B_TOPK * 128;
    static constexpr uint32_t STS_TOKEN_STRIDE_BYTES = NUM_ROWS_PER_WAVEFRONT * 128;

    typename K::SharedMemoryPlan &smem;
    uint32_t local_warp_idx, row_in_wavefront, idx_in_row, chunk_rotation;
    uint32_t sts_offsets[NUM_STS_OFFSETS];
    uint32_t orig_scale_offset, extra_scale_offset, scale_token_stride;

    __device__ __forceinline__ DequantRows(typename K::SharedMemoryPlan &smem_, uint32_t dq_warp_idx, uint32_t lane_idx, uint32_t cta_idx)
        : smem(smem_), local_warp_idx(dq_warp_idx),
          row_in_wavefront(lane_idx % NUM_ROWS_PER_WAVEFRONT), idx_in_row(lane_idx / NUM_ROWS_PER_WAVEFRONT),
          chunk_rotation(lane_idx % NUM_ROWS_PER_WAVEFRONT / 4 * 4) {
        using namespace cute;
        Tensor sKV = make_tensor(make_smem_ptr(smem.kv_slots[0]), ku::make_umma_canonical_k_major_layout<B_TOPK, K::D_QK / K::CLUSTER_SIZE, 128>());
        const uint32_t row = get_row_idx(0);
        if constexpr (K::IS_2CTA) {
            // Resolve the asymmetric scale transport once, outside the block loop, so both CTAs
            // execute the same dequantization body without duplicating its instruction cache footprint.
            auto init_scale_offset = [&]<typename F>() {
                if (cta_idx == 0) {
                    return K::SCALE_STAGE_OFFSET + row * K::SCALE_STAGE_ROW_BYTES;
                }
                using Cta1Format = typename K::template KVFormatCta<F::Base::MODEL_TYPE, 1>;
                return row / 4 * K::RAW_KV_GROUP_BYTES + row % 4 * F::RAW_TOKEN_SMEM_STRIDE + Cta1Format::SCALE_OFFSET;
            };
            orig_scale_offset = init_scale_offset.template operator()<OrigFormat>();
            extra_scale_offset = init_scale_offset.template operator()<ExtraFormat>();
            scale_token_stride = NUM_ROWS_PER_WAVEFRONT * (cta_idx == 0 ? K::SCALE_STAGE_ROW_BYTES : K::RAW_KV_GROUP_BYTES / 4);
        }
        // In the SW128 K-major layout a row is 128 B of every swizzle-atom column, and its 16 B part p lies at part index
        // p ^ (row % 8). The layout gives the row's first part; the other entries are derived from it with 32-bit
        // arithmetic. Deriving every entry from the layout takes 64-bit arithmetic per entry, and with all entries live at
        // once ptxas spills the table
        const uint32_t row_swizzle = row % 8;
        const uint32_t row_start = (uint32_t)((&sKV(row, 0) - smem.kv_slots[0]) * sizeof(bf16)) - row_swizzle * 16;
        // Offset of the (part_idx % 8)-th part of atom column part_idx / 8 of the row
        auto part_offset = [&](uint32_t part_idx) {
            return row_start + ((part_idx % 8 ^ row_swizzle) * 16) + part_idx / 8 * STS_ATOM_COL_STRIDE_BYTES;
        };
        CUTE_UNROLL
        for (uint32_t p = 0; p < 8; ++p) {
            sts_offsets[p] = part_offset(p);
        }
    }

    __device__ __forceinline__ uint32_t get_row_idx(uint32_t token_idx) const {
        return local_warp_idx * NUM_ROWS_PER_WARP + token_idx * NUM_ROWS_PER_WAVEFRONT + row_in_wavefront;
    }
    // Quantized chunk g of the lane is chunk get_chunk_base(g / 4) + g % 4 of the row, so an address is a per-lane base
    // plus an immediate. The rows 4..7 (chunk_rotation = 4) rotate the order by 4 chunks: a rotation of the row's chunk
    // indices when a lane owns at most 4 chunks, a swap of its two halves (g ^ 4) when it owns 8. For rows 0..3
    // (chunk_rotation = 0) the lane's chunks are [idx_in_row * N, +N)
    template<typename F>
    __device__ __forceinline__ uint32_t get_chunk_base(uint32_t half) const {
        if constexpr (F::NUM_QUANT_CHUNKS_PER_LANE <= 4) {
            return (idx_in_row * F::NUM_QUANT_CHUNKS_PER_LANE + chunk_rotation) % F::NUM_QUANT_CHUNKS_PER_ROW;
        } else {
            return idx_in_row * F::NUM_QUANT_CHUNKS_PER_LANE + (half * 4 ^ chunk_rotation);
        }
    }
    // Offset of part j of quantized chunk g of the lane's first token in the slot
    template<typename F>
    __device__ __forceinline__ uint32_t quant_sts_offset(uint32_t g, uint32_t j) const {
        // chunk_base is a multiple of 4 (a multiple of CHUNKS_PER_ATOM_ROW), so the part index is selected at compile time
        static constexpr uint32_t CHUNKS_PER_ATOM_ROW = chunks_per_atom_row<F>();
        uint32_t atom_col = get_chunk_base<F>(g / 4) / CHUNKS_PER_ATOM_ROW + g % 4 / CHUNKS_PER_ATOM_ROW;
        return sts_offsets[g % CHUNKS_PER_ATOM_ROW * num_sts_per_chunk<F>() + j] + atom_col * STS_ATOM_COL_STRIDE_BYTES;
    }

    // Dequantize this thread's rows of one block whose raw data and scales are in the slot; the first
    // NUM_FP8_TOKENS of its tokens are fp8, the rest fp4. Reads the scales (staged separately on CTA0 of h128,
    // interleaved with raw data otherwise), releases the indices buffer, then reads raw data via LDS.128.
    // A named barrier ensures every lane has read its input before bf16 write-back starts. Per quantized
    // chunk: fp8: 8x F2FP (e4m3x2 x ue8m0 -> bf16x2); fp4: 16x F2FP (e2m1x2 -> bf16x2) + 16x HMUL2.BF16 (x the
    // bf16 of the e4m3 scale, exact: the product has at most 2 + 4 significant bits). Then the STS.128s, the fence,
    // and the kv_slot_full arrive. The caller passes the block's slot and indices buffer and advances its own ring state
    template<uint32_t NUM_FP8_TOKENS>
    __device__ __forceinline__ void body(uint32_t kv_slot_idx, uint32_t indices_buf_idx) const {
        using cutlass::arch::NamedBarrier;
        auto for_each_token = [&](auto callable) {
            cute::for_each(cute::make_int_sequence<NUM_TOKENS_PER_THREAD>{}, [&](auto token_idx) {
                if constexpr (token_idx < NUM_FP8_TOKENS) {
                    callable.template operator()<OrigFormat>(token_idx);
                } else {
                    callable.template operator()<ExtraFormat>(token_idx);
                }
            });
        };
        uint8_t *slot_base = (uint8_t*)smem.kv_slots[kv_slot_idx];

        uint32_t cached_scales[NUM_TOKENS_PER_THREAD][SCALE_WORDS_PER_LANE];
        for_each_token([&]<typename F>(uint32_t token_idx) {
            static constexpr uint32_t SCALE_BYTES_PER_LANE = scale_bytes_per_lane<F>();
            const uint8_t *scale_src = [&]() {
                if constexpr (K::IS_2CTA) {
                    return slot_base + (F::IS_FP4 ? extra_scale_offset : orig_scale_offset) + token_idx * scale_token_stride;
                } else {
                    uint32_t row = get_row_idx(token_idx);
                    return slot_base + row / 4 * K::RAW_KV_GROUP_BYTES + row % 4 * F::RAW_TOKEN_SMEM_STRIDE + F::SCALE_OFFSET;
                }
            }();
            if constexpr (F::IS_FP4) {
                const uint8_t *src = scale_src + get_chunk_base<F>(0) * 2;
                if constexpr (SCALE_BYTES_PER_LANE == 8) {
                    *(uint64_t*)cached_scales[token_idx] = *(const uint64_t*)src;
                } else {
                    static_assert(SCALE_BYTES_PER_LANE == 4);
                    cached_scales[token_idx][0] = *(const uint32_t*)src;
                }
            } else {
                if constexpr (F::NUM_QUANT_CHUNKS_PER_LANE == 8) {
                    // A rotated lane swaps the two halves of its chunks, so also the two halves of its scale bytes
                    if constexpr (SCALE_BYTES_PER_LANE == 4) {
                        uint32_t scales = *(uint32_t*)(scale_src + idx_in_row * 4);
                        cached_scales[token_idx][0] = chunk_rotation ? __byte_perm(scales, scales, 0x1032) : scales;
                    } else {
                        static_assert(SCALE_BYTES_PER_LANE == 2);
                        uint32_t scales = *(uint16_t*)(scale_src + idx_in_row * 2);
                        cached_scales[token_idx][0] = chunk_rotation ? __byte_perm(scales, scales, 0x0001) : scales;
                    }
                } else {
                    // A rotated lane's chunks are the next lane's (modulo the row), so are its scale bytes
                    static_assert(F::NUM_QUANT_CHUNKS_PER_LANE == 4 && (SCALE_BYTES_PER_LANE == 2 || SCALE_BYTES_PER_LANE == 1));
                    const uint8_t *src = scale_src + get_chunk_base<F>(0) / chunks_per_tile<F>();
                    cached_scales[token_idx][0] = SCALE_BYTES_PER_LANE == 2 ? *(uint16_t*)src : *src;
                }
            }
        });
        smem.bar_indices_empty[indices_buf_idx].arrive();

        // A format with fewer quantized chunks per lane uses the first entries
        uint32_t cached_input[NUM_TOKENS_PER_THREAD][MAX_QUANT_CHUNKS_PER_LANE][4];
        for_each_token([&]<typename F>(uint32_t token_idx) {
            uint32_t row = get_row_idx(token_idx);
            uint8_t *row_base = slot_base + row / 4 * K::RAW_KV_GROUP_BYTES + row % 4 * F::RAW_TOKEN_SMEM_STRIDE;
            CUTE_UNROLL
            for (uint32_t g = 0; g < F::NUM_QUANT_CHUNKS_PER_LANE; ++g) {
                *(__int128_t*)(cached_input[token_idx][g]) = ku::ld_shared(row_base + (get_chunk_base<F>(g / 4) + g % 4) * 16);
            }
        });
        NamedBarrier::arrive_and_wait(128, K::barrier_ids::WG1_DEQUANT_SYNC);  // Make sure everyone has finished reading

        for_each_token([&]<typename F>(uint32_t token_idx) {
            static constexpr uint32_t NUM_STS_PER_CHUNK = num_sts_per_chunk<F>();
            uint8_t *token_base = slot_base + token_idx * STS_TOKEN_STRIDE_BYTES;
            ku::nvbf16x2 fp4_scales[2];   // fp4: the two scales of each of the current pair of chunks, as bf16
            // Each 16 B store is produced as four independent 32-bit values and stored as a uint4
            // aggregate, not through ku::st_shared's b128 form. That form passes the 16 B as one
            // 128-bit asm operand ("q"), so ptxas must place the four values in an aligned register
            // quad and, with several store chains in progress, gathers them with MOVs; from four
            // independent sources it emits the same single STS.128 with no MOVs
            static_assert(!F::IS_FP4 || (F::QUANT_TILE_SIZE / 2) % 4 == 0,
                          "a store quad must fall entirely inside one quantization tile");
            CUTE_UNROLL
            for (uint32_t g = 0; g < F::NUM_QUANT_CHUNKS_PER_LANE; ++g) {
                ku::nvbf16x2 scale_lo, scale_hi;
                __nv_fp8_e8m0 fp8_scale;
                if constexpr (F::IS_FP4) {
                    if (g % 2 == 0) {
                        fp8x4_to_bf16x2x2(cached_scales[token_idx][g / 2], fp4_scales);
                    }
                    scale_lo = __low2bfloat162(fp4_scales[g % 2]);
                    scale_hi = __high2bfloat162(fp4_scales[g % 2]);
                } else {
                    fp8_scale = ((__nv_fp8_e8m0*)cached_scales[token_idx])[g / chunks_per_tile<F>()];
                }
                CUTE_UNROLL
                for (uint32_t j = 0; j < NUM_STS_PER_CHUNK; ++j) {
                    ku::nvbf16x2 quad[4];
                    if constexpr (F::IS_FP4) {
                        fp4x8_to_bf16x2x4(cached_input[token_idx][g][j], quad);
                        ku::nvbf16x2 sc = j * 4 < F::QUANT_TILE_SIZE / 2 ? scale_lo : scale_hi;
                        CUTE_UNROLL
                        for (uint32_t k = 0; k < 4; ++k) quad[k] = __hmul2(quad[k], sc);
                    } else {
                        CUTE_UNROLL
                        for (uint32_t k = 0; k < 4; ++k) {
                            quad[k] = fp8x2_to_bf16x2_with_scale(((ku::nve4m3x2*)cached_input[token_idx][g])[j * 4 + k], fp8_scale);
                        }
                    }
                    *(uint4*)(token_base + quant_sts_offset<F>(g, j)) = *(const uint4*)quad;
                }
            }
        });

        cutlass::arch::fence_view_async_shared();
        K::arrive_on_cta0_barrier(smem.bar_kv_slot_full[kv_slot_idx]);
    }

    // Dispatch on this thread's number of fp8 tokens (warp-uniform since topk % 8 == 0) so a token's
    // conversion code is fixed at compile time by its format. When both caches have the same format
    // (both caches fp8, or one cache) every token takes the same code, so one body serves every value
    __device__ __forceinline__ void run_block(uint32_t num_fp8_tokens, uint32_t kv_slot_idx, uint32_t indices_buf_idx) const {
        if constexpr (std::is_same_v<OrigFormat, ExtraFormat>) {
            body<NUM_TOKENS_PER_THREAD>(kv_slot_idx, indices_buf_idx);
        } else {
            [&]<uint32_t... Ks>(std::integer_sequence<uint32_t, Ks...>) {
                ((num_fp8_tokens == Ks ? body<Ks>(kv_slot_idx, indices_buf_idx) : void()), ...);
            }(std::make_integer_sequence<uint32_t, NUM_TOKENS_PER_THREAD + 1>{});
        }
    }
};

// Gather front of one KV block (warp 11, see there): one gather4 per 4-row group of the block's raw rows into the KV
// slot, in bursts of 8 (the 8 int4 coordinates of a burst take 32 registers; all groups at once would not fit the
// 72-register allocation of that warpgroup). A block mixes the two caches (`mixed_block`) only when topk % B_TOPK != 0:
// its groups from `first_extra_group` on read the extra cache (`map_from_split`), the ones before it the original one
// (`map_before_split`). Every other block reads one cache: the MIXED_BLOCK = false instantiation takes that cache's map as
// `map_before_split` and has no per-group map select. The maps are byte offsets into tma_params: a select between two
// pointers into the kernel parameters whose result is an operand of the gather asm makes cicc (CUDA 13.1) crash, in a
// lambda as well as in a function
template<bool MIXED_BLOCK, uint32_t NUM_GATHER_GROUPS, uint32_t RAW_KV_GROUP_BYTES>
__device__ __forceinline__ void issue_raw_gathers(
    const int4 *coords, const uint8_t *maps_base, uint32_t map_before_split, uint32_t map_from_split, uint32_t first_extra_group,
    transac_bar_t &bar_raw_kv_full, uint8_t *slot_base
) {
    static_assert(NUM_GATHER_GROUPS % 8 == 0);
    // One burst of code, run NUM_GATHER_GROUPS / 8 times. The front runs once per block, and the per-block code of the h128
    // kernels is close to the instruction cache's capacity; unrolled bursts would be NUM_GATHER_GROUPS / 8 times the code
    // (h128: 3 x 8 gather4s, at two call sites)
    CUTE_NO_UNROLL
    for (uint32_t burst = 0; burst < NUM_GATHER_GROUPS / 8; ++burst) {
        int4 burst_coords[8];
        CUTE_UNROLL
        for (uint32_t i = 0; i < 8; ++i) {
            burst_coords[i] = coords[burst * 8 + i];
        }
        CUTE_UNROLL
        for (uint32_t i = 0; i < 8; ++i) {
            const uint32_t g = burst * 8 + i;
            uint32_t map_offset = map_before_split;
            if constexpr (MIXED_BLOCK) {
                map_offset = g < first_extra_group ? map_before_split : map_from_split;
            }
            ku::tma_gather4(
                maps_base + map_offset,
                bar_raw_kv_full,
                slot_base + g * RAW_KV_GROUP_BYTES,
                0,
                burst_coords[i],
                (int64_t)TMA::CacheHintSm90::EVICT_FIRST
            );
        }
    }
}

template<Config CONFIG>
__device__ __forceinline__
void Kernel<CONFIG>::devfunc(const Params &params, const TMAParams &tma_params, const AuxParams &aux_params) {
#ifdef KERUTILS_ENABLE_SM100A
    const uint32_t cta_idx = IS_2CTA ? blockIdx.x % 2 : 0;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t warpgroup_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    const uint32_t idx_in_warpgroup = threadIdx.x % 128;
    const uint32_t lane_idx = threadIdx.x % 32;

    extern __shared__ char smem_buf[];
    SharedMemoryPlan &smem = *reinterpret_cast<SharedMemoryPlan*>(smem_buf);

    if constexpr (IS_2CTA) {
        ku::barrier_cluster_arrive_relaxed();
        ku::barrier_cluster_wait_acquire();
    }

    if (warp_idx == 0 && elect_one_sync()) {
        // Prefetch TMA descriptors
        if constexpr (!IS_DECODE) {
            cute::prefetch_tma_descriptor(&tma_params.tensor_map_kv);
        }
    } else if (warp_idx == 1 && elect_one_sync()) {
        // Init barriers
        CUTE_UNROLL
        for (uint32_t i = 0; i < NUM_KV_SLOTS; ++i) {
            // bar_kv_slot_full:
            //   Prefill: 1 arrive (arrive_and_expect_tx from the MMA warp) + TMA transactions of the whole KV block
            //   Decode: 128 arrives (the dequant warpgroup) from each CTA + 1 arrive (the MMA warp) from CTA0; no TMA transactions
            smem.bar_kv_slot_full[i].init(IS_DECODE ? 128*CLUSTER_SIZE + 1 : 1);  // bar_kv_full: Every CTA -> CTA0
            smem.bar_kv_slot_empty[i].init(1); // bar_kv_empty: CTA0 -> Every CTA
        }
        CUTE_UNROLL
        for (uint32_t i = 0; i < NUM_INDICES_BUFS; ++i) {
            smem.bar_indices_full[i].init(32);  // CTA-local
            smem.bar_indices_empty[i].init(IS_DECODE ? 128 + 128 + 32 : 128);  // CTA-local: WG1 + WG3 + warp 11
        }
        CUTE_UNROLL
        for (uint32_t i = 0; i < NUM_P_BUFS; ++i) {
            smem.bar_tP_full[i].init(1);       // CTA0 -> Every CTA
            if constexpr (NEED_TP_EMPTY_BAR) {
                smem.bar_tP_empty[i].init(128*CLUSTER_SIZE);    // Every CTA -> CTA0
            }
        }
        if constexpr (ENABLE_Q_NORM) {
            smem.bar_q_sqr_sum_full.init(128); // CTA-local: WG0 has written the job's sums of squares to q_sqr_sum_buf
        }
        smem.bar_clc_full.init(1);      // CTA0 -> Every CTA
        smem.bar_clc_empty.init(cta_idx == 1 ? 1 : NUM_WORKING_THREADS);   // Every CTA -> CTA0
        smem.bar_tQ_full.init(128*CLUSTER_SIZE);    // Every CTA -> CTA0
        smem.bar_tQ_empty.init(1+128);      // CTA0 -> Every CTA (arrive by MMA thread), as well as CTA-local (arrived by exp warpgroup)
        smem.bar_tO_full.init(1);       // CTA0 -> Every CTA
        smem.bar_tO_empty.init(128*CLUSTER_SIZE);   // Every CTA -> CTA0
        smem.bar_SO_full.init(128*CLUSTER_SIZE);    // Every CTA -> CTA0
        smem.bar_SO_empty.init(1);      // CTA0 -> Every CTA
        smem.bar_li_mi_full.init(128);  // CTA-local
        smem.bar_li_mi_empty.init(128); // CTA-local
        if constexpr (IS_DECODE) {
            CUTE_UNROLL
            for (uint32_t i = 0; i < NUM_RAW_BARS; ++i) {
                smem.bar_raw_kv_full[i].init(IS_2CTA && cta_idx == 0 ? 33 : 1);  // TMA leader + CTA0's 32 deferred cp.async arrivals
                smem.bar_wg1_past_raw[i].init(4);      // CTA-local: one elected arrive per dequant warp
            }
        }
        fence_barrier_init();
    } else if (warp_idx == 3) {
        // Allocate TMEM
        AllocatorT().allocate(512, smem.tmem_start_addr.data());
        AllocatorT().release_allocation_lock();
        KU_TRAP_ONLY_DEVICE_ASSERT(smem.tmem_start_addr.data()[0] == 0);
    }

    if constexpr (IS_2CTA) {
        ku::barrier_cluster_arrive_relaxed();
        ku::barrier_cluster_wait_acquire();
    } else {
        __syncthreads();
    }

    struct OuterloopArgs {
        bool is_valid;
        uint32_t s_q_idx;
        uint32_t job_idx_mod_2;
        uint32_t topk_length;
        uint32_t num_kv_blocks;
        // Decoding only:
        uint32_t extra_topk_length;     // Number of valid extra-topk entries of the current request
        uint32_t num_orig_slots;        // Number of slots occupied by the orig KV in the unified slot space, i.e. slots < num_orig_slots come from `kv`/`indices` (will be set to -1 if we don't have so many valid indices) while the others come from `extra_kv`/`extra_indices`. 0xFFFFFFFF when there is no extra KV (so that every slot belongs to the orig KV)
    };

    auto _make_outer_loop_args = [&](uint32_t job_idx_mod_2, uint32_t cta_x_idx) -> OuterloopArgs {
        uint32_t s_q_idx = cta_x_idx / CLUSTER_SIZE;
        if constexpr (IS_DECODE) {
            uint32_t topk_length = params.topk_length ? (uint32_t)__ldg(params.topk_length + s_q_idx) : (uint32_t)params.topk;
            uint32_t extra_topk_length = params.extra_topk_length ? (uint32_t)__ldg(params.extra_topk_length + s_q_idx) : (uint32_t)params.extra_topk;
            bool have_extra_kv = params.extra_topk > 0;
            uint32_t num_orig_slots, num_kv_blocks;
            if (have_extra_kv) {
                num_orig_slots = (uint32_t)params.topk;
                num_kv_blocks = ku::ceil_div(num_orig_slots + extra_topk_length, (uint32_t)B_TOPK); // The orig slots always use the full topk, so round topk + extra_topk_length up to whole blocks
            } else {
                num_orig_slots = 0xFFFFFFFFu;
                num_kv_blocks = ku::ceil_div(topk_length, (uint32_t)B_TOPK);
            }
            num_kv_blocks = std::max(num_kv_blocks, 1u);
            return {
                true,
                s_q_idx,
                job_idx_mod_2,
                topk_length,
                num_kv_blocks,
                extra_topk_length,
                num_orig_slots
            };
        } else {
            uint32_t topk_length = params.topk_length ? __ldg(params.topk_length + s_q_idx) : params.topk;
            uint32_t num_kv_blocks = std::max(ku::ceil_div(topk_length, (uint32_t)B_TOPK), 1u);
            return {
                true,
                s_q_idx,
                job_idx_mod_2,
                topk_length,
                num_kv_blocks
            };
        }
    };

    // Decode only: how block kv_block_idx of a job splits between the two KV caches. Its first
    // num_orig_rows rows come from `kv` (fp8), the rest from `extra_kv`
    auto get_num_orig_rows = [&](const OuterloopArgs &job, uint32_t kv_block_idx) -> uint32_t {
        uint32_t block_start = kv_block_idx * B_TOPK;
        return job.num_orig_slots >= block_start + B_TOPK ? B_TOPK
             : job.num_orig_slots <= block_start ? 0
             : job.num_orig_slots - block_start;
    };

    auto get_first_job = [&]() -> OuterloopArgs {
        return _make_outer_loop_args(0, blockIdx.x);
    };
    auto get_next_job = [&](const OuterloopArgs &cur_args) -> OuterloopArgs {
        smem.bar_clc_full.wait(cur_args.job_idx_mod_2);
        ku::CLCResult next_cta0_idx = ku::get_clc_query_response<true>(smem.clc_response_obj);
        arrive_on_cta0_barrier(smem.bar_clc_empty);

        if (!next_cta0_idx.is_valid) {
            return OuterloopArgs {false};
        } else {
            return _make_outer_loop_args(
                cur_args.job_idx_mod_2^1,
                next_cta0_idx.x
            );
        }
    };

    if (warpgroup_idx == 0) {
        /*
        Q fetching & Epilogue warpgroup

        The timeline of this warpgroup is as follows:
        Q0 Q1 O0 Q2 O1 Q3 O2 Q4 O3 ... Qn O(n-1) On

        Where
        - Qi means loading the Q of the i-th request, computing the sum of squares of each head on the fly,
          performing RoPE transformation, then writing Q to TMEM
        - Oi means reading the O of the i-th request from TMEM, performing RoPE transformation,
          quantizing to FP8, and writing back to global memory

        About cached_cos and cached_sin:
        - D_ROPE is always 64, so the i-th lane caches the i-th cos and sin dimension of the
          corresponding position
        - We always cache the cos and sin of the i-th request and the (i-1)-th request, so cached_cos
          and cached_sin have two slots
        - Every time a new Q is loaded, cached_cos/sin[1] (the first slot) is assigned to the zeroth slot
          (and "sin" is negated to prepare for the conjugate RoPE of O),
          and the cos/sin of the new Q is saved to the 1st slot
        - This way, O RoPE only needs to read from the 0th slot (except for the last O)
        */
        cutlass::arch::warpgroup_reg_alloc<WG0_REGS>();
        // The thread's TMEM lane is 32 * warp + lane; the 128 lanes are FOLD_FACTOR groups of H_Q_PER_CTA rows (config.h). Warp w
        // belongs to lane group w / (H_Q_PER_CTA / 32) and holds the K-view of rail w / 2 of Q (groups 0 | 1 = rails 0 | 1),
        // and its row is head idx % H_Q_PER_CTA. Both are derived from warp_idx, not from idx_in_warpgroup, so the compiler
        // sees warp-uniform values
        static_assert(NUM_MRGEMM_RAILS == 2, "rail = warp_idx / 2 below");
        const uint32_t rail = warp_idx / 2;

        #pragma nv_diag_suppress 549    // Uninitialized variable. The following two variables are indeed initialized but the compiler is too dumb to release that
        float cached_cos[2], cached_sin[2];
        auto shift_cached_cos_and_cached_sin = [&]() {
            cached_cos[0] = cached_cos[1];
            cached_sin[0] = -cached_sin[1];
        };

        auto load_q_and_save_to_tmem = [&](const OuterloopArgs &cur_job) {
            // A thread holds the K-view of one rail for its head: D_QK / NUM_MRGEMM_RAILS bf16
            static constexpr uint32_t NUM_CACHED_Q_BF16_PER_THREAD = D_QK / NUM_MRGEMM_RAILS;
            static constexpr uint32_t NUM_BF16_PER_LOAD = 256 / 16; // LDG 256
            bf16 cached_q[NUM_CACHED_Q_BF16_PER_THREAD];
            float q_sqr_sum = 0.0f;

            uint32_t cur_q_position = __ldg(params.token_positions + cur_job.s_q_idx);
            cached_cos[1] = __ldg(params.cos_sin_cache + cur_q_position * D_ROPE + lane_idx);
            cached_sin[1] = __ldg(params.cos_sin_cache + cur_q_position * D_ROPE + D_ROPE / 2 + lane_idx);

            bf16 *q_token_base = params.q + (uint64_t)cur_job.s_q_idx * params.stride_q_s_q;

            static constexpr uint32_t TILE_SIZE = 64;
            // GROUP_Q_LOADS: the 16 loads of a thread (4 warps x 16 x LDG.256 = 64 KB per CTA) are issued one 16 KB tile at a
            // time. More than ~24 KB of global loads issued but not completed on an SM stalls every shared-memory instruction
            // of the SM (STS, LDS, proxy fence, mbarrier arrive) until the data has returned. This load follows get_next_job,
            // while WG3 stores the S of the next job's second block (in decode also while WG1 writes the dequantized KV
            // block), so a 64 KB burst delays that S and the SV that waits on it by ~2000 cycles in prefill. Each tile's
            // addresses carry one word of every load of the previous tile, masked by a run-time zero, so ptxas issues a tile
            // only after the previous tile has arrived. Every load contributes a word: given the word of one load only, ptxas
            // hoists that load and merges the other loads into one burst. The serial tiles cost ~3 load latencies per job,
            // inside this warpgroup's ~10k cycles of slack
            uint32_t q_load_dep = 0;
            const uint32_t q_zero = cur_job.topk_length >> 31;   // topk_length < 2^31, so 0 at run time; a value ptxas cannot fold away
            CUTE_UNROLL
            for (uint32_t local_tile_idx = 3; local_tile_idx != 0xFFFFFFFF; --local_tile_idx) {
                // Rail j holds the 64-dim tiles j, j+2, j+4, j+6 (1 CTA) or the CTA's half of the dims in order (2 CTAs)
                uint32_t tile_idx = CLUSTER_SIZE == 1 ? local_tile_idx * 2 + rail : rail * 4 + local_tile_idx;
                CUTE_UNROLL
                for (uint32_t i = 0; i < TILE_SIZE / NUM_BF16_PER_LOAD; ++i) {
                    uint32_t h_q_idx = cta_idx * H_Q_PER_CTA + idx_in_warpgroup % H_Q_PER_CTA;
                    uint32_t d_q_idx = tile_idx * TILE_SIZE + i * NUM_BF16_PER_LOAD;
                    KU_LDG_256(
                        q_token_base + h_q_idx * NUM_BF16_PER_LOAD + d_q_idx * H_Q + (GROUP_Q_LOADS ? q_load_dep : 0u), // always 0
                        cached_q + local_tile_idx * TILE_SIZE + i * NUM_BF16_PER_LOAD,
                        ".nc", "no_allocate", "evict_first", "256B"
                    );
                }
                if constexpr (GROUP_Q_LOADS) {
                    uint32_t tile_words = 0;
                    CUTE_UNROLL
                    for (uint32_t i = 0; i < TILE_SIZE / NUM_BF16_PER_LOAD; ++i) {
                        tile_words ^= *(const uint32_t*)(cached_q + local_tile_idx * TILE_SIZE + i * NUM_BF16_PER_LOAD + NUM_BF16_PER_LOAD - 2);
                    }
                    q_load_dep = tile_words & q_zero;
                }
                // Perform RoPE. This tile's sum of squares is taken here, before the rotation, from the values the
                // loop converts anyway; the other tiles' sums are taken after the TMEM store (below)
                if (local_tile_idx == 3 && tile_idx == D_VO / TILE_SIZE - 1) {
                    float2 cur_q_sqr_sum = {0.0f, 0.0f};
                    CUTE_UNROLL
                    for (uint32_t j = 0; j < TILE_SIZE; j += 2) {
                        float2 x = __bfloat1622float2(*(nv_bfloat162*)(cached_q+local_tile_idx*TILE_SIZE+j));
                        if constexpr (ENABLE_Q_NORM) {
                            cur_q_sqr_sum = ku::float2_fma(x, x, cur_q_sqr_sum);
                        }
                        float cur_cos = __shfl_sync(0xFFFFFFFF, cached_cos[1], j/2);
                        float cur_sin = __shfl_sync(0xFFFFFFFF, cached_sin[1], j/2);
                        float2 y = apply_rope(x, cur_cos, cur_sin);
                        *(nv_bfloat162*)(cached_q+local_tile_idx*TILE_SIZE+j) = nv_bfloat162{__float2bfloat16_rn(y.x), __float2bfloat16_rn(y.y)};
                    }
                    q_sqr_sum += cur_q_sqr_sum.x + cur_q_sqr_sum.y;
                } else if constexpr (ENABLE_Q_NORM && !COMPACT_PER_JOB_CODE) {
                    // This tile's sum of squares, one fma.rn.f32.bf16 per element, NUM_CACHED_Q_BF16_PER_THREAD - TILE_SIZE = 192 instructions, With
                    // COMPACT_PER_JOB_CODE the sum is taken from TMEM after the store instead to save icache
                    CUTE_UNROLL
                    for (uint32_t i = 0; i < TILE_SIZE; ++i)
                        asm volatile ("fma.rn.f32.bf16 %0, %1, %1, %0;\n" : "+f"(q_sqr_sum) : "h"(*(uint16_t*)(cached_q+local_tile_idx*TILE_SIZE+i)));
                }
            }

            smem.bar_tQ_empty.wait(cur_job.job_idx_mod_2^1);
            ku::tcgen05_after_thread_sync();
            
            static constexpr uint32_t NUM_CACHED_UINT32 = NUM_CACHED_Q_BF16_PER_THREAD/2;
            ku::tmem_st_32dp32bNx<NUM_CACHED_UINT32/2>(tmem_cols::Q, cached_q);
            ku::tmem_st_32dp32bNx<NUM_CACHED_UINT32/2>(tmem_cols::Q+NUM_CACHED_UINT32/2, cached_q+NUM_CACHED_UINT32/2*2);   // Two stores: one store of all NUM_CACHED_UINT32 words does not compile (insufficient registers)
            cutlass::arch::fence_view_async_tmem_store();

            ku::tcgen05_before_thread_sync();
            arrive_on_cta0_barrier(smem.bar_tQ_full);

            if constexpr (ENABLE_Q_NORM && COMPACT_PER_JOB_CODE) {
                // Sum of squares of the tiles other than the RoPE one, read back from TMEM one tile per iteration, in the
                // order the tiles were loaded (3 unless it is the RoPE tile, then 2, 1, 0), each element added with
                // fma.rn.f32.bf16: the same operations in the same order as the in-register sum of the load loop, so the
                // same fp32 result, in one tile of code
                // Local tile 3 is the RoPE tile for the warps whose tile_idx of it is the last one (the formula of the load loop)
                const uint32_t tile_idx_of_local_3 = CLUSTER_SIZE == 1 ? 3 * 2 + rail : rail * 4 + 3;
                const uint32_t first_tile = tile_idx_of_local_3 == D_VO / TILE_SIZE - 1 ? 2 : 3;
                CUTE_NO_UNROLL
                for (uint32_t local_tile_idx = first_tile; local_tile_idx != 0xFFFFFFFF; --local_tile_idx) {
                    uint32_t q_words[TILE_SIZE / 2];
                    ku::tmem_ld_32dp32bNx<TILE_SIZE / 2>(tmem_cols::Q + local_tile_idx * (TILE_SIZE / 2), q_words);
                    cutlass::arch::fence_view_async_tmem_load();
                    CUTE_UNROLL
                    for (uint32_t i = 0; i < TILE_SIZE; ++i) {
                        asm volatile ("fma.rn.f32.bf16 %0, %1, %1, %0;\n" : "+f"(q_sqr_sum) : "h"(((const uint16_t*)q_words)[i]));
                    }
                }
            }
            if constexpr (ENABLE_Q_NORM) {
                smem.q_sqr_sum_buf[cur_job.job_idx_mod_2][idx_in_warpgroup] = q_sqr_sum;
                smem.bar_q_sqr_sum_full.arrive();
            }
        }; 
        auto store_o = [&](const OuterloopArgs &cur_job, const bool &is_last_job) { 
            smem.bar_li_mi_full.wait(cur_job.job_idx_mod_2);
            float mi = smem.rowwise_mi_buf[idx_in_warpgroup % H_Q_PER_CTA];
            // The row sum over the FOLD_FACTOR token slices (softmax warp w holds slice q(w), see there); the same sum in the same
            // order in the prefill and the decode kernel
            float li = smem.rowwise_li_buf[idx_in_warpgroup] + smem.rowwise_li_buf[idx_in_warpgroup ^ 64];
            smem.bar_li_mi_empty.arrive();

            if (idx_in_warpgroup < H_Q_PER_CTA) {
                uint32_t global_index = cur_job.s_q_idx * H_Q + cta_idx * H_Q_PER_CTA + idx_in_warpgroup;
                float cur_lse = fmaf(mi, CUDART_LN2_F, logf(li));
                cur_lse = cur_lse == -CUDART_INF_F ? +CUDART_INF_F : cur_lse;
                params.lse[global_index] = cur_lse;
            }

            float attn_sink = (params.attn_sink == nullptr) ? -CUDART_INF_F : __ldg(params.attn_sink + cta_idx * H_Q_PER_CTA + idx_in_warpgroup % H_Q_PER_CTA) * CUDART_L2E_F;
            float output_scale = li == 0.0f ? 0.0f : __fdividef(1.0f, li + exp2f(attn_sink - mi));

            smem.bar_tO_full.wait(cur_job.job_idx_mod_2);
            ku::tcgen05_after_thread_sync();
            
            static constexpr uint32_t MMA_ATOM_N = 256;
            static constexpr uint32_t NUM_MMA_ATOMS = D_VO / MMA_ATOM_N;
            static constexpr uint32_t NUM_O_TMEM_COLS_PER_ATOM = MMA_ATOM_N / FOLD_FACTOR;
            static constexpr uint32_t EPILOGUE_TILE_SIZE = O_QUANT_TILE_SIZE;
            static constexpr uint32_t NUM_EPILOGUE_TILES_PER_ATOM = NUM_O_TMEM_COLS_PER_ATOM / EPILOGUE_TILE_SIZE;
            static_assert(NUM_O_TMEM_COLS_PER_ATOM % EPILOGUE_TILE_SIZE == 0);
            // Since the weight is per-32 scaled and deep_gemm.einsum requires A and B to have the same scale granularity, the output sf is always stored in a per-32 scaled format, although it will be actually (numerically) per-128 scaled when num_per_channels is 128
            static constexpr uint32_t OUTPUT_SAVE_AS_SCALE_GRAN = 32;
            static_assert(EPILOGUE_TILE_SIZE == OUTPUT_SAVE_AS_SCALE_GRAN, "one tile = one 32 B output store + one sf byte");

            uint32_t head_idx = cta_idx*H_Q_PER_CTA + idx_in_warpgroup%H_Q_PER_CTA;
            uint32_t wv_group_idx = head_idx / WV_GROUP_SIZE;
            uint32_t head_idx_in_wv_group = head_idx % WV_GROUP_SIZE;

            // Layout of O in Tensor Memory:
            // For HEAD_DIM_QK = 64 (head64):
            //   - Atom 0 computes O[:, 0:256]; Atom 1 computes O[:, 256:512]
            //   - Mapping to TMEM:
            //       O[0:128]   -> TMEM[0:64,   0:128]
            //       O[128:256] -> TMEM[64:128,  0:128]
            //       O[256:384] -> TMEM[0:64,   128:256]
            //       O[384:512] -> TMEM[64:128, 128:256]
            //   - Visually (label the four 128-wide O chunks as 0..3):
            //        +---+---+
            //        | 0 | 2 |
            //        +---+---+
            //        | 1 | 3 |
            //        +---+---+
            // For HEAD_DIM_QK = 128 (head128):
            //   - CTA0 holds V[:, 0:256]; CTA1 holds V[:, 256:512]
            //   - Atom 0 computes O[:, 0:128] and O[:, 256:384]
            //     Atom 1 computes O[:, 128:256] and O[:, 384:512]
            //   - Mapping to TMEM:
            //       O[0:128]   -> TMEM[0:64,   0:128]
            //       O[128:256] -> TMEM[0:64,   128:256]
            //       O[256:384] -> TMEM[64:128, 0:128]
            //       O[384:512] -> TMEM[64:128, 128:256]
            //   - Visually (label the four 128-wide O chunks as 0..3):
            //        +---+---+
            //        | 0 | 1 |
            //        +---+---+
            //        | 2 | 3 |
            //        +---+---+
            //
            // A tile is 32 columns of one atom: TMEM load, conjugate RoPE, quantization, its 32 B output store and its sf
            // byte. Two forms of the loop below: with EPILOGUE_TILES_AHEAD > 0 the TMEM loads run ahead of the conversion,
            // otherwise one tile is finished before the next one starts. finish_tile is a tile's work after its TMEM load;
            // the load and the tO_empty arrive belong to the loop
            auto finish_tile = [&](const uint32_t mma_atom_idx, const uint32_t epilogue_tile_idx_in_atom, float (&output)[EPILOGUE_TILE_SIZE], const float reduce_result_by_tmem_ld) {
                // RoPE (conjugate); the condition is warp-uniform
                bool should_perform_rope;
                {
                    static_assert(D_ROPE % EPILOGUE_TILE_SIZE == 0);
                    // The RoPE dims [D_VO - D_ROPE, D_VO) are the last atom's last two tiles of the last lane group (the group that holds
                    // the atom's last dims, warps (FOLD_FACTOR - 1) * H_Q_PER_CTA / 32 on, i.e. warps 2 and 3, tiles 2 and 3 of 4).
                    // The condition is warp-uniform; lane j/2 of cos/sin holds pair j/2 of the 64 RoPE dims
                    static_assert(D_ROPE / EPILOGUE_TILE_SIZE == 2 && NUM_EPILOGUE_TILES_PER_ATOM >= 2);
                    should_perform_rope =
                        mma_atom_idx + 1 == NUM_MMA_ATOMS &&
                        epilogue_tile_idx_in_atom >= NUM_EPILOGUE_TILES_PER_ATOM - D_ROPE/EPILOGUE_TILE_SIZE &&
                        warp_idx >= (FOLD_FACTOR - 1) * (H_Q_PER_CTA / 32);
                    if (should_perform_rope) {
                        CUTE_UNROLL
                        for (uint32_t j = 0; j < EPILOGUE_TILE_SIZE; j += 2) {
                            float2 x = *(float2*)(output + j);
                            uint32_t src_lane = j/2 + (epilogue_tile_idx_in_atom + 1 == NUM_EPILOGUE_TILES_PER_ATOM ? EPILOGUE_TILE_SIZE / 2 : 0);
                            float cur_cos = __shfl_sync(0xFFFFFFFF, cached_cos[0], src_lane);
                            float cur_sin = __shfl_sync(0xFFFFFFFF, cached_sin[0], src_lane);
                            float2 y = apply_rope(x, cur_cos, cur_sin);
                            *(float2*)(output + j) = y;
                        }
                    }
                }

                // Cast to FP8
                float output_abs_max;
                if (!IS_TMEM_LD_WITH_RED_AVAILABLE || should_perform_rope) {
                    output_abs_max = get_max<EPILOGUE_TILE_SIZE, true>(output) * output_scale;
                } else {
                    output_abs_max = reduce_result_by_tmem_ld * output_scale;
                }
                output_abs_max = max(O_QUANT_CLAMP_MIN_VALUE, output_abs_max);
                float sf = output_abs_max / 448.0f;
                uint32_t sf_as_uint32 = *reinterpret_cast<uint32_t*>(&sf);
                uint32_t exp_sf = (int32_t)((sf_as_uint32-1) >> 23) + (1 - 127);
                uint32_t sf_inv_as_uint32 = (127 - exp_sf) << 23;
                float sf_inv = *reinterpret_cast<float*>(&sf_inv_as_uint32);
                float cur_multiplier = output_scale * sf_inv;
                float2 cur_multiplier_float2 = float2(cur_multiplier, cur_multiplier);

                fp8_e4m3 tile_fp8[EPILOGUE_TILE_SIZE];
                CUTE_UNROLL
                for (uint32_t j = 0; j < EPILOGUE_TILE_SIZE; j += 2) {
                    float2 x = *(float2*)(output + j);
                    x = ku::float2_mul(x, cur_multiplier_float2);
                    *(__nv_fp8x2_storage_t*)(tile_fp8 + j) = __nv_cvt_float2_to_fp8x2(
                        x,
                        __NV_SATFINITE,
                        __nv_fp8_interpretation_t::__NV_E4M3
                    );  // NOTE. Here we don't use cvt.f8x4type.f32 since it only has .rs mode, which affects accuracy
                }

                // Store this tile's sf byte and its 32 output bytes
                // Lane group g (= warp_idx / (H_Q_PER_CTA / 32)) holds dims [g * MMA_ATOM_N / FOLD_FACTOR, +MMA_ATOM_N / FOLD_FACTOR) of each
                // atom (1 CTA), or the CTA's half of the dims: [g * MMA_ATOM_N, +MMA_ATOM_N) of each half-atom (2 CTAs)
                uint32_t head_dim_idx_base =
                    CLUSTER_SIZE == 1 ?
                    mma_atom_idx * MMA_ATOM_N + (warp_idx/(H_Q_PER_CTA/32)) * (MMA_ATOM_N/FOLD_FACTOR) :
                    mma_atom_idx * NUM_O_TMEM_COLS_PER_ATOM + (warp_idx/(H_Q_PER_CTA/32)) * MMA_ATOM_N;
                uint32_t head_dim_idx = head_dim_idx_base + epilogue_tile_idx_in_atom * EPILOGUE_TILE_SIZE;
                uint32_t sf_block_idx = head_idx_in_wv_group + (head_dim_idx/OUTPUT_SAVE_AS_SCALE_GRAN) * WV_GROUP_SIZE;
                *((uint8_t*)(params.out_sf + cur_job.s_q_idx + wv_group_idx*params.stride_out_sf_wv_group + (sf_block_idx/4)*params.stride_out_sf_head_dim) + sf_block_idx%4) = (uint8_t)(exp_sf + 127);  // TODO Optimize
                KU_STG_256(
                    params.out_fp8 + (uint64_t)cur_job.s_q_idx*(H_Q*D_VO) + wv_group_idx*(WV_GROUP_SIZE*D_VO) + head_idx_in_wv_group*32 + head_dim_idx*WV_GROUP_SIZE,
                    tile_fp8,
                    "no_allocate",
                    "evict_first"
                );
            };

            if constexpr (EPILOGUE_TILES_AHEAD > 0) {
                // The TMEM loads run EPILOGUE_TILES_AHEAD tiles ahead of the conversion. A tcgen05.wait::ld covers every load
                // issued so far, so the wait of tile t covers the load of tile t + EPILOGUE_TILES_AHEAD - 1, and the tO_empty
                // arrive follows the wait that covers the last tile's load: the next job's first SV waits on tO_empty and
                // starts while the remaining tiles are converted and stored from registers
                static constexpr uint32_t NUM_TILES = NUM_MMA_ATOMS * NUM_EPILOGUE_TILES_PER_ATOM;
                static_assert(!COMPACT_PER_JOB_CODE && EPILOGUE_TILES_AHEAD <= NUM_TILES);
                float outputs[NUM_TILES][EPILOGUE_TILE_SIZE];
                float reduce_results_by_tmem_ld[NUM_TILES];
                auto issue_tile = [&](const uint32_t t) {
                    const uint32_t atom = t / NUM_EPILOGUE_TILES_PER_ATOM, tile = t % NUM_EPILOGUE_TILES_PER_ATOM;
                    const uint32_t col = tmem_cols::O + atom * NUM_O_TMEM_COLS_PER_ATOM + tile * EPILOGUE_TILE_SIZE;
                    if constexpr (IS_TMEM_LD_WITH_RED_AVAILABLE) {
                        ku::tmem_ld_red_32dp32bNx<EPILOGUE_TILE_SIZE, true, true, true>(col, outputs[t], reduce_results_by_tmem_ld[t]);
                    } else {
                        ku::tmem_ld_32dp32bNx<EPILOGUE_TILE_SIZE>(col, outputs[t]);
                    }
                };
                cute::for_each(cute::make_int_sequence<EPILOGUE_TILES_AHEAD>{}, [&](auto t) { issue_tile(t); });
                cute::for_each(cute::make_int_sequence<NUM_TILES>{}, [&](auto t_) {
                    constexpr uint32_t t = decltype(t_)::value;
                    cutlass::arch::fence_view_async_tmem_load();   // the loads of tiles <= min(t + EPILOGUE_TILES_AHEAD - 1, NUM_TILES - 1) have completed
                    if constexpr (t + EPILOGUE_TILES_AHEAD < NUM_TILES) {
                        issue_tile(t + EPILOGUE_TILES_AHEAD);
                    }
                    if constexpr (t + EPILOGUE_TILES_AHEAD - 1 == NUM_TILES - 1) {
                        // Notify tO's emptiness: every tile of O has been read out of TMEM
                        ku::tcgen05_before_thread_sync();
                        if (!is_last_job) {
                            // Don't arrive on the barrier if this job is the last job, to avoid "cluster target block not present"
                            arrive_on_cta0_barrier(smem.bar_tO_empty);
                        }
                    }
                    finish_tile(t / NUM_EPILOGUE_TILES_PER_ATOM, t % NUM_EPILOGUE_TILES_PER_ATOM, outputs[t], reduce_results_by_tmem_ld[t]);
                });
            } else {
                // With COMPACT_PER_JOB_CODE the loops are not unrolled, so the body exists once in the binary and no per-tile
                // state is live across iterations; the loop control adds nothing measurable (this warpgroup is busy about a
                // tenth of the time). Otherwise they are unrolled
                #pragma unroll (COMPACT_PER_JOB_CODE ? 1 : NUM_MMA_ATOMS)
                for (uint32_t mma_atom_idx = 0; mma_atom_idx < NUM_MMA_ATOMS; ++mma_atom_idx)
                #pragma unroll (COMPACT_PER_JOB_CODE ? 1 : NUM_EPILOGUE_TILES_PER_ATOM)
                for (uint32_t epilogue_tile_idx_in_atom = 0; epilogue_tile_idx_in_atom < NUM_EPILOGUE_TILES_PER_ATOM; ++epilogue_tile_idx_in_atom) {
                    const bool is_last_tile = mma_atom_idx + 1 == NUM_MMA_ATOMS && epilogue_tile_idx_in_atom + 1 == NUM_EPILOGUE_TILES_PER_ATOM;

                    // Fetch output from TMEM
                    uint32_t tmem_col_base = tmem_cols::O + mma_atom_idx * NUM_O_TMEM_COLS_PER_ATOM + epilogue_tile_idx_in_atom * EPILOGUE_TILE_SIZE;
                    float output[EPILOGUE_TILE_SIZE];
                    float reduce_result_by_tmem_ld;
                    if constexpr (IS_TMEM_LD_WITH_RED_AVAILABLE) {
                        ku::tmem_ld_red_32dp32bNx<EPILOGUE_TILE_SIZE, true, true, true>(tmem_col_base, output, reduce_result_by_tmem_ld);
                    } else {
                        ku::tmem_ld_32dp32bNx<EPILOGUE_TILE_SIZE>(tmem_col_base, output);
                    }
                    cutlass::arch::fence_view_async_tmem_load();

                    // Notify tO's emptiness
                    if (is_last_tile) {
                        ku::tcgen05_before_thread_sync();
                        if (!is_last_job) {
                            // Don't arrive on the barrier if this job is the last job, to avoid "cluster target block not present"
                            arrive_on_cta0_barrier(smem.bar_tO_empty);
                        }
                    }
                    finish_tile(mma_atom_idx, epilogue_tile_idx_in_atom, output, reduce_result_by_tmem_ld);
                }
            }
        };

        OuterloopArgs cur_job = get_first_job();
        load_q_and_save_to_tmem(cur_job);
        do {
            OuterloopArgs next_job = get_next_job(cur_job);
            shift_cached_cos_and_cached_sin();
            if (next_job.is_valid) {
                load_q_and_save_to_tmem(next_job);
            }
            store_o(cur_job, !next_job.is_valid);
            cur_job = next_job;
        } while (cur_job.is_valid);

        NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);
        if (warp_idx == 0) {
            AllocatorT().free(0, 512);
        }
    } else if (warpgroup_idx == 3) {
        // Scale & Exp warpgroup
        cutlass::arch::warpgroup_reg_alloc<WG3_REGS>();

        OuterloopArgs cur_job = get_first_job();
        uint32_t local_warp_idx = warp_idx - 12;
        // After retrieve_mask_and_reduce_p, warp w holds every head's tokens [q(w) * B_TOPK / FOLD_FACTOR, +B_TOPK / FOLD_FACTOR):
        // q(w) = w / 2 (warps 0, 1 | 2, 3)
        // S is K-major with 8-row core matrices (INTER layout), so a head's 8 tokens are 16 B at row * 16 B of the token group's block
        static_assert(FOLD_FACTOR == 2);
        const uint32_t token_slice = local_warp_idx >= 2 ? 1u : 0u;
        bf16 *sS_base = smem.s + token_slice * (H_Q_PER_CTA * (B_TOPK / FOLD_FACTOR)) + (idx_in_warpgroup % H_Q_PER_CTA) * 8;
        RingBufferState rs;
        do {
            // For definition and consistency about `mi`, `li`, and `real_mi`, plz refer to head64 prefill
            static constexpr uint32_t NUM_ELEMS_PER_THREAD = B_TOPK * H_Q_PER_CTA / 128;
            float mi = MAX_INIT_VAL;
            float li = 0.0f;
            float real_mi = -CUDART_INF_F;

            float score_multiplier; // qk_scale * rms_norm's denominator (if q norm is enabled)
            if constexpr (ENABLE_Q_NORM) {
                smem.bar_q_sqr_sum_full.wait(cur_job.job_idx_mod_2);

                // WG0 thread t summed the squares of its rail's dims of head t % H_Q_PER_CTA; threads t and t ^ 64 hold the two rails of
                // the same head
                static_assert(NUM_MRGEMM_RAILS == 2);
                score_multiplier = smem.q_sqr_sum_buf[cur_job.job_idx_mod_2][idx_in_warpgroup] + smem.q_sqr_sum_buf[cur_job.job_idx_mod_2][idx_in_warpgroup^64];
                score_multiplier = params.sm_scale_div_log2 * rsqrtf(score_multiplier / D_QK + params.rms_norm_eps);    // rsqrt is translated to `MUFU.RSQ`
            } else {
                score_multiplier = params.sm_scale_div_log2;
            }

            smem.bar_tQ_empty.arrive(); // Must arrive on the empty barrier here, to prevent smem.bar_tQ_full being phase-skipped

            CUTE_NO_UNROLL
            for (uint32_t kv_block_idx = 0; kv_block_idx < cur_job.num_kv_blocks; ++kv_block_idx) {
                auto [indices_buf_idx, indices_bar_phase] = rs.get<NUM_INDICES_BUFS>();
                auto [p_buf_idx, p_bar_phase] = rs.get<NUM_P_BUFS>();
                smem.bar_tP_full[p_buf_idx].wait(p_bar_phase);
                smem.bar_indices_full[indices_buf_idx].wait(indices_bar_phase);
                ku::tcgen05_after_thread_sync();

                float p[NUM_ELEMS_PER_THREAD];
                retrieve_mask_and_reduce_p<
                    H_Q_PER_CTA,
                    NUM_ELEMS_PER_THREAD,
                    barrier_ids::WG3_WARP02_SYNC,
                    barrier_ids::WG3_WARP13_SYNC
                >(
                    tmem_cols::get_p(p_buf_idx),
                    (char*)&smem.is_k_valid[indices_buf_idx],
                    local_warp_idx,
                    lane_idx,
                    [&]() {
                        if constexpr (NEED_TP_EMPTY_BAR) {
                            arrive_on_cta0_barrier(smem.bar_tP_empty[p_buf_idx]);
                        }
                    },
                    smem.p_exchange_buf,
                    p
                );

                float cur_pi_max = get_max<NUM_ELEMS_PER_THREAD>(p);
                cur_pi_max *= score_multiplier;

                // Row max over the lane groups holding the head's other token slices: the partner warp (w ^ 2, a 64-thread
                // barrier per pair). max is exact in any order
                smem.rowwise_max_buf[idx_in_warpgroup] = cur_pi_max;
                NamedBarrier::arrive_and_wait(32 * FOLD_FACTOR, barrier_ids::WG3_WARP02_SYNC + (local_warp_idx & 1));
                smem.bar_indices_empty[indices_buf_idx].arrive();   // Put it here to give the compiler more room for code reordering
                CUTE_UNROLL
                for (uint32_t g = 1; g < FOLD_FACTOR; ++g) {
                    cur_pi_max = max(cur_pi_max, smem.rowwise_max_buf[idx_in_warpgroup ^ (g * H_Q_PER_CTA)]);
                }
                real_mi = max(real_mi, cur_pi_max);
                bool should_scale_o = __any_sync(0xffffffff, cur_pi_max - mi > 6.0f);

                float new_max, scale_for_old;
                if (!should_scale_o) {
                    // Don't scale O
                    scale_for_old = 1.0f;
                    new_max = mi;
                } else {
                    new_max = max(cur_pi_max, mi);
                    scale_for_old = exp2f(mi - new_max);
                }
                mi = new_max;   // mi is still identical within each row

                // Calculate S
                nv_bfloat16 s[NUM_ELEMS_PER_THREAD];
                float cur_sum = get_s_from_p<NUM_ELEMS_PER_THREAD>((nv_bfloat162*)s, p, score_multiplier, new_max);
                li = fmaf(li, scale_for_old, cur_sum);

                // Store S
                smem.bar_SO_empty.wait(rs.get<1>().second^1);
                CUTE_UNROLL
                for (int i = 0; i < NUM_ELEMS_PER_THREAD/8; ++i) {
                    ku::st_shared(sS_base + i*8*H_Q_PER_CTA, *(__int128_t*)(s + i*8));
                }

                // Rescale O
                if (kv_block_idx > 0 && should_scale_o) {
                    ku::tcgen05_after_thread_sync();
                    rescale_O<D_VO / FOLD_FACTOR, 32, tmem_cols::O>(scale_for_old);
                    ku::tcgen05_before_thread_sync();
                }

                fence_view_async_shared();
                ku::tcgen05_before_thread_sync();
                arrive_on_cta0_barrier(smem.bar_SO_full);
                rs.update();
            }

            if (real_mi == -CUDART_INF_F) {
                // No valid TopK indices
                li = 0.0f;
                mi = -CUDART_INF_F;
            }

            smem.bar_li_mi_empty.wait(cur_job.job_idx_mod_2^1);
            static_assert(H_Q_PER_CTA % 32 == 0);
            if (local_warp_idx < H_Q_PER_CTA / 32) {
                if constexpr (!IS_DECODE) {
                    uint32_t global_index = cur_job.s_q_idx * H_Q + cta_idx * H_Q_PER_CTA + idx_in_warpgroup;
                    params.max_logits[global_index] = real_mi * CUDART_LN2_F;
                }
                smem.rowwise_mi_buf[idx_in_warpgroup] = mi;
            }
            smem.rowwise_li_buf[idx_in_warpgroup] = li;
            smem.bar_li_mi_full.arrive();

            cur_job = get_next_job(cur_job);
        } while (cur_job.is_valid);
    } else if (warpgroup_idx == 2) {
        cutlass::arch::warpgroup_reg_dealloc<WG2_REGS>();
        if (warp_idx == 8 && cta_idx == 0 && elect_one_sync()) {
            // MMA warp (CTA0 only)
            auto tiled_mma_qk = TiledMMA_QK{};
            auto tiled_mma_sv = TiledMMA_SV{};
            Tensor tQ = tiled_mma_qk.get_slice(_0{}).make_fragment_A(
                partition_shape_A(tiled_mma_qk, Shape<Int<H_Q_PER_CTA>, Int<D_QK / NUM_MRGEMM_RAILS>>{})
            );
            Tensor tP = partition_fragment_C(tiled_mma_qk, Shape<Int<H_Q_PER_CTA>, Int<B_TOPK * NUM_MRGEMM_RAILS>>{});
            Tensor sS = make_tensor(
                make_smem_ptr(smem.s),
                ku::make_umma_canonical_k_major_layout<H_Q_PER_CTA, B_TOPK, 0>()
            );
            Tensor tO = partition_fragment_C(tiled_mma_sv, Shape<Int<H_Q_PER_CTA>, Int<D_VO>>{});
            tQ.data().get() = tmem_cols::Q;
            tO.data().get() = tmem_cols::O;
            // tP.data() will be assigned in the loop since it has double buffers

            RingBufferState rs_qk, rs_sv;
            // Issue QK on the current kv slot. The caller has waited on kv_slot_full, on tQ_full for the
            // first block of a job, and on tP_empty when P is single-buffered
            auto issue_qk = [&](const OuterloopArgs &job, uint32_t kv_block_idx) {
                auto kv_slot_idx = rs_qk.get<NUM_KV_SLOTS>().first;
                Tensor sK = make_tensor(
                    make_smem_ptr(smem.kv_slots[kv_slot_idx]),
                    ku::make_umma_canonical_k_major_layout<B_TOPK / CLUSTER_SIZE * NUM_MRGEMM_RAILS, D_QK / NUM_MRGEMM_RAILS, 128>()
                );
                auto p_buf_idx = rs_qk.get<NUM_P_BUFS>().first;
                tP.data().get() = tmem_cols::get_p(p_buf_idx);

                ku::tcgen05_after_thread_sync();
                ku::utcmma_ts(tiled_mma_qk, tQ, sK, tP, true);
                umma_arrive_on_every_cta(smem.bar_tP_full[p_buf_idx]);

                if (kv_block_idx == job.num_kv_blocks-1) {
                    umma_arrive_on_every_cta(smem.bar_tQ_empty);
                }
                rs_qk.update();
            };
            // Prefill: the KV block arrives by TMA, so the arrive carries its transaction bytes. Decode: the dequant warps
            // write the whole slot and arrive on the same barrier, so this is a plain arrive
            auto arrive_kv_slot_full = [&](uint32_t kv_slot_idx) {
                if constexpr (IS_DECODE) {
                    smem.bar_kv_slot_full[kv_slot_idx].arrive();
                } else {
                    smem.bar_kv_slot_full[kv_slot_idx].arrive_and_expect_tx(B_TOPK*D_QK*sizeof(bf16));
                }
            };
            // Issue SV on the current kv slot. The caller has waited on SO_full and, for the first block of
            // a job, on tO_empty
            auto issue_sv = [&](const OuterloopArgs &job, uint32_t kv_block_idx) {
                auto kv_slot_idx = rs_sv.get<NUM_KV_SLOTS>().first;
                Tensor sV = make_tensor(
                    make_smem_ptr(smem.kv_slots[kv_slot_idx]),
                    ku::make_umma_canonical_mn_major_layout<D_VO / CLUSTER_SIZE, B_TOPK, 128>()
                );
                ku::tcgen05_after_thread_sync();
                ku::utcmma_ss(tiled_mma_sv, sS, sV, tO, kv_block_idx == 0);
                umma_arrive_on_every_cta(smem.bar_kv_slot_empty[kv_slot_idx]);
                umma_arrive_on_every_cta(smem.bar_SO_empty);
                if (kv_block_idx == job.num_kv_blocks-1) {
                    umma_arrive_on_every_cta(smem.bar_tO_full);
                }
                rs_sv.update();
            };

            if constexpr (IS_DECODE && !NEED_TP_EMPTY_BAR) {
                // Adaptive issue. In the fixed order SV(g) is issued after QK(g+1)'s kv_slot_full wait, so a late
                // dequant of block g+1 delays SV(g) and the release of slot g to the KV producer. The two gemms
                // wait on different barriers (kv_slot_full vs SO_full), so the warp polls both and issues
                // whichever is ready.
                // P is double-buffered and QK(g) overwrites P[g%2], the buffer of block g-2.
                // num_qk_issued <= num_sv_issued + 1 issues QK(g) only after SV(g-2), and SV(g-2) waited on
                // SO_full, so softmax(g-2) has finished reading it.
                // The polls are test_wait, which returns at once, so a pending SV input does not keep a ready QK
                // from being issued
                OuterloopArgs qk_job = get_first_job();
                OuterloopArgs sv_job = qk_job;
                uint32_t qk_block_idx = 0, sv_block_idx = 0;    // block index within qk_job / sv_job
                uint32_t num_qk_issued = 0, num_sv_issued = 0;  // gemms issued so far, across jobs
                uint32_t qk_job_idx = 0, sv_job_idx = 0;        // index of qk_job / sv_job among this CTA's jobs
                bool qk_all_done = false;
                bool kv_slot_full_arrived = false;  // arrive_kv_slot_full has been issued for the pending QK block (once per block, before its first poll)
                while (!qk_all_done || num_sv_issued < num_qk_issued) {
                    if (num_sv_issued < num_qk_issued && smem.bar_SO_full.test_wait(rs_sv.get<1>().second)
                        && (sv_block_idx > 0 || smem.bar_tO_empty.test_wait(sv_job.job_idx_mod_2 ^ 1))) {
                        issue_sv(sv_job, sv_block_idx);
                        ++num_sv_issued;
                        if (++sv_block_idx == sv_job.num_kv_blocks) {
                            // Each SV is issued after its QK, so QK has finished this job; the sv_job_idx == qk_job_idx
                            // guard below stops QK from starting the job after next, so qk_job is the SV side's next job
                            sv_block_idx = 0;
                            ++sv_job_idx;
                            sv_job = qk_job;
                        }
                        continue;
                    }
                    // QK starts a job only after SV has finished the job before it (sv_job_idx == qk_job_idx),
                    // so qk_job is never more than one job ahead of sv_job. With a 1-block job QK could otherwise
                    // finish two jobs between consecutive SV issues and the sv_job = qk_job copy above would skip one
                    if (!qk_all_done && num_qk_issued <= num_sv_issued + 1 && (qk_block_idx > 0 || sv_job_idx == qk_job_idx)) {
                        auto [kv_slot_idx, kv_bar_phase] = rs_qk.get<NUM_KV_SLOTS>();
                        if (!kv_slot_full_arrived) {
                            kv_slot_full_arrived = true;
                            arrive_kv_slot_full(kv_slot_idx);
                        }
                        bool ready = smem.bar_kv_slot_full[kv_slot_idx].test_wait(kv_bar_phase);
                        if (ready && qk_block_idx == 0) {
                            ready = smem.bar_tQ_full.test_wait(qk_job.job_idx_mod_2);
                        }
                        if (ready) {
                            issue_qk(qk_job, qk_block_idx);
                            kv_slot_full_arrived = false;
                            ++num_qk_issued;
                            if (++qk_block_idx == qk_job.num_kv_blocks) {
                                qk_block_idx = 0;
                                ++qk_job_idx;
                                qk_job = get_next_job(qk_job);
                                qk_all_done = !qk_job.is_valid;
                            }
                        }
                    }
                }
            } else {
                // The adaptive loop replaces the tP_empty wait with the issue-distance bound
                // num_qk_issued <= num_sv_issued + 1, so it needs two P buffers; the 2-CTA decode has one, and
                // prefill loads KV by TMA with no dequant to run late. Both keep the fixed interleaving
                // QK(0); { QK(k), SV(k-1) }*; SV(last)
                auto run_qk_gemm = [&](const OuterloopArgs &job, uint32_t kv_block_idx) {
                    if (kv_block_idx == 0) {
                        smem.bar_tQ_full.wait(job.job_idx_mod_2);
                    }
                    auto [kv_slot_idx, kv_bar_phase] = rs_qk.get<NUM_KV_SLOTS>();
                    arrive_kv_slot_full(kv_slot_idx);
                    smem.bar_kv_slot_full[kv_slot_idx].wait(kv_bar_phase);
                    if constexpr (NEED_TP_EMPTY_BAR) {
                        auto [p_buf_idx, p_bar_phase] = rs_qk.get<NUM_P_BUFS>();
                        smem.bar_tP_empty[p_buf_idx].wait(p_bar_phase ^ 1);
                    }
                    issue_qk(job, kv_block_idx);
                };
                auto run_sv_gemm = [&](const OuterloopArgs &job, uint32_t kv_block_idx) {
                    if (kv_block_idx == 0) {
                        smem.bar_tO_empty.wait(job.job_idx_mod_2^1);
                    }
                    smem.bar_SO_full.wait(rs_sv.get<1>().second);
                    issue_sv(job, kv_block_idx);
                };

                OuterloopArgs cur_job = get_first_job();
                run_qk_gemm(cur_job, 0);
                do {
                    CUTE_NO_UNROLL
                    for (uint32_t kv_block_idx = 1; kv_block_idx < cur_job.num_kv_blocks; ++kv_block_idx) {
                        run_qk_gemm(cur_job, kv_block_idx);
                        run_sv_gemm(cur_job, kv_block_idx-1);
                    }

                    OuterloopArgs next_job = get_next_job(cur_job);
                    if (next_job.is_valid) {
                        run_qk_gemm(next_job, 0);
                    }
                    run_sv_gemm(cur_job, cur_job.num_kv_blocks-1);
                    
                    cur_job = next_job;
                } while (cur_job.is_valid);
            }
        } else if (warp_idx == 9 && elect_one_sync()) {
            // CLC warp
            bool phase = 0;
            while (true) {
                if (cta_idx == 0) {
                    smem.bar_clc_empty.wait(phase^1);
                    ku::issue_clc_query_multicast_cluster_all(smem.bar_clc_full, smem.clc_response_obj);
                }
                smem.bar_clc_full.arrive_and_expect_tx(sizeof(smem.clc_response_obj));
                
                smem.bar_clc_full.wait(phase&1);
                ku::CLCResult clc_result = ku::get_clc_query_response<true>(smem.clc_response_obj);
                arrive_on_cta0_barrier(smem.bar_clc_empty);
                if (!clc_result.is_valid)
                    break;

                phase ^= 1;
            }
            if constexpr (IS_2CTA) {
                if (cta_idx == 0) {
                    smem.bar_clc_empty.wait(phase); // Wait for all threads' arrival on `bar_clc_empty`, which means that there will be no further operations on distributed shared memory (including barrier arrive and MMA), avoiding the "cluster target block not present" error
                    smem.bar_clc_empty.arrive(1u);  // Transfer the signal above to CTA1
                } else {
                    smem.bar_clc_empty.wait(0);
                }
            }
        } else if (warp_idx == 10) {
            // Indices generator
            // (also generates TMA coordinates for the KV producer)
            OuterloopArgs cur_job = get_first_job();
            RingBufferState rs;
            static constexpr uint32_t NUM_INDICES_PER_THREAD = B_TOPK / 32;
            static_assert(B_TOPK % 32 == 0);

            do {
                if constexpr (!IS_DECODE) {
                    auto body = [&]<bool CHECK_TOPK_SUBSCRIPT>() {
                        CUTE_NO_UNROLL
                        for (uint32_t kv_block_idx = 0; kv_block_idx < cur_job.num_kv_blocks; ++kv_block_idx) {
                            auto [indices_buf_idx, indices_bar_phase] = rs.get<NUM_INDICES_BUFS>();
                            smem.bar_indices_empty[indices_buf_idx].wait(indices_bar_phase^1);

                            CUTE_UNROLL
                            for (uint32_t i = 0; i < NUM_INDICES_PER_THREAD; ++i) {
                                uint32_t pos = kv_block_idx * B_TOPK + i * 32 + lane_idx;
                                int cur_index;
                                if constexpr (CHECK_TOPK_SUBSCRIPT) {
                                    // Predicate the load on `pos < topk_length` to prevent IMA
                                    cur_index = pos < cur_job.topk_length ? __ldg(params.indices + cur_job.s_q_idx * params.stride_indices_s_q + pos) : -1;
                                } else {
                                    // topk_length % B_TOPK == 0, so every pos is within the row
                                    cur_index = __ldg(params.indices + cur_job.s_q_idx * params.stride_indices_s_q + pos);
                                }
                                bool is_index_valid = (uint32_t)cur_index < (uint32_t)params.s_kv;  // Don't need to check `index >= 0`, since if `index < 0` holds, `(uint32_t)index` must lies in 2147483648 ~ 4294967295, which is definitely greater than `params.s_kv`
                                uint32_t mask = __ballot_sync(0xFFFFFFFF, is_index_valid);
                                if (lane_idx == 0) {
                                    *((uint32_t*)smem.is_k_valid[indices_buf_idx] + i) = mask;
                                }
                            }

                            smem.bar_indices_full[indices_buf_idx].arrive();
                            rs.update();
                        }
                    };
                    if (cur_job.topk_length % B_TOPK == 0 && cur_job.topk_length != 0)
                        body.template operator()<false>();
                    else
                        body.template operator()<true>();
                } else {
                    int *indices_base = params.indices + (int64_t)cur_job.s_q_idx * params.stride_indices_s_q;
                    int *extra_indices_base = params.extra_indices + (int64_t)cur_job.s_q_idx * params.stride_extra_indices_s_q;
                    // The indices of a block are loaded one block ahead, into registers: the load is a global read
                    // with no shared-memory destination, so it needs neither the metadata buffer nor a barrier, and
                    // its DRAM round trip overlaps with this block's coordinate and scale arithmetic. Inside the
                    // per-index loop the load could not be issued early: the __ballot_sync at the end of each
                    // iteration orders the next iteration's load behind it. The prefetch resolves the source at
                    // run time (pos >= num_orig_slots holds for every KVLocation; num_orig_slots is 0xFFFFFFFF
                    // without an extra cache); the consume side keeps the compile-time dispatch. The first block
                    // of a job is not prefetched: its indices row is unknown until the CLC response
                    auto load_indices = [&](uint32_t kv_block_idx, int (&dst)[NUM_INDICES_PER_THREAD]) {
                        CUTE_UNROLL
                        for (uint32_t i = 0; i < NUM_INDICES_PER_THREAD; ++i) {
                            uint32_t pos = kv_block_idx * B_TOPK + i * 32 + lane_idx;
                            bool in_extra = pos >= cur_job.num_orig_slots;
                            uint32_t local_pos = in_extra ? pos - cur_job.num_orig_slots : pos;
                            uint32_t valid_len = in_extra ? cur_job.extra_topk_length : cur_job.topk_length;
                            // The last KV block may run beyond the end of the indices row (e.g. topk = 128 with
                            // B_TOPK = 96), so an unconditional load could read out of bounds of `indices` /
                            // `extra_indices`. Slots beyond valid_len are masked out anyway, so they get -1 without a load
                            dst[i] = local_pos < valid_len ? __ldg((in_extra ? extra_indices_base : indices_base) + local_pos) : -1;
                        }
                    };
                    int prefetched[NUM_INDICES_PER_THREAD];
                    load_indices(0, prefetched);
                    auto metadata_block = [&]<KVLocation LOC>(uint32_t kv_block_idx) {
                        auto [indices_buf_idx, indices_bar_phase] = rs.get<NUM_INDICES_BUFS>();
                        smem.bar_indices_empty[indices_buf_idx].wait(indices_bar_phase^1);
                        int cur_indices[NUM_INDICES_PER_THREAD];
                        CUTE_UNROLL
                        for (uint32_t i = 0; i < NUM_INDICES_PER_THREAD; ++i) cur_indices[i] = prefetched[i];
                        if (kv_block_idx + 1 < cur_job.num_kv_blocks) {
                            load_indices(kv_block_idx + 1, prefetched);
                        }
                        // Which cache a slot of this block comes from; a compile-time constant for ORIG / EXTRA blocks
                        auto pos_in_extra = [&](uint32_t pos) {
                            if constexpr (LOC == KVLocation::ORIG) {
                                return false;
                            } else if constexpr (LOC == KVLocation::EXTRA) {
                                return true;
                            } else {
                                return pos >= cur_job.num_orig_slots;
                            }
                        };
                        CUTE_UNROLL
                        for (uint32_t i = 0; i < NUM_INDICES_PER_THREAD; ++i) {
                            uint32_t pos = kv_block_idx * B_TOPK + i * 32 + lane_idx;
                            bool in_extra = pos_in_extra(pos);
                            int cur_index = cur_indices[i];
                            bool is_index_valid = cur_index >= 0;

                            // Share coordinate generation instead of repeating it in all dequant warps.
                            // The per-cache constants are selected by value, not by reference. Selecting between
                            // the two FastDivmod objects by reference makes ptxas lose the `divisor != 1` fold of
                            // fast_divmod and emit a full integer division per index; the divmod is written out
                            // here on the selected fields so the fold survives. Likewise the quotient
                            // stride / BYTES_PER_TOKEN is formed per format below: the format's stride is a
                            // compile-time constant, so each quotient is a multiply-shift, while dividing the
                            // selected stride by the selected BYTES_PER_TOKEN would be a runtime 64-bit division
                            const uint32_t dm_mul = in_extra ? aux_params.fast_divmod_extra_page_block_size.multiplier : aux_params.fast_divmod_page_block_size.multiplier;
                            const uint32_t dm_shr = in_extra ? aux_params.fast_divmod_extra_page_block_size.shift_right : aux_params.fast_divmod_page_block_size.shift_right;
                            const int32_t dm_div = in_extra ? aux_params.fast_divmod_extra_page_block_size.divisor : aux_params.fast_divmod_page_block_size.divisor;
                            uint32_t token_idx = is_index_valid ? (uint32_t)cur_index : 0;
                            int block_idx = dm_div != 1 ? (int)(__umulhi(token_idx, dm_mul) >> dm_shr) : (int)token_idx;
                            int idx_in_block = (int)token_idx - block_idx * dm_div;
                            uint32_t row = i * 32 + lane_idx;
                            // The extra KV cache may have another format (HAS_FP4_KV); see above for the per-format quotient
                            const int blocks_per_stride = in_extra ? params.stride_extra_kv_block / (int)ExtraKVFormat::BYTES_PER_TOKEN
                                                                   : params.stride_kv_block / (int)OrigKVFormat::BYTES_PER_TOKEN;
                            smem.decode_tma_coords[indices_buf_idx][row] = is_index_valid
                                ? blocks_per_stride * block_idx + idx_in_block
                                : -1;

                            uint32_t mask = __ballot_sync(0xFFFFFFFF, is_index_valid);
                            if (lane_idx == 0) {
                                *((uint32_t*)smem.is_k_valid[indices_buf_idx] + i) = mask;
                            }
                        }

                        smem.bar_indices_full[indices_buf_idx].arrive();
                        rs.update();
                    };
                    if constexpr (HAS_FP4_KV) {
                        // Two copies of the body. The fp4 blocks, most of a job's blocks, run the EXTRA copy: the
                        // source folds to the extra cache at compile time so
                        // it is this kernel's per-block hot code and stays small. The fp8 blocks and the at most
                        // one mixed block share the run-time-dispatched copy: they run once per job, so their
                        // code is per-job footprint, and one copy instead of two (ORIG and mixed) is less to
                        // fetch through the TPC-shared instruction cache at every job boundary
                        const uint32_t num_full_orig_blocks = std::min(cur_job.num_kv_blocks, cur_job.num_orig_slots / B_TOPK);
                        const uint32_t num_general_blocks = std::min(cur_job.num_kv_blocks,
                            num_full_orig_blocks + (cur_job.num_orig_slots % B_TOPK != 0 ? 1u : 0u));
                        CUTE_NO_UNROLL
                        for (uint32_t kv_block_idx = 0; kv_block_idx < num_general_blocks; ++kv_block_idx) {
                            metadata_block.template operator()<KVLocation::ORIG_AND_EXTRA>(kv_block_idx);
                        }
                        CUTE_NO_UNROLL
                        for (uint32_t kv_block_idx = num_general_blocks; kv_block_idx < cur_job.num_kv_blocks; ++kv_block_idx) {
                            metadata_block.template operator()<KVLocation::EXTRA>(kv_block_idx);
                        }
                    } else {
                        // fp8 extra cache (or none): both sources have the same format, so the run-time form
                        // computes the same arithmetic as the ORIG / EXTRA specializations and one copy serves
                        // every block
                        CUTE_NO_UNROLL
                        for (uint32_t kv_block_idx = 0; kv_block_idx < cur_job.num_kv_blocks; ++kv_block_idx) {
                            metadata_block.template operator()<KVLocation::ORIG_AND_EXTRA>(kv_block_idx);
                        }
                    }
                }
                cur_job = get_next_job(cur_job);
            } while (cur_job.is_valid);
        } else if (warp_idx == 11) {
            if constexpr (IS_DECODE) {
                // Gather front of the KV producer, for every block of every job (the dequant warps run only the
                // back, see WG1). Per block: wait until the dequant warps have released this block's index of the
                // alternating barrier pair (bar_wg1_past_raw), then for the block's KV slot and indices; issue the
                // raw gather4s and CTA0 scale copies, both tracked by bar_raw_kv_full[index]
                RingBufferState rs;
                auto issue_front = [&](const OuterloopArgs &job, uint32_t kv_block_idx) {
                    uint32_t raw_buf = rs.get<NUM_RAW_BARS>().first;
                    auto [kv_slot_idx, kv_bar_phase] = rs.get<NUM_KV_SLOTS>();
                    smem.bar_kv_slot_empty[kv_slot_idx].wait(kv_bar_phase^1);
                    auto [indices_buf_idx, indices_bar_phase] = rs.get<NUM_INDICES_BUFS>();
                    smem.bar_indices_full[indices_buf_idx].wait(indices_bar_phase);
                    uint8_t *slot_base = (uint8_t*)smem.kv_slots[kv_slot_idx];

                    // CTA0's lanes prefetch scales while its elected lane issues the raw gathers.
                    if constexpr (IS_2CTA) {
                        if (cta_idx == 0) {
                            CUTE_UNROLL
                            for (uint32_t i = 0; i < B_TOPK / 32; ++i) {
                                uint32_t row = i * 32 + lane_idx;
                                int coord = smem.decode_tma_coords[indices_buf_idx][row];
                                bool in_extra = kv_block_idx * B_TOPK + row >= job.num_orig_slots;
                                const uint8_t *base = in_extra ? (const uint8_t*)params.extra_kv + ExtraKVFormat::QUANT_BYTES
                                                               : (const uint8_t*)params.kv + OrigKVFormat::QUANT_BYTES;
                                uint32_t stride = in_extra ? ExtraKVFormat::BYTES_PER_TOKEN : OrigKVFormat::BYTES_PER_TOKEN;
                                uint8_t *dst = slot_base + SCALE_STAGE_OFFSET + row * SCALE_STAGE_ROW_BYTES;
                                copy_kv_scales_async<SCALE_STAGE_ROW_BYTES>(dst, base, coord, stride);
                            }
                            cutlass::arch::cpasync_barrier_arrive_noinc(reinterpret_cast<uint64_t const*>(&smem.bar_raw_kv_full[raw_buf]));
                        }
                    }
                    bool elected = elect_one_sync();
                    if (elected) {
                        // The tensor map is a property of the 4-row group (topk % 8 == 0, so a group never straddles the
                        // orig/extra boundary); the block's gather4s are issued by issue_raw_gathers
                        // The maps are passed as byte offsets into tma_params (issue_raw_gathers adds the base)
                        const uint8_t *maps_base = (const uint8_t*)&tma_params;
                        const uint32_t orig_map = cta_idx == 0 ? offsetof(TMAParams, tensor_map_kv_fp8_part_cta0) : offsetof(TMAParams, tensor_map_kv_fp8_part_cta1);
                        // The extra cache's map follows its format: fp4 (HAS_FP4_KV) or fp8 (an fp8 extra cache)
                        const uint32_t extra_map = HAS_FP4_KV
                            ? (cta_idx == 0 ? offsetof(TMAParams, tensor_map_extra_kv_fp4_part_cta0) : offsetof(TMAParams, tensor_map_extra_kv_fp4_part_cta1))
                            : (cta_idx == 0 ? offsetof(TMAParams, tensor_map_extra_kv_fp8_part_cta0) : offsetof(TMAParams, tensor_map_extra_kv_fp8_part_cta1));
                        const int4 *coords = (const int4*)smem.decode_tma_coords[indices_buf_idx];
                        const uint32_t num_orig_rows = get_num_orig_rows(job, kv_block_idx);
                        if (0 < num_orig_rows && num_orig_rows < B_TOPK) {
                            issue_raw_gathers<true, B_TOPK / 4, RAW_KV_GROUP_BYTES>(coords, maps_base, orig_map, extra_map, num_orig_rows / 4, smem.bar_raw_kv_full[raw_buf], slot_base);
                        } else {
                            issue_raw_gathers<false, B_TOPK / 4, RAW_KV_GROUP_BYTES>(coords, maps_base, num_orig_rows == 0 ? extra_map : orig_map, 0, 0, smem.bar_raw_kv_full[raw_buf], slot_base);
                        }
                        // The raw row widths of this CTA (both caches have the same width unless the extra cache is fp4)
                        const uint32_t orig_stride = cta_idx == 0 ? KVFormatCta<MODEL_TYPE, 0>::RAW_TOKEN_SMEM_STRIDE : KVFormatCta<MODEL_TYPE, CLUSTER_SIZE - 1>::RAW_TOKEN_SMEM_STRIDE;
                        const uint32_t extra_stride = cta_idx == 0 ? KVFormatCta<EXTRA_MODEL_TYPE, 0>::RAW_TOKEN_SMEM_STRIDE : KVFormatCta<EXTRA_MODEL_TYPE, CLUSTER_SIZE - 1>::RAW_TOKEN_SMEM_STRIDE;
                        smem.bar_raw_kv_full[raw_buf].arrive_and_expect_tx(num_orig_rows * orig_stride + (B_TOPK - num_orig_rows) * extra_stride);
                    }

                    // Each lane releases its coordinates after its reads.
                    smem.bar_indices_empty[indices_buf_idx].arrive();
                    rs.update();
                };

                // Block 0's front is issued before the loop; the iteration for block issued_block_idx issues the
                // front of the block after it, so that at a job boundary the CLC result is read before the next
                // job's first front
                OuterloopArgs cur_job = get_first_job();
                issue_front(cur_job, 0);
                uint32_t issued_block_idx = 0;   // index across jobs of the last block whose front was issued
                do {
                    OuterloopArgs next_job;
                    next_job.is_valid = false;
                    CUTE_NO_UNROLL
                    for (uint32_t kv_block_idx = 0; kv_block_idx < cur_job.num_kv_blocks; ++kv_block_idx) {
                        // The next front uses pair index (issued_block_idx + 1) % NUM_RAW_BARS, last used by the block
                        // NUM_RAW_BARS blocks before it. The dequant warps arrive on bar_wg1_past_raw once they have
                        // passed the completion wait of that block, so after this wait no waiter is left behind on the
                        // barrier and the next front's arrives cannot complete a phase ahead of one. The barrier is
                        // at most at that block's phase (the next front has not been issued), so the parity is
                        // unambiguous
                        if (issued_block_idx + 1 >= NUM_RAW_BARS) {
                            auto [past_raw_buf, past_raw_phase] = RingBufferState{issued_block_idx + 1 - NUM_RAW_BARS}.get<NUM_RAW_BARS>();
                            smem.bar_wg1_past_raw[past_raw_buf].wait(past_raw_phase);
                        }
                        bool has_next = true;
                        if (kv_block_idx + 1 == cur_job.num_kv_blocks) {
                            next_job = get_next_job(cur_job);
                            has_next = next_job.is_valid;
                        }
                        const OuterloopArgs &next_block_job = kv_block_idx + 1 < cur_job.num_kv_blocks ? cur_job : next_job;
                        uint32_t next_block_idx = kv_block_idx + 1 < cur_job.num_kv_blocks ? kv_block_idx + 1 : 0;
                        if (has_next) {
                            issue_front(next_block_job, next_block_idx);
                        }
                        ++issued_block_idx;
                    }
                    cur_job = next_job;
                } while (cur_job.is_valid);
            }
        }
    } else if (warpgroup_idx == 1) {
        // setmaxnreg.inc raises the count from the initial 128 and .dec lowers it; the direction is part of the instruction
        if constexpr (WG1_REGS >= 128) {
            cutlass::arch::warpgroup_reg_alloc<WG1_REGS>();
        } else {
            cutlass::arch::warpgroup_reg_dealloc<WG1_REGS>();
        }
        if constexpr (IS_DECODE) {
            // Warp 11 loads raw data and scales. WG1 waits once for both transports before
            // dequantizing in place. The orig/extra format boundary is resolved per 8 rows.
            uint32_t local_warp_idx = warp_idx - 4;
            auto run_dequant_warps = [&](auto &dq_rows) {
                using DQ = std::remove_reference_t<decltype(dq_rows)>;
                RingBufferState rs_back;
                OuterloopArgs cur_job = get_first_job();
                do {
                    OuterloopArgs next_job;
                    next_job.is_valid = false;
                    CUTE_NO_UNROLL
                    for (uint32_t kv_block_idx = 0; kv_block_idx < cur_job.num_kv_blocks; ++kv_block_idx) {
                        auto [raw_buf, raw_bar_phase] = rs_back.get<NUM_RAW_BARS>();
                        smem.bar_raw_kv_full[raw_buf].wait(raw_bar_phase);
                        if (elect_one_sync()) {
                            smem.bar_wg1_past_raw[raw_buf].arrive();
                        }
                        if (kv_block_idx + 1 == cur_job.num_kv_blocks) {
                            next_job = get_next_job(cur_job);
                        }
                        // The number of fp8 tokens of this thread is warp-uniform since topk % 8 == 0 (asserted by `run()`);
                        // DequantRows::run_block dispatches on it
                        uint32_t num_fp8_tokens = (uint32_t)std::clamp(
                            (int)get_num_orig_rows(cur_job, kv_block_idx) - (int)(local_warp_idx * DQ::NUM_ROWS_PER_WARP), 0, (int)DQ::NUM_ROWS_PER_WARP
                        ) / DQ::NUM_ROWS_PER_WAVEFRONT;
                        auto [indices_buf_idx, indices_bar_phase] = rs_back.get<NUM_INDICES_BUFS>();
                        auto [kv_slot_idx, kv_bar_phase] = rs_back.get<NUM_KV_SLOTS>();
                        dq_rows.run_block(num_fp8_tokens, kv_slot_idx, indices_buf_idx);
                        rs_back.update();
                    }
                    cur_job = next_job;
                } while (cur_job.is_valid);
            };
            // Both CTAs of a cluster hold their half of the token's dims, so one instantiation of DequantRows serves both
            DequantRows<Kernel, 0> dq_rows(smem, local_warp_idx, lane_idx, cta_idx);
            run_dequant_warps(dq_rows);
        } else {
            // KV Producer (prefill): gathers the whole bf16 KV block via TMA gather4
            if (elect_one_sync()) {
                OuterloopArgs cur_job = get_first_job();
                RingBufferState rs;
                uint32_t local_warp_idx = warp_idx - 4;
                do {
                    for (uint32_t i = 0; i < cur_job.num_kv_blocks; ++i) {
                        static_assert(B_TOPK % (4*4) == 0);
                        static constexpr uint32_t NUM_ROW_PER_WARP = B_TOPK / 4;
                        int4 topk_idxs[NUM_ROW_PER_WARP / 4];
                        CUTE_UNROLL
                        for (uint32_t local_row = 0; local_row < NUM_ROW_PER_WARP / 4; local_row += 1) {
                            uint32_t row = local_row * 4 * 4 + local_warp_idx * 4;
                            uint32_t pos = i * B_TOPK + row;
                            // Predicate the load on `pos < topk_length` to avoid reading OOB of the
                            // indices row in the last (partial) KV block. Chunks with no slot below
                            // topk_length are masked out by the indices generator warp anyway, so feed
                            // them invalid indices (-1), which tma_gather4 bounds-checks into zero-fill
                            topk_idxs[local_row] = pos < cur_job.topk_length
                                ? __ldg((int4*)(params.indices + cur_job.s_q_idx * params.stride_indices_s_q + pos))
                                : int4{-1, -1, -1, -1};
                        }

                        auto [kv_slot_idx, kv_bar_phase] = rs.get<NUM_KV_SLOTS>();
                        smem.bar_kv_slot_empty[kv_slot_idx].wait(kv_bar_phase^1);
                        CUTE_UNROLL
                        for (uint32_t local_row = 0; local_row < NUM_ROW_PER_WARP / 4; local_row += 1) {
                            uint32_t row = local_row * 4 * 4 + local_warp_idx * 4;
                            CUTE_UNROLL
                            for (uint32_t tile_idx = 0; tile_idx < (CLUSTER_SIZE == 2 ? (D_QK/2/64) : (D_QK/64)); ++tile_idx) {
                                /*
                                For cases where CLUSTER_SIZE == 1, each CTA reads the full K/V
                                For cases where CLUSTER_SIZE == 2, CTA0 reads KV[:, :D_QK/2] and CTA1 reads KV[:, D_QK/2:], since dual GeM< is used
                                */
                                if constexpr (CLUSTER_SIZE == 1) {
                                    ku::tma_gather4(
                                        &tma_params.tensor_map_kv,
                                        smem.bar_kv_slot_full[kv_slot_idx],
                                        smem.kv_slots[kv_slot_idx] + tile_idx * B_TOPK * 64 + row * 64,
                                        tile_idx * 64,
                                        topk_idxs[local_row],
                                        (int64_t)TMA::CacheHintSm90::EVICT_LAST
                                    );
                                } else {
                                    ku::tma_gather4_cta_group_2<true>(
                                        &tma_params.tensor_map_kv,
                                        smem.bar_kv_slot_full[kv_slot_idx],
                                        smem.kv_slots[kv_slot_idx] + tile_idx * B_TOPK * 64 + row * 64,
                                        tile_idx * 64 + cta_idx * (D_QK/2),
                                        topk_idxs[local_row],
                                        (int64_t)TMA::CacheHintSm90::EVICT_LAST
                                    );
                                }
                            }
                        }
                        rs.update();
                    }
                    cur_job = get_next_job(cur_job);
                } while (cur_job.is_valid);
            }
        }
    }

#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm100");
    }
#endif
}


template<typename Kernel>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, Kernel::CLUSTER_SIZE)
fwd_kernel(__grid_constant__ const typename Kernel::Params params, __grid_constant__ const typename Kernel::TMAParams tma_params, __grid_constant__ const typename Kernel::AuxParams aux_params) {
    Kernel::devfunc(params, tma_params, aux_params);
}


template<Config CONFIG>
void Kernel<CONFIG>::run(const Params &params) {
    KU_ASSERT(params.h_q == H_Q);
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.d_qk == D_QK);
    KU_ASSERT(params.d_v == D_VO);
    KU_ASSERT(params.stride_indices_s_q*sizeof(int) % 32 == 0, "indices.stride(0) must be 32B aligned, got %d elements (%d bytes)", params.stride_indices_s_q, (int)(params.stride_indices_s_q*sizeof(int)));

    KU_ASSERT(params.enable_q_norm == ENABLE_Q_NORM);
    KU_ASSERT(params.is_rope_neox_style == false && params.rope_dim == 64);
    KU_ASSERT(params.wv_group_size == WV_GROUP_SIZE);
    KU_ASSERT(params.num_per_channels == O_QUANT_TILE_SIZE);
    KU_ASSERT(params.use_tma_aligned_col_major_sf == true && params.round_sf == true && params.use_packed_ue8m0 == true);

    TMAParams tma_params = {};
    if constexpr (IS_DECODE) {
        KU_ASSERT(params.b == 1, "Only batch size 1 is supported for the fused decoding kernel");
        KU_ASSERT(params.model_type == MODEL_TYPE && params.extra_model_type == EXTRA_MODEL_TYPE);
        if (params.extra_topk > 0) {
            // A KV block may straddle the orig/extra boundary (see KVLocation::ORIG_AND_EXTRA). Since
            // one TMA gather4 covers 4 consecutive rows sharing one tensor map, the boundary (i.e.
            // topk) must be aligned to 4 rows; the common dequant path resolves the format per 8 rows
            KU_ASSERT(params.topk % (HAS_FP4_KV ? 8 : 4) == 0, "topk (%d) must be a multiple of %d when the extra KV cache is used", params.topk, HAS_FP4_KV ? 8 : 4);
        }
        auto make_kv_tensor_maps = []<ModelType MT>(bool is_extra, void *kv_ptr, int num_blocks, int64_t block_stride_bytes, int row_stride_bytes) -> std::pair<CUtensorMap, CUtensorMap> {
            auto make_map_for_cta = [&]<uint32_t CTA>() {
                using FC = KVFormatCta<MT, CTA>;
                return make_kv_quant_part_tensor_map<FC, FC::RAW_TOKEN_SMEM_STRIDE>(
                    is_extra ? "extra_kv" : "kv", kv_ptr, num_blocks, block_stride_bytes, row_stride_bytes,
                    CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
            };
            CUtensorMap map_cta1 = {};
            if constexpr (IS_2CTA) {
                map_cta1 = make_map_for_cta.template operator()<1>();
            }
            return {make_map_for_cta.template operator()<0>(), map_cta1};
        };
        {
            auto [map_cta0, map_cta1] = make_kv_tensor_maps.template operator()<MODEL_TYPE>(false, params.kv, params.num_blocks, params.stride_kv_block, params.stride_kv_row);
            tma_params.tensor_map_kv_fp8_part_cta0 = map_cta0;
            tma_params.tensor_map_kv_fp8_part_cta1 = map_cta1;
        }
        if (params.extra_topk > 0) {
            auto [map_cta0, map_cta1] = make_kv_tensor_maps.template operator()<EXTRA_MODEL_TYPE>(true, params.extra_kv, params.extra_num_blocks, params.stride_extra_kv_block, params.stride_extra_kv_row);
            if constexpr (HAS_FP4_KV) {
                tma_params.tensor_map_extra_kv_fp4_part_cta0 = map_cta0;
                tma_params.tensor_map_extra_kv_fp4_part_cta1 = map_cta1;
            } else {
                tma_params.tensor_map_extra_kv_fp8_part_cta0 = map_cta0;
                tma_params.tensor_map_extra_kv_fp8_part_cta1 = map_cta1;
            }
        }
    } else {
        tma_params.tensor_map_kv = ku::make_tensor_map(
            {(uint64_t)D_QK, (uint64_t)params.s_kv},
            {(uint64_t)params.stride_kv_s_kv * sizeof(bf16)},
            {64, 1},
            params.kv,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        );
    }

    auto aux_params = AuxParams {
    };  
    if constexpr (IS_DECODE) {
        aux_params.fast_divmod_page_block_size = cutlass::FastDivmod(params.page_block_size);
        aux_params.fast_divmod_extra_page_block_size = cutlass::FastDivmod(params.extra_kv != nullptr ? params.extra_page_block_size : 1);
    }

    auto kernel = &fwd_kernel<Kernel>;
    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    cutlass::ClusterLaunchParams launch_params = {
        dim3(params.s_q * CLUSTER_SIZE, 1, 1),
        dim3(NUM_THREADS, 1, 1),
        dim3(CLUSTER_SIZE, 1, 1),
        smem_size,
        params.stream
    };
    KU_CUTLASS_CHECK(cutlass::launch_kernel_on_cluster(
        launch_params, (void*)kernel, params, tma_params, aux_params
    ));
}


template<Config CONFIG>
void run_fused_norm_rope_attn_rope_cast_fwd_kernel(const ParamT<CONFIG.FWD_MODE>& params) {
    using KernelType = Kernel<CONFIG>;
    KernelType::run(params);
}


}
