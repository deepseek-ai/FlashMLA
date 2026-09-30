#pragma once

#include "params.h"

// Each V4.1 token stores [raw data, 1-byte scales], including the RoPE dims:
// FP8: 512 B e4m3 + 16 B ue8m0; FP4: 256 B e2m1 + 32 B e4m3. See tests/quant.py.
template<ModelType MT>
struct KVCacheFormat {
    static constexpr ModelType MODEL_TYPE = MT;
    static constexpr bool IS_FP4 = MT == ModelType::V41_FP4;
    static constexpr int D_QK = 512;
    static constexpr int D_ROPE = 64;
    static constexpr int D_NOPE = D_QK - D_ROPE;
    static constexpr int D_FP4 = IS_FP4 ? D_QK : 0;    // Dimensions stored as fp4 e2m1
    static constexpr int D_FP8 = IS_FP4 ? 0 : D_QK;    // Dimensions stored as fp8 e4m3, needing dequant
    static constexpr int QUANT_TILE_SIZE = IS_FP4 ? 16 : 32;                            // Dimensions sharing one scale
    static constexpr int NUM_SCALES_EACH_TOKEN = D_QK / QUANT_TILE_SIZE;                // 16 (V41) / 32 (fp4)
    static constexpr int QUANT_BYTES = D_FP4 / 2 + D_FP8;                               // The quantized (fp4 / fp8) data of a token
    static constexpr int BYTES_PER_TOKEN = QUANT_BYTES + NUM_SCALES_EACH_TOKEN;   // 528 / 288
};

// A CTA's contiguous share of a token. In a pair, CTA0 loads its scales separately;
// CTA1's TMA view includes the trailing scales. Shared-memory padding is kernel-specific.
template<typename F, int NUM_CTAS, int CTA>
struct KVCachePart : F {
    static_assert((NUM_CTAS == 1 || NUM_CTAS == 2) && 0 <= CTA && CTA < NUM_CTAS);
    static constexpr int QUANT_DIMS = F::D_QK / NUM_CTAS;
    static constexpr int QUANT_BYTES = F::QUANT_BYTES / NUM_CTAS;
    static constexpr int NUM_SCALES = F::NUM_SCALES_EACH_TOKEN / NUM_CTAS;
    static constexpr int RAW_TOKEN_OFFSET = CTA * QUANT_BYTES;
    static constexpr int SCALE_TOKEN_OFFSET = F::QUANT_BYTES + CTA * NUM_SCALES;
    static constexpr bool SEPARATE_SCALES = NUM_CTAS == 2 && CTA == 0;
    static constexpr int RAW_TOKEN_DATA_BYTES = QUANT_BYTES + (SEPARATE_SCALES ? 0 : F::NUM_SCALES_EACH_TOKEN);
    static constexpr int SCALE_OFFSET = SCALE_TOKEN_OFFSET - RAW_TOKEN_OFFSET;
};

// Runtime counterpart of KVCacheFormat<MT>::BYTES_PER_TOKEN
constexpr int kv_cache_bytes_per_token(ModelType mt) {
    switch (mt) {
        case ModelType::V41: return KVCacheFormat<ModelType::V41>::BYTES_PER_TOKEN;
        case ModelType::V41_FP4: return KVCacheFormat<ModelType::V41_FP4>::BYTES_PER_TOKEN;
    }
    return 0;
}

// The (kv, extra_kv) format pairs that exist: extra_kv has the format of kv, or is the fp4 cache next to a (fp8) kv.
constexpr bool is_valid_kv_format_pair(ModelType kv, ModelType extra_kv) {
    return extra_kv == kv || (kv == ModelType::V41 && extra_kv == ModelType::V41_FP4);
}

template<ModelType KV, ModelType EXTRA_KV = KV>
struct KVFormatPair {
    static_assert(is_valid_kv_format_pair(KV, EXTRA_KV));
    static constexpr ModelType kv = KV, extra_kv = EXTRA_KV;
};

// A list of KVFormatPair, see dispatch_kv_formats in csrc/api/common.h
template<typename... Pairs>
struct KVFormatPairs {};
