#pragma once

#include <cute/tensor.hpp>
#include "defines.h"

namespace nv_smallbatch::sm100 {

using namespace cute;

using _72 = Int<72>;
using _576 = Int<576>;

template<
    typename TMA,
    typename Tensor0,
    typename Tensor1
>
CUTE_DEVICE
void launch_tma_copy(
    const TMA &tma_copy,
    Tensor0 src,
    Tensor1 dst,
    transac_bar_t &bar,
    const cute::TMA::CacheHintSm90 &cache_hint = cute::TMA::CacheHintSm90::EVICT_NORMAL
) {
    auto thr_tma = tma_copy.get_slice(_0{});
    cute::copy(
        tma_copy.with(reinterpret_cast<typename transac_bar_t::ValueType&>(bar), 0, cache_hint),
        thr_tma.partition_S(src),
        thr_tma.partition_D(dst)
    );
}  // namespace nv_smallbatch::sm100

template<
    typename TiledMMA,
    typename TensorA,
    typename TensorB,
    typename TensorFragC
>
CUTE_DEVICE
void utcmma_ss(
    TiledMMA &tiled_mma,
    TensorA sA,
    TensorB sB,
    TensorFragC tC_frag,
    bool clear_accum
) {
    tiled_mma.accumulate_ = clear_accum ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
    ThrMMA thr_mma = tiled_mma.get_slice(_0{}); // Since A/B/C are already CTA-local tiles, this number does not matter
    auto sA_frag = thr_mma.partition_fragment_A(sA);
    auto sB_frag = thr_mma.partition_fragment_B(sB);
    static_assert(size<2>(sA_frag) == size<2>(sB_frag));
    static_assert(size<1>(sA_frag) == size<1>(tC_frag));
    static_assert(size<1>(sB_frag) == size<2>(tC_frag));
    CUTE_UNROLL
    for (int k = 0; k < size<2>(sA_frag); ++k) {
        cute::gemm(
            tiled_mma,
            sA_frag(_, _, k),
            sB_frag(_, _, k),
            tC_frag
        );
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
    }
}

template<
    typename TiledMMA,
    typename TensorA,
    typename TensorB,
    typename TensorFragC
>
CUTE_DEVICE
void utcmma_ts(
    TiledMMA &tiled_mma,
    TensorA tA_frag,
    TensorB sB,
    TensorFragC tC_frag,
    bool clear_accum
) {
    tiled_mma.accumulate_ = clear_accum ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
    ThrMMA thr_mma = tiled_mma.get_slice(_0{}); // Since A/B/C are already CTA-local tiles, this number does not matter
    auto sB_frag = thr_mma.partition_fragment_B(sB);
    static_assert(size<2>(tA_frag) == size<2>(sB_frag));
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tA_frag); ++k) {
        cute::gemm(
            tiled_mma,
            tA_frag(_, _, k),
            sB_frag(_, _, k),
            tC_frag
        );
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
    }
}

}
