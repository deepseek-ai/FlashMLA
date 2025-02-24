#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

using namespace cute;

#include "named_barrier.h"
#include "utils.h"
#include "softmax.h"
#include "static_switch.h"
#include "flash_mla.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Helper: Decide K-Layout at SMEM level given type and dimension.
/// Swizzling is determined primarily by alignment constraints.
/// Return GMMA Layout at compile time.
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PrecType, int DIM, int DIM2 = DIM>
constexpr auto getSmemLayoutK() {
    constexpr int headSizeBytes  = sizeof(PrecType) * DIM;
    constexpr int headSizeBytes2 = sizeof(PrecType) * DIM2;

    if constexpr (headSizeBytes % 128 == 0 && headSizeBytes2 % 128 == 0) {
        return GMMA::Layout_K_SW128_Atom<PrecType>{};
    } else if constexpr (headSizeBytes % 64 == 0 && headSizeBytes2 % 64 == 0) {
        return GMMA::Layout_K_SW64_Atom<PrecType>{};
    } else {
        return GMMA::Layout_K_SW32_Atom<PrecType>{};
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Kernel Trait: FWD MLA for Flash Attention
///  - Templated on HeadDim (kHeadDim_), block tiling, warp usage, etc.
///  - Provides all necessary sub-layouts for Q/K/V, softmax partials, etc.
////////////////////////////////////////////////////////////////////////////////////////////////////
template <
    int kHeadDim_,
    int kBlockM_,
    int kBlockN_,
    int kNumWarps_,
    typename ElemType = cutlass::bfloat16_t,
    int kHeadDimV_    = 0
>
struct FlashFwdKernelTraitsMLA {
    using Element      = ElemType;
    using ElementAccum = float;
    using IndexT       = int64_t;

    // Warp organization
    static constexpr int kNumWarps          = kNumWarps_;
    static constexpr int kNumThreads        = kNumWarps * 32;
    static constexpr int kNumWarpsSoftmax   = 4;
    static constexpr int kNumThreadsSoftmax = kNumWarpsSoftmax * 32;

    // Tiling in M, N, K
    static constexpr int kBlockM    = kBlockM_;
    static constexpr int kBlockN    = kBlockN_;
    static constexpr int kHeadDim   = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);

    // Possibly distinct V-dimension
    static constexpr int kHeadDimV = (kHeadDimV_ != 0) ? kHeadDimV_ : kHeadDim;
    static_assert(kHeadDimV % 32 == 0);
    static_assert(kHeadDimV <= kHeadDim);

    // SMEM swizzling for partial K/V
    static constexpr int kBlockKSmem = (kHeadDim % 64 == 0) ? 64 : 32;
    static constexpr int kSwizzle    = (kBlockKSmem == 32) ? 2 : 3;

    // GMMA Tiled Mma
    // Q*K -> S
    using TiledMma = decltype(make_tiled_mma(
        cute::GMMA::ss_op_selector<
            Element, Element, ElementAccum,
            Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>,
            GMMA::Major::K, GMMA::Major::K
        >(),
        Layout<Shape<Int<kNumWarpsSoftmax / 4>, _1, _1>>{}
    ));

    // S*V -> O
    // For the O “outer product,” we define the shape in [M, HeadDimV, N].
    static constexpr int AtomLayoutNO = kNumThreads / kNumThreadsSoftmax;
    using TiledMmaO = decltype(make_tiled_mma(
        cute::GMMA::rs_op_selector<
            Element, Element, ElementAccum,
            Shape<Int<kBlockM>, Int<kHeadDimV / AtomLayoutNO>, Int<kBlockN>>,
            GMMA::Major::K, GMMA::Major::MN
        >(),
        Layout<Shape<Int<kNumWarpsSoftmax / 4>, Int<AtomLayoutNO>, _1>>{}
    ));

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// SMEM Layout definitions: Q/K/V, P, row-scale, etc.
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    using SmemLayoutQ = decltype(
        tile_to_shape(
            getSmemLayoutK<Element, kHeadDim>(),
            Shape<Int<kBlockM>, Int<kHeadDim>>{}
        )
    );

    using SmemLayoutK = decltype(
        tile_to_shape(
            getSmemLayoutK<Element, kHeadDim, kHeadDimV>(),
            Shape<Int<kBlockN>, Int<kHeadDim>>{}
        )
    );

    using SmemLayoutV = decltype(
        tile_to_shape(
            getSmemLayoutK<Element, kHeadDim, kHeadDimV>(),
            Shape<Int<kBlockN>, Int<kHeadDimV>>{}
        )
    );
    using SmemLayoutVtransposed = decltype(
        composition(
            SmemLayoutV{},
            make_layout(
                Shape<Int<kHeadDimV>, Int<kBlockN>>{},
                GenRowMajor{}
            )
        )
    );

    // For partial S data (softmax region)
    using SmemLayoutP   = Layout<Shape<Shape<_2, _2>, Int<kNumThreadsSoftmax>, _1, Int<kBlockN / 8>>>;
    using SmemLayoutRow = Layout<Shape<_2, Int<kNumThreadsSoftmax>>, Stride<_1, _2>>;

    // Layout for the O tile in smem
    using SmemLayoutAtomO = decltype(
        composition(
            Swizzle<kSwizzle, 3, 3>{},
            Layout<Shape<Int<8>, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}
        )
    );
    using SmemLayoutO = decltype(
        tile_to_shape(
            SmemLayoutAtomO{},
            Shape<Int<kBlockM>, Int<kHeadDimV>>{}
        )
    );

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Copy Atoms for SMEM read/write
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    using SmemCopyAtomO        = Copy_Atom<SM90_U32x4_STSM_N, Element>;
    using SmemCopyAtomOaccum   = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// GMEM Tiled Copies for Q/K/V
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    static constexpr int kGmemElemsPerLoad  = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must align with vector load size");
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    using GmemCopyStruct        = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    static constexpr int kNumThreadsLoad    = kNumThreads - kNumThreadsSoftmax;
    static_assert(kNumThreadsLoad % kGmemThreadsPerRow == 0, "Thread counts must match row partitions");

    using GmemLayoutAtom = Layout<
        Shape<Int<kNumThreadsLoad / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
        Stride<Int<kGmemThreadsPerRow>, _1>
    >;
    using GmemTiledCopy = decltype(
        make_tiled_copy(
            Copy_Atom<GmemCopyStruct, Element>{},
            GmemLayoutAtom{},
            Layout<Shape<_1, _8>>{} // 8 vals per read
        )
    );

    // For storing O to GMEM
    using GmemLayoutAtomO = Layout<
        Shape<Int<kNumThreadsSoftmax / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
        Stride<Int<kGmemThreadsPerRow>, _1>
    >;
    using GmemTiledCopyO = decltype(
        make_tiled_copy(
            Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
            GmemLayoutAtomO{},
            Layout<Shape<_1, _8>>{} // 8 vals per store
        )
    );

    // For accumulation path (split)
    static constexpr int kGmemElemsPerLoadAccum  = sizeof(cute::uint128_t) / sizeof(ElementAccum);
    static constexpr int kGmemThreadsPerRowAccum = kBlockKSmem / kGmemElemsPerLoadAccum;
    using GmemLayoutAtomOaccum = Layout<
        Shape<Int<kNumThreadsSoftmax / kGmemThreadsPerRowAccum>, Int<kGmemThreadsPerRowAccum>>,
        Stride<Int<kGmemThreadsPerRowAccum>, _1>
    >;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(
            Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
            GmemLayoutAtomOaccum{},
            Layout<Shape<_1, _4>>{} // 4 vals per store
        )
    );
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Shared Storage Container for MLA
///  - Re-used union across Q/K/P/O or row sums, etc.
////////////////////////////////////////////////////////////////////////////////////////////////////
namespace flash {

using namespace cute;

template <typename KernelTraits>
struct SharedStorageMLA {
    union {
        struct {
            cute::array_aligned<typename KernelTraits::Element,
                cute::cosize_v<typename KernelTraits::SmemLayoutQ>> smem_q;
            cute::array_aligned<typename KernelTraits::Element,
                cute::cosize_v<typename KernelTraits::SmemLayoutK> * 2> smem_k;  // double buffer
            cute::array_aligned<typename KernelTraits::Element,
                cute::cosize_v<typename KernelTraits::SmemLayoutP>> smem_p;
            cute::array_aligned<typename KernelTraits::ElementAccum,
                cute::cosize_v<typename KernelTraits::SmemLayoutRow>> smem_scale;
        };
        struct {
            cute::array_aligned<typename KernelTraits::ElementAccum,
                cute::cosize_v<typename KernelTraits::SmemLayoutRow>> smem_max;
            cute::array_aligned<typename KernelTraits::ElementAccum,
                cute::cosize_v<typename KernelTraits::SmemLayoutRow>> smem_sum;
            cute::array_aligned<typename KernelTraits::ElementAccum,
                cute::cosize_v<typename KernelTraits::SmemLayoutO>> smem_o;
        };
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// store() Epilogue for partial or non-partial results
///  - Manages writing O/accumulation to global memory + writing out LSE for row block.
////////////////////////////////////////////////////////////////////////////////////////////////////
template <
    typename KernelTraits,
    bool Split,
    typename SharedStorage,
    typename AccO,
    typename Softmax
>
__forceinline__ __device__
void store(
    const Flash_fwd_mla_params &params,
    const int batch_id,
    const int head_id,
    const int m_block,
    const int n_split_idx,
    SharedStorage &shared_storage,
    AccO tOrO,
    Softmax softmax
) {
    constexpr int kBlockM       = KernelTraits::kBlockM;
    constexpr int kHeadDimV     = KernelTraits::kHeadDimV;
    constexpr int kNumThreadsS  = KernelTraits::kNumThreadsSoftmax;
    using Element               = typename KernelTraits::Element;
    using ElementAccum          = typename KernelTraits::ElementAccum;
    using IndexT                = typename KernelTraits::IndexT;

    const int tidx = threadIdx.x;

    typename KernelTraits::TiledMmaO tiled_mma_o;
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);

    // Softmax LSE for final normalization
    auto lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, Split>(tOrO, params.scale_softmax);

    // Decide if writing ephemeral partial results (float accumulation) or final (Element).
    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;

    // Prepare SMEM for O
    Tensor sOaccum = make_tensor(
        make_smem_ptr(reinterpret_cast<ElementO *>(shared_storage.smem_o.data())),
        typename KernelTraits::SmemLayoutO{}
    );
    auto smem_tiled_copy_Oaccum = make_tiled_copy_C(
        std::conditional_t<!Split,
            typename KernelTraits::SmemCopyAtomO,
            typename KernelTraits::SmemCopyAtomOaccum>{},
        tiled_mma_o
    );

    auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor rO      = flash::convert_type<ElementO>(tOrO);
    Tensor taccOrO = smem_thr_copy_Oaccum.retile_S(rO);
    Tensor taccOsO = smem_thr_copy_Oaccum.partition_D(sOaccum);

    __syncthreads();
    cute::copy(smem_tiled_copy_Oaccum, taccOrO, taccOsO);

    // Compute GMEM offsets
    const IndexT row_offset_o = batch_id * params.o_batch_stride
                                + m_block * kBlockM * params.o_row_stride
                                + head_id * params.o_head_stride;
    const IndexT row_offset_oaccum = (((__ldg(params.num_splits_ptr + batch_id) + n_split_idx)
                                       * params.h + head_id)
                                      * params.seqlen_q + (m_block * kBlockM)) * params.d_v;
    const IndexT row_offset_lse = (batch_id * params.h + head_id) * params.seqlen_q + m_block * kBlockM;
    const IndexT row_offset_lseaccum = (((__ldg(params.num_splits_ptr + batch_id) + n_split_idx)
                                         * params.h + head_id)
                                        * params.seqlen_q + (m_block * kBlockM));

    // Prepare GMEM for final or partial O
    Tensor gOaccum = make_tensor(
        make_gmem_ptr(
            reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr)
            + (Split ? row_offset_oaccum : row_offset_o)
        ),
        Shape<Int<kBlockM>, Int<kHeadDimV>>{},
        make_stride(Split ? kHeadDimV : params.o_row_stride, _1{})
    );

    // Prepare GMEM LSE
    Tensor gLSEaccum = make_tensor(
        make_gmem_ptr(
            reinterpret_cast<ElementAccum *>(
                Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr
            ) + (Split ? row_offset_lseaccum : row_offset_lse)
        ),
        Shape<Int<kBlockM>>{},
        Stride<_1>{}
    );

    // Tiled copy from SMEM -> GMEM for O
    using GmemTiledCopyOAccum = std::conditional_t<
        !Split,
        typename KernelTraits::GmemTiledCopyO,
        typename KernelTraits::GmemTiledCopyOaccum
    >;
    GmemTiledCopyOAccum gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);

    Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

    __syncthreads();

    // If out of range of the "softmax" portion, do not store
    if (tidx >= kNumThreadsS) { return; }

    // Load from SMEM
    Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
    cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);

    // Write out the LSE
    auto caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});
    auto taccOcO = thr_mma_o.partition_C(caccO);
    auto taccOcO_row = taccOcO(make_coord(0, _, 0), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));

    if (get<1>(taccOcO_row(0)) == 0) {
#pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < params.seqlen_q - m_block * kBlockM) {
                gLSEaccum(row) = lse(mi);
            }
        }
    }

    // Identity layout for sO
    auto cO = make_identity_tensor(
        make_shape(size<0>(sOaccum), size<1>(sOaccum))
    );
    auto tOcO = gmem_thr_copy_Oaccum.partition_D(cO);
    auto tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));

    // Copy final O back to GMEM
    flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO,
        params.seqlen_q - m_block * kBlockM
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// compute_attn_1rowblock_splitkv_mla()
///  - Core logic for Q*K -> S -> Softmax -> S*V -> O
///  - Includes partial accumulation for splits and optional causal masking.
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename KernelTraits, bool IsCausal, typename SharedStorage>
__forceinline__ __device__
void compute_attn_1rowblock_splitkv_mla(
    const Flash_fwd_mla_params &params,
    const int batch_id,
    const int head_id,
    const int m_block,
    const int n_split_idx,
    const int seqlen_k,
    const int n_block_min,
    const int n_block_max,
    const bool no_split,
    SharedStorage &shared_storage
) {
    constexpr int kBlockM         = KernelTraits::kBlockM;
    constexpr int kBlockN         = KernelTraits::kBlockN;
    constexpr int kHeadDim        = KernelTraits::kHeadDim;
    constexpr int kHeadDimV       = KernelTraits::kHeadDimV;
    constexpr int kNumThreads     = KernelTraits::kNumThreads;
    constexpr int kNumThreadsS    = KernelTraits::kNumThreadsSoftmax;
    using Element                 = typename KernelTraits::Element;
    using IndexT                  = typename KernelTraits::IndexT;

    static_assert(kNumThreads == 256 && kNumThreadsS == 128, "Expected 256 main threads, 128 softmax threads.");

    const int tidx = threadIdx.x;
    int n_block    = n_block_max - 1;

    // Smem pointers for Q, K, V, partial S, etc.
    Tensor sQ = make_tensor(
        make_smem_ptr(shared_storage.smem_q.data()),
        typename KernelTraits::SmemLayoutQ{}
    );
    Tensor sK = make_tensor(
        make_smem_ptr(shared_storage.smem_k.data()),
        typename KernelTraits::SmemLayoutK{}
    );
    Tensor sV = make_tensor(
        make_smem_ptr(shared_storage.smem_k.data()),
        typename KernelTraits::SmemLayoutV{}
    );
    Tensor sVt = make_tensor(
        make_smem_ptr(shared_storage.smem_k.data()),
        typename KernelTraits::SmemLayoutVtransposed{}
    );

    // Softmax partial
    Tensor sP  = make_tensor(
        make_smem_ptr(shared_storage.smem_p.data()),
        typename KernelTraits::SmemLayoutP{}
    );
    Tensor tPsP = sP(_, tidx % kNumThreadsS, _, _);

    // Row-based scale, sum, etc.
    Tensor sScale  = make_tensor(
        make_smem_ptr(shared_storage.smem_scale.data()),
        typename KernelTraits::SmemLayoutRow{}
    );
    Tensor tScale = sScale(_, tidx % kNumThreadsS);
    Tensor sRowMax = make_tensor(
        make_smem_ptr(shared_storage.smem_max.data()),
        typename KernelTraits::SmemLayoutRow{}
    );
    Tensor tRowMax = sRowMax(_, tidx % kNumThreadsS);
    Tensor sRowSum = make_tensor(
        make_smem_ptr(shared_storage.smem_sum.data()),
        typename KernelTraits::SmemLayoutRow{}
    );
    Tensor tRowSum = sRowSum(_, tidx % kNumThreadsS);

    // Mma for O
    typename KernelTraits::TiledMmaO tiled_mma_o;
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);
    Tensor tOrVt    = thr_mma_o.partition_fragment_B(sVt);
    Tensor tOrO     = partition_fragment_C(tiled_mma_o, Shape<Int<kBlockM>, Int<kHeadDimV>>{});
    clear(tOrO);

    // Combined softmax utility
    flash::Softmax<2 * size<1>(tOrO)> softmax;

    // Warp group logic: warpGroupIdx=0 does Q*K->S, warpGroupIdx=1 does async loads for next iteration
    int warpGroupIdx = cutlass::canonical_warp_group_idx();
    if (warpGroupIdx == 0) {
        // Main matmul Q*K -> S
        typename KernelTraits::TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_thread_slice(tidx);

        Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
        Tensor tSrK = thr_mma.partition_fragment_B(sK);

        // If n_block is odd => shift for double-buffer
        if (n_block % 2 == 1) {
            constexpr int sKOffset = size(sK);
            tSrK.data()   += (sKOffset / 8);
            tOrVt.data()  += (sKOffset / 8);
        }

        // We have a loop from n_block_max-1 down to n_block_min
        // Need to do “masking step(s)” for partial or causal scenarios.
        constexpr int nMaskingSteps = !IsCausal
                                      ? 1
                                      : cute::ceil_div(kBlockM, kBlockN) + 1;

#pragma unroll 1
        for (int masking
    const int hs = params.h * params.seqlen_q;
    const int batch_idx = bidx / hs;
    const int hs_idx = bidx % hs;

    const int split_offset = __ldg(params.num_splits_ptr + batch_idx);
    const int actual_num_splits = __ldg(params.num_splits_ptr + batch_idx + 1) - split_offset;
    FLASH_DEVICE_ASSERT(actual_num_splits <= kMaxSplits);
    if (actual_num_splits == 1) return;

    __shared__ ElementAccum sLseScale[kMaxSplits];

    const index_t row_offset_lseaccum = split_offset * hs + hs_idx;
    const index_t row_offset_lse = bidx;
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lseaccum_ptr) + row_offset_lseaccum),
                                   Shape<Int<kMaxSplits>>{}, make_stride(hs));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<_1>{}, Stride<_1>{});

    int warp_idx = cutlass::canonical_warp_idx_sync();
    if (warp_idx == 0) {
        constexpr int kNLsePerThread = cute::ceil_div(kMaxSplits, 32);

        float local_lse[kNLsePerThread];
        for (int i = 0; i < kNLsePerThread; ++i) {
            const int split = i * 32 + tidx;
            local_lse[i] = split < actual_num_splits ? gLSEaccum(split) : -INFINITY;
        }

        float max_lse = -INFINITY;
        for (int i = 0; i < kNLsePerThread; ++i) max_lse = max(max_lse, local_lse[i]);
        for (int offset = 16; offset >= 1; offset /= 2) max_lse = max(max_lse, __shfl_xor_sync(uint32_t(-1), max_lse, offset));
        max_lse = max_lse == -INFINITY ? 0.0f : max_lse;  // In case all local LSEs are -inf

        float sum_lse = 0;
        for (int i = 0; i < kNLsePerThread; ++i) sum_lse = sum_lse + expf(local_lse[i] - max_lse);
        for (int offset = 16; offset >= 1; offset /= 2) sum_lse = sum_lse + __shfl_xor_sync(uint32_t(-1), sum_lse, offset);

        float global_lse = (sum_lse == 0.f || sum_lse != sum_lse) ? INFINITY : logf(sum_lse) + max_lse;
        if (tidx == 0) gLSE(0) = global_lse;

        for (int i = 0; i < kNLsePerThread; ++i) {
            const int split = i * 32 + tidx;
            if (split < actual_num_splits) sLseScale[split] = expf(local_lse[i] - global_lse);
        }
    }
    __syncthreads();

    static_assert(kHeadDimV % kNThreads == 0);
    constexpr int Elements = kHeadDimV / kNThreads;
    const index_t row_offset_oaccum = (split_offset * hs + hs_idx) * kHeadDimV;
    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.oaccum_ptr) + row_offset_oaccum),
                                 Shape<Int<kHeadDimV>>{}, Stride<_1>{});
    using GmemTiledCopyOaccum = decltype(make_tiled_copy(
            Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
            Layout<Shape<Int<kNThreads>>>{},
            Layout<Shape<Int<Elements>>>{}));
    GmemTiledCopyOaccum gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_S(gOaccum);
    Tensor tOrOaccum = make_tensor<ElementAccum>(shape(tOgOaccum));
    Tensor tOrO = make_tensor<ElementAccum>(shape(tOgOaccum));
    clear(tOrO);

    for (int split = 0; split < actual_num_splits; ++split) {
        cute::copy(tOgOaccum, tOrOaccum);
        ElementAccum lse_scale = sLseScale[split];
        for (int i = 0; i < size(tOrO); ++i) {
            tOrO(i) += lse_scale * tOrOaccum(i);
        }
        tOgOaccum.data() = tOgOaccum.data() + hs * kHeadDimV;
    }

    Tensor rO = flash::convert_type<Element>(tOrO);
    const int head_idx = (bidx - batch_idx * hs) / params.seqlen_q;
    const int row = bidx - batch_idx * hs - head_idx * params.seqlen_q;
    auto o_ptr = reinterpret_cast<Element *>(params.o_ptr) + batch_idx * params.o_batch_stride + head_idx * params.o_head_stride + row * params.o_row_stride;
    Tensor gO = make_tensor(make_gmem_ptr(o_ptr + tidx * Elements), Shape<Int<decltype(size<0>(rO))::value>>{}, Stride<_1>{});
    cute::copy(rO, gO);
}

} // namespace flash

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, typename SharedStorage>
void run_flash_splitkv_fwd_mla(Flash_fwd_mla_params &params, cudaStream_t stream) {
    FLASH_ASSERT(params.page_block_size == Kernel_traits::kBlockN);
    const int num_m_block = cute::ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        auto kernel = &flash::flash_fwd_splitkv_mla_kernel<Kernel_traits, Is_causal, SharedStorage>;
        constexpr size_t smem_size = sizeof(SharedStorage);
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        kernel<<<dim3(num_m_block, params.h, params.num_sm_parts), Kernel_traits::kNThreads, smem_size, stream>>>(params);
    });
    CHECK_CUDA_KERNEL_LAUNCH();

    dim3 grid_combine(params.b * params.h * params.seqlen_q);
    MLA_NUM_SPLITS_SWITCH(params.num_sm_parts, kMaxSplits, [&] {
        auto combine_kernel = &flash::flash_fwd_splitkv_mla_combine_kernel<
                typename Kernel_traits::Element, typename Kernel_traits::ElementAccum, typename Kernel_traits::index_t, Kernel_traits::kHeadDimV, kMaxSplits>;
        combine_kernel<<<grid_combine, 128, 0, stream>>>(params);
    });
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<typename T, int Headdim>
void run_mha_fwd_splitkv_mla(Flash_fwd_mla_params &params, cudaStream_t stream) {
    static_assert(Headdim == 576);
    FLASH_ASSERT(params.d_v == 512);
    FLASH_ASSERT(params.k_ptr == params.v_ptr);  // Shared_KV
    using Kernel_traits = Flash_fwd_kernel_traits_mla<576, 64, 64, 8, T, 512>;
    run_flash_splitkv_fwd_mla<Kernel_traits, flash::SharedStorageMLA<Kernel_traits>>(params, stream);
}

static constexpr int MaxBatchSize = 4096;

__global__ void __launch_bounds__(256, 1, 1)
get_mla_metadata_kernel(__grid_constant__ const Mla_metadata_params params) {
    int *seqlens_k_ptr = params.seqlens_k_ptr;
    int *tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr;
    int *num_splits_ptr = params.num_splits_ptr;
    int batch_size = params.batch_size;
    int block_size_n = params.block_size_n;
    int fixed_overhead_num_blocks = params.fixed_overhead_num_blocks;
    int num_sm_parts = params.num_sm_parts;

    __shared__ int num_blocks_shared[MaxBatchSize];
    __shared__ int num_splits_shared[MaxBatchSize];

    int total_num_blocks = 0;
    for (int i = threadIdx.x; i < batch_size; i += 32) {
        int num_blocks = cutlass::ceil_div(seqlens_k_ptr[i], block_size_n);
        total_num_blocks += num_blocks + fixed_overhead_num_blocks;
        num_blocks_shared[i] = num_blocks;
    }
    for (int offset = 16; offset >= 1; offset /= 2) {
        total_num_blocks += __shfl_xor_sync(uint32_t(-1), total_num_blocks, offset);
    }
    __syncwarp();

    if (threadIdx.x == 0) {
        int payload = cutlass::ceil_div(total_num_blocks, num_sm_parts) + fixed_overhead_num_blocks;

        int now_idx = 0, now_block = 0, now_n_split_idx = 0, cum_num_splits = 0;
        num_splits_shared[0] = 0;
        for (int i = 0; i < num_sm_parts; ++i) {
            int tile_scheduler_metadata0[4], tile_scheduler_metadata1;
            tile_scheduler_metadata0[0] = now_idx;
            tile_scheduler_metadata0[1] = now_block * block_size_n;
            tile_scheduler_metadata1 = now_n_split_idx;
            int remain_payload = payload;
            while (now_idx < batch_size) {
                int num_blocks = num_blocks_shared[now_idx];
                int now_remain_blocks = num_blocks - now_block;
                if (remain_payload >= now_remain_blocks + fixed_overhead_num_blocks) {
                    cum_num_splits += now_n_split_idx + 1;
                    num_splits_shared[now_idx + 1] = cum_num_splits;
                    remain_payload -= now_remain_blocks + fixed_overhead_num_blocks;
                    ++now_idx;
                    now_block = 0;
                    now_n_split_idx = 0;
                } else {
                    if (remain_payload - fixed_overhead_num_blocks > 0) {
                        now_block += remain_payload - fixed_overhead_num_blocks;
                        ++now_n_split_idx;
                        remain_payload = 0;
                    }
                    break;
                }
            }
            tile_scheduler_metadata0[2] = now_block > 0 ? now_idx : now_idx - 1;
            tile_scheduler_metadata0[3] = now_block > 0 ? now_block * block_size_n : seqlens_k_ptr[now_idx - 1];
            *reinterpret_cast<int4 *>(tile_scheduler_metadata_ptr + i * TileSchedulerMetaDataSize) = *reinterpret_cast<int4 *>(tile_scheduler_metadata0);
            tile_scheduler_metadata_ptr[i * TileSchedulerMetaDataSize + 4] = tile_scheduler_metadata1;
        }
        FLASH_DEVICE_ASSERT(now_idx == batch_size && now_block == 0 && now_n_split_idx == 0);
    }
    __syncwarp();

    for (int i = threadIdx.x; i <= batch_size; i += 32) {
        num_splits_ptr[i] = num_splits_shared[i];
    }
}

void get_mla_metadata_func(Mla_metadata_params &params, cudaStream_t stream) {
    FLASH_ASSERT(params.batch_size < MaxBatchSize);
    get_mla_metadata_kernel<<<1, 32, 0, stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}
