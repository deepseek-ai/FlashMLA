#pragma once

#include <stdexcept>
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


template<typename PrecType, int DIM, int DIM2 = DIM>
constexpr auto getSmemLayoutK() {
    constexpr int headSizeBytes = sizeof(PrecType) * DIM;
    constexpr int headSizeBytes2 = sizeof(PrecType) * DIM2;

    if constexpr (headSizeBytes % 128 == 0 && headSizeBytes2 % 128 == 0) {
        return GMMA::Layout_K_SW128_Atom<PrecType>{};
    } else if constexpr (headSizeBytes % 64 == 0 && headSizeBytes2 % 64 == 0) {
        return GMMA::Layout_K_SW64_Atom<PrecType>{};
    } else {
        return GMMA::Layout_K_SW32_Atom<PrecType>{};
    }
}

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::bfloat16_t, int kHeadDimV_ = 0>
struct Flash_fwd_kernel_traits_mla {
    using Element = elem_type;
    using ElementAccum = float;
    using index_t = int64_t;

    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;
    // sm86 legacy fix: a BlockM=16 tile uses a single M-warp (m16 MMA), the original BlockM=32 uses 2.
    static constexpr int kNWarpsM = kBlockM_ >= 32 ? 2 : 1;
    static constexpr int kNWarpsN = kNWarps / kNWarpsM;
    static_assert(kNWarps == kNWarpsM * kNWarpsN);

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kHeadDimV = kHeadDimV_ != 0 ? kHeadDimV_ : kHeadDim;
    static_assert(kHeadDimV % 32 == 0);
    static_assert(kHeadDimV <= kHeadDim);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
    static constexpr int kNumInnerStagesK = 3;
    static_assert(kHeadDim % kNumInnerStagesK == 0, "kHeadDim must be divisible by kNumInnerStagesK");

    using MMA_Atom_Arch = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
    using TiledMma = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<Int<kNWarpsM>, Int<kNWarpsN>, _1>>,
        Tile<Int<16 * kNWarpsM>, Int<8 * kNWarpsN>, _16>>;

    // static constexpr int AtomLayoutNO = kNThreads / kNThreadsS;
    // using TiledMmaO = decltype(make_tiled_mma(
    //         cute::GMMA::rs_op_selector<Element, Element, ElementAccum, Shape<Int<kBlockM>, Int<kHeadDimV / AtomLayoutNO>, Int<kBlockN>>,
    //                 GMMA::Major::K, GMMA::Major::MN>(),
    //         Layout<Shape<Int<kNWarpsS / 4>, Int<AtomLayoutNO>, _1>>{}));

    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemLayoutSplitQ = decltype(composition(
        SmemLayoutQ{},
        make_layout(Shape<Int<kBlockM>, Int<kHeadDim / kNumInnerStagesK>, Int<kNumInnerStagesK>>{})));

    using kP = Int<2>; // pipeline count
    using SmemLayoutK = decltype(tile_to_shape(
            getSmemLayoutK<Element, kHeadDim, kHeadDimV>(),
            Shape<Int<kBlockN>, Int<kHeadDim>, kP>{}));
    using SmemLayoutSplitK = decltype(composition(
        SmemLayoutK{}, make_layout(Shape<Int<kBlockN>, Int<kHeadDim/kNumInnerStagesK>, Int<kNumInnerStagesK>, kP>{})));

    using SmemLayoutV = decltype(tile_to_shape(
            getSmemLayoutK<Element, kHeadDim, kHeadDimV>(),
            Shape<Int<kBlockN>, Int<kHeadDimV>, kP>{}));
    // Use SmemLayoutK (not SmemLayoutV) so the pipeline stride matches the actual K buffer layout.
    // SmemLayoutV has pipeline stride cosize(32,512)=16384, but K data for pipe=1 starts at cosize(32,576)=18432.
    using SmemLayoutVtransposed = decltype(composition(SmemLayoutK{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>, kP>{}, Stride<Int<kBlockN>,_1, Int<kHeadDim * kBlockN>>{})));

    static constexpr int kBlockKSmemP = kBlockN % 64 == 0 ? 64 : 32;
    static constexpr int kSwizzleP = kBlockKSmemP == 32 ? 2 : 3;
    using SmemLayoutAtomP = decltype(
        composition(Swizzle<kSwizzleP, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmemP>>,
                           Stride<Int<kBlockKSmemP>, _1>>{}));
    using SmemLayoutP = decltype(tile_to_shape(
        SmemLayoutAtomP{},
        Shape<Int<kBlockM>, Int<kBlockN>>{}));

    using SmemLayoutAtomO = decltype(composition(
            Swizzle<kSwizzle, 3, 3>{},
            Layout<Shape<Int<8>, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
            SmemLayoutAtomO{},
            Shape<Int<kBlockM>, Int<kHeadDimV>>{}));
    
    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, elem_type>;
    // Smaller copy atoms for B operand: each warp covers N=8 (one MMA atom),
    // so B fragment has only 4 vals/thread instead of 8.
    using SmemCopyAtomB = Copy_Atom<SM75_U32x2_LDSM_N, Element>;
    using SmemCopyAtomBTransposed = Copy_Atom<SM75_U16x4_LDSM_T, elem_type>;
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    using Gmem_copy_struct = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    // static constexpr int kNThreadsLoad = kNThreads - kNThreadsS;
    // static_assert(kNThreadsLoad % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");

    using GmemLayoutAtom = Layout<
            Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
            Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopy = decltype(make_tiled_copy(
            Copy_Atom<Gmem_copy_struct, Element>{},
            GmemLayoutAtom{},
            Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read

    using GmemLayoutAtomO = Layout<
            Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
            Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyO = decltype(make_tiled_copy(
            Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
            GmemLayoutAtomO{},
            Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store

    static constexpr int kGmemElemsPerLoadAccum = sizeof(cute::uint128_t) / sizeof(ElementAccum);
    static constexpr int kGmemThreadsPerRowAccum = kBlockKSmem / kGmemElemsPerLoadAccum;
    using GmemLayoutAtomOaccum = Layout<
            Shape<Int<kNThreads / kGmemThreadsPerRowAccum>, Int<kGmemThreadsPerRowAccum>>,
            Stride<Int<kGmemThreadsPerRowAccum>, _1>>;
    using GmemTiledCopyOaccum = decltype(make_tiled_copy(
            Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
            GmemLayoutAtomOaccum{},
            Layout<Shape<_1, _4>>{}));  // Val layout, 4 vals per store
};

namespace flash {

using namespace cute;

template<typename Kernel_traits>
struct SharedStorageMLA {
    union {
        struct {
            cute::array_aligned<typename Kernel_traits::Element, cute::cosize_v<typename Kernel_traits::SmemLayoutQ>> smem_q;
            cute::array_aligned<typename Kernel_traits::Element, cute::cosize_v<typename Kernel_traits::SmemLayoutK>> smem_k;  // Double buffer
        };
        struct {
            cute::array_aligned<typename Kernel_traits::ElementAccum, cute::cosize_v<typename Kernel_traits::SmemLayoutO>> smem_o;
        };
    };
    // P intermediate buffer for PV GEMM (kBlockM x kBlockN bf16)
    cute::array_aligned<typename Kernel_traits::Element, cute::cosize_v<typename Kernel_traits::SmemLayoutP>> smem_p;
    // Cross-warp reduction buffer: kBlockM rows x (kNWarpsN+1) stride (padding avoids bank conflicts)
    cute::array_aligned<typename Kernel_traits::ElementAccum, Kernel_traits::kBlockM * (Kernel_traits::kNWarpsN + 1)> smem_reduce;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Split, typename SharedStorage, typename AccO, typename Softmax>
__forceinline__ __device__ void store(const Flash_fwd_mla_params &params, const int bidb, const int bidh, const int m_block, const int n_split_idx,
                                      SharedStorage &shared_storage, AccO tOrO, Softmax softmax,
                                      float *smem_reduce, int n_warp_idx, const int *row_indices, int reduce_stride) {
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarpsN = Kernel_traits::kNWarpsN;
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    const int tidx = threadIdx.x;

    typename Kernel_traits::TiledMma tiled_mma_o;
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);

    // Epilogue

    const int split_offset = __ldg(params.num_splits_ptr + bidb);

    Tensor lse = softmax.template normalize_softmax_lse_cross_warp</*Is_dropout=*/false, Split, kNWarpsN>(
        tOrO, params.scale_softmax, smem_reduce, n_warp_idx, row_indices, reduce_stride);

    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;
    Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(shared_storage.smem_o.data())), typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    using SmemTiledCopyO = std::conditional_t<
            !Split,
            typename Kernel_traits::SmemCopyAtomO,
            typename Kernel_traits::SmemCopyAtomOaccum
    >;
    auto smem_tiled_copy_Oaccum = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma_o);
    auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor rO = flash::convert_type<ElementO>(tOrO);
    Tensor taccOrOaccum = smem_thr_copy_Oaccum.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsOaccum = smem_thr_copy_Oaccum.partition_D(sOaccum);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    cute::copy(smem_tiled_copy_Oaccum, taccOrOaccum, taccOsOaccum);

    const index_t row_offset_o = bidb * params.o_batch_stride + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_oaccum = (((split_offset + n_split_idx) * params.h + bidh) * params.seqlen_q + m_block * kBlockM) * params.d_v;
    const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
    const index_t row_offset_lseaccum = ((split_offset + n_split_idx) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;

    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                                 Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                 make_stride(Split ? kHeadDimV : params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + (Split ? row_offset_lseaccum : row_offset_lse)),
                                   Shape<Int<kBlockM>>{}, Stride<_1>{});

    using GmemTiledCopyO = std::conditional_t<!Split, typename Kernel_traits::GmemTiledCopyO, typename Kernel_traits::GmemTiledCopyOaccum>;
    GmemTiledCopyO gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

    __syncthreads();

    Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
    cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma_o.partition_C(caccO);                           // ((MMA=4, X), MMA_M, MMA_K=1)
    Tensor taccOcO_row = taccOcO(make_coord(0, _), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
    if (get<1>(taccOcO_row(0)) == 0) {
#pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < params.seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
        }
    }

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, params.seqlen_q - m_block * kBlockM
    );
}

template<typename Kernel_traits, bool Is_causal, bool Split, typename SharedStorage>
__forceinline__ __device__ void compute_attn_1rowblock_splitkv_mla(const Flash_fwd_mla_params &params,
                                                                   const int bidb, const int bidh, const int m_block,
                                                                   const int n_split_idx, const int seqlen_k,
                                                                   const int n_block_min, const int n_block_max,
                                                                   SharedStorage &shared_storage) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNumInnerStagesK = Kernel_traits::kNumInnerStagesK;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarpsN = Kernel_traits::kNWarpsN;
    constexpr int kNWarpsM = Kernel_traits::kNWarpsM;
    constexpr int kReduceStride = kNWarpsN + 1;  // padding to avoid bank conflicts

    // Cross-warp reduction setup
    const int warp_idx = tidx / 32;
    const int n_warp_idx = warp_idx / kNWarpsM;  // CuTe default column-major: stride=(1,kNWarpsM)
    float *smem_reduce = reinterpret_cast<float *>(shared_storage.smem_reduce.data());

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    // We move K and V to the last block.
    int n_block = n_block_max - 1;
    const int *block_table = params.block_table + bidb * params.block_table_batch_stride;
    int cur_block_table = __ldg(&block_table[n_block]);

    const index_t row_offset_q = bidb * params.q_batch_stride + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                                Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                make_stride(params.q_row_stride, _1{}));
    const index_t row_offset_k = (bidh / params.h_h_k_ratio) * params.k_head_stride;
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), typename Kernel_traits::SmemLayoutQ{});
    Tensor sQ_split = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), typename Kernel_traits::SmemLayoutSplitQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), typename Kernel_traits::SmemLayoutK{});
    Tensor sK_split = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), typename Kernel_traits::SmemLayoutSplitK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), typename Kernel_traits::SmemLayoutVtransposed{});

    typename Kernel_traits::GmemTiledCopy gmem_tiled_copy;
    auto gmem_thr_copy = gmem_tiled_copy.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy.partition_D(sK);
    auto K_PIPE_MAX = size<3>(tKsK); // pipeline count

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ_split(_, _, 0));            // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK_split(_, _, 0, 0));         // (MMA,MMA_N,MMA_K)
    Tensor tOrVt = thr_mma.partition_fragment_B(sVt(_, _, 0));                 // (MMA,MMA_M,MMA_N)

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ_split);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK_split);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomBTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    // P smem infrastructure for PV GEMM (route P through smem to avoid convert_layout_acc_Aregs issue)
    Tensor sP = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), typename Kernel_traits::SmemLayoutP{});
    // C-fragment -> smem (store P after softmax)
    auto smem_tiled_copy_PwriteC = make_tiled_copy_C(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto smem_thr_copy_PwriteC = smem_tiled_copy_PwriteC.get_thread_slice(tidx);
    // smem -> A-fragment (load P as GEMM A-operand)
    auto smem_tiled_copy_PreadA = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_PreadA = smem_tiled_copy_PreadA.get_thread_slice(tidx);
    Tensor tSrP = thr_mma.partition_fragment_A(sP);           // A-fragment for P
    Tensor tPsP_read = smem_thr_copy_PreadA.partition_S(sP);  // smem source for A-load

    // Extract row indices for cross-warp softmax reduction
    Tensor cS_id = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tScS_id = thr_mma.partition_C(cS_id);
    auto scores_id_rowcol = make_tensor(tScS_id.data(), flash::convert_layout_acc_rowcol(tScS_id.layout()));
    constexpr int kSoftmaxRows = decltype(size<0>(scores_id_rowcol))::value;
    int row_indices[kSoftmaxRows];
    #pragma unroll
    for (int mi = 0; mi < kSoftmaxRows; ++mi) {
        row_indices[mi] = int(get<0>(scores_id_rowcol(mi, 0)));
    }

    // PREDICATES
    //
    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Prologue
    // Read Q from gmem to smem, optionally apply rotary embedding.
    // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    flash::copy</*Is_even_MN*/false, /*Is_even_K*/true>(gmem_tiled_copy, tQgQ, tQsQ, tQcQ, tQpQ,
                                        params.seqlen_q - m_block * kBlockM);

    // Current pipe index in smem to read from
    int smem_pipe_read  = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = K_PIPE_MAX-1;
    // We need to clear the sK smem tiles because K is V.
    const index_t offset_k = cur_block_table * params.k_batch_stride;
    tKgK.data() = tKgK.data() + offset_k;
    Tensor tKsK_p = tKsK(_,_,_,0);
    flash::copy</*Is_even_MN*/false, /*Is_even_K*/true, /*Clear_OOB_MN=*/true>(gmem_tiled_copy, tKgK, tKsK_p, tKVcKV, tKVpKV,
                                       seqlen_k - n_block * kBlockN);
    tKgK.data() = tKgK.data() + -offset_k;
    cute::cp_async_fence();

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o);

    flash::Softmax<2 * size<1>(acc_o)> softmax;

    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    constexpr int n_masking_steps = !Is_causal ? 1 : cute::ceil_div(kBlockM, kBlockN) + 1;
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        if (n_block - 1 >= n_block_min) {
            // Advance gK
            const int *block_table = params.block_table + bidb * params.block_table_batch_stride;
            cur_block_table = __ldg(&block_table[n_block - 1]);
            const index_t offset_k = cur_block_table * params.k_batch_stride;
            tKgK.data() = tKgK.data() + offset_k;
            Tensor tKsK_p = tKsK(_,_,_,smem_pipe_write);
            flash::copy</*Is_even_MN=*/true, /*Is_even_K=*/true>(gmem_tiled_copy, tKgK, tKsK_p, tKVcKV, tKVpKV);
            tKgK.data() = tKgK.data() + -offset_k;
        }
        // This cp_async_fence needs to be in the if block, otherwise the synchronization
        // isn't right and we get race conditions.
        cute::cp_async_fence();
        flash::cp_async_wait<1>();
        __syncthreads();

        #pragma unroll
        for (int slice = 0; slice < kNumInnerStagesK; ++slice) {
            Tensor tSsQ_p = tSsQ(_,_,_,slice);
            Tensor tSsK_p = tSsK(_,_,_,slice, smem_pipe_read);
            flash::gemm_8x<false, false>(acc_s, tSrQ, tSrK, tSsQ_p, tSsK_p, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K);
        }

        const bool is_masking_step = masking_step >= 0;
        const bool is_first_masking_step = masking_step == 0;

        if (is_masking_step) {
            Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
            Tensor tScS = thr_mma.partition_C(cS);
#pragma unroll
            for (int i = 0; i < size(acc_s); ++i) {
                if constexpr (!Is_causal) {  // Just masking based on col
                    if (int(get<1>(tScS(i))) >= int(seqlen_k - n_block * kBlockN)) acc_s(i) = -INFINITY;
                } else {
                    // Ensure seqlen_k - 1 - (n_block * kBlockN + col) >= (seqlen_q - 1 - (m_block * kBlockM + row)) / ngroups
                    // col <= seqlen_k - 1 - n_block * kBlockN - (seqlen_q - 1 - (m_block * kBlockM + row)) / ngroups
                    int row = int(get<0>(tScS(i)));
                    int col_limit_right = seqlen_k - 1 - n_block * kBlockN - (params.seqlen_q - 1 - (m_block * kBlockM + row)) / params.ngroups;
                    if (int(get<1>(tScS(i))) > col_limit_right) acc_s(i) = -INFINITY;
                }
            }
        }

        // We have key_padding_mask so we'll need to Check_inf
        Tensor scale_o = is_first_masking_step
                            ? softmax.template softmax_cross_warp</*Is_first=*/true,  /*Check_inf=*/Is_causal, kNWarpsN>(acc_s, params.scale_softmax_log2, smem_reduce, n_warp_idx, row_indices, kReduceStride)
                            : is_masking_step ?
                            softmax.template softmax_cross_warp</*Is_first=*/false, /*Check_inf=*/Is_causal, kNWarpsN>(acc_s, params.scale_softmax_log2, smem_reduce, n_warp_idx, row_indices, kReduceStride)
                                            : softmax.template softmax_cross_warp</*Is_first=*/false, /*Check_inf=*/false, kNWarpsN>(acc_s, params.scale_softmax_log2, smem_reduce, n_warp_idx, row_indices, kReduceStride);

        // Convert acc_s from fp32 to fp16/bf16, store P to smem, then PV GEMM via smem
        Tensor rP = flash::convert_type<Element>(acc_s);
        flash::rescale_o(acc_o, scale_o);
        // Store P to shared memory (C-fragment layout -> smem)
        Tensor tPrP_src = smem_thr_copy_PwriteC.retile_S(rP);
        Tensor tPsP_dst = smem_thr_copy_PwriteC.partition_D(sP);
        cute::copy(smem_tiled_copy_PwriteC, tPrP_src, tPsP_dst);
        __syncthreads();
        // PV GEMM: both A (P) and B (Vt) from shared memory
        Tensor tOsVt_p = tOsVt(_,_,_,smem_pipe_read);
        flash::gemm_8x<false, false>(acc_o, tSrP, tOrVt, tPsP_read, tOsVt_p, tiled_mma, smem_tiled_copy_PreadA, smem_tiled_copy_V,
            smem_thr_copy_PreadA, smem_thr_copy_V);

        // Advance the smem pipe
        smem_pipe_write = smem_pipe_read;
        ++smem_pipe_read;
        smem_pipe_read = smem_pipe_read % K_PIPE_MAX;

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        if (n_block - 1 >= n_block_min) {
            // Advance gK
            const int *block_table = params.block_table + bidb * params.block_table_batch_stride;
            cur_block_table = __ldg(&block_table[n_block - 1]);
            const index_t offset_k = cur_block_table * params.k_batch_stride;
            tKgK.data() = tKgK.data() + offset_k;
            Tensor tKsK_p = tKsK(_,_,_,smem_pipe_write);
            flash::copy</*Is_even_MN=*/true, /*Is_even_K=*/true>(gmem_tiled_copy, tKgK, tKsK_p, tKVcKV, tKVpKV);
            tKgK.data() = tKgK.data() + -offset_k;
        }
        cute::cp_async_fence();
        flash::cp_async_wait<1>();
        __syncthreads();

        #pragma unroll
        for (int slice = 0; slice < kNumInnerStagesK; ++slice) {
            Tensor tSsQ_p = tSsQ(_,_,_,slice);
            Tensor tSsK_p = tSsK(_,_,_,slice, smem_pipe_read);
            flash::gemm_8x<false, false>(acc_s, tSrQ, tSrK, tSsQ_p, tSsK_p, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K);
        }

        Tensor scale_o = softmax.template softmax_cross_warp</*Is_first=*/false, /*Check_inf=*/false, kNWarpsN>(acc_s, params.scale_softmax_log2, smem_reduce, n_warp_idx, row_indices, kReduceStride);
        Tensor rP = flash::convert_type<Element>(acc_s);

        flash::rescale_o(acc_o, scale_o);
        // Store P to shared memory (C-fragment layout -> smem)
        {
            Tensor tPrP_src = smem_thr_copy_PwriteC.retile_S(rP);
            Tensor tPsP_dst = smem_thr_copy_PwriteC.partition_D(sP);
            cute::copy(smem_tiled_copy_PwriteC, tPrP_src, tPsP_dst);
        }
        __syncthreads();
        // PV GEMM: both A (P) and B (Vt) from shared memory
        Tensor tOsVt_p = tOsVt(_,_,_,smem_pipe_read);
        flash::gemm_8x<false, false>(acc_o, tSrP, tOrVt, tPsP_read, tOsVt_p, tiled_mma, smem_tiled_copy_PreadA, smem_tiled_copy_V,
            smem_thr_copy_PreadA, smem_thr_copy_V);
        smem_pipe_write = smem_pipe_read;
        ++smem_pipe_read;
        smem_pipe_read = smem_pipe_read % K_PIPE_MAX;
    }

    // Epilogue
    if(!Split) {
        store<Kernel_traits, false>(params, bidb, bidh, m_block, n_split_idx, shared_storage, acc_o, softmax,
                                    smem_reduce, n_warp_idx, row_indices, kReduceStride);
    } else {
        store<Kernel_traits, true>(params, bidb, bidh, m_block, n_split_idx, shared_storage, acc_o, softmax,
                                   smem_reduce, n_warp_idx, row_indices, kReduceStride);
    }
}

template<typename Kernel_traits, bool Is_causal, typename SharedStorage>
__global__ void __launch_bounds__(Kernel_traits::kNThreads)
flash_fwd_splitkv_mla_kernel(__grid_constant__ const Flash_fwd_mla_params params) {
    constexpr int kBlockN = Kernel_traits::kBlockN;
    const int m_block = blockIdx.x;
    const int bidh = blockIdx.y;
    const int partition_idx = blockIdx.z;

    extern __shared__ char shared_memory[];
    auto &shared_storage = *reinterpret_cast<SharedStorage *>(shared_memory);

    int *tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr + partition_idx * TileSchedulerMetaDataSize;
    int4 tile_scheduler_metadata = __ldg(reinterpret_cast<int4 *>(tile_scheduler_metadata_ptr));
    int begin_idx = tile_scheduler_metadata.x;
    int begin_seqlen = tile_scheduler_metadata.y;
    int end_idx = tile_scheduler_metadata.z;
    int end_seqlen = tile_scheduler_metadata.w;
    if (begin_idx >= params.b) return;
    int begin_n_split_idx = __ldg(tile_scheduler_metadata_ptr + 4);

#pragma unroll 1
    for (int batch_id = begin_idx; batch_id <= end_idx; ++batch_id) {
        const int n_split_idx = batch_id == begin_idx ? begin_n_split_idx : 0;
        const int seqlen_k = __ldg(params.cu_seqlens_k + batch_id);
        const int n_block_min = batch_id == begin_idx ? begin_seqlen / kBlockN : 0;
        const int n_block_max = batch_id == end_idx ? cute::ceil_div(end_seqlen, kBlockN) : cute::ceil_div(seqlen_k, kBlockN);
        const bool NoSplit = n_block_min == 0 && n_block_max == cute::ceil_div(seqlen_k, kBlockN);
        if (batch_id > begin_idx) {
            __syncthreads();  // Barrier between two tiles.
        }
        if (!NoSplit) {
            flash::compute_attn_1rowblock_splitkv_mla<Kernel_traits, Is_causal, true>(params, batch_id, bidh, m_block, n_split_idx, seqlen_k, n_block_min, n_block_max, shared_storage);
        } else {
            flash::compute_attn_1rowblock_splitkv_mla<Kernel_traits, Is_causal, false>(params, batch_id, bidh, m_block, n_split_idx, seqlen_k, n_block_min, n_block_max, shared_storage);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Element, typename ElementAccum, typename index_t, int kHeadDimV, int kMaxSplits>
__global__ void __launch_bounds__(256)
flash_fwd_splitkv_mla_combine_kernel(__grid_constant__ const Flash_fwd_mla_params params) {
    constexpr int kNThreads = 128;

    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
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
struct mha_fwd_splitkv_mla<T, Headdim, false> {
    static void run(Flash_fwd_mla_params &params, cudaStream_t stream) {
        static_assert(Headdim == 576);
        FLASH_ASSERT(params.d_v == 512);
        FLASH_ASSERT(params.k_ptr == params.v_ptr);  // Shared_KV
        // Original tile (BlockM=32, 8 warps) needs ~113 KB of shared memory: fine on A100 (164 KB) but not on sm_86
        // (99 KB opt-in limit; cudaFuncSetAttribute fails with "invalid argument"). Pick the largest tile that fits.
        int dev = 0, max_optin = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        using Traits32 = Flash_fwd_kernel_traits_mla<576, 32, 32, 8, T, 512>;
        using Traits16 = Flash_fwd_kernel_traits_mla<576, 16, 32, 4, T, 512>;
        if (sizeof(flash::SharedStorageMLA<Traits32>) <= (size_t)max_optin) {
            run_flash_splitkv_fwd_mla<Traits32, flash::SharedStorageMLA<Traits32>>(params, stream);
        } else if (sizeof(flash::SharedStorageMLA<Traits16>) <= (size_t)max_optin) {
            run_flash_splitkv_fwd_mla<Traits16, flash::SharedStorageMLA<Traits16>>(params, stream);
        } else {
            throw std::runtime_error("legacy sm80 MLA kernel does not fit in shared memory on this device");
        }
    }
};
