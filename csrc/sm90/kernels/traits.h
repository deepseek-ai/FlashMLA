#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/barrier.h>

#include "config.h"

using TMABarrier = cutlass::arch::ClusterTransactionBarrier;
using namespace cute;

template<typename InputT_>
struct Traits {
    using InputT = InputT_;
    
    static constexpr int BLOCK_SIZE_M = Config::BLOCK_SIZE_M;
    static constexpr int PAGE_BLOCK_SIZE = Config::PAGE_BLOCK_SIZE;
    static constexpr int HEAD_DIM_K = Config::HEAD_DIM_K;
    static constexpr int HEAD_DIM_V = Config::HEAD_DIM_V;

    static constexpr int NUM_THREADS = 256;

    static_assert(std::is_same_v<InputT, cutlass::bfloat16_t> || std::is_same_v<InputT, cutlass::half_t>);

    using TiledMMA_QK_sQ = decltype(make_tiled_mma(
        GMMA::ss_op_selector<InputT, InputT, float, Shape<Int<BLOCK_SIZE_M>, Int<PAGE_BLOCK_SIZE>, Int<HEAD_DIM_K>>, GMMA::Major::K, GMMA::Major::K>(),
        Layout<Shape<_1, _1, _1>>{}
    ));

    using TiledMMA_QK_rQ = decltype(make_tiled_mma(
        GMMA::rs_op_selector<InputT, InputT, float, Shape<Int<BLOCK_SIZE_M>, Int<PAGE_BLOCK_SIZE>, Int<HEAD_DIM_K>>, GMMA::Major::K, GMMA::Major::K>(),
        Layout<Shape<_1, _1, _1>>{}
    ));

    using TiledMMA_PV_LocalP = decltype(make_tiled_mma(
        GMMA::rs_op_selector<InputT, InputT, float, Shape<Int<BLOCK_SIZE_M>, Int<HEAD_DIM_V/2>, Int<PAGE_BLOCK_SIZE>>, GMMA::Major::K, GMMA::Major::MN>(),
        Layout<Shape<_1, _1, _1>>{}
    ));

    using TiledMMA_PV_RemoteP = decltype(make_tiled_mma(
        GMMA::ss_op_selector<InputT, InputT, float, Shape<Int<BLOCK_SIZE_M>, Int<HEAD_DIM_V/2>, Int<PAGE_BLOCK_SIZE>>, GMMA::Major::K, GMMA::Major::MN>(),
        Layout<Shape<_1, _1, _1>>{}
    ));

    using SmemLayoutQ = decltype(tile_to_shape(
        GMMA::Layout_K_SW128_Atom<InputT>{},
        Shape<Int<BLOCK_SIZE_M>, Int<HEAD_DIM_K>>{}
    ));

    using SmemLayoutK = decltype(tile_to_shape(
        GMMA::Layout_K_SW128_Atom<InputT>{},
        Shape<Int<PAGE_BLOCK_SIZE>, Int<HEAD_DIM_K>>{}
    ));

    using SmemLayoutV = decltype(composition(
        SmemLayoutK{},
        make_layout(Shape<Int<HEAD_DIM_V>, Int<PAGE_BLOCK_SIZE>>{}, GenRowMajor{})
    ));	// A transposed version of SmemLayoutK

    using SmemLayoutP0 = decltype(tile_to_shape(
        GMMA::Layout_K_SW128_Atom<InputT>{},
        Shape<Int<BLOCK_SIZE_M>, Int<PAGE_BLOCK_SIZE>>{}
    ));

    using rP0Layout = decltype(layout(partition_fragment_C(
        TiledMMA_QK_sQ{},
        Shape<Int<BLOCK_SIZE_M>, Int<PAGE_BLOCK_SIZE>>{}
    )));

    struct SharedMemoryPlan {
        cute::array_aligned<InputT, cosize_v<SmemLayoutQ>> smem_sQ;
        cute::array_aligned<InputT, cosize_v<SmemLayoutK>> smem_sK0;
        cute::array_aligned<InputT, cosize_v<SmemLayoutK>> smem_sK1;
        cute::array_aligned<InputT, cosize_v<SmemLayoutP0>> smem_sP0;
        cute::array_aligned<float, BLOCK_SIZE_M> smem_sM;
        cute::array_aligned<float, 2*BLOCK_SIZE_M> sL_reduction_wksp;
        cute::array_aligned<float, BLOCK_SIZE_M> smem_sScale0;
        cute::array_aligned<float, BLOCK_SIZE_M> smem_sScale1;
        TMABarrier barriers_K0[HEAD_DIM_K/64];
        TMABarrier barriers_K1[HEAD_DIM_K/64];
        TMABarrier barrier_Q;
    };

};

template<
    typename ShapeQ, typename TMA_Q,
    typename ShapeK, typename TMA_K,
    typename ShapeO, typename TMA_O
>
struct TmaParams {
    ShapeQ shape_Q;
    TMA_Q tma_Q;
    ShapeK shape_K;
    TMA_K tma_K;
    ShapeO shape_O;
    TMA_O tma_O;
};

enum NamedBarriers : int {
    sScale0Ready = 0,
    sScale1Ready = 1,
    sP0Ready = 2,
    rO1sP0sV0RIssued = 3,
    sMInitialized = 4,
};
