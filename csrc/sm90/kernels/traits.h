#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/barrier.h>

#include "config.h"

using TMABarrier = cutlass::arch::ClusterTransactionBarrier;
using namespace cute;

using TMABarrier = cutlass::arch::ClusterTransactionBarrier;
template<typename T,int D,int D2=D,GMMA::Major M=GMMA::Major::K>
constexpr auto get_smem_layoutK() {
    constexpr int b0 = sizeof(T)*D, b1=sizeof(T)*D2;
    if constexpr(M==GMMA::Major::K){
        if constexpr(b0%128==0 && b1%128==0) return GMMA::Layout_K_SW128_Atom<T>{};
        else if constexpr(b0%64==0 && b1%64==0) return GMMA::Layout_K_SW64_Atom<T>{};
        else                                   return GMMA::Layout_K_SW32_Atom<T>{};
    } else {
        if constexpr(b0%128==0 && b1%128==0) return GMMA::Layout_MN_SW128_Atom<T>{};
        else if constexpr(b0%64==0 && b1%64==0) return GMMA::Layout_MN_SW64_Atom<T>{};
        else                                   return GMMA::Layout_MN_SW32_Atom<T>{};
    }
}

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

template<typename T_, typename OutputT_>
struct TraitsFP8 {
    using InputT = T_;
    using OutputT = OutputT_;
    static constexpr bool IsFp8 = std::is_same_v<InputT, cutlass::float_e4m3_t>;

    static constexpr int BLOCK_SIZE_M    = Config::BLOCK_SIZE_M;      // 64
    static constexpr int PAGE_BLOCK_SIZE = Config::PAGE_BLOCK_SIZE;   // 64
    static constexpr int HEAD_DIM_K      = Config::HEAD_DIM_K;        // 576
    static constexpr int HEAD_DIM_V      = Config::HEAD_DIM_V;        // 512
    static constexpr int NUM_THREADS     = 256;

    static_assert( std::is_same_v<InputT, cutlass::float_e4m3_t> );

    static_assert( std::is_same_v<OutputT, cutlass::bfloat16_t>  ||
                   std::is_same_v<OutputT, cutlass::half_t> );

    using TiledMMA_QK_sQ = decltype(make_tiled_mma(
         GMMA::ss_op_selector<InputT,InputT,float,
               Shape<Int<BLOCK_SIZE_M>,Int<PAGE_BLOCK_SIZE>,Int<HEAD_DIM_K>>,
               GMMA::Major::K, GMMA::Major::K>(),
         Layout<Shape<_1,_1,_1>>{}));

    using TiledMMA_PV_LocalP = decltype(make_tiled_mma(
         GMMA::rs_op_selector<InputT,InputT,float,
               Shape<Int<BLOCK_SIZE_M>,Int<HEAD_DIM_V / 2>,Int<PAGE_BLOCK_SIZE>>,
               GMMA::Major::K, GMMA::Major::K>(),
         Layout<Shape<_1,_1,_1>>{}));


    using TiledMMA_PV_RemoteP = decltype(make_tiled_mma(
         GMMA::ss_op_selector<InputT,InputT,float,
               Shape<Int<BLOCK_SIZE_M>,Int<HEAD_DIM_V / 2>,Int<PAGE_BLOCK_SIZE>>,
               GMMA::Major::K, GMMA::Major::K>(),
         Layout<Shape<_1,_1,_1>>{}));

    using SmemLayoutQ = decltype(tile_to_shape(
         get_smem_layoutK<InputT,HEAD_DIM_K>(),
         Shape<Int<BLOCK_SIZE_M>,Int<HEAD_DIM_K>>{}));

    using SmemLayoutK = decltype(tile_to_shape(
         get_smem_layoutK<InputT, HEAD_DIM_K, HEAD_DIM_V>(),
         Shape<Int<PAGE_BLOCK_SIZE>,Int<HEAD_DIM_K>>{}));
 
    using SmemLayoutV          = SmemLayoutK;
    using SmemLayoutV_Trans    = decltype(
         composition(SmemLayoutK{},
             make_layout(Shape<Int<HEAD_DIM_V>,Int<PAGE_BLOCK_SIZE>>{},GenRowMajor{})));

    using SmemLayoutP0 = decltype(tile_to_shape(
        get_smem_layoutK<InputT,HEAD_DIM_K>(),
        Shape<Int<BLOCK_SIZE_M>, Int<PAGE_BLOCK_SIZE>>{}
    ));

    using rP0Layout = decltype(layout(partition_fragment_C(
        TiledMMA_QK_sQ{},
        Shape<Int<BLOCK_SIZE_M>, Int<PAGE_BLOCK_SIZE>>{}
    )));
                       
    using SmemLayoutPi = Layout<Shape <_128,Shape <_2, _4>>,
                                Stride< _2,Stride<_1,_256>>>;

    using RegLayout = Layout<Shape <Shape<Shape<_2,_4>,_1>,_1,_1>,
                            Stride<Stride<Stride<_1,_2>,_0>,_0,_0>>;

    struct SharedMemoryPlan {
        alignas(16) InputT smem_sQ[BLOCK_SIZE_M * HEAD_DIM_K];
        alignas(16) InputT smem_sK0[PAGE_BLOCK_SIZE * HEAD_DIM_K]; // overlap Sout
        alignas(16) InputT smem_sK1[PAGE_BLOCK_SIZE * HEAD_DIM_K]; // overlap Sout
        alignas(16) InputT smem_vt0[PAGE_BLOCK_SIZE * HEAD_DIM_V]; // overlap Sout
        alignas(16) InputT smem_vt1[PAGE_BLOCK_SIZE * HEAD_DIM_V]; // overlap Sout
        alignas(16) uint16_t smem_sP0[64 * 32];
        alignas(16) uint16_t smem_sP1[64 * 32];

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
    sPreV0ZeroReady = 5,
    sPreV1ZeroReady = 6,
    sV0ZeroReady = 7,
    sV1ZeroReady = 8
};
