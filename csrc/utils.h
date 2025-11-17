#pragma once

#include <cuda_bf16.h>
#include <cute/tensor.hpp>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include "sm90/decode/dense/fp8_transpose_v.h"

using namespace cute;

// Fill out-of-bound V with 0.0
// We must fill it since it may contain NaN, which may propagate to the final result
template<
    typename T
>
CUTLASS_DEVICE void fill_oob_KV(
    typename T::InputT* sV_ptr, // ptr of tensor(tile_to_shape(Shape<Int<T::HEAD_DIM_V/8>,Int<T::PAGE_BLOCK_SIZE>>{}))
    int valid_window_size,
    int idx_in_warpgroup
) {
    Tensor sV_int64 = make_tensor(
        make_smem_ptr((int64_t*)(sV_ptr)),
        tile_to_shape(
            GMMA::Layout_K_SW64_Atom<cute::int64_t>{},
            Shape<Int<64>, Int<8>>{} // (64, 64/(64/8))
        )
    );
    valid_window_size = max(valid_window_size, 0);
    int head_dim_size = 8;	// 128%head_dim_size == 0 should holds
    for (int token_idx = valid_window_size + (idx_in_warpgroup/head_dim_size); token_idx < size<0>(sV_int64); token_idx += (128/head_dim_size)) {
        sV_int64(token_idx, idx_in_warpgroup%head_dim_size) = 0;
    }
}


template <typename T>
CUTLASS_DEVICE void fp8_transpose_v(typename T::InputT* sK_ptr, 
                     typename T::InputT* sVt_ptr, 
                     int tile_id)
{
    // every tile: (64, 64)
    using Fp8Trans  = SmemTransposeFp8_64x64<T::PAGE_BLOCK_SIZE, T::HEAD_DIM_V>;
    Fp8Trans trans;
    Tensor src = cute::as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(sK_ptr),
                    typename Fp8Trans::SmemLayoutTransposeV{}));

    Tensor dst = cute::as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(sVt_ptr),
                    typename Fp8Trans::SmemLayoutTransposeVt{}));
    trans.transpose(
        flatten(src(_, _0{}, tile_id)),
        flatten(dst(_, _0{}, tile_id)));
}


template <typename Fragment>
CUTLASS_DEVICE void permute_Cregs_128_to_64(Fragment &frag) {
    // frag has shape ((2, 2, N / 8), MMA_M, MMA_N), each element is 32 bits
    static_assert(decltype(size<0, 0>(frag))::value == 2);
    static_assert(decltype(size<0, 1>(frag))::value == 2);
    static_assert(decltype(size<0, 2>(frag))::value % 2 == 0);
    static_assert(decltype(stride<0, 0>(frag))::value == 1);
    static_assert(sizeof(typename Fragment::value_type) == 4);
    Tensor frag_64b = group_modes<1, 3>(recast<uint2>(frag));  // ((1, 2, N / 8), (MMA_M, MMA_N))
    #pragma unroll
    for (int mi = 0; mi < size<1>(frag_64b); ++mi) {
        #pragma unroll
        for (int i = 0; i < size<0, 2>(frag_64b) / 2; ++i) {
            cutlass::swap(frag_64b(make_coord(_0{}, _1{}, 2 * i), mi), frag_64b(make_coord(_0{}, _0{}, 2 * i + 1), mi));
        }
    }
}


template <typename Engine, typename Layout, typename EngineOut>
CUTLASS_DEVICE void convert_type_out(Tensor<Engine, Layout> const &tensor, Tensor<EngineOut, Layout> &out) {
    // Somehow if we allocate out inside this function and return it, e2e is slower and the output can be wrong.
    using From_type = typename Engine::value_type;
    using To_type = typename EngineOut::value_type;
    static constexpr int FragmentSize = std::max(sizeof(From_type) / sizeof(To_type), sizeof(To_type) / sizeof(From_type));
    static_assert(CUTE_STATIC_V(size(tensor)) % FragmentSize == 0, "Fragment size does not vectorize properly");
    Tensor frag = recast<cutlass::Array<From_type, FragmentSize> const>(tensor);
    Tensor out_frg = recast<cutlass::Array<To_type, FragmentSize>>(out);
    static_assert(size(frag) == size(out_frg));
    cutlass::NumericArrayConverter<To_type, From_type, FragmentSize> convert_op;
    #pragma unroll
    for (int i = 0; i < size(frag); ++i) { out_frg[i] = convert_op(frag[i]); }
}

#define CHECK_CUDA(call)                                                                                  \
    do {                                                                                                  \
        cudaError_t status_ = call;                                                                       \
        if (status_ != cudaSuccess) {                                                                     \
            fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
            exit(1);                                                                              \
        }                                                                                                 \
    } while(0)

#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(cudaGetLastError())


#define FLASH_ASSERT(cond)                                                                                \
    do {                                                                                                  \
        if (not (cond)) {                                                                                 \
            fprintf(stderr, "Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, #cond);                 \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)


#define FLASH_DEVICE_ASSERT(cond)                                                                         \
    do {                                                                                                  \
        if (not (cond)) {                                                                                 \
            printf("Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, #cond);                          \
            asm("trap;");                                                                                 \
        }                                                                                                 \
    } while(0)

#define println(fmt, ...) { print(fmt, ##__VA_ARGS__); print("\n"); }

template<typename T>
__inline__ __host__ __device__ T ceil_div(const T &a, const T &b) {
    return (a + b - 1) / b;
}

#ifndef TRAP_ONLY_DEVICE_ASSERT
#define TRAP_ONLY_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) \
        asm("trap;"); \
} while (0)
#endif

// For development, we define both IS_SM100 and IS_SM90 when using CLion or VSCode IDEs so code highlighting will be correct.
#if defined(__CLION_IDE__) || defined(__VSCODE_IDE__)
#define IS_SM100 1
#define IS_SM90 1
#else

// We define the following macros to detect the CUDA architecture, so that we can enable/disable certains kernels that depends on specific architectures.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
#define IS_SM100 1
#else
#define IS_SM100 0
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)
#define IS_SM90 1
#else
#define IS_SM90 0
#endif

#endif  // defined(__CLION_IDE__) || defined(__VSCODE_IDE__)