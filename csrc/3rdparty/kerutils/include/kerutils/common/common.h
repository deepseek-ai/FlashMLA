#pragma once

namespace kerutils {}

#define KU_PRINTLN(fmt, ...) { cute::print(fmt, ##__VA_ARGS__); print("\n"); }

namespace ku = kerutils;

#ifdef __CUDACC__
#define KERUTILS_IS_BUILD_ON_CUDA
#endif

#if __has_include("kernel_operator.h")
#define KERUTILS_IS_BUILD_ON_ASCEND
#endif

#ifdef KERUTILS_IS_BUILD_ON_CUDA
#ifdef KERUTILS_IS_BUILD_ON_ASCEND
#error "KERUTILS_IS_BUILD_ON_CUDA and KERUTILS_IS_BUILD_ON_ASCEND is defined at the same time!"
#endif
#endif

#ifndef KERUTILS_IS_BUILD_ON_CUDA
#ifndef KERUTILS_IS_BUILD_ON_ASCEND
#error "Neither KERUTILS_IS_BUILD_ON_CUDA and KERUTILS_IS_BUILD_ON_ASCEND is defined!"
#endif
#endif

namespace kerutils {

template<typename T>
inline constexpr T ceil_div(const T &a, const T &b) {
    return (a + b - 1) / b;
}

template<typename T>
inline constexpr T ceil(const T &a, const T &b) {
    return (a + b - 1) / b * b;
}

}
