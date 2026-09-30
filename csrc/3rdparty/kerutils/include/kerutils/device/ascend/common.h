#pragma once

namespace kerutils {

template<typename T>
inline __aicore__ constexpr T ceil_div(const T &a, const T &b) {
    return (a + b - 1) / b;
}

template<typename T>
inline __aicore__ constexpr T ceil(const T &a, const T &b) {
    return (a + b - 1) / b * b;
}

}   // namespace kerutils
