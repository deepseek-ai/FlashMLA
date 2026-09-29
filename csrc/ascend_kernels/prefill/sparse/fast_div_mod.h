#pragma once

#include <kerutils/kerutils_for_ascend_npu.h>

// Copied from cutlass::FastDivMod
template<typename T>
static constexpr inline T clz(T x) {
    for (int i = 31; i >= 0; --i) {
        if (x & (1 << i))
            return T(31-i);
    }
    return T(32);
}

template<typename T>
static constexpr inline T find_log2(T x) {
  int a = int(31-clz(x));
  a += (x&(x-1)) != 0;  // Round up, add 1 if not a power of 2.
  return a;
}

// Use `umulhi(x, multiplier) >> right_shift_amount` to calculate `x / divisor`
// Require `divisor` > 1 and x < 2**31
struct FastDivMod {
    uint32_t multiplier;
    uint32_t right_shift_amount;

    FastDivMod() {}
    
    FastDivMod(int divisor) {
        KU_ASSERT_NPU(divisor > 1, "divisor must > 1 for FastDivMod");
        uint32_t p = 31 + find_log2(divisor);
        uint32_t m = uint32_t(((1ull << p) + uint32_t(divisor) - 1) / uint32_t(divisor));
        multiplier = m;
        right_shift_amount = p - 32;
    }
};
