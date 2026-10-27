#pragma once

#include "kerutils/device/cuda/common.h"
#include "kerutils/device/cuda/sm80/intrinsics.cuh"

namespace kerutils {

// Retrieve the value of `%smid` and check its range
CUTE_DEVICE
uint32_t get_sm_id_with_range_check(uint32_t num_physical_sms) {
    uint32_t sm_id = get_sm_id();
    if (!(sm_id < num_physical_sms)) {
        trap();
    }
    return sm_id;
}

#ifndef KU_TRAP_ONLY_DEVICE_ASSERT
#define KU_TRAP_ONLY_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) \
        asm("trap;"); \
} while (0)
#endif

// Construct a `float2` from a single `float` by duplicating the value 
CUTE_DEVICE
float2 float2float2(const float &x) {
    return float2 {x, x};
}

CUTE_DEVICE
void st_shared(void* ptr, int128_t val) {
#ifdef _MSC_VER
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :
                 : "r"(cute::cast_smem_ptr_to_uint(ptr)), "r"(val.x), "r"(val.y),
                   "r"(val.z), "r"(val.w));
#else
    asm volatile("st.shared.b128 [%0], %1;" :: "r"(cute::cast_smem_ptr_to_uint(ptr)), "q"(val));
#endif
}

CUTE_DEVICE
void st_shared(void* ptr, float4 val) {
    st_shared(ptr, *reinterpret_cast<int128_t*>(&val));
}

CUTE_DEVICE
int128_t ld_shared(const void* ptr) {
    int128_t val;
#ifdef _MSC_VER
    asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
                 : "r"(cute::cast_smem_ptr_to_uint(ptr)));
#else
    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(val) : "r"(cute::cast_smem_ptr_to_uint(ptr)));
#endif
    return val;
}

CUTE_DEVICE
float4 ld_shared_float4(const void* ptr) {
    auto temp = ld_shared(ptr);
    return *reinterpret_cast<float4*>(&temp);
}

}
