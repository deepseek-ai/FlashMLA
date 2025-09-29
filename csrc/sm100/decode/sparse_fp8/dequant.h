#pragma once

#include <cuda_fp8.h>
#include <cuda_bf16.h>

#include "sm100/defines.h"

namespace sm100 {

struct fp8x8 {
    __nv_fp8x4_e4m3 lo;
    __nv_fp8x4_e4m3 hi;
};

struct fp8x32 {
    fp8x8 a0, a1, a2, a3;
};

struct fp8x16 {
    fp8x8 a0, a1;
};

__device__ __forceinline__
bf16x8 cvt_fp8x8_bf16x8(const fp8x8 &inputs, const float &scale) {
    __nv_bfloat162 scale_bf162 = __float2bfloat162_rn(scale);
    
    #define DEQUANT_FP8x4(OUTPUT_BF16_LO, OUTPUT_BF16_HI, FP8x4) \
    { \
        float4 fp32x4 = (float4)(FP8x4); \
        OUTPUT_BF16_LO = __float22bfloat162_rn({fp32x4.x, fp32x4.y})*scale_bf162; \
        OUTPUT_BF16_HI = __float22bfloat162_rn({fp32x4.z, fp32x4.w})*scale_bf162; \
    }

    bf16x8 result;
    DEQUANT_FP8x4(result.a01, result.a23, inputs.lo);
    DEQUANT_FP8x4(result.a45, result.a67, inputs.hi);

    return result;
}

__device__ __forceinline__
fp8x32 ldg_256_fp8x32(void* src_ptr) {
    int32x8_t val;
    asm volatile("ld.global.nc.L1::evict_normal.L2::evict_normal.L2::256B.v8.s32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=r"(val.a0), "=r"(val.a1), "=r"(val.a2), "=r"(val.a3),
          "=r"(val.a4), "=r"(val.a5), "=r"(val.a6), "=r"(val.a7)
        : "l"(src_ptr)
    );
    return *reinterpret_cast<fp8x32*>(&val);
}

__device__ __forceinline__
fp8x16 ldg_128_fp8x16(void* src_ptr) {
    int4 ret;
    asm volatile("ld.global.nc.L1::evict_first.v4.s32 {%0, %1, %2, %3}, [%4];"
        : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
        : "l"(src_ptr));
    return *reinterpret_cast<fp8x16*>(&ret);
}

}
