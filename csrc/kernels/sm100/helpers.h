#pragma once

#include <cuda_fp16.h>

#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

namespace sm100 {

using namespace cute;

struct bf16x8 {
    __nv_bfloat162 a01;
    __nv_bfloat162 a23;
    __nv_bfloat162 a45;
    __nv_bfloat162 a67;
};

CUTE_DEVICE
int int4_max(int4 t) {
    return max(max(t.x, t.y), max(t.z, t.w));
}

CUTE_DEVICE
int int4_min(int4 t) {
    return min(min(t.x, t.y), min(t.z, t.w));
}

struct int32x8_t {
    int a0, a1, a2, a3, a4, a5, a6, a7;
};

struct float8 {
    float2 a01, a23, a45, a67;
};

// CUDA 13.2 (PTX ISA 9.2) adds a direct {e4m3x2, e2m1x2} -> bf16x2 conversion, i.e.
// `cvt.rn.bf16x2.e4m3x2` / `cvt.rn.bf16x2.e2m1x2`. Older toolchains only offer the .f16x2
// destination, so the value has to be widened through the FP32 domain instead.
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
#define SM100_HAS_NATIVE_BF16X2_CVT 1
#else
#define SM100_HAS_NATIVE_BF16X2_CVT 0
#endif

// 2x f16 -> 2x bf16 (exact: f16 is a subset of f32, and every f16 the KV cache can store is representable in bf16)
CUTE_DEVICE
ku::nvbf16x2 f16x2_to_bf16x2(uint32_t f16x2) {
    return __float22bfloat162_rn(__half22float2(*(__half2*)&f16x2));
}

// 2x fp8_e4m3 -> 2x bf16, without scaling
CUTE_DEVICE
ku::nvbf16x2 fp8x2_to_bf16x2(ku::nve4m3x2 data) {
    uint32_t out;
#if SM100_HAS_NATIVE_BF16X2_CVT
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;"
        : "=r"(out)
        : "h"(*(uint16_t*)&data));
#else
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;"
        : "=r"(out)
        : "h"(*(uint16_t*)&data));
    const ku::nvbf16x2 bf16 = f16x2_to_bf16x2(out);
    out = *(uint32_t*)&bf16;
#endif
    return *(ku::nvbf16x2*)&out;
}

// 1x ue8m0 scale -> 2x bf16 (the same scale in both halves)
CUTE_DEVICE
ku::nvbf16x2 ue8m0_to_bf16x2(__nv_fp8_e8m0 scale_e8m0) {
    uint16_t packed = (uint16_t)(*(uint8_t*)&scale_e8m0) * 0x0101;
    uint32_t out;
    asm("cvt.rn.bf16x2.ue8m0x2 %0, %1;"
        : "=r"(out)
        : "h"(packed));
    return *(ku::nvbf16x2*)&out;
}

// 2x fp8_e4m3 -> 2x bf16, multiplied by an ue8m0 scale
CUTE_DEVICE
ku::nvbf16x2 fp8x2_to_bf16x2_with_scale(ku::nve4m3x2 data, __nv_fp8_e8m0 scale_e8m0) {
    return __hmul2(fp8x2_to_bf16x2(data), ue8m0_to_bf16x2(scale_e8m0));
}

// 8x fp4_e2m1 (packed in 32 bits, 2 per byte) -> 4x bf16x2
// Written as one asm block so that ptxas selects the source byte with the .B0-.B3 operand selector of F2FP instead of emitting PRMTs
CUTE_DEVICE
void fp4x8_to_bf16x2x4(uint32_t packed, ku::nvbf16x2 *out) {
    uint32_t o0, o1, o2, o3;
#if SM100_HAS_NATIVE_BF16X2_CVT
    asm(
        "{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "mov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;\n"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;\n"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;\n"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;\n"
        "}\n"
        : "=r"(o0), "=r"(o1), "=r"(o2), "=r"(o3)
        : "r"(packed)
    );
    out[0] = *(ku::nvbf16x2*)&o0;
    out[1] = *(ku::nvbf16x2*)&o1;
    out[2] = *(ku::nvbf16x2*)&o2;
    out[3] = *(ku::nvbf16x2*)&o3;
#else
    asm(
        "{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "mov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
        "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
        "cvt.rn.f16x2.e2m1x2 %3, b3;\n"
        "}\n"
        : "=r"(o0), "=r"(o1), "=r"(o2), "=r"(o3)
        : "r"(packed)
    );
    const ku::nvbf16x2 b0 = f16x2_to_bf16x2(o0), b1 = f16x2_to_bf16x2(o1);
    const ku::nvbf16x2 b2 = f16x2_to_bf16x2(o2), b3 = f16x2_to_bf16x2(o3);
    out[0] = b0; out[1] = b1; out[2] = b2; out[3] = b3;
#endif
}

// 4x fp8_e4m3 -> 2x bf16x2 without scaling, used for the e4m3 scales of the fp4 KV cache
CUTE_DEVICE
void fp8x4_to_bf16x2x2(uint32_t packed, ku::nvbf16x2 *out) {
    uint32_t o0, o1;
#if SM100_HAS_NATIVE_BF16X2_CVT
    asm(
        "{\n"
        ".reg .b16 h0, h1;\n"
        "mov.b32 {h0, h1}, %2;\n"
        "cvt.rn.bf16x2.e4m3x2 %0, h0;\n"
        "cvt.rn.bf16x2.e4m3x2 %1, h1;\n"
        "}\n"
        : "=r"(o0), "=r"(o1)
        : "r"(packed)
    );
    out[0] = *(ku::nvbf16x2*)&o0;
    out[1] = *(ku::nvbf16x2*)&o1;
#else
    asm(
        "{\n"
        ".reg .b16 h0, h1;\n"
        "mov.b32 {h0, h1}, %2;\n"
        "cvt.rn.f16x2.e4m3x2 %0, h0;\n"
        "cvt.rn.f16x2.e4m3x2 %1, h1;\n"
        "}\n"
        : "=r"(o0), "=r"(o1)
        : "r"(packed)
    );
    const ku::nvbf16x2 b0 = f16x2_to_bf16x2(o0), b1 = f16x2_to_bf16x2(o1);
    out[0] = b0; out[1] = b1;
#endif
}

// 2x fp8_e4m3 -> bf16x2 without scaling, used for the e4m3 scale of one quant tile of the fp4 KV cache
CUTE_DEVICE
ku::nvbf16x2 e4m3x2_to_bf16x2(uint16_t packed) {
    return fp8x2_to_bf16x2(*(ku::nve4m3x2*)&packed);
}

}
