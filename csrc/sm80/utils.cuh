#pragma once

#include <cstdint>
#include <cutlass/numeric_types.h>

namespace sm80 {

__device__ __forceinline__ uint32_t cvta_to_shared_u32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cp_async_16(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_ptr)
    );
}

// L1-bypassing 16-byte cp.async (cache at L2 only). Useful for streaming
// reads that won't be reused at the L1 level, e.g. KV cache in attention.
__device__ __forceinline__ void cp_async_16_cg(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_ptr)
    );
}

__device__ __forceinline__ void cp_async_16_zfill_oob(uint32_t smem_addr, const void* gmem_ptr, bool in_bounds) {
    int src_size = in_bounds ? 16 : 0;
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(src_size)
    );
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n");
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// ldmatrix.x4 .m8n8 .shared .b16
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&out)[4], uint32_t smem_addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
        : "r"(smem_addr)
    );
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&out)[4], uint32_t smem_addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
        : "r"(smem_addr)
    );
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t (&out)[2], uint32_t smem_addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(out[0]), "=r"(out[1])
        : "r"(smem_addr)
    );
}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t (&out)[2], uint32_t smem_addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(out[0]), "=r"(out[1])
        : "r"(smem_addr)
    );
}

// Pack two FP32 values into one BF16x2 (.b32) register via PTX cvt.
__device__ __forceinline__ uint32_t pack_bf16x2(float a, float b) {
    uint32_t out;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(out) : "f"(b), "f"(a));
    return out;
}

// Pack two FP32 values into one FP16x2 (.b32) register.
__device__ __forceinline__ uint32_t pack_fp16x2(float a, float b) {
    uint32_t out;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(out) : "f"(b), "f"(a));
    return out;
}

template<typename T>
__device__ __forceinline__ uint32_t pack_2xfp32_to_b32(float a, float b);

template<>
__device__ __forceinline__ uint32_t pack_2xfp32_to_b32<cutlass::bfloat16_t>(float a, float b) {
    return pack_bf16x2(a, b);
}

template<>
__device__ __forceinline__ uint32_t pack_2xfp32_to_b32<cutlass::half_t>(float a, float b) {
    return pack_fp16x2(a, b);
}

// mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 (accumulating into D)
template<typename T>
__device__ __forceinline__ void mma_m16n8k16_acc(
    float (&D)[4], const uint32_t (&A)[4], const uint32_t (&B)[2]);

template<>
__device__ __forceinline__ void mma_m16n8k16_acc<cutlass::bfloat16_t>(
    float (&D)[4], const uint32_t (&A)[4], const uint32_t (&B)[2]
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1])
    );
}

template<>
__device__ __forceinline__ void mma_m16n8k16_acc<cutlass::half_t>(
    float (&D)[4], const uint32_t (&A)[4], const uint32_t (&B)[2]
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1])
    );
}

// XOR swizzle for SMEM K-major layout (BF16/FP16). Within each row, the
// element offset (in BF16 units) is XOR-permuted by ((row & 7) << 3) so that
// 8 consecutive rows accessing the same logical column land on 8 different
// 16-byte chunks. This eliminates the 32-way bank conflict that the
// non-swizzled K-major layout suffers from.
//
// Granularity: 16-byte chunks (= 8 BF16 elements). All ldmatrix and cp.async
// accesses in this kernel are 16-byte aligned, so the XOR is well-defined.
__device__ __forceinline__ int swizzle_col_bf16(int row, int col) {
    return col ^ ((row & 7) << 3);
}

// 16-byte SMEM store (st.shared.b128)
__device__ __forceinline__ void st_shared_b128(uint32_t smem_addr, const uint32_t (&data)[4]) {
    asm volatile(
        "st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n"
        :: "r"(smem_addr), "r"(data[0]), "r"(data[1]), "r"(data[2]), "r"(data[3])
    );
}

}
