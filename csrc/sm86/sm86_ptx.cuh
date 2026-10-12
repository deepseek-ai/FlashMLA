// Inline-PTX helpers for the sm_86 MLA decode kernels (CUDA C++ only, no CuTe needed).
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace mla_sm86 {

__device__ __forceinline__ uint32_t smem_u32(const void *p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// 16-byte cp.async (L2 only, .cg).  pred == false -> src-size 0 -> destination is zero-filled, global is not read.
__device__ __forceinline__ void cp_async_16(uint32_t dst, const void *src, bool pred) {
    const int sz = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}
__device__ __forceinline__ void ldsm_x4_trans(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}

// ---- mbarrier (sm_80+): spin with test_wait (try_wait needs sm_90) ----
__device__ __forceinline__ void mbar_init(uint32_t addr, int count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(addr), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t addr) {
    uint64_t st;
    asm volatile("mbarrier.arrive.shared.b64 %0, [%1];\n" : "=l"(st) : "r"(addr) : "memory");
}
__device__ __forceinline__ bool mbar_test_wait_parity(uint32_t addr, uint32_t parity) {
    uint32_t ok;
    asm volatile("{\n.reg .pred p;\nmbarrier.test_wait.parity.shared.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(addr), "r"(parity) : "memory");
    return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t addr, uint32_t parity) {
    while (!mbar_test_wait_parity(addr, parity)) {}
}
// Arrive on `addr` when all cp.async issued so far by this thread have landed (pending count is not incremented).
__device__ __forceinline__ void cp_async_mbar_arrive_noinc(uint32_t addr) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"r"(addr) : "memory");
}

__device__ __forceinline__ void bar_sync(int id, int nthreads) {
    asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(nthreads) : "memory");
}

__device__ __forceinline__ float ex2_approx(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint32_t ldg_nc_u32(const void *p) {
    uint32_t v;
    asm volatile("ld.global.nc.b32 %0, [%1];\n" : "=r"(v) : "l"(p));
    return v;
}

template <typename T> struct ElemTraits;

template <> struct ElemTraits<__nv_bfloat16> {
    static __device__ __forceinline__ uint32_t pack(float lo, float hi) {
        __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
        return *reinterpret_cast<uint32_t *>(&v);
    }
    static __device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
};

template <> struct ElemTraits<__half> {
    static __device__ __forceinline__ uint32_t pack(float lo, float hi) {
        __half2 v = __floats2half2_rn(lo, hi);
        return *reinterpret_cast<uint32_t *>(&v);
    }
    static __device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
};

}  // namespace mla_sm86
