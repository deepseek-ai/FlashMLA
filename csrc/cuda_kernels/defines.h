#pragma once

#include <cutlass/bfloat16.h>
#include <cutlass/arch/barrier.h>

using bf16 = cutlass::bfloat16_t;
using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;
using cutlass::arch::fence_view_async_shared;
using cutlass::arch::fence_barrier_init;
using cutlass::arch::NamedBarrier;
using fp8_e4m3 = cutlass::float_e4m3_t;

