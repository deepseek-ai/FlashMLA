#pragma once

#include <cstdint>

#include <cutlass/arch/barrier.h>

namespace deep_gemm::comm {

// Keep the timeout constant used by the fused ROOT epilogue.
constexpr int64_t kNumTimeoutCycles = 60ll * 2000000000ll;

// CSA uses only the cluster rendezvous from the public barrier header. Keeping
// this small overlay avoids pulling distributed-workspace layouts into the
// wheel snapshot when their grid/NVLink barrier APIs are never instantiated.
CUTLASS_DEVICE void cluster_sync_with_relaxed_arrive() {
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
}

} // namespace deep_gemm::comm
