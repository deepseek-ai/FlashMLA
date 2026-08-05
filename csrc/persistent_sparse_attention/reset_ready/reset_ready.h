// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cuda_runtime_api.h>
cudaError_t launch_reset_ready(void* counters, size_t counter_bytes,
    void* ready, size_t ready_bytes, void* producer_clocks,
    size_t producer_clock_bytes, void* consumer_clocks,
    size_t consumer_clock_bytes, void* qnorm_clocks,
    size_t qnorm_clock_bytes, int num_descriptors, int num_sms,
    cudaStream_t stream);
cudaError_t query_reset_ready_max_active_blocks_per_sm(int* max_active_blocks);
size_t reset_ready_residency_shared_bytes();
