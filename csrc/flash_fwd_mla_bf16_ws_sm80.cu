// Warp-specialized SM80 MLA kernel - BF16 template instantiation
#include "flash_fwd_mla_kernel_sm80.h"   // for combine kernel (must be first)
#include "flash_fwd_mla_kernel_sm80_ws.h"
#include <stdexcept>
#include <string>

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, typename SharedStorage>
void run_flash_splitkv_fwd_mla_ws(Flash_fwd_mla_params &params, cudaStream_t stream) {
    FLASH_ASSERT(params.page_block_size == Kernel_traits::kBlockN);
    const int num_m_block = cute::ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        auto kernel = &flash::flash_fwd_splitkv_mla_ws_kernel<Kernel_traits, Is_causal, SharedStorage>;
        constexpr size_t smem_size = sizeof(SharedStorage);
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        kernel<<<dim3(num_m_block, params.h, params.num_sm_parts), Kernel_traits::kNThreads, smem_size, stream>>>(params);
    });
    CHECK_CUDA_KERNEL_LAUNCH();

    // Reuse the combine kernel from the SM80 header (flash namespace)
    dim3 grid_combine(params.b * params.h * params.seqlen_q);
    MLA_NUM_SPLITS_SWITCH(params.num_sm_parts, kMaxSplits, [&] {
        auto combine_kernel = &flash::flash_fwd_splitkv_mla_combine_kernel<
                typename Kernel_traits::Element, typename Kernel_traits::ElementAccum, typename Kernel_traits::index_t, Kernel_traits::kHeadDimV, kMaxSplits>;
        combine_kernel<<<grid_combine, 128, 0, stream>>>(params);
    });
    CHECK_CUDA_KERNEL_LAUNCH();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
void mha_fwd_splitkv_mla_ws<cutlass::bfloat16_t, 576>::run(Flash_fwd_mla_params &params, cudaStream_t stream) {
    FLASH_ASSERT(params.d_v == 512);
    FLASH_ASSERT(params.k_ptr == params.v_ptr);  // Shared_KV
    // kNWarpsS: consumer/producer warp split. Valid values with kBlockN=32: 2 or 4.
    // kNWarpsS=4 → 4 consumer + 4 producer (default)
    // kNWarpsS=2 → 2 consumer + 6 producer
    using Kernel_traits = Flash_fwd_kernel_traits_mla_ws<576, 32, 32, 8, 4, cutlass::bfloat16_t, 512>;
    // sm86 note: needs ~113 KB of shared memory, more than the 99 KB opt-in limit of sm_86. The 8-warp 2Mx4N PV tiling is
    // hard-wired to BlockM=32, so there is no minimal fix; fail with a clear error instead of exit(1) from CHECK_CUDA.
    int dev = 0, max_optin = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    if (sizeof(flash::SharedStorageMLA_WS<Kernel_traits>) > (size_t)max_optin) {
        throw std::runtime_error("legacy warp-specialized sm80 MLA kernel needs " +
                                 std::to_string(sizeof(flash::SharedStorageMLA_WS<Kernel_traits>)) +
                                 " B of shared memory, device allows " + std::to_string(max_optin) + " B");
    }
    run_flash_splitkv_fwd_mla_ws<Kernel_traits, flash::SharedStorageMLA_WS<Kernel_traits>>(params, stream);
}

template struct mha_fwd_splitkv_mla_ws<cutlass::bfloat16_t, 576>;
