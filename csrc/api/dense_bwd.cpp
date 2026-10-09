#include "common.h"

#ifdef FLASH_MLA_IS_BUILD_ON_CUDA
#include "cuda_kernels/sm100/prefill/dense/interface.h"
#endif  // FLASH_MLA_IS_BUILD_ON_CUDA

void register_dense_bwd(pybind11::module_& m) {
#ifdef FLASH_MLA_IS_BUILD_ON_CUDA
    m.def("dense_prefill_bwd",
        &FMHACutlassSM100BwdRun,
        "Run Dense Attention Prefill Backward (cutlass FMHA)");
#else
    (void)m;    // The dense attention kernels are CUDA-only
#endif  // FLASH_MLA_IS_BUILD_ON_CUDA
}
