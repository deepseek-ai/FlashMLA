#include "common.h"

#include "kernels/sm100/prefill/dense/interface.h"

STABLE_TORCH_LIBRARY_IMPL(flash_mla, CUDA, m) {
    m.impl("dense_prefill_fwd", TORCH_BOX(&FMHACutlassSM100FwdRun));
}
