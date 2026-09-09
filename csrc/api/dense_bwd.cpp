#include "common.h"

#include "kernels/sm100/prefill/dense/interface.h"

void register_dense_bwd(pybind11::module_& m) {
    m.def("dense_prefill_bwd",
        &FMHACutlassSM100BwdRun,
        "Run Dense Attention Prefill Backward (cutlass FMHA)");
}
