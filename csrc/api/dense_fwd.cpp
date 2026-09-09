#include "common.h"

#include "kernels/sm100/prefill/dense/interface.h"

void register_dense_fwd(pybind11::module_& m) {
    m.def("dense_prefill_fwd",
        &FMHACutlassSM100FwdRun,
        "Run Dense Attention Prefill Forward (cutlass FMHA)");
}
