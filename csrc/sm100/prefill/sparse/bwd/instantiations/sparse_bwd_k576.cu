// Instantiation TU. Pulls the kernel struct + host wrapper into a real
// compilation unit so type errors surface at build time (vs hanging in header-
// only land).

#include "../sparse_bwd_host.cuh"

namespace sm100::sparse_bwd {

// Non-inline forwarder so the linker pulls in run_sparse_bwd_kernel.
// Currently this is the only external entry point.
void launch(SparseAttnBwdParams& params) {
    run_sparse_bwd_kernel(params);
}

}  // namespace sm100::sparse_bwd
