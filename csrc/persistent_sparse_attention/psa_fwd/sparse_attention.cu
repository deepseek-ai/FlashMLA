#include "sparse_attention.cuh"

namespace persistent_sparse_attention::sparse_attention {

void launch_sparse_attention(const SparseAttnFwdParams& params) {
    using Kernel = KernelTemplate<SparseAttnFwdMode::Prefill, 512>;
    constexpr int kBoundedGridSchedule = 10;
    Kernel::run(params, kBoundedGridSchedule);
}

void launch_sparse_attention_prefix(const SparseAttnFwdParams& params) {
    using Kernel = KernelTemplate<SparseAttnFwdMode::Prefill, 512>;
    // GB200 keeps the native problem-sized prefix. B200 production uses the
    // grid108 schedule so 40 SMs remain available for the steady36 ready
    // producer without relying on CUDA Graph child-node priority. Grid148 is
    // retained below only for isolated schedule-11 experiments.
    if (params.num_sm == 152) {
        Kernel::run(params, 0);
    } else if (params.num_sm == 148) {
        constexpr int kB200WqBAdaptivePrefixSchedule = 11;
        Kernel::run(params, kB200WqBAdaptivePrefixSchedule);
    } else {
        constexpr int kB200ReservedPrefixSchedule = 10;
        Kernel::run(params, kB200ReservedPrefixSchedule);
    }
}

void launch_sparse_attention_tail(const SparseAttnFwdParams& params) {
    using Kernel = KernelTemplate<SparseAttnFwdMode::Prefill, 512>;
    constexpr int kBoundedTailSchedule = 10;
    Kernel::run(params, kBoundedTailSchedule);
}

}  // namespace persistent_sparse_attention::sparse_attention
