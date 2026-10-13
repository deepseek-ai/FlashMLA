#include <acl/acl.h>

// torch_npu's NPUStream API transitively includes unstable libtorch headers.
// Opt out of the torch stable ABI by temporarily unsetting TORCH_TARGET_VERSION
#pragma push_macro("TORCH_TARGET_VERSION")
#undef TORCH_TARGET_VERSION
#include <torch_npu/csrc/core/npu/NPUStream.h>
#pragma pop_macro("TORCH_TARGET_VERSION")

extern "C" aclrtStream flash_mla_get_current_npu_stream() {
    return c10_npu::getCurrentNPUStream().stream();
}
