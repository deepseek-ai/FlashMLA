#pragma once

#include <acl/acl.h>

// The stable ABI cannot expose a backend-native stream before PyTorch 2.13, so the Ascend
// build obtains the current `aclrtStream` through this small bridge. Its implementation lives
// in ascend_stream.cpp, the only translation unit that is allowed to include torch_npu's
// unstable stream headers.
extern "C" aclrtStream flash_mla_get_current_npu_stream();
