#pragma once

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>

namespace kerutils {

// Return the backend-native stream through PyTorch's stable accelerator API
// (available when targeting PyTorch 2.13 or newer).
template <typename NativeStream>
inline NativeStream get_current_stream(const torch::stable::Tensor &tensor) {
    return static_cast<NativeStream>(
        torch::stable::accelerator::getCurrentStream(tensor.get_device_index()).nativeHandle());
}

}  // namespace kerutils
