#pragma once

#include <torch/extension.h>

#define SWCSA_CHECK_CUDA(tensor) \
  TORCH_CHECK((tensor).is_cuda(), #tensor " must be a CUDA tensor")

#define SWCSA_CHECK_CONTIGUOUS(tensor) \
  TORCH_CHECK((tensor).is_contiguous(), #tensor " must be contiguous")

#define SWCSA_CHECK_DEVICE_MATCH(lhs, rhs) \
  TORCH_CHECK((lhs).device() == (rhs).device(), \
              #lhs " and " #rhs " must be on the same device")
