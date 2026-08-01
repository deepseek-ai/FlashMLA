#pragma once

#include <functional>
#include <initializer_list>

#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include "kerutils/common/common.h"

namespace kerutils {

// Check whether the given tensor or optional tensor satisfies the given condition
// If tensor_or_opt is a tensor, check_fn is applied directly
// If tensor_or_opt is an optional tensor, check_fn is applied only when the optional has value
template<typename T>
static inline bool _check_optional_tensor(const T& tensor_or_opt, const std::function<bool(const torch::stable::Tensor&)>& check_fn) {
    if constexpr (std::is_same<T, torch::stable::Tensor>::value) {
        return check_fn(tensor_or_opt);
    } else {
        if (tensor_or_opt.has_value()) {
            return check_fn(tensor_or_opt.value());
        } else {
            return true;
        }
    }
}

// Get the pointer of the given tensor
// Return (PtrT*)tensor.data_ptr() if the tensor is defined, nullptr otherwise
template<typename PtrT>
static inline PtrT* get_tensor_ptr(const torch::stable::Tensor& tensor) {
    if (tensor.defined()) {
        return (PtrT*)tensor.data_ptr();
    } else {
        return nullptr;
    }
}

// Get the pointer of the given tensor or optional tensor
// Return (PtrT*)tensor.data_ptr() if tensor_or_opt has value and points to a valid tensor, return nullptr otherwise
template<typename PtrT, typename T>
static inline PtrT* get_optional_tensor_ptr(const T& tensor_or_opt) {
    if constexpr (std::is_same<T, torch::stable::Tensor>::value) {
        return get_tensor_ptr<PtrT>(tensor_or_opt);
    } else {
        if (tensor_or_opt.has_value()) {
            return get_tensor_ptr<PtrT>(*tensor_or_opt);
        } else {
            return nullptr;
        }
    }
}

}

// Check whether the given tensor (or optional<tensor>) is on cuda
#define KU_CHECK_DEVICE(tensor) STD_TORCH_CHECK(ku::_check_optional_tensor(tensor, [](const torch::stable::Tensor& t) { return t.is_cuda(); }), #tensor " must be on CUDA")

// Check whether the given tensor (or optional<tensor>) has the given number of dimensions
#define KU_CHECK_NDIM(tensor, ndim) STD_TORCH_CHECK(ku::_check_optional_tensor(tensor, [&](const torch::stable::Tensor& t) { return t.dim() == (ndim); }), #tensor " must have " #ndim " dimensions")

// Check whether the given tensor (or optional<tensor>) has the given shape
#define KU_CHECK_SHAPE(tensor, ...) STD_TORCH_CHECK(ku::_check_optional_tensor(tensor, [&](const torch::stable::Tensor& t) { return t.sizes().equals({__VA_ARGS__}); }), #tensor " must have shape (" #__VA_ARGS__ ")")

// Check whether the given tensor (or optional<tensor>) is contiguous
#define KU_CHECK_CONTIGUOUS(tensor) STD_TORCH_CHECK(ku::_check_optional_tensor(tensor, [](const torch::stable::Tensor& t) { return t.is_contiguous(); }), #tensor " must be contiguous")

// Check whether the last dimention of the given tensor (or optional<tensor>)
#define KU_CHECK_LAST_DIM_CONTIGUOUS(tensor) STD_TORCH_CHECK(ku::_check_optional_tensor(tensor, [](const torch::stable::Tensor& t) { return t.size(-1) == 1 || t.stride(-1) == 1; }), #tensor " must have contiguous last dimension")

// Check whether the given tensor (or optional<tensor>) has the specified dtype
#define KU_CHECK_DTYPE(tensor, target_dtype) STD_TORCH_CHECK(ku::_check_optional_tensor(tensor, [](const torch::stable::Tensor& t) { return t.scalar_type() == (target_dtype); }), #tensor " must have dtype " #target_dtype)
