#pragma once

#include <pybind11/pybind11.h>

namespace flash_mla::csa::deepgemm_adapters {

void register_apis(pybind11::module_& module);

}  // namespace flash_mla::csa::deepgemm_adapters
