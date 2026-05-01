#include <pybind11/pybind11.h>

#include "dense_decode.h"

#if !defined(FLASH_MLA_DISABLE_SM90)
#include "sparse_fwd.h"
#include "sparse_decode.h"
#endif

#if !defined(FLASH_MLA_DISABLE_SM100)
#include "dense_fwd.h"
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA";
    m.def("dense_decode_fwd", &dense_attn_decode_interface);
#if !defined(FLASH_MLA_DISABLE_SM90)
    m.def("sparse_decode_fwd", &sparse_attn_decode_interface);
    m.def("sparse_prefill_fwd", &sparse_attn_prefill_interface);
#endif
#if !defined(FLASH_MLA_DISABLE_SM100)
    m.def("dense_prefill_fwd", &FMHACutlassSM100FwdRun);
    m.def("dense_prefill_bwd", &FMHACutlassSM100BwdRun);
#endif
}
