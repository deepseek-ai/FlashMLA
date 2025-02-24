import os
from pathlib import Path
from datetime import datetime
import subprocess

from setuptools import setup

from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    IS_WINDOWS,
)


def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or "32"
    return nvcc_extra_args + ["--threads", nvcc_threads]


subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"])

cc_flag = []
cc_flag.append("-gencode")
cc_flag.append("arch=compute_90a,code=sm_90a")

this_dir = os.path.dirname(os.path.abspath(__file__))

if IS_WINDOWS:
    cxx_args = ["/O2", "/std:c++17", "/DNDEBUG", "/W0"]
else:
    cxx_args = ["-O3", "-std=c++17", "-DNDEBUG", "-Wno-deprecated-declarations"]

ext_modules = []
ext_modules.append(
    CUDAExtension(
        name="flash_mla.flash_mla_cuda",
        sources=[
            "csrc/flash_api.cpp",
            "csrc/flash_fwd_mla_bf16_sm90.cu",
        ],
        extra_compile_args={
            "cxx": cxx_args,
            "nvcc": append_nvcc_threads(
                [
                    "-O3",
                    "-std=c++17",
                    "-DNDEBUG",
                    "-D_USE_MATH_DEFINES",
                    "-Wno-deprecated-declarations",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v,--register-usage-level=10"
                ]
                + cc_flag
            ),
        },
        include_dirs=[
            Path(this_dir) / "csrc",
            Path(this_dir) / "csrc" / "cutlass" / "include",
        ],
    )
)


try:
    cmd = ['git', 'rev-parse', '--short', 'HEAD']
    rev = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
except Exception as _:
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    rev = '+' + date_time_str


setup(
    name="flash-mla",
    version="1.0.0" + rev,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
