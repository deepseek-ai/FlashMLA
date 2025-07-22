import os
from pathlib import Path
from datetime import datetime
import subprocess

from setuptools import setup, find_packages

from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    IS_WINDOWS,
)


def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or "32"
    return nvcc_extra_args + ["--threads", nvcc_threads]


def get_features_args():
    features_args = []
    DISABLE_FP16 = os.getenv("FLASH_MLA_DISABLE_FP16", "FALSE") in ["TRUE", "1"]
    if DISABLE_FP16:
        features_args.append("-DFLASH_MLA_DISABLE_FP16")
    return features_args


subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"])

cc_flag_sm90 = []
cc_flag_sm90.append("-gencode")
cc_flag_sm90.append("arch=compute_90a,code=sm_90a")

cc_flag_sm100 = []
cc_flag_sm100.append("-gencode")
cc_flag_sm100.append("arch=compute_100a,code=sm_100a")

this_dir = os.path.dirname(os.path.abspath(__file__))

if IS_WINDOWS:
    cxx_args = ["/O2", "/std:c++17", "/DNDEBUG", "/W0"]
else:
    cxx_args = ["-O3", "-std=c++17", "-DNDEBUG", "-Wno-deprecated-declarations"]

ext_modules = []
ext_modules.append(
    CUDAExtension(
        name="flash_mla_sm90",
        sources=[
            "csrc/sm90/flash_api.cpp",
            "csrc/sm90/kernels/get_mla_metadata.cu",
            "csrc/sm90/kernels/mla_combine.cu",
            "csrc/sm90/kernels/splitkv_mla.cu",
        ],
        extra_compile_args={
            "cxx": cxx_args + get_features_args(),
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
                + cc_flag_sm90
            ) + get_features_args(),
        },
        include_dirs=[
            Path(this_dir) / "csrc" / "sm90",
            Path(this_dir) / "csrc" / "cutlass" / "include",
        ],
    )
)

ext_modules.append(
    CUDAExtension(
        name="flash_mla_sm100",
        sources=[
            "csrc/sm100/pybind.cu",
            "csrc/sm100/fmha_cutlass_fwd_sm100.cu",
            "csrc/sm100/fmha_cutlass_bwd_sm100.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17", "-DNDEBUG", "-Wno-deprecated-declarations"],
            "nvcc": append_nvcc_threads(
                [
                    "-O3",
                    "-std=c++17",
                    "-DNDEBUG",
                    "-Wno-deprecated-declarations",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "-lineinfo",
                    "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage",
                ]
                + cc_flag_sm100
            ),
        },
        include_dirs=[
            Path(this_dir) / "csrc" / "sm100",
            Path(this_dir) / "csrc" / "cutlass" / "include",
            Path(this_dir) / "csrc" / "cutlass" / "tools" / "util" / "include",
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
    name="flash_mla",
    version="1.0.0" + rev,
    packages=find_packages(include=['flash_mla']),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
