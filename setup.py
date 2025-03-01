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

DISABLE_FP16 = os.getenv("FLASH_MLA_DISABLE_FP16", "FALSE") == "TRUE"
DISABLE_FP8  = os.getenv("FLASH_MLA_DISABLE_FP8", "FALSE") == "TRUE"


def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or "32"
    return nvcc_extra_args + ["--threads", nvcc_threads]


def get_sources():
    sources = [
        "csrc/flash_api.cpp",
        "csrc/flash_fwd_mla_bf16_sm90.cu",
        "csrc/flash_fwd_mla_metadata.cu",
    ]

    if not DISABLE_FP16:
        sources.append("csrc/flash_fwd_mla_fp16_sm90.cu")
    if not DISABLE_FP8:
        sources.append("csrc/flash_fwd_mla_fp8_sm90.cu")

    return sources


def get_features_args():
    features_args = []
    if DISABLE_FP16:
        features_args.append("-DFLASH_MLA_DISABLE_FP16")
    if DISABLE_FP8:
        features_args.append("-DFLASH_MLA_DISABLE_FP8")
    return features_args


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
        name="flash_mla_cuda",
        sources=get_sources(),
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
                    "--ptxas-options=-v,--register-usage-level=10",
                    "--ftemplate-backtrace-limit=0"
                ]
                + cc_flag
            ) + get_features_args(),
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
    name="flash_mla",
    version="1.0.0" + rev,
    packages=find_packages(include=['flash_mla']),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
