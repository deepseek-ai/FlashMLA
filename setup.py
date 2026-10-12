import os
from pathlib import Path
from datetime import datetime
import subprocess

from setuptools import setup, find_packages

import torch

# Copied from https://github.com/Dao-AILab/flash-attention/blob/main/hopper/setup.py
# HACK: we monkey patch pytorch's _write_ninja_file to pass
# "-gencode arch=compute_sm90a,code=sm_90a" to files ending in '_sm90.cu',
# and pass "-gencode arch=compute_sm80,code=sm_80" to files ending in '_sm80.cu'
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    IS_HIP_EXTENSION,
    COMMON_HIP_FLAGS,
    SUBPROCESS_DECODE_ARGS,
    IS_WINDOWS,
    get_cxx_compiler,
    _join_rocm_home,
    _join_cuda_home,
    _is_cuda_file,
    _maybe_write,
)

def _write_ninja_file(path,
                      cflags,
                      post_cflags,
                      cuda_cflags,
                      cuda_post_cflags,
                      cuda_dlink_post_cflags,
                      sources,
                      objects,
                      ldflags,
                      library_target,
                      with_cuda,
                      **kwargs  # kwargs (ignored) to absorb new flags in torch.utils.cpp_extension.
) -> None:
    r"""Write a ninja file that does the desired compiling and linking.

    `path`: Where to write this file
    `cflags`: list of flags to pass to $cxx. Can be None.
    `post_cflags`: list of flags to append to the $cxx invocation. Can be None.
    `cuda_cflags`: list of flags to pass to $nvcc. Can be None.
    `cuda_postflags`: list of flags to append to the $nvcc invocation. Can be None.
    `sources`: list of paths to source files
    `objects`: list of desired paths to objects, one per source.
    `ldflags`: list of flags to pass to linker. Can be None.
    `library_target`: Name of the output library. Can be None; in that case,
                      we do no linking.
    `with_cuda`: If we should be compiling with CUDA.
    """
    def sanitize_flags(flags):
        if flags is None:
            return []
        else:
            return [flag.strip() for flag in flags]

    cflags = sanitize_flags(cflags)
    post_cflags = sanitize_flags(post_cflags)
    cuda_cflags = sanitize_flags(cuda_cflags)
    cuda_post_cflags = sanitize_flags(cuda_post_cflags)
    cuda_dlink_post_cflags = sanitize_flags(cuda_dlink_post_cflags)
    ldflags = sanitize_flags(ldflags)

    # Sanity checks...
    assert len(sources) == len(objects)
    assert len(sources) > 0

    compiler = get_cxx_compiler()

    # Version 1.3 is required for the `deps` directive.
    config = ['ninja_required_version = 1.3']
    config.append(f'cxx = {compiler}')
    if with_cuda or cuda_dlink_post_cflags:
        if IS_HIP_EXTENSION:
            nvcc = _join_rocm_home('bin', 'hipcc')
        else:
            nvcc = _join_cuda_home('bin', 'nvcc')
        if "PYTORCH_NVCC" in os.environ:
            nvcc_from_env = os.getenv("PYTORCH_NVCC") # user can set nvcc compiler with ccache using the environment variable here.
        else:
            nvcc_from_env = nvcc
        config.append(f'nvcc_from_env = {nvcc_from_env}')
        config.append(f'nvcc = {nvcc}')

    if IS_HIP_EXTENSION:
        post_cflags = COMMON_HIP_FLAGS + post_cflags
    flags = [f'cflags = {" ".join(cflags)}']
    flags.append(f'post_cflags = {" ".join(post_cflags)}')
    if with_cuda:
        flags.append(f'cuda_cflags = {" ".join(cuda_cflags)}')
        flags.append(f'cuda_post_cflags = {" ".join(cuda_post_cflags)}')
        cuda_post_cflags_sm80 = [s if s != 'arch=compute_90a,code=sm_90a' else 'arch=compute_80,code=sm_80' for s in cuda_post_cflags]
        flags.append(f'cuda_post_cflags_sm80 = {" ".join(cuda_post_cflags_sm80)}')
        cuda_post_cflags_sm80_sm90 = cuda_post_cflags + (
            ['-gencode', 'arch=compute_80,code=sm_80'] if 'arch=compute_90a,code=sm_90a' in cuda_post_cflags else [])
        flags.append(f'cuda_post_cflags_sm80_sm90 = {" ".join(cuda_post_cflags_sm80_sm90)}')
        cuda_post_cflags_sm100 = [s if s != 'arch=compute_90a,code=sm_90a' else 'arch=compute_100a,code=sm_100a' for s in cuda_post_cflags]
        flags.append(f'cuda_post_cflags_sm100 = {" ".join(cuda_post_cflags_sm100)}')
    flags.append(f'cuda_dlink_post_cflags = {" ".join(cuda_dlink_post_cflags)}')
    flags.append(f'ldflags = {" ".join(ldflags)}')

    # Turn into absolute paths so we can emit them into the ninja build
    # file wherever it is.
    sources = [os.path.abspath(file) for file in sources]

    # See https://ninja-build.org/build.ninja.html for reference.
    compile_rule = ['rule compile']
    if IS_WINDOWS:
        compile_rule.append(
            '  command = cl /showIncludes $cflags -c $in /Fo$out $post_cflags')
        compile_rule.append('  deps = msvc')
    else:
        compile_rule.append(
            '  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags')
        compile_rule.append('  depfile = $out.d')
        compile_rule.append('  deps = gcc')

    if with_cuda:
        cuda_compile_rule = ['rule cuda_compile']
        nvcc_gendeps = ''
        # --generate-dependencies-with-compile is not supported by ROCm
        # Nvcc flag `--generate-dependencies-with-compile` is not supported by sccache, which may increase build time.
        if torch.version.cuda is not None and os.getenv('TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES', '0') != '1':
            cuda_compile_rule.append('  depfile = $out.d')
            cuda_compile_rule.append('  deps = gcc')
            # Note: non-system deps with nvcc are only supported
            # on Linux so use --generate-dependencies-with-compile
            # to make this work on Windows too.
            nvcc_gendeps = '--generate-dependencies-with-compile --dependency-output $out.d'
        cuda_compile_rule_sm80 = ['rule cuda_compile_sm80'] + cuda_compile_rule[1:] + [
            f'  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags_sm80'
        ]
        cuda_compile_rule_sm80_sm90 = ['rule cuda_compile_sm80_sm90'] + cuda_compile_rule[1:] + [
            f'  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags_sm80_sm90'
        ]
        cuda_compile_rule_sm100 = ['rule cuda_compile_sm100'] + cuda_compile_rule[1:] + [
            f'  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags_sm100'
        ]
        cuda_compile_rule.append(
            f'  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags')

    # Emit one build rule per source to enable incremental build.
    build = []
    for source_file, object_file in zip(sources, objects):
        is_cuda_source = _is_cuda_file(source_file) and with_cuda
        if is_cuda_source:
            if source_file.endswith('_sm90.cu'):
                rule = 'cuda_compile'
            elif source_file.endswith('_sm80.cu'):
                rule = 'cuda_compile_sm80'
            elif source_file.endswith('_sm100.cu'):
                rule = 'cuda_compile_sm100'
            else:
                rule = 'cuda_compile_sm80_sm90'
        else:
            rule = 'compile'
        if IS_WINDOWS:
            source_file = source_file.replace(':', '$:')
            object_file = object_file.replace(':', '$:')
        source_file = source_file.replace(" ", "$ ")
        object_file = object_file.replace(" ", "$ ")
        build.append(f'build {object_file}: {rule} {source_file}')

    if cuda_dlink_post_cflags:
        devlink_out = os.path.join(os.path.dirname(objects[0]), 'dlink.o')
        devlink_rule = ['rule cuda_devlink']
        devlink_rule.append('  command = $nvcc $in -o $out $cuda_dlink_post_cflags')
        devlink = [f'build {devlink_out}: cuda_devlink {" ".join(objects)}']
        objects += [devlink_out]
    else:
        devlink_rule, devlink = [], []

    if library_target is not None:
        link_rule = ['rule link']
        if IS_WINDOWS:
            cl_paths = subprocess.check_output(['where',
                                                'cl']).decode(*SUBPROCESS_DECODE_ARGS).split('\r\n')
            if len(cl_paths) >= 1:
                cl_path = os.path.dirname(cl_paths[0]).replace(':', '$:')
            else:
                raise RuntimeError("MSVC is required to load C++ extensions")
            link_rule.append(f'  command = "{cl_path}/link.exe" $in /nologo $ldflags /out:$out')
        else:
            link_rule.append('  command = $cxx $in $ldflags -o $out')

        link = [f'build {library_target}: link {" ".join(objects)}']

        default = [f'default {library_target}']
    else:
        link_rule, link, default = [], [], []

    # 'Blocks' should be separated by newlines, for visual benefit.
    blocks = [config, flags, compile_rule]
    if with_cuda:
        blocks.append(cuda_compile_rule)  # type: ignore[possibly-undefined]
        blocks.append(cuda_compile_rule_sm80)  # type: ignore[possibly-undefined]
        blocks.append(cuda_compile_rule_sm80_sm90)  # type: ignore[possibly-undefined]
        blocks.append(cuda_compile_rule_sm100)  # type: ignore[possibly-undefined]
    blocks += [devlink_rule, link_rule, build, devlink, link, default]
    content = "\n\n".join("\n".join(b) for b in blocks)
    # Ninja requires a new line at the end of the .ninja file.
    content += "\n"
    _maybe_write(path, content)

# Monkey patching.
torch.utils.cpp_extension._write_ninja_file = _write_ninja_file

DISABLE_FP16 = os.getenv("FLASH_MLA_DISABLE_FP16", "FALSE") == "TRUE"
DEBUG_BUILD = os.getenv("FLASH_MLA_DEBUG", "FALSE") == "TRUE"

# FLASH_MLA_ARCH: ""/"90" -> original build (sm_90a + sm_80 sources).
#                 "80"/"86" -> Ampere-only build: skips every *_sm90*.cu source, passes
#                 -gencode arch=compute_<ARCH>,code=sm_<ARCH>, defines FLASH_MLA_NO_SM90 and
#                 drops the 12.x-only ptxas flag --register-usage-level (nvcc < 12.4 compatible).
ARCH = os.getenv("FLASH_MLA_ARCH", "")
assert ARCH in ("", "80", "86", "90"), f"unsupported FLASH_MLA_ARCH={ARCH!r} (use 80, 86 or 90)"
AMPERE_ONLY = ARCH in ("80", "86")
# FLASH_MLA_LINEINFO=1 adds -lineinfo (source correlation for ncu) to optimized builds.
LINEINFO = os.getenv("FLASH_MLA_LINEINFO", "0") == "1"


def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or "32"
    return nvcc_extra_args + ["--threads", nvcc_threads]


def _glob_sources(subdir, patterns=("*.cu", "*.cpp")):
    """Sorted, repo-relative paths of every source in csrc/<subdir> (empty when the directory does not exist yet)."""
    root = Path(__file__).resolve().parent
    d = root / "csrc" / subdir
    out = []
    for pat in patterns:
        out += [p.relative_to(root).as_posix() for p in d.glob(pat)]
    return sorted(out)


def get_sources():
    sources = [
        "csrc/flash_api.cpp",
        "csrc/flash_fwd_mla_bf16_sm80.cu",
        "csrc/flash_fwd_mla_bf16_ws_sm80.cu",
        "csrc/flash_fwd_mla_metadata.cu",
    ]

    if not AMPERE_ONLY:
        sources.append("csrc/flash_fwd_mla_bf16_sm90.cu")
        if not DISABLE_FP16:
            sources.append("csrc/flash_fwd_mla_fp16_sm90.cu")

    # New sm86 kernels: every csrc/sm86/*.cu|*.cpp is picked up automatically (no setup.py edit needed to add files).
    # Drop `_sm90`-suffixed files in Ampere builds, they would not compile for sm_86.
    sources += [s for s in _glob_sources("sm86") if not (AMPERE_ONLY and "_sm90" in s)]
    return sources


def get_fused_sources():
    """Sources of the separate `flash_mla_fused_cuda` extension (decode prologue, quantize-on-write): csrc/fused/*.cu|*.cpp."""
    return _glob_sources("fused")


def get_features_args():
    features_args = []
    if DISABLE_FP16:
        features_args.append("-DFLASH_MLA_DISABLE_FP16")
    if AMPERE_ONLY:
        features_args.append("-DFLASH_MLA_NO_SM90")
    return features_args


# Only touch the network if CUTLASS has not been checked out yet (worktrees / offline servers).
if not (Path(__file__).resolve().parent / "csrc" / "cutlass" / "include" / "cutlass" / "cutlass.h").exists():
    subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"])

cc_flag = []
cc_flag.append("-gencode")
if AMPERE_ONLY:
    cc_flag.append(f"arch=compute_{ARCH},code=sm_{ARCH}")
else:
    cc_flag.append("arch=compute_90a,code=sm_90a")

this_dir = os.path.dirname(os.path.abspath(__file__))

if DEBUG_BUILD:
    if IS_WINDOWS:
        cxx_args = ["/Od", "/Zi", "/std:c++17", "/W0"]
    else:
        cxx_args = ["-O0", "-g", "-std=c++17", "-Wno-deprecated-declarations"]
else:
    if IS_WINDOWS:
        cxx_args = ["/O2", "/std:c++17", "/DNDEBUG", "/W0"]
    else:
        cxx_args = ["-O3", "-std=c++17", "-DNDEBUG", "-Wno-deprecated-declarations"]

nvcc_args = append_nvcc_threads(
    [
        "-O0" if DEBUG_BUILD else "-O3",
        "-std=c++17",
        "-D_USE_MATH_DEFINES",
        "-Wno-deprecated-declarations",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]
    + ([] if DEBUG_BUILD else ["-DNDEBUG", "--use_fast_math"])
    + (["-g", "-G", "-lineinfo"] if DEBUG_BUILD else (["-lineinfo"] if LINEINFO else []))
    + ["--ptxas-options=-v" + ("" if AMPERE_ONLY else ",--register-usage-level=10")]
    + cc_flag
) + get_features_args()

ext_modules = []
ext_modules.append(
    CUDAExtension(
        name="flash_mla_cuda",
        sources=get_sources(),
        extra_compile_args={
            "cxx": cxx_args + get_features_args(),
            "nvcc": nvcc_args,
        },
        include_dirs=[
            Path(this_dir) / "csrc",
            Path(this_dir) / "csrc" / "cutlass" / "include",
        ],
    )
)

# Separate extension for the fused decode prologue / quantize-on-write kernels (csrc/fused), so it never has to touch
# csrc/flash_api.cpp. Built only when csrc/fused has sources. Python: `import flash_mla_fused_cuda`.
if get_fused_sources():
    ext_modules.append(
        CUDAExtension(
            name="flash_mla_fused_cuda",
            sources=get_fused_sources(),
            extra_compile_args={
                "cxx": cxx_args + get_features_args(),
                "nvcc": nvcc_args,
            },
            include_dirs=[
                Path(this_dir) / "csrc",
                Path(this_dir) / "csrc" / "fused",
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
