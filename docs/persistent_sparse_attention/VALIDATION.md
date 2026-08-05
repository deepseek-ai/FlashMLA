# Persistent sparse-attention suite validation

## Acceptance criteria

The migration is accepted only when:

1. the unified CUDA 13 extension builds from the pinned DeepGEMM and CUTLASS
   submodules;
2. migrated outputs match their qualified source implementations exactly;
3. median target/source latency remains within 25%;
4. sparse-attention regression tests pass after the module merge;
5. an isolated wheel imports and runs without DeepGEMM, dependency headers,
   NVCC, or NVRTC at runtime.

## Environment

The recorded comparison uses one B200 GPU with 148 SMs, CUDA 13.2, and a
PyTorch 2.12 development build. Timings use alternating CUDA-event samples
after warmup to reduce ordering bias.

The suite requires an SM100 GPU, CUDA 12.9 or newer, a matching CUDA-enabled
PyTorch build, and initialized `csrc/cutlass` and `csrc/deep_gemm` submodules.
CUDA 13.x uses the toolkit's `include/cccl` layout directly.

## Build

Focused full-suite build:

~~~bash
git submodule update --init csrc/cutlass csrc/deep_gemm
env FLASH_MLA_BUILD_PERSISTENT_SPARSE_ATTENTION_SUITE=1 FLASH_MLA_DISABLE_SM90=1 MAX_JOBS=8 python setup.py build_ext --inplace
~~~

Wheel build:

~~~bash
git submodule update --init csrc/cutlass csrc/deep_gemm
env FLASH_MLA_BUILD_PERSISTENT_SPARSE_ATTENTION_SUITE=1 MAX_JOBS=8 python setup.py bdist_wheel
~~~

The resulting wheel must contain
flash_mla/persistent_sparse_attention_cuda*.so and must not contain csa_cuda,
DeepGEMM/CUTLASS/CuTe headers, or a runtime include snapshot.

## Static and package gates

~~~bash
python tests/persistent_sparse_attention/test_packaging_contract.py
~~~

The packaging contract also verifies one source tree, one Python package, one
native extension, fixed submodule revisions, the explicit overlay file set,
and absence of JIT initialization.

## Source-to-port comparisons

Migration qualification loads the pre-merge and target extensions in the same
process, checks every output field, and measures alternating CUDA-event samples
after warmup. These source-oracle harnesses and their raw JSON records are
development artifacts and are intentionally not versioned. The stable,
reviewable summary is recorded below.

## Recorded results

The final CUDA 13.2 rerun on B200 is authoritative. Each comparison used three
warmup iterations followed by 15 alternating CUDA-event samples. `source` is
the qualified pre-merge implementation and `target` is the unified
`persistent_sparse_attention_cuda` implementation.

| Operation | Exact outputs | Source median (ms) | Target median (ms) | Target/source | Result |
| --- | --- | ---: | ---: | ---: | --- |
| Persistent frontend | 7/7 tensors | 0.626464 | 0.620832 | 0.991 | PASS |
| Logits adapter | logits | 0.013376 | 0.010336 | 0.773 | PASS |
| WqB-ready | output, counters, ready, clock mask | 1.621536 | 1.616864 | 0.997 | PASS |
| WqB-ready, packed | output, counters, ready, clock mask | 1.611360 | 1.602016 | 0.994 | PASS |

All compared outputs are bitwise exact and every latency ratio is within the
25% acceptance band. The JSON records contain the raw samples and full output
field lists; a large timing deviation remains a migration failure even when
correctness passes.

## GPU regression gates

~~~bash
env FLASH_MLA_CSA_ACCEPTANCE=1 pytest -q tests/persistent_sparse_attention/test_native_gpu.py
env FLASH_MLA_CSA_FAITHFUL_ACCEPTANCE=1 pytest -q tests/persistent_sparse_attention/test_faithful_leaves_gpu.py
env FLASH_MLA_CSA_ROOT_ACCEPTANCE=1 pytest -q tests/persistent_sparse_attention/test_root_fusion_gpu.py
python tests/persistent_sparse_attention/test_private_extension.py
~~~

The CSA environment variable names are retained only as test compatibility
switches; all tests import the unified persistent_sparse_attention module.

The standalone persistent sparse-attention prefill/decode regression passed on
B200. Prefill matched exactly (LSE maximum error about 4.77e-7); decode output
maximum error was 0.001953125 and LSE maximum error about 9.54e-7.

## Wheel isolation result

The CUDA 13.2 wheel was installed with `--no-deps` into a temporary empty
site-packages directory and tested with Python isolated mode. Both native
extensions imported, `index_logits_out` completed on B200 with finite output,
and the installed tree contained neither a `deep_gemm` package nor a dependency
header snapshot. Archive inspection also found no `.h`, `.hpp`, `.cuh`, or
`.cu` files and no legacy `csa_cuda` extension. The unified extension links
against the CUDA driver/runtime and PyTorch libraries, with no NVRTC dependency.
