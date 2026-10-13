# FlashMLA

***Adapted from：*** https://github.com/deepseek-ai/FlashMLA/

FlashMLA was initially developed based on Hopper(can refer to:https://github.com/deepseek-ai/FlashMLA/), and I adapted it to Ampere GPUs. Due to the different architectures, the performance of Ampere is currently poor due to register overflow. Welcome to add good optimization ideas.

Currently released:
- BF16
- Paged kvcache with block size of 32
- Warp-specialized SM80 kernel (`warp_spec=True`): splits 8 warps into 4 consumer (QK^T GEMM + softmax) and 4 producer (global memory loads), with all 8 warps cooperating on PV GEMM. Uses double-buffered K/V loads and raw PTX barriers for SM80 compatibility.

## Quick start

### Install

```bash
python setup.py install
```

### Benchmark

```bash
# amphere gpus
python tests/test_flash_mla_sm80.py

# hopper gpus
python tests/test_flash_mla_sm90.py
```

It is able up to 464 GB/s in memory-bound configuration and 59 TFLOPS in computation-bound configuration on A100 SXM, using CUDA 12.8.
For [reference](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf), the peak bandwidth and fp16 FLOPS of A100 SXM are 2039 GB/s and 312 TFLOPS respectively. More efforts are needed to optimize the performance.

### Usage

```python
from flash_mla import get_mla_metadata, flash_mla_with_kvcache

tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, s_q * h_q // h_kv, h_kv)

for i in range(num_layers):
    ...
    # Cooperative kernel (default)
    o_i, lse_i = flash_mla_with_kvcache(
        q_i, kvcache_i, block_table, cache_seqlens, dv,
        tile_scheduler_metadata, num_splits, causal=True,
    )
    # Warp-specialized kernel (SM80 only; the old `warp_spec=True` flag is now `impl="legacy_ws"`)
    o_i, lse_i = flash_mla_with_kvcache(
        q_i, kvcache_i, block_table, cache_seqlens, dv,
        tile_scheduler_metadata, num_splits, causal=True, impl="legacy_ws",
    )
    ...
```

## Ampere (sm_86)

The original Ampere port cannot launch on a GeForce Ampere card (its tile needs 113 KB of shared memory, the sm_86 limit is 99 KB, and it only
accepts 32-token pages). This branch adds a native sm_86 decode kernel (`impl="v2"`, the default), a graph-safe API, FP16, 16/32/64-token pages,
an optional FP8 / INT8 KV cache, a fused decode prologue and a DeepSeek-V2-Lite decode layer. On an RTX 3090 (measured peaks: 873.8 GB/s read-only,
70 TFLOPS bf16) it reaches 93-99% of the DRAM peak at h_q=16, b>=64, s_k>=4096 and 73-75% (eager) / 66-72% (CUDA Graph) of the bf16 tensor peak at h_q=128, b>=16, s_k>=4096; see
[docs/ampere_opt_report.md](docs/ampere_opt_report.md) for design, measurements and what did not work, and
[docs/bench_matrix.md](docs/bench_matrix.md) for the full matrix against legacy, Triton and FlashInfer.

```bash
# build for sm_86 only (nvcc >= 12.x; tested with nvcc 12.4 + torch 2.7.1+cu126), in place
FLASH_MLA_ARCH=86 python setup.py build_ext --inplace       # or tools/build.sh
#   FLASH_MLA_LINEINFO=1  adds -lineinfo (ncu source view);  FLASH_MLA_DEFINES="FLASH_MLA_SM86_NT_SMALL=32 ..." sets kernel tunables
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_flash_mla_sm86.py tests/test_flash_mla_sm86_quant.py \
    tests/test_fused_prologue.py tests/test_fused_layer.py tests/test_kv_quant.py tests/test_decode_layer_e2e.py \
    tests/test_flash_mla_sm80.py -q                          # the last one: the fork's own test, impl = legacy and v2
```

```python
import torch, flash_mla

# DeepSeek MLA decode: q [b, s_q, h_q, 576] (512 absorbed latent + 64 rope), paged cache [num_blocks, page, 1, 576], page in {16, 32, 64}
md, nsplit = flash_mla.get_mla_metadata(cache_seqlens, s_q * h_q, 1)            # device-side, no host sync
ws = flash_mla.alloc_mla_workspace(b, s_q * h_q, 1, md.shape[0])                 # zero-filled once, reused by every layer / step
out = torch.empty(b, s_q, h_q, 512, dtype=q.dtype, device=q.device)
lse = torch.empty(b, h_q, s_q, dtype=torch.float32, device=q.device)
o, lse = flash_mla.flash_mla_with_kvcache(
    q, kv_cache, block_table, cache_seqlens, 512, md, nsplit,
    softmax_scale=(128 + 64) ** -0.5,        # DeepSeek: pre-absorption head dim, NOT 576 ** -0.5
    causal=s_q > 1,                          # MTP (s_q = 2) needs the causal tail
    out=out, lse=lse, workspace=ws,          # with out/lse/workspace given: no allocation, no sync, CUDA-Graph capturable
    impl="v2")                               # "v2" (default) | "legacy" (fork kernel, BlockM=16 fallback, page 32, bf16) | "legacy_ws"

# quantized KV cache (docs/kv_quant_format.md): uint8 [num_blocks, page, 640] + fp32 scale [num_blocks, page]; q stays bf16/fp16
kvq, scale = flash_mla.quantize_kv(kv_native.view(-1, page, 576), "fp8")         # or "int8" (more accurate, same bytes)
md, nsplit = flash_mla.get_mla_metadata(cache_seqlens, s_q * h_q, 1, kv_dtype="fp8")
o, lse = flash_mla.flash_mla_with_kvcache(q, kvq, block_table, cache_seqlens, 512, md, nsplit, softmax_scale=(128 + 64) ** -0.5,
                                           kv_scale=scale, kv_dtype="fp8")

# a whole DeepSeek-V2-Lite decode layer: q_proj/kv_a_proj -> absorption bmm -> fused prologue (RoPE, RMSNorm, concat,
# paged-cache write, optional quantize-on-write) -> v2 attention -> W_UV bmm -> o_proj, whole step capturable in one CUDA Graph
from flash_mla import MLAConfig, MLADecodeLayer, alloc_cache, make_state, capture_layer_graph
```

Backends and requirements: `impl="v2"` needs `d=576, d_v=512`, a shared K/V cache and 16-byte aligned cache rows; 8-bit KV needs `impl="v2"` and
`kv_dtype`. Benchmarks: `benchmarks/bench_decode.py` (eager and CUDA-Graph timing, `--impls v2,legacy,triton`, FlashInfer through
`benchmarks/bench_flashinfer.py` in a separate env), `benchmarks/bench_layer.py`, profiling helpers in `tools/` (ncu, nsys).

## Requirements

- Ampere GPUs
- CUDA 12.3 and above
- PyTorch 2.0 and above

## Acknowledgement

FlashMLA is inspired by [FlashAttention 2&3](https://github.com/dao-AILab/flash-attention/) and [cutlass](https://github.com/nvidia/cutlass) projects.

## Citation

```bibtex
@misc{flashmla2025,
      title={FlashMLA: Efficient MLA decoding kernel},
      author={Jiashi Li},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/FlashMLA}},
}
```