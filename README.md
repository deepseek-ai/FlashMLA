# FlashMLA

## Performance Update (2025.04.22)

We're excited to announce the new release of Flash MLA, which delivers 5% ~ 15% performance improvement on compute-bound workloads, achieving up to 660 TFlops on NVIDIA H800 SXM5 GPUs. The interface of the new version is fully compatible with the old one. Just switch to the new version and enjoy the instant speedup! 🚀🚀🚀

Besides, we'd love to share the technical details behind the new kernel! Check out our deep-dive write-up here: <LINK>

The new kernel primarily targets compute-intensive settings (where the number of q heads $\times$ the number of q sequences per request (if MTP is disabled then it's 1) $\ge 64$). For memory-bound cases, we recommend using version [b31bfe7](https://github.com/deepseek-ai/FlashMLA/tree/b31bfe72a83ea205467b3271a5845440a03ed7cb) for optimal performance.

## Introduction

FlashMLA is an efficient MLA decoding kernel for Hopper GPUs, optimized for variable-length sequences serving.

Currently released:
- BF16, FP16
- Paged kvcache with block size of 64

## Requirements

- Hopper GPUs
- CUDA 12.3 and above
    - **But we highly recommend 12.8 or above for the best performance**
- PyTorch 2.0 and above

## Quick start

### Install

```bash
python setup.py install
```

### Benchmark

```bash
python tests/test_flash_mla.py
```

It is able up to 3000 GB/s in memory-bound configuration and 660 TFLOPS in computation-bound configuration on H800 SXM5, using CUDA 12.8.

Note. For memory-bound cases, we recommend using version [b31bfe7](https://github.com/deepseek-ai/FlashMLA/tree/b31bfe72a83ea205467b3271a5845440a03ed7cb) for optimal performance.

### Usage

```python
from flash_mla import get_mla_metadata, flash_mla_with_kvcache

tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, s_q * h_q // h_kv, h_kv)

for i in range(num_layers):
    ...
    o_i, lse_i = flash_mla_with_kvcache(
        q_i, kvcache_i, block_table, cache_seqlens, dv,
        tile_scheduler_metadata, num_splits, causal=True,
    )
    ...
```

## Acknowledgement

FlashMLA is inspired by [FlashAttention 2&3](https://github.com/dao-AILab/flash-attention/) and [cutlass](https://github.com/nvidia/cutlass) projects.

## Community Support

### MetaX
For MetaX GPUs, visit the official website: [MetaX](https://www.metax-tech.com).

The corresponding FlashMLA version can be found at: [MetaX-MACA/FlashMLA](https://github.com/MetaX-MACA/FlashMLA)


### Moore Threads
For the Moore Threads GPU, visit the official website: [Moore Threads](https://www.mthreads.com/).

The corresponding FlashMLA version is available on GitHub: [MooreThreads/MT-flashMLA](https://github.com/MooreThreads/MT-flashMLA).


### Hygon DCU
For the Hygon DCU, visit the official website: [Hygon Developer](https://developer.sourcefind.cn/).

The corresponding FlashMLA version is available here: [OpenDAS/MLAttention](https://developer.sourcefind.cn/codes/OpenDAS/MLAttention).


### Intellifusion
For the Intellifusion NNP, visit the official website: [Intellifusion](https://www.intellif.com).

The corresponding FlashMLA version is available on Gitee: [Intellifusion/tyllm](https://gitee.com/Intellifusion_2025/tyllm/blob/master/python/tylang/flash_mla.py).


### Iluvatar Corex
For Iluvatar Corex GPUs, visit the official website: [Iluvatar Corex](https://www.iluvatar.com).

The corresponding FlashMLA version is available on GitHub: [Deep-Spark/FlashMLA](https://github.com/Deep-Spark/FlashMLA/tree/iluvatar_flashmla)


### AMD Instinct
For AMD Instinct GPUs, visit the official website: [AMD Instinct](https://www.amd.com/en/products/accelerators/instinct.html).

The corresponding FlashMLA version can be found at: [AITER/MLA](https://github.com/ROCm/aiter/blob/main/aiter/mla.py)

## Citation

```bibtex
@misc{flashmla2025,
      title={FlashMLA: Efficient MLA decoding kernels},
      author={Jiashi Li, Shengyu Liu},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/FlashMLA}},
}
```
