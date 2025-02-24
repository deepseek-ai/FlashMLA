# FlashMLA

FlashMLA is an efficient MLA decoding kernel for Hopper GPUs, optimized for variable-length sequences serving.

Currently released:
- BF16
- Paged kvcache with block size of 64

## Quick start

### Install

```bash
python3 -m pip install --upgrade pip setuptools
python3 -m pip install torch pybind11 --index-url https://download.pytorch.org/whl/cu126
python3 -m pip install --no-build-isolation --editable .
```

### Benchmark

```bash
python tests/test_flash_mla.py
```

Achieving up to 3000 GB/s in memory-bound configuration and 580 TFLOPS in computation-bound configuration on H800 SXM5, using CUDA 12.6.

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

## Requirements

- Hopper GPUs
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
