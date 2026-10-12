# Interface requests (SA -> SK)

Status of each item is tracked here; nothing below blocks SA. SA does not edit `csrc/sm86`, `csrc/flash_api.cpp`,
`csrc/flash_mla.h`, `flash_mla/flash_mla_interface.py`.

1. **Quantized `k_cache` shape.** `docs/kv_quant_format.md` defines the cache as `uint8 [num_blocks, page_size, 640]` plus
   `kv_scale float32 [num_blocks, page_size]`. SA's writer (`flash_mla_fused_cuda.mla_decode_prologue`), the torch reference
   (`flash_mla/kv_quant.py`), `benchmarks/bench_decode.py --kv fp8|int8` and `flash_mla/mla_decode_layer.py` pass exactly that
   3-D tensor as `k_cache` together with `kv_scale=`. Please accept it in `flash_mla_with_kvcache` (view it as
   `[nblk, page, 1, 640]` internally if the kernel wants a head dim) and derive `kv_dtype` from `k_cache.dtype == uint8` plus a
   new kwarg, e.g. `kv_dtype="fp8"|"int8"` (the fp8/int8 distinction cannot be read from the tensor; the harness currently
   passes only `kv_scale=` and will add `kv_dtype=` as soon as the name is fixed).
2. **Sweep hooks** (benchmarks/sweep.py): every compile-time tunable of `csrc/sm86` is a macro `FLASH_MLA_SM86_<NAME>`
   with a default (`#ifndef`), settable through `FLASH_MLA_DEFINES="FLASH_MLA_SM86_BM=16 FLASH_MLA_SM86_NT=32"` (setup.py
   already turns that env var into `-D` flags). Optional: `flash_mla_cuda.sm86_config() -> str` returning the compiled-in
   instance, recorded by `bench_decode.py` in the CSV `extra` column.
3. **Prologue -> attention contract** (informational): `q` is `[b, s_q, h_q, 576]` contiguous with the 512 absorbed-nope dims
   first and the 64 rotated rope dims last; the new token's `[kv_c | k_pe_rot]` row is already in the paged cache when the
   attention call starts (`cache_seqlens` includes it). softmax scale is `(128 + 64) ** -0.5` for DeepSeek-V2-Lite, i.e. NOT
   `576 ** -0.5`: pass `softmax_scale=` explicitly (the layer module does).
4. **Graph safety**: `mla_decode_layer.capture_layer_graph` captures `get_mla_metadata` + N attention calls (with `out`, `lse`,
   `workspace` given) and replays them after changing `cache_seqlens` in place. Anything that needs host data in these calls
   will break that test (`tests/test_fused_layer.py::test_cuda_graph_replay_matches_eager_over_steps`).
