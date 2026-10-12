# SK -> SA: answers to `docs/interface_requests.md`, and the final kernel-side contract

All of this is implemented on branch `wt/kernel` (`csrc/sm86/`, `csrc/flash_api.cpp`, `csrc/flash_mla.h`,
`flash_mla/flash_mla_interface.py`) and covered by `tests/test_flash_mla_sm86.py` and `tests/test_flash_mla_sm86_quant.py`.

## 1. Quantized `k_cache` (request 1) -- done

```python
out, lse = flash_mla_with_kvcache(q, k_cache_u8, block_table, cache_seqlens, 512, md, ns,
                                  softmax_scale=..., kv_scale=kv_scale, kv_dtype="fp8")   # or "int8"
```

* `k_cache_u8`: `torch.uint8`, `[num_blocks, page, 640]` (a unit head dim `[num_blocks, page, 1, 640]` is accepted too),
  contiguous, 16-byte aligned base, `page in {16, 32, 64}`. Row layout exactly as in `docs/kv_quant_format.md`
  (512 B latent + 128 B rope in the query dtype).
* `kv_scale`: `float32 [num_blocks, page]`, contiguous.
* `kv_dtype`: `"fp8"` (e4m3fn) or `"int8"` -- **required** whenever `k_cache` is `uint8` (it cannot be derived from the tensor).
  `q` may be bf16 or fp16 (the rope part of the cache must be the same dtype as `q`); `q` rows must be 8-byte aligned.
* `get_mla_metadata(seqlens, M, h_k, kv_dtype=None)`: the optional 4th argument is only a hint (it selects the CTA count of
  the quantized kernels); omitting it is fine and gives metadata that works with every kernel.
* `impl` must be `"v2"` for quantized caches (`"legacy"` has no 8-bit path).
* Numerics: the kernel reproduces attention over the *exactly dequantized* cache (`q_i * scale`, fp32 product) to
  cos_diff ~ 2e-6 .. 2e-5; the remaining error w.r.t. the unquantized cache is the format's own quantization error. Measured
  here on synthetic heavy-tailed rows: fp8 2.6e-3 .. 3.9e-3, int8 1.6e-4 .. 5.8e-4 (your gates: fp8 < 5e-3, int8 < 5e-4 on
  realistic data).

## 2. Sweep hooks (request 2) -- done

Compile-time macros with `#ifndef` defaults (see `csrc/sm86/mla_sm86.h`); pass them through `FLASH_MLA_DEFINES`:

| macro | default | meaning |
|---|---|---|
| `FLASH_MLA_SM86_BM_SMALL` / `_BM_LARGE` | 16 / 32 | rows per CTA when `s_q*h_q/h_k <= 16` / otherwise |
| `FLASH_MLA_SM86_NT` | 16 | tokens per KV tile (16 or 32; clamped to the page size) |
| `FLASH_MLA_SM86_NS` | 3 | smem ring stages |
| `FLASH_MLA_SM86_MINB` | 1 | CTAs per SM (sets `num_sm_parts`) |
| `FLASH_MLA_SM86_SPEC` / `_PIPE` | 0 / 0 | 1 = v2b (producer warp + mbarrier) / v2c (software-pipelined) |
| `FLASH_MLA_SM86_FUSE` | 0 | max splits merged by the fused last-arriver combine (0 = always use the combine kernel) |
| `FLASH_MLA_SM86_OV`, `_MINBLK` | 2, 0 | scheduler: per-segment overhead / minimum 32-token blocks per active part |
| `FLASH_MLA_SM86_QNS`, `_QMINB` | 4, 1 | stages / CTAs per SM of the fp8/int8 kernels |

Only the combinations instantiated in the dispatch tables (`mla_sm86_run`, `mla_sm86_run_q_impl`) are valid; others abort
with `Assertion failed ... launched`. Every macro can also be overridden **at run time** by an environment variable of the same
name (read on every call), and `FLASH_MLA_SM86_CFG="bm,nt,ns[,minb[,spec[,pipe]]]"` / `FLASH_MLA_SM86_QCFG="bm,ns[,minb]"` set a
whole plan at once -- handy for sweeps without rebuilding (set them before `get_mla_metadata` as well: `minb` changes
`num_sm_parts`). `flash_mla.sm86_config(num_q_tokens_per_head_k, page)` returns the selected configuration as a string.

## 3. softmax scale / prologue contract (requests 3, 4) -- acknowledged

* `softmax_scale=None` still defaults to `q.shape[-1] ** -0.5` (576); DeepSeek callers pass `(128 + 64) ** -0.5` explicitly.
* Everything is device-side and graph-safe: `get_mla_metadata` (shapes depend only on `(b, M, h_k)`), `flash_mla_with_kvcache`
  with `out=`, `lse=`, `workspace=` allocates nothing and never synchronises. Allocate the workspace with
  `flash_mla.alloc_mla_workspace(b, M, h_k, num_sm_parts, device)` (zero-initialised; the counters it contains are
  self-resetting) or size your own with `get_mla_workspace_bytes(...)` -- it **must be zero-filled once**. `out` must be
  contiguous `(b, s_q, h_q, 512)`; `lse` `(b, h_q, s_q)` fp32.
* Without `workspace=` the call allocates its partial buffers per call (fine eagerly, not graph-capturable).

## 4. Other contract changes

* `warp_spec` is gone: `impl="legacy"` (BlockM=16 fallback) and `impl="legacy_ws"` select the fork's kernels. They accept the
  same metadata as `"v2"` (the scheduler granule is 32 tokens).
* `get_mla_metadata` now builds balanced parts (`floor(p*T/P)` cuts), stores `seqlen_k` of the first segment in slot 5 of each
  row, and is a single-block kernel that is graph safe.
* Alignment requirements of `impl="v2"`: `q`/`out` rows 4-byte aligned (8 for quantized `q`), bf16/fp16 `k_cache` 16-byte
  aligned rows, `page in {16, 32, 64}`, `d == 576`, `d_v == 512`, shared K/V cache.
