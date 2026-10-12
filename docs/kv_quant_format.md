# Quantized MLA KV-cache format (sm86 work)

Status: **interface contract** shared by the kernel (SK), fusion/quantization (SF) and system (SA) work.
Selected by `Flash_fwd_mla_params::kv_dtype` (see `csrc/flash_mla.h`); `kv_dtype == 0` keeps the native
bf16/fp16 layout (`[num_blocks, page_size, 1, 576]`).

| `kv_dtype` | name | latent dims 0..511 | rope dims 512..575 | scale |
|---|---|---|---|---|
| 0 | `kKVNative` | bf16/fp16 | bf16/fp16 | none |
| 1 | `kKVFp8E4m3` | fp8 e4m3 (1 B each) | bf16 (2 B each) | fp32, per token |
| 2 | `kKVInt8` | int8 (1 B each) | bf16 (2 B each) | fp32, per token |

Bytes per token: 512 + 64*2 = **640 B** (+ 4 B scale = **644 B**) vs 1152 B for bf16, i.e. 0.559x of the traffic.
The query stays bf16/fp16 and is never quantized. K and V share the same cache (MLA): K = all 576 dims,
V = the 512 latent dims, so the latent part is quantized once and used for both Q.K^T and P.V.

## Storage

Two tensors, both addressed through the same `block_table`:

1. **KV bytes** `kv_q`: `uint8 [num_blocks, page_size, 640]` (`k_ptr`, `v_ptr` point at it; strides are in bytes).
   One token row = 640 B:

   | byte offset in row | size | content |
   |---|---|---|
   | 0   | 512 B | latent dims 0..511, one byte per dim (e4m3 bit pattern or int8) |
   | 512 | 128 B | rope dims 512..575, bf16 (native-endian), **not** scaled |

2. **Scales** `kv_scale`: `float32 [num_blocks, page_size]` (`kv_scale_ptr`); `kv_scale[block, slot]` belongs to token
   `kv_q[block, slot]`.

Each token row is 640 B = 5 x 128 B, so every row starts on a 128 B boundary (full cache-line `cp.async`).
The latent part (512 B) and the rope part (128 B) are both multiples of 16 B and 16 B-aligned, so all global->shared copies
can use 16 B `cp.async.cg` with no tail handling. A tile of `Nt` consecutive slots inside one page is a single
contiguous `Nt*640` B range; the matching scales are a contiguous `Nt*4` B range (64 B for Nt=16, one 4 x 16 B cp.async).

Byte offsets inside a page (`page_stride = page_size * 640` B, always a multiple of 128 B; base alignment of the tensor: 128 B):

| page_size | token t row offset | latent of t | rope of t | page bytes | scale bytes / page |
|---|---|---|---|---|---|
| 16 | `t*640` | `[t*640, t*640+512)` | `[t*640+512, t*640+640)` | 10 240 | 64 |
| 32 | `t*640` | same | same | 20 480 | 128 |
| 64 | `t*640` | same | same | 40 960 | 256 |

Page `p` starts at `p * page_stride`; the scale of slot `t` of page `p` is at float index `p * page_size + t`.

## Quantization (write path, per token)

`amax = max_i |x_i|` over the 512 latent dims (fp32 accumulate of the bf16 input).

* fp8 e4m3: `scale = amax / 448` (448 = max finite e4m3fn); `q_i = cvt_e4m3_rn(x_i / scale)` (saturating).
* int8:     `scale = amax / 127`; `q_i = clamp(rne(x_i / scale), -127, 127)`.
* if `amax == 0` store `scale = 1.0` and `q = 0`. `scale` is always a finite positive fp32.
* rope dims are copied as bf16 unchanged.

**Dequantization: `x_i ~ q_i * scale`** (fp32 product).

## Kernel-side guidance (sm86 has no fp8 tensor cores)

* Every e4m3 and every int8 (|q| <= 127) value is exactly representable in bf16, so the byte -> bf16 conversion is exact
  (software convert via `prmt` / `lop3` bit tricks; no `cvt.e4m3` on sm_86). MMA stays bf16 x bf16 -> fp32.
* Because the scale is per *token* (the N dimension of S = Q.K^T and the reduction dimension of O = P.V), it can be applied in
  fp32 outside the MMAs and avoids a second bf16 rounding: `S[:, t] = scale_t * (Q_nope . q_t) + Q_rope . k_rope_t`,
  and `O += (P[:, t] * scale_t) . q_t`. Both forms equal `q * scale` dequantization mathematically.
* Accuracy gate: cos_diff of the output vs the bf16 reference must stay below the threshold defined by the quantization
  test (to be fixed by SF from measured error; not specified here).
