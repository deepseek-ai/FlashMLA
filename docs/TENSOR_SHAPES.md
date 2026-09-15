# Tensor Shapes Reference - FlashMLA

This document provides a comprehensive reference for tensor shapes expected by FlashMLA functions.

## Shape Notation

| Symbol | Meaning |
|--------|---------|
| B | Batch size |
| S | Sequence length (tokens in request) |
| P | Prompt length (prefill tokens) |
| T | Decode step count (incremental tokens) |
| H | Attention heads |
| Hq | Query heads |
| Hk | Key/Value heads (Hk <= Hq for grouped-query attention) |
| D | Per-head hidden dimension |
| G | Groups for grouped-query attention (G = Hk; Hq = G * ratio) |
| Nblk | Number of KV blocks: `ceil((P + T) / 64)` |
| Blk | Block size for paged cache (always 64) |

**Layout**: Row-major contiguous unless noted; strides allowed if stated.

## Function Reference

### 1. `flash_mla_with_kvcache` - MLA Decoding with Paged KV Cache

```python
flash_mla_with_kvcache(
    q:           [B, Hq, 1, D],           # Current token query
    k_cache:     [B, Hk, Nblk, 64, D],    # Paged K blocks
    v_cache:     [B, Hk, Nblk, 64, D],    # Paged V blocks
    block_table: [B, Nblk],               # Maps block index -> physical cache block
    seq_lens:    [B],                     # Prompt + decoded length per batch
    mask:        [B, 1, 1, P+T],          # Optional; causal if omitted
    metadata:    get_mla_metadata(...),
) -> out: [B, Hq, 1, D]
```

**Layout Requirements**:
- `q`: contiguous
- `k_cache`/`v_cache`: contiguous within last two dims (Blk, D)
- `block_table`: must be contiguous

**Notes**:
- MLA uses grouped-query attention: typically Hq = multiple of Hk (e.g., 8:4)
- Decoding assumes the last cache block may be partially filled; unused slots ignored via `seq_lens`

### 2. `get_mla_metadata` - Metadata for Decoding

```python
get_mla_metadata(
    q_shape:     tuple[B, Hq, 1, D],
    kv_shape:    tuple[B, Hk, Nblk, 64, D],
    block_table: [B, Nblk],
    seq_lens:    [B],
) -> metadata
```

**Notes**:
- Pure shape/stride validation; does not materialize tensors
- Requires consistent Hq/Hk grouping and block_table coverage up to max `seq_lens`

### 3. `flash_mla_prefill` - Prefill Attention with Sparse Patterns

```python
flash_mla_prefill(
    q:             [B, Hq, P, D],
    k:             [B, Hk, P, D],
    v:             [B, Hk, P, D],
    sparse_layout: optional pattern (e.g., block-sparse mask),
    attn_mask:     [B, 1, P, P] or [B, Hq, P, P],  # Causal or custom
) -> out: [B, Hq, P, D]
```

**Layout Requirements**:
- q/k/v: contiguous on last dim; leading dims can be strided but consistent
- Sparse pattern typically block-aligned to 64 for reuse with paged cache

### 4. `mha_fwd_kvcache` - MHA Prefill with KV Cache

```python
mha_fwd_kvcache(
    q:           [B, H, P, D],
    k_cache:     [B, H, Nblk, 64, D],
    v_cache:     [B, H, Nblk, 64, D],
    block_table: [B, Nblk],
    attn_mask:   [B, 1, P, P] or [B, H, P, P],
) -> out: [B, H, P, D]
```

**Notes**:
- Standard MHA (no MLA head-grouping)
- Prefill writes into paged cache; P must align with block_table coverage (`ceil(P/64)` blocks)

## Memory Layout Requirements

| Requirement | Details |
|-------------|---------|
| Contiguous dims | Inner dims (D and Blk) must be contiguous for q/k/v and caches |
| Block size | Fixed at 64; `block_table` must enumerate all blocks in order |
| Mixed precision | FP8/BF16/FP16 supported; metadata and indices are integer/FP32 |
| Strides | Batch/head/sequence dims may be strided if monotonic; non-monotonic unsupported |

## Visual Layout Diagrams

### Paged KV Cache (per batch, per head)

```
k_cache[b, h] -> [ blk0 | blk1 | ... | blk(Nblk-1) ]
                   |
                   v
               blkX: 64 rows x D cols (contiguous slab)
```

### Grouped-Query Attention (Hq > Hk)

```
Hq heads (e.g., 8)
├── group0 (q heads 0-1) -> maps to kv head 0
├── group1 (q heads 2-3) -> maps to kv head 1
├── group2 (q heads 4-5) -> maps to kv head 2
└── group3 (q heads 6-7) -> maps to kv head 3
```

### Prefill to Cache Mapping (P tokens)

```
tokens 0..63   -> block_table[b, 0]
tokens 64..127 -> block_table[b, 1]
tokens 128..191 -> block_table[b, 2]
...
```

## Common Shape Mismatch Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| Hq vs Hk mismatch | `Hq` is not an integer multiple of `Hk` | Adjust q reshape or head-splitting for MLA |
| Wrong block count | `Nblk` != `ceil(seq_len/64)` | Update block_table or cache allocation |
| Non-contiguous inner dims | q/k_cache/v_cache not contiguous on D/Blk | Call `.contiguous()` or re-pack before kernel call |
| Mask length short | `attn_mask` last dim < `P+T` | Pad mask or recompute with correct seq length |
| Seq_lens vs cache | `seq_lens[b]` > `Nblk*64` | Grow cache or truncate sequence |
| Mixed precision | q/k/v and cache have different dtypes | Cast consistently to FP8/BF16/FP16 |

## Quick Checks Before Calling

```python
# Before flash_mla_with_kvcache
assert q.shape == (B, Hq, 1, D)
assert k_cache.shape == (B, Hk, Nblk, 64, D)
assert Hq % Hk == 0  # MLA grouping
assert block_table.shape == (B, Nblk)
assert all(seq_lens[b] <= Nblk * 64 for b in range(B))
assert q.is_contiguous()
```
