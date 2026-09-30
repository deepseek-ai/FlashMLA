import enum
from typing import Tuple

import torch

import kernelkit as kk

class KVCacheLayout(enum.Enum):
    V41_FP8Sparse = 0
    V41_FP4Sparse = 1

    def get_meta(self) -> Tuple[int, int, int, int, int]:
        # Return: (d, d_nope, d_rope, tile_size, num_tiles)
        return {
            KVCacheLayout.V41_FP8Sparse: (512, 448, 64, 32, 16),  # 14 NoPE + 2 RoPE tiles
            KVCacheLayout.V41_FP4Sparse: (512, 448, 64, 16, 32),  # 28 NoPE + 4 RoPE tiles, all fp4
        }[self]

    def get_bytes_per_token(self) -> int:
        d, d_nope, d_rope, tile_size, num_tiles = self.get_meta()
        return {
            KVCacheLayout.V41_FP8Sparse: d_nope + d_rope + num_tiles,
            KVCacheLayout.V41_FP4Sparse: d // 2 + num_tiles,
        }[self]

def _cast_scale_inv_to_ue8m0(scales_inv: torch.Tensor, out_dtype = torch.float32) -> torch.Tensor:
    return torch.pow(2, torch.clamp_min(scales_inv, 1e-4).log2().ceil()).to(out_dtype)    # This 1e-4 aligns with Tile Kernel's FP8_AMAX_MARGIN

_E2M1_MAGNITUDES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)  # Indexed by the 3 low bits of the code, bit 3 is the sign

def _quantize_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """
    Round to the nearest fp4_e2m1 value with the semantics of PTX `cvt.rn.satfinite.e2m1x2.f32` (ties to even, saturating to +-6)
    and return the 4-bit codes as uint8. NaN is mapped to code 0 (fp4 has no NaN; the caller keeps the NaN in the scale instead)
    """
    x = x.float()
    if x.numel() > (1 << 26):
        # The tie detection below broadcasts an elements x 7 intermediate; quantize big caches in chunks
        out = torch.empty(x.shape, dtype=torch.uint8, device=x.device)
        for i in range(0, x.numel(), 1 << 26):
            out.reshape(-1)[i:i + (1 << 26)] = _quantize_to_e2m1(x.reshape(-1)[i:i + (1 << 26)])
        return out
    mags = _E2M1_MAGNITUDES.to(x.device)
    sign = (torch.signbit(x)).to(torch.uint8) << 3
    a = torch.nan_to_num(x.abs(), nan=0.0, posinf=6.0).clamp_max(6.0)
    mids = (mags[:-1] + mags[1:]) / 2
    code = torch.bucketize(a, mids, right=True)
    on_tie = (a.unsqueeze(-1) == mids).any(dim=-1)
    tie_code = torch.bucketize(a, mids, right=False)    # On a tie: the lower of the two candidate codes
    code = torch.where(on_tie, tie_code + (tie_code & 1), code)
    return (sign | code.to(torch.uint8)).to(torch.uint8)

def _dequantize_e2m1(codes: torch.Tensor) -> torch.Tensor:
    mags = _E2M1_MAGNITUDES.to(codes.device)
    val = mags[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -val, val)

def quantize_k_cache(
    input_k_cache: torch.Tensor,    # (num_blocks, block_size, h_k, d)
    kvcache_layout: KVCacheLayout,
) -> torch.Tensor:
    """
    Quantize the k-cache
    Each token stores its quantized raw data followed immediately by its scales.
    """
    d, d_nope, d_rope, tile_size, num_tiles = kvcache_layout.get_meta()
    assert input_k_cache.shape[-1] == d
    num_blocks, block_size, h_k, _ = input_k_cache.shape
    assert h_k == 1
    input_k_cache = input_k_cache.squeeze(2)    # [num_blocks, block_size, d]

    if kvcache_layout == KVCacheLayout.V41_FP8Sparse:
        # Token layout: [512 B fp8 data, 16 B ue8m0 scales].
        bytes_per_token = kvcache_layout.get_bytes_per_token()
        result = torch.empty((num_blocks, block_size, bytes_per_token), dtype=torch.float8_e4m3fn, device=input_k_cache.device)
        result_k_nope_rope_part = result[..., :d_nope+d_rope]
        result_k_scale_factor_view_as_dtype = torch.int8 if kk.is_on_ascend_platform() else torch.float8_e8m0fnu    # torch_npu has bug when copying from/to `torch.float8_e8m0fnu`
        result_k_scale_factor = result[..., d_nope+d_rope:].view(result_k_scale_factor_view_as_dtype)  # [num_blocks, block_size, 16]

        is_ascend = kk.is_on_ascend_platform()
        for tile_idx in range(0, 16):
            cur_scale_factors_inv = torch.abs(input_k_cache[..., tile_idx*tile_size:(tile_idx+1)*tile_size]).max(dim=-1).values.float() / 448.0
            cur_scale_factors_inv = _cast_scale_inv_to_ue8m0(cur_scale_factors_inv)
            if is_ascend:
                # torch_npu can not copy from float32 to float8_e8m0fnu while preserving the bit pattern,
                # so compute the e8m0 byte pattern (= bf16 bits >> 7) on CPU and write it as int8 bits.
                scale_bits = cur_scale_factors_inv.view(torch.int32) // 2**23
                result_k_scale_factor[:, :, tile_idx] = scale_bits.to(torch.int8)
            else:
                result_k_scale_factor[:, :, tile_idx] = cur_scale_factors_inv.to(torch.float8_e8m0fnu)
            cur_scale_factors_inv = cur_scale_factors_inv.view(num_blocks, block_size, 1)
            cur_quantized_nope = (input_k_cache[..., tile_idx*tile_size:(tile_idx+1)*tile_size].float() / cur_scale_factors_inv.float()).to(torch.float8_e4m3fn)
            result_k_nope_rope_part[:, :, tile_idx*tile_size:(tile_idx+1)*tile_size] = cur_quantized_nope

        result = result.view(num_blocks, block_size, 1, -1)
        return result

    elif kvcache_layout == KVCacheLayout.V41_FP4Sparse:
        # Token layout: [256 B fp4 data, 32 B e4m3 scales]. Element i of a row is in byte i//2, even
        # elements in the low nibble. The scale is amax / 6 (6 = max magnitude of e2m1) rounded to e4m3, without a per-tensor scale
        bytes_per_token = kvcache_layout.get_bytes_per_token()
        result = torch.empty((num_blocks, block_size, bytes_per_token), dtype=torch.float8_e4m3fn, device=input_k_cache.device)
        result_k_data = result[..., :d//2].view(torch.uint8)
        result_k_scale = result[..., d//2:]

        # Chunk along blocks: the fp32 intermediates below are 4x the size of the (bf16) cache, and a
        # whole-cache version would need several of them alive at once (e.g. 3 x 30.7GB for a 15GB cache)
        blocks_per_chunk = max(1, (1 << 26) // (block_size * d))
        for b0 in range(0, num_blocks, blocks_per_chunk):
            b1 = min(b0 + blocks_per_chunk, num_blocks)
            n = b1 - b0
            x = input_k_cache[b0: b1].float()
            amax = torch.nan_to_num(x.abs(), nan=float("inf")).view(n, block_size, num_tiles, tile_size).amax(dim=-1)   # A NaN element poisons the whole tile
            scale = torch.clamp(amax / 6.0, 2.0**-9, 448.0).to(torch.float8_e4m3fn)     # Clamp to the e4m3 range first: torch maps overflow to NaN
            if kk.is_on_ascend_platform():
                # torch_npu cannot fill FP8 tensors; write the E4M3 NaN encoding through its byte view.
                scale.view(torch.int8).masked_fill_(torch.isinf(amax), 0x7f)
            else:
                scale = torch.where(torch.isinf(amax), torch.full_like(scale, float("nan")), scale)
            normalized = x.view(n, block_size, num_tiles, tile_size) / scale.float().unsqueeze(-1)
            if input_k_cache.device.type == "npu":
                import torch_npu
                # Accept the native cast's FP32 rounding near E2M1 midpoints.
                result_k_data[b0: b1] = torch_npu.npu_dtype_cast(
                    normalized, torch_npu.float4_e2m1fn_x2
                ).view(n, block_size, d // 2)
            else:
                codes = _quantize_to_e2m1(normalized).view(n, block_size, d)
                result_k_data[b0: b1] = codes[..., 0::2] | (codes[..., 1::2] << 4)
            result_k_scale[b0: b1] = scale

        result = result.view(num_blocks, block_size, 1, -1)
        return result

    else:
        raise NotImplementedError(f"Unsupported kvcache_layout: {kvcache_layout}")
    

def dequantize_k_cache(
    quant_k_cache: torch.Tensor,    # (num_blocks, block_size, 1, bytes_per_token)
    kvcache_layout: KVCacheLayout,
) -> torch.Tensor:
    """
    De-quantize the k-cache
    """
    d, d_nope, d_rope, tile_size, num_tiles = kvcache_layout.get_meta()
    num_blocks, block_size, h_k, _ = quant_k_cache.shape
    assert h_k == 1
    result = torch.empty((num_blocks, block_size, d), dtype=torch.bfloat16, device=quant_k_cache.device)

    if kvcache_layout == KVCacheLayout.V41_FP8Sparse:
        quant_k_cache = quant_k_cache.view(num_blocks, block_size, -1)
        input_nope_rope = quant_k_cache[..., :d_nope+d_rope]
        input_scale = quant_k_cache[..., d_nope+d_rope:].view(torch.float8_e8m0fnu)  # [num_blocks, block_size, 16]

        # Dequant NoPE (tiles 0-13, each tile size 32)
        is_ascend = kk.is_on_ascend_platform()
        for tile_idx in range(0, 16):
            cur_nope_rope = input_nope_rope[..., tile_idx*tile_size:(tile_idx+1)*tile_size].to(torch.bfloat16)
            if is_ascend:
                cur_scales = (input_scale[:, :, tile_idx].view(torch.int8).to(torch.int16) * 2**7).view(torch.bfloat16) # torch_npu can not cast float8_e8m0 to bf16 on NPU
            else:
                cur_scales = input_scale[:, :, tile_idx].to(torch.bfloat16)
            cur_scales = cur_scales.unsqueeze(-1)
            result[..., tile_idx*tile_size:(tile_idx+1)*tile_size] = cur_nope_rope * cur_scales
            
    elif kvcache_layout == KVCacheLayout.V41_FP4Sparse:
        quant_k_cache = quant_k_cache.view(num_blocks, block_size, -1)
        input_data = quant_k_cache[..., :d//2].view(torch.uint8)
        input_scale = quant_k_cache[..., d//2:].view(torch.float8_e4m3fn)

        # Chunk along blocks to bound intermediate memory for large caches.
        blocks_per_chunk = max(1, (1 << 26) // (block_size * d))
        for b0 in range(0, num_blocks, blocks_per_chunk):
            b1 = min(b0 + blocks_per_chunk, num_blocks)
            if quant_k_cache.device.type == "npu":
                import torch_npu
                # Packed FP4 cast needs contiguous input; raw page rows skip the scale region.
                values = torch_npu.npu_dtype_cast(
                    input_data[b0:b1].contiguous(), torch.bfloat16, input_dtype=torch_npu.float4_e2m1fn_x2
                ).view(b1 - b0, block_size, num_tiles, tile_size)
            else:
                codes = torch.empty((b1 - b0, block_size, d), dtype=torch.uint8, device=quant_k_cache.device)
                codes[..., 0::2] = input_data[b0:b1] & 0xF
                codes[..., 1::2] = input_data[b0:b1] >> 4
                values = _dequantize_e2m1(codes).view(b1 - b0, block_size, num_tiles, tile_size)
            # e2m1 x e4m3 has at most 2 + 4 significant bits, so the product is exact in bf16, as in the kernel
            result[b0:b1] = (values * input_scale[b0:b1].to(values.dtype).unsqueeze(-1)).view(b1 - b0, block_size, d).to(torch.bfloat16)

    else:
        raise NotImplementedError(f"Unsupported kvcache_layout: {kvcache_layout}")
    
    result = result.view(num_blocks, block_size, 1, d)
    return result


def abs_indices2indices_in_kvcache(
    abs_indices: torch.Tensor,  # [b, s_q, topk]
    block_table: torch.Tensor,  # [b, /]
    block_size: int,
) -> torch.Tensor:
    """
    Convert abs_indices (logical index, ranging from 0 to s_k-1) to index expected by the sparse attn kernel
    Equivalent to:
    
    b, s_q, topk = abs_indices.shape
    indices_in_kvcache = torch.empty_like(abs_indices)
    for i in range(b):
        cur_abs_indices = abs_indices[i, :, :].clone()  # [s_q, topk]
        invalid_mask = cur_abs_indices == -1
        cur_abs_indices[invalid_mask] = 0
        cur_indices_in_kvcache = block_table[i].index_select(0, cur_abs_indices.flatten()//block_size).view(s_q, topk)*block_size + cur_abs_indices%block_size
        cur_indices_in_kvcache[invalid_mask] = -1
        indices_in_kvcache[i] = cur_indices_in_kvcache
    return indices_in_kvcache

    """
    b, s_q, topk = abs_indices.shape
    _, max_blocks_per_seq = block_table.shape

    abs_indices = abs_indices.clone()
    invalid_mask = abs_indices == -1
    abs_indices[invalid_mask] = 0

    real_block_idxs = block_table.view(-1).index_select(0, (abs_indices//block_size + torch.arange(0, b).view(b, 1, 1)*max_blocks_per_seq).view(-1))
    indices_in_kvcache = real_block_idxs.view(b, s_q, topk)*block_size + abs_indices%block_size
    indices_in_kvcache[invalid_mask] = -1

    return indices_in_kvcache
