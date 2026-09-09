import enum
from typing import Tuple

import torch

import kernelkit as kk

class KVCacheLayout(enum.Enum):
    V32_FP8 = 0
    V32_FP8Sparse = 1
    V4_FP8Sparse = 2
    V41_FP8Sparse = 3
    V41_FP4 = 4

    def get_meta(self) -> Tuple[int, int, int, int, int]:
        # Return: (d, d_nope, d_rope, tile_size, num_tiles)
        return {
            KVCacheLayout.V32_FP8: (576, 512, 64, 128, 4),
            KVCacheLayout.V32_FP8Sparse: (576, 512, 64, 128, 4),
            KVCacheLayout.V4_FP8Sparse: (512, 448, 64, 64, 7),
            KVCacheLayout.V41_FP8Sparse: (512, 448, 64, 32, 16),  # 14 NoPE + 2 RoPE tiles
            KVCacheLayout.V41_FP4: (512, 448, 64, 16, 32),  # 28 NoPE + 4 RoPE tiles, all fp4
        }[self]

    def get_bytes_per_token(self) -> int:
        d, d_nope, d_rope, tile_size, num_tiles = self.get_meta()
        return {
            KVCacheLayout.V32_FP8: d_nope + num_tiles*4 + 2*d_rope,
            KVCacheLayout.V32_FP8Sparse: d_nope + num_tiles*4 + 2*d_rope,
            KVCacheLayout.V4_FP8Sparse: d_nope + 2*d_rope + num_tiles + 1,
            KVCacheLayout.V41_FP8Sparse: d_nope + d_rope + num_tiles,
            KVCacheLayout.V41_FP4: d // 2 + num_tiles,
        }[self]

def _cast_scale_inv_to_ue8m0(scales_inv: torch.Tensor, out_dtype = torch.float32) -> torch.Tensor:
    return torch.pow(2, torch.clamp_min(scales_inv, 1e-4).log2().ceil()).to(out_dtype)    # This 1e-4 align with Tile Kernel's FP8_AMAX_MARGIN

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
    For more detail about the layout of K/V, please refer to comments in flash_mla_interface.py
    """
    d, d_nope, d_rope, tile_size, num_tiles = kvcache_layout.get_meta()
    assert input_k_cache.shape[-1] == d
    num_blocks, block_size, h_k, _ = input_k_cache.shape
    assert h_k == 1
    input_k_cache = input_k_cache.squeeze(2)    # [num_blocks, block_size, d]
    input_elem_size = input_k_cache.element_size()

    if kvcache_layout == KVCacheLayout.V32_FP8:
        bytes_per_block = block_size*d_nope + block_size*num_tiles*4 + block_size*input_elem_size*d_rope
        result = kk.gen_non_contiguous_tensor((num_blocks, bytes_per_block), dtype=torch.float8_e4m3fn, device=input_k_cache.device)
        result_k_nope_part = result[..., :block_size*d_nope].view(num_blocks, 512//16, block_size, 16)
        result_k_scale_factor = result[..., block_size*d_nope:block_size*(d_nope+4*num_tiles)].view(torch.float32).view(num_blocks, num_tiles, block_size).permute(0, 2, 1) # [num_blocks, block_size, num_tiles]
        result_k_rope_part = result[..., block_size*(d_nope+4*num_tiles):].view(input_k_cache.dtype).view(num_blocks, block_size, d_rope)

        result_k_rope_part[:] = input_k_cache[..., d_nope:]

        for tile_idx in range(0, num_tiles):
            cur_scale_factors_inv = torch.abs(input_k_cache[..., tile_idx*tile_size:(tile_idx+1)*tile_size]).max(dim=-1).values.float() / 448.0 # [num_blocks, block_size]
            cur_scale_factors_inv = _cast_scale_inv_to_ue8m0(cur_scale_factors_inv)
            result_k_scale_factor[:, :, tile_idx] = cur_scale_factors_inv

            cur_scale_factors_inv = cur_scale_factors_inv.view(num_blocks, block_size, 1)
            cur_quantized_nope = (input_k_cache[..., tile_idx*tile_size:(tile_idx+1)*tile_size].float() / cur_scale_factors_inv.float()).to(torch.float8_e4m3fn)
            result_k_nope_part[:, tile_idx*tile_size//16:(tile_idx+1)*tile_size//16, :, :].permute(0, 2, 1, 3)[:] = cur_quantized_nope.view(num_blocks, block_size, tile_size//16, 16)

        result = result.view(num_blocks, block_size, 1, -1)
        return result

    elif kvcache_layout == KVCacheLayout.V32_FP8Sparse:
        bytes_per_token = d_nope + num_tiles*4 + input_elem_size*d_rope
        result = torch.empty((num_blocks, block_size+1, bytes_per_token), dtype=torch.float8_e4m3fn, device=input_k_cache.device)[:, :block_size, :]
        result_k_nope_part = result[..., :d_nope]
        result_k_scale_factor = result[..., d_nope: d_nope + num_tiles*4].view(torch.float32)
        result_k_rope_part = result[..., d_nope + num_tiles*4:].view(input_k_cache.dtype)
        result_k_rope_part[:] = input_k_cache[..., d_nope:]

        for tile_idx in range(0, num_tiles):
            cur_scale_factors_inv = torch.abs(input_k_cache[..., tile_idx*tile_size:(tile_idx+1)*tile_size]).max(dim=-1).values.float() / 448.0 # [num_blocks, block_size]
            cur_scale_factors_inv = _cast_scale_inv_to_ue8m0(cur_scale_factors_inv)
            result_k_scale_factor[:, :, tile_idx] = cur_scale_factors_inv

            cur_scale_factors_inv.unsqueeze_(-1)    # [num_blocks, block_size, 1]
            cur_quantized_nope = (input_k_cache[..., tile_idx*tile_size:(tile_idx+1)*tile_size].float() / cur_scale_factors_inv.float()).to(torch.float8_e4m3fn)
            result_k_nope_part[..., tile_idx*tile_size:(tile_idx+1)*tile_size] = cur_quantized_nope
        
        result = result.view(num_blocks, block_size, 1, -1)
        return result
    
    elif kvcache_layout == KVCacheLayout.V4_FP8Sparse:
        bytes_per_token = d_nope + 2*d_rope + num_tiles + 1
        size_per_block_padded = (block_size*bytes_per_token + 576-1) // 576 * 576
        result = torch.empty((num_blocks, size_per_block_padded), dtype=torch.float8_e4m3fn, device=input_k_cache.device)[:, :block_size*bytes_per_token]
        result_k_nope_rope_part = result[:, :block_size*(d_nope+2*d_rope)].view(num_blocks, block_size, d_nope + 2*d_rope)
        result_k_nope = result_k_nope_rope_part[:, :, :d_nope]  # [num_blocks, block_size, d_nope]
        result_k_rope = result_k_nope_rope_part[:, :, d_nope:].view(input_k_cache.dtype)  # [num_blocks, block_size, d_rope]
        result_k_scale_factor = result[:, block_size*(d_nope+2*d_rope):].view(num_blocks, block_size, 8)[:, :, :7].view(torch.float8_e8m0fnu)   # [num_blocks, block_size, num_tiles]

        result_k_rope[:] = input_k_cache[..., d_nope:]
        for tile_idx in range(0, num_tiles):
            cur_scale_factors_inv = torch.abs(input_k_cache[..., tile_idx*tile_size:(tile_idx+1)*tile_size]).max(dim=-1).values.float() / 448.0 # [num_blocks, block_size]
            cur_scale_factors_inv = _cast_scale_inv_to_ue8m0(cur_scale_factors_inv)
            result_k_scale_factor[:, :, tile_idx] = cur_scale_factors_inv.to(torch.float8_e8m0fnu)

            cur_scale_factors_inv = cur_scale_factors_inv.view(num_blocks, block_size, 1)
            cur_quantized_nope = (input_k_cache[..., tile_idx*tile_size:(tile_idx+1)*tile_size].float() / cur_scale_factors_inv.float()).to(torch.float8_e4m3fn)
            result_k_nope[:, :, tile_idx*tile_size:(tile_idx+1)*tile_size] = cur_quantized_nope

        result = result.view(num_blocks, block_size, 1, -1)
        return result

    elif kvcache_layout == KVCacheLayout.V41_FP8Sparse:
        bytes_per_token = d_nope + d_rope + num_tiles
        size_per_block_padded = (block_size*bytes_per_token + 512-1) // 512 * 512
        result = torch.empty((num_blocks, size_per_block_padded), dtype=torch.float8_e4m3fn, device=input_k_cache.device)[:, :block_size*bytes_per_token]
        result_k_nope_rope_part = result[:, :block_size*(d_nope+d_rope)].view(num_blocks, block_size, d_nope + d_rope)
        result_k_scale_factor = result[:, block_size*(d_nope+d_rope):].view(num_blocks, block_size, num_tiles).view(torch.float8_e8m0fnu)  # [num_blocks, block_size, 16]

        for tile_idx in range(0, 16):
            cur_scale_factors_inv = torch.abs(input_k_cache[..., tile_idx*tile_size:(tile_idx+1)*tile_size]).max(dim=-1).values.float() / 448.0
            cur_scale_factors_inv = _cast_scale_inv_to_ue8m0(cur_scale_factors_inv)
            result_k_scale_factor[:, :, tile_idx] = cur_scale_factors_inv.to(torch.float8_e8m0fnu)
            cur_scale_factors_inv = cur_scale_factors_inv.view(num_blocks, block_size, 1)
            cur_quantized_nope = (input_k_cache[..., tile_idx*tile_size:(tile_idx+1)*tile_size].float() / cur_scale_factors_inv.float()).to(torch.float8_e4m3fn)
            result_k_nope_rope_part[:, :, tile_idx*tile_size:(tile_idx+1)*tile_size] = cur_quantized_nope

        result = result.view(num_blocks, block_size, 1, -1)
        return result

    elif kvcache_layout == KVCacheLayout.V41_FP4:
        # Block layout: [block_size x 256 B fp4 rows][block_size x 32 B e4m3 scale rows]. Element i of a row is in byte i//2, even
        # elements in the low nibble. The scale is amax / 6 (6 = max magnitude of e2m1) rounded to e4m3, without a per-tensor scale
        bytes_per_token = d // 2 + num_tiles
        size_per_block_padded = (block_size*bytes_per_token + 512-1) // 512 * 512
        result = torch.empty((num_blocks, size_per_block_padded), dtype=torch.float8_e4m3fn, device=input_k_cache.device)[:, :block_size*bytes_per_token]
        result_k_data = result[:, :block_size*(d//2)].view(torch.uint8).view(num_blocks, block_size, d//2)
        result_k_scale = result[:, block_size*(d//2):].view(num_blocks, block_size, num_tiles)

        x = input_k_cache.float()
        amax = torch.nan_to_num(x.abs(), nan=float("inf")).view(num_blocks, block_size, num_tiles, tile_size).amax(dim=-1)   # A NaN element poisons the whole tile
        scale = torch.clamp(amax / 6.0, 2.0**-9, 448.0).to(torch.float8_e4m3fn)     # Clamp to the e4m3 range first: torch maps overflow to NaN
        scale = torch.where(torch.isinf(amax), torch.full_like(scale, float("nan")), scale)
        codes = _quantize_to_e2m1(x.view(num_blocks, block_size, num_tiles, tile_size) / scale.float().unsqueeze(-1))
        codes = codes.view(num_blocks, block_size, d)
        result_k_data[:] = codes[..., 0::2] | (codes[..., 1::2] << 4)
        result_k_scale[:] = scale

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

    if kvcache_layout == KVCacheLayout.V32_FP8:
        quant_k_cache = quant_k_cache.view(num_blocks, -1)  # [num_blocks, ...]
        input_nope = quant_k_cache[..., :block_size*d_nope].view(num_blocks, 512//16, block_size, 16).view(torch.float8_e4m3fn)
        input_scale = quant_k_cache[..., block_size*d_nope:block_size*(d_nope+4*num_tiles)].view(torch.float32).view(num_blocks, num_tiles, block_size).permute(0, 2, 1).contiguous() # [num_blocks, block_size, num_tiles]
        input_rope = quant_k_cache[..., block_size*(d_nope+4*num_tiles):].view(torch.bfloat16).view(num_blocks, block_size, d_rope)

        result[..., d_nope:] = input_rope
        for tile_idx in range(0, num_tiles):
            cur_nope = input_nope[:, tile_idx*tile_size//16:(tile_idx+1)*tile_size//16, :, :].to(torch.float32).permute(0, 2, 1, 3).contiguous().view(num_blocks, block_size, tile_size)
            cur_scales = input_scale[:, :, tile_idx].unsqueeze(-1)
            result[..., tile_idx*tile_size:(tile_idx+1)*tile_size] = cur_nope * cur_scales

    elif kvcache_layout == KVCacheLayout.V32_FP8Sparse:
        quant_k_cache = quant_k_cache.view(num_blocks, block_size, -1)

        input_nope = quant_k_cache[..., :d_nope].view(torch.float8_e4m3fn)
        input_scale = quant_k_cache[..., d_nope:d_nope + num_tiles*4].view(torch.float32)
        input_rope = quant_k_cache[..., d_nope + num_tiles*4:].view(torch.bfloat16)
        result[..., d_nope:] = input_rope

        for tile_idx in range(0, num_tiles):
            cur_nope = input_nope[..., tile_idx*tile_size:(tile_idx+1)*tile_size].to(torch.float32)
            cur_scales = input_scale[..., tile_idx].unsqueeze(-1)
            result[..., tile_idx*tile_size:(tile_idx+1)*tile_size] = cur_nope * cur_scales

    elif kvcache_layout == KVCacheLayout.V4_FP8Sparse:
        quant_k_cache = quant_k_cache.view(num_blocks, -1)  # [num_blocks, ...]  
        input_nope_rope = quant_k_cache[:, :block_size*(d_nope+2*d_rope)].view(num_blocks, block_size, d_nope + 2*d_rope)
        input_nope = input_nope_rope[:, :, :d_nope].view(torch.float8_e4m3fn)
        input_rope = input_nope_rope[:, :, d_nope:].view(torch.bfloat16)
        input_scale = quant_k_cache[:, block_size*(d_nope+2*d_rope):].view(num_blocks, block_size, 8)[:, :, :7].view(torch.float8_e8m0fnu)   # [num_blocks, block_size, num_tiles]

        result[..., d_nope:] = input_rope
        for tile_idx in range(0, num_tiles):
            cur_nope = input_nope[..., tile_idx*tile_size:(tile_idx+1)*tile_size].to(torch.bfloat16)
            cur_scales = input_scale[:, :, tile_idx].to(torch.bfloat16).unsqueeze(-1)
            result[..., tile_idx*tile_size: (tile_idx+1)*tile_size] = cur_nope * cur_scales

    elif kvcache_layout == KVCacheLayout.V41_FP8Sparse:
        quant_k_cache = quant_k_cache.view(num_blocks, -1)  # [num_blocks, ...]
        input_nope_rope = quant_k_cache[:, :block_size*(d_nope+d_rope)].view(num_blocks, block_size, d_nope + d_rope)
        input_scale = quant_k_cache[:, block_size*(d_nope+d_rope):].view(num_blocks, block_size, num_tiles).view(torch.float8_e8m0fnu)  # [num_blocks, block_size, 16]

        # Dequant NoPE (tiles 0-13, each tile size 32)
        for tile_idx in range(0, 16):
            cur_nope_rope = input_nope_rope[..., tile_idx*tile_size:(tile_idx+1)*tile_size].to(torch.bfloat16)
            cur_scales = input_scale[:, :, tile_idx].to(torch.bfloat16).unsqueeze(-1)
            result[..., tile_idx*tile_size:(tile_idx+1)*tile_size] = cur_nope_rope * cur_scales
            
    elif kvcache_layout == KVCacheLayout.V41_FP4:
        quant_k_cache = quant_k_cache.view(num_blocks, -1)
        input_data = quant_k_cache[:, :block_size*(d//2)].view(torch.uint8).view(num_blocks, block_size, d//2)
        input_scale = quant_k_cache[:, block_size*(d//2):block_size*(d//2 + num_tiles)].view(torch.float8_e4m3fn).view(num_blocks, block_size, num_tiles)

        # Chunked along the blocks: the fp32 intermediates are ~12x the size of the cache
        blocks_per_chunk = max(1, (1 << 26) // (block_size * d))
        for b0 in range(0, num_blocks, blocks_per_chunk):
            b1 = min(b0 + blocks_per_chunk, num_blocks)
            codes = torch.empty((b1 - b0, block_size, d), dtype=torch.uint8, device=quant_k_cache.device)
            codes[..., 0::2] = input_data[b0:b1] & 0xF
            codes[..., 1::2] = input_data[b0:b1] >> 4
            values = _dequantize_e2m1(codes).view(b1 - b0, block_size, num_tiles, tile_size)
            # e2m1 x e4m3 has at most 2 + 4 significant bits, so the product is exact in bf16, as in the kernel
            result[b0:b1] = (values * input_scale[b0:b1].float().unsqueeze(-1)).view(b1 - b0, block_size, d).to(torch.bfloat16)

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