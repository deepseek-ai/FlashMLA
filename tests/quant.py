import torch

def quantize_k_cache(
    input_k_cache: torch.Tensor,    # (num_blocks, block_size, h_k, d)
    dv: int,
    tile_size: int = 128,
) -> torch.Tensor:
    """
    Quantize the k-cache
    Return a tensor with shape (num_blocks, block_size, h_k, dv + 4(dv/tile_size) + t(d-dv)) of dtype float8_e4m3fn, where t = input_k_cache.element_size()
    For more detail about the layout of K/V, please refer to comments in flash_mla_interface.py or README.md
    """
    assert dv % tile_size == 0
    num_tiles = dv // tile_size
    num_blocks, block_size, h_k, d = input_k_cache.shape
    assert h_k == 1
    input_k_cache = input_k_cache.squeeze(2)    # [num_blocks, block_size, d]
    input_elem_size = input_k_cache.element_size()

    result = torch.empty((num_blocks, block_size, dv + num_tiles * 4 + input_elem_size * (d - dv)), dtype=torch.float8_e4m3fn, device=input_k_cache.device)
    result_k_nope_part = result[..., :dv]
    result_k_scale_factor = result[..., dv:dv + num_tiles * 4].view(torch.float32)
    result_k_rope_part = result[..., dv + num_tiles * 4:].view(input_k_cache.dtype)
    result_k_rope_part[:] = input_k_cache[..., dv:]

    for tile_idx in range(0, num_tiles):
        cur_scale_factors_inv = torch.abs(input_k_cache[..., tile_idx * tile_size:(tile_idx + 1) * tile_size]).max(dim=-1).values / 448.0  # [num_blocks, block_size]
        result_k_scale_factor[:, :, tile_idx] = cur_scale_factors_inv

        cur_scale_factors_inv.unsqueeze_(-1)    # [num_blocks, block_size, 1]
        cur_quantized_nope = (input_k_cache[..., tile_idx * tile_size:(tile_idx + 1) * tile_size].float() / cur_scale_factors_inv.float()).to(torch.float8_e4m3fn)
        result_k_nope_part[..., tile_idx * tile_size:(tile_idx + 1) * tile_size] = cur_quantized_nope

    result = result.view(num_blocks, block_size, 1, -1)
    return result


def dequantize_k_cache(
    quant_k_cache: torch.Tensor,    # (num_blocks, block_size, 1, bytes_per_token)
    dv: int = 512,
    tile_size: int = 128,
    d: int = 576
) -> torch.Tensor:
    """
    De-quantize the k-cache
    """
    assert dv % tile_size == 0
    num_tiles = dv // tile_size
    num_blocks, block_size, h_k, _ = quant_k_cache.shape
    assert h_k == 1
    result = torch.empty((num_blocks, block_size, d), dtype=torch.bfloat16, device=quant_k_cache.device)

    quant_k_cache = quant_k_cache.view(num_blocks, block_size, -1)

    input_nope = quant_k_cache[..., :dv]
    input_scale = quant_k_cache[..., dv:dv + num_tiles * 4].view(torch.float32)
    input_rope = quant_k_cache[..., dv + num_tiles * 4:].view(torch.bfloat16)
    result[..., dv:] = input_rope

    for tile_idx in range(0, num_tiles):
        cur_nope = input_nope[..., tile_idx * tile_size:(tile_idx + 1) * tile_size].to(torch.float32)
        cur_scales = input_scale[..., tile_idx].unsqueeze(-1)
        result[..., tile_idx * tile_size:(tile_idx + 1) * tile_size] = cur_nope * cur_scales

    result = result.view(num_blocks, block_size, 1, d)
    return result
