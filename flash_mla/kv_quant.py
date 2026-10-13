"""Reference (pure torch) implementation of the quantized MLA KV-cache format, see docs/kv_quant_format.md.

Row = 576 dims: 512 latent dims quantized to 1 byte each (fp8 e4m3 or int8) with ONE fp32 scale per token, and 64 rope dims
kept as bf16.  The CUDA write path (csrc/fused/mla_decode_prologue.cu) and the in-kernel dequant of the attention kernel
must agree bit for bit with `quantize_kv` / `dequantize_kv` below.

    kvq   uint8 [..., 640]   bytes 0..511 = quantized latent, bytes 512..639 = 64 bf16 rope values (native endian)
    scale float32 [...]      amax / 448 (e4m3) or amax / 127 (int8); amax == 0 -> scale 1, q = 0
    x ~ q * scale            (fp32 product)
"""
import torch

LATENT, ROPE = 512, 64
ROW_BYTES = LATENT + 2 * ROPE  # 640
KIND_ID = {"native": 0, "fp8": 1, "int8": 2}
E4M3_MAX, INT8_MAX = 448.0, 127.0


def kind_id(kind):
    return KIND_ID[kind] if isinstance(kind, str) else int(kind)


def bytes_per_token(kind):
    """HBM bytes read per cached token (scale included)."""
    return 2 * (LATENT + ROPE) if kind_id(kind) == 0 else ROW_BYTES + 4


_QMAX = {}


def _qmax(k, device):
    """0-dim fp32 device tensor holding 448 / 127, created once per device: an H2D copy per call would make quantize_kv
    impossible to capture in a CUDA graph (the warm-up iterations of capture_layer_graph populate the cache)."""
    key = (k, str(device))
    t = _QMAX.get(key)
    if t is None:
        t = _QMAX[key] = torch.tensor(E4M3_MAX if k == 1 else INT8_MAX, dtype=torch.float32, device=device)
    return t


def quantize_kv(x, kind):
    """x: [..., 576] bf16 -> (kvq uint8 [..., 640], scale fp32 [...])."""
    k = kind_id(kind)
    assert k in (1, 2), "kind must be 'fp8' or 'int8'"
    assert x.dtype == torch.bfloat16 and x.shape[-1] == LATENT + ROPE
    lat = x[..., :LATENT].float()
    amax = lat.abs().amax(dim=-1)
    qmax = _qmax(k, x.device)
    # IEEE fp32 division (tensor / 0-dim tensor): `amax / 448.0` with a python scalar would be computed by torch as
    # amax * (1/448), which is not correctly rounded and differs from the kernel's __fdiv_rn in the last bit.
    scale = torch.where(amax > 0, torch.div(amax, qmax), torch.ones_like(amax))
    y = lat / scale[..., None]
    if k == 1:
        q = y.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8)
    else:
        q = y.round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8).view(torch.uint8)  # torch.round = half to even
    rope = x[..., LATENT:].contiguous().view(torch.uint8)  # [..., 128]
    return torch.cat([q, rope], dim=-1).contiguous(), scale.contiguous()


def dequantize_kv(kvq, scale, kind, dtype=torch.bfloat16):
    """(kvq uint8 [..., 640], scale [...]) -> [..., 576] in `dtype` (fp32 product, rounded once)."""
    k = kind_id(kind)
    q = kvq[..., :LATENT].contiguous()
    qf = q.view(torch.float8_e4m3fn).float() if k == 1 else q.view(torch.int8).float()
    lat = qf * scale[..., None].float()
    rope = kvq[..., LATENT:].contiguous().view(torch.bfloat16).float()
    return torch.cat([lat, rope], dim=-1).to(dtype)


def alloc_quant_cache(num_blocks, page, device="cuda", fill_scale=1.0):
    """Zeroed quantized cache tensors in the layout the kernels expect: kvq uint8 [nblk, page, 640], scale fp32 [nblk, page]."""
    kvq = torch.zeros(num_blocks, page, ROW_BYTES, dtype=torch.uint8, device=device)
    scale = torch.full((num_blocks, page), fill_scale, dtype=torch.float32, device=device)
    return kvq, scale


def quant_error(x, kind):
    """Per-row and global error statistics of quantize->dequantize of x ([N, 576] bf16)."""
    kvq, sc = quantize_kv(x, kind)
    xd = dequantize_kv(kvq, sc, kind, torch.float32)
    xf = x.float()
    lat_err = (xd[..., :LATENT] - xf[..., :LATENT])
    rel = lat_err.norm(dim=-1) / xf[..., :LATENT].norm(dim=-1).clamp_min(1e-30)
    return dict(rel_rms_mean=rel.mean().item(), rel_rms_max=rel.max().item(),
                max_abs_over_amax=(lat_err.abs().amax(dim=-1) / xf[..., :LATENT].abs().amax(dim=-1).clamp_min(1e-30)).max().item(),
                rope_exact=bool(torch.equal(xd[..., LATENT:], xf[..., LATENT:])))
