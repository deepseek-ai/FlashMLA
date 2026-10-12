"""A DeepSeek-V2-Lite-shaped MLA attention block for decode (one token per sequence, or s_q tokens for MTP).

    hidden [T, 2048]
      q  = q_proj(hidden)            -> [T, 16, 192] = [q_nope 128 | q_pe 64]
      kv = kv_a_proj(hidden)         -> [T, 576]     = [kv_c 512 | k_pe 64]          (q_proj and kv_a_proj can be one GEMM)
      q_nope_abs = q_nope @ W_UK     -> [T, 16, 512]  (matrix absorption, bmm over heads)
      --- prologue: RMSNorm(kv_c), RoPE(q_pe), RoPE(k_pe), q = [q_nope_abs | q_pe_rot], paged-cache write [kv_c | k_pe_rot]
      o  = MLA attention over the paged cache (latent 576 / value 512)     -> [T, 16, 512]
      y  = o @ W_UV                  -> [T, 16, 128]
      out = o_proj(y.flatten)        -> [T, 2048]

Two implementations of the same math (shared weights):
    fused=False  "unfused PyTorch": HF-style elementwise ops for norm / rope / concat / cache write (~30 small kernels)
    fused=True   the prologue above is ONE kernel (csrc/fused), q_proj+kv_a_proj is one GEMM
Attention is pluggable (`attn`): "torch" (fp32 reference), "triton" (vendored vLLM/SGLang kernel, benchmarks/baselines),
"legacy" / "v2" / "pkg" (flash_mla.flash_mla_with_kvcache with that impl; page size 32 for legacy).
The whole step is CUDA-graph capturable (`capture_layer_graph`): metadata, positions and slots are computed on device.
"""
import math
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .kv_quant import dequantize_kv, kind_id
from .mla_prologue import apply_rope_ref, make_rope_cache, mla_decode_prologue, rmsnorm_ref


@dataclass
class MLAConfig:
    hidden: int = 2048
    n_heads: int = 16
    qk_nope: int = 128
    qk_rope: int = 64
    v_head: int = 128
    kv_lora: int = 512
    rope_base: float = 10000.0
    rms_eps: float = 1e-6
    n_layers: int = 27

    @property
    def q_head(self):
        return self.qk_nope + self.qk_rope

    @property
    def softmax_scale(self):  # MLA scales by the pre-absorption head dim (nope + rope)
        return self.q_head ** -0.5


class DecodeState:
    """Per-step data shared by all layers: page table, sequence lengths (incl. the new tokens), rope cache, scheduler metadata."""

    def __init__(self, cfg, block_table, cache_seqlens, s_q=1, max_pos=None, attn="legacy"):
        self.cfg, self.block_table, self.cache_seqlens, self.s_q = cfg, block_table, cache_seqlens, s_q
        self.B = cache_seqlens.shape[0]
        self.T = self.B * s_q
        self.attn = attn
        max_pos = max_pos or (block_table.shape[1] * 64 + 64)
        self.cos_sin = make_rope_cache(max_pos, cfg.qk_rope, cfg.rope_base, device=cache_seqlens.device)
        self.meta = self.nsplit = None
        self.workspace = None
        self.positions = self.slots = None
        self.page = None

    def refresh(self, page):
        """Device-side per-step bookkeeping (graph capturable): token positions, cache slots, MLA scheduler metadata."""
        self.page = page
        t = torch.arange(self.T, device=self.cache_seqlens.device)
        b, j = t // self.s_q, t % self.s_q
        self.positions = self.cache_seqlens[b].long() - self.s_q + j
        self.slots = self.block_table[b, self.positions // page].long() * page + self.positions % page
        if self.attn in ("legacy", "legacy_ws", "v2", "pkg"):
            import flash_mla
            self.meta, self.nsplit = flash_mla.get_mla_metadata(self.cache_seqlens, self.s_q * self.cfg.n_heads, 1)
            if self.workspace is None and self.attn in ("v2", "pkg") and hasattr(flash_mla, "alloc_mla_workspace"):
                # persistent scratch, allocated once (first call is the eager warm-up, before any graph capture)
                self.workspace = flash_mla.alloc_mla_workspace(self.B, self.s_q * self.cfg.n_heads, 1, self.meta.shape[0])


def _rope_hf(x, cos_sin, pos):
    """HF-style rotate_half rope on [N, ..., 64] in several elementwise ops (the 'unfused' baseline)."""
    cs = cos_sin[pos]
    c, s = cs[:, :32], cs[:, 32:]
    shape = [x.shape[0]] + [1] * (x.dim() - 2) + [32]
    c, s = torch.cat([c, c], -1).view(shape[:-1] + [64]), torch.cat([s, s], -1).view(shape[:-1] + [64])
    x1, x2 = x[..., :32], x[..., 32:]
    rot = torch.cat([-x2, x1], dim=-1)
    return (x.float() * c + rot.float() * s).to(x.dtype)


class MLADecodeLayer(torch.nn.Module):
    def __init__(self, cfg=None, dtype=torch.bfloat16, device="cuda", seed=0, fused=True, merge_proj=True, attn="legacy",
                 kv_dtype="native"):
        super().__init__()
        self.cfg = cfg = cfg or MLAConfig()
        self.dtype, self.fused, self.merge_proj, self.attn, self.kv_dtype = dtype, fused, merge_proj, attn, kv_dtype
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        H = cfg.n_heads

        def w(*shape, std=0.02):
            return (torch.randn(*shape, generator=g, device=device) * std).to(dtype)

        self.w_q = w(H * cfg.q_head, cfg.hidden)
        self.w_kv_a = w(cfg.kv_lora + cfg.qk_rope, cfg.hidden)
        self.w_qkv_a = torch.cat([self.w_q, self.w_kv_a], 0).contiguous()  # q_proj + kv_a_proj as one GEMM
        self.kv_norm_w = (1 + 0.1 * torch.randn(cfg.kv_lora, generator=g, device=device)).to(dtype)
        self.w_uk = w(H, cfg.qk_nope, cfg.kv_lora, std=0.05)  # absorbed K up-projection  [H, 128, 512]
        self.w_uv = w(H, cfg.kv_lora, cfg.v_head, std=0.05)   # absorbed V up-projection  [H, 512, 128]
        self.w_o = w(cfg.hidden, H * cfg.v_head)

    # ------------------------------------------------------------------------------------------------------------------
    def _attend(self, q_full, kv_cache, kv_scale, st):
        """q_full [T, H, 576] -> o [T, H, 512]"""
        cfg, T, H = self.cfg, st.T, self.cfg.n_heads
        dv = cfg.kv_lora
        a = self.attn
        if a in ("legacy", "legacy_ws", "v2", "pkg"):
            import flash_mla
            kw = {} if a == "pkg" else {"impl": a}
            if st.workspace is not None:
                kw["workspace"] = st.workspace
            if self.kv_dtype != "native":
                kw["kv_scale"] = kv_scale
            o, _ = flash_mla.flash_mla_with_kvcache(q_full.view(st.B, st.s_q, H, -1), kv_cache, st.block_table, st.cache_seqlens,
                                                    dv, st.meta, st.nsplit, softmax_scale=cfg.softmax_scale,
                                                    causal=st.s_q > 1, **kw)
            return o.view(T, H, dv)
        if a == "triton":
            assert st.s_q == 1 and self.kv_dtype == "native", "triton baseline: s_q == 1, native cache"
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks"))
            from baselines.triton_decode_attention import decode_attention_fwd
            if getattr(st, "_triton_buf", None) is None:
                splits = 8 if st.B >= 8 else 32
                st._triton_buf = (torch.empty(st.B, H, splits, dv + 1, dtype=torch.float32, device=q_full.device), splits,
                                  torch.ones((), dtype=torch.float32, device=q_full.device))
            logits, splits, one = st._triton_buf
            o = torch.empty(st.B, H, dv, dtype=q_full.dtype, device=q_full.device)
            lse = torch.empty(st.B, H, dtype=torch.float32, device=q_full.device)
            decode_attention_fwd(q_full.view(st.B, H, -1), kv_cache, kv_cache[..., :dv], o, lse, st.block_table,
                                 st.cache_seqlens, logits, splits, cfg.softmax_scale, kv_cache.shape[1], k_scale=one, v_scale=one)
            return o.view(T, H, dv)
        if a == "torch":
            return self._attend_torch(q_full, kv_cache, kv_scale, st)
        raise ValueError(a)

    def _attend_torch(self, q_full, kv_cache, kv_scale, st):
        """fp32 reference attention over the paged cache (dequantizes quantized caches)."""
        cfg, H, dv = self.cfg, self.cfg.n_heads, self.cfg.kv_lora
        page = kv_cache.shape[1]
        out = torch.empty(st.T, H, dv, dtype=q_full.dtype, device=q_full.device)
        npages = st.block_table.shape[1]
        kv_all = kv_cache if self.kv_dtype == "native" else dequantize_kv(kv_cache, kv_scale, self.kv_dtype, torch.float32)
        lens = st.cache_seqlens.tolist()  # host sync: reference only (not graph capturable)
        for b in range(st.B):
            kv = kv_all[st.block_table[b, :npages].long()].reshape(npages * page, -1)[: lens[b]].float()
            for j in range(st.s_q):
                n = lens[b] - (st.s_q - 1 - j)  # causal prefix visible to token j
                q = q_full[b * st.s_q + j].float()
                s = (q @ kv[:n].T) * cfg.softmax_scale
                out[b * st.s_q + j] = (torch.softmax(s, -1) @ kv[:n, :dv]).to(out.dtype)
        return out

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, hidden, kv_cache, st, kv_scale=None):
        cfg, T, H = self.cfg, st.T, self.cfg.n_heads
        if self.fused:
            if self.merge_proj:
                qkv = F.linear(hidden, self.w_qkv_a)
                nq = H * cfg.q_head
                q = qkv[:, :nq].view(T, H, cfg.q_head)
                kv_c, k_pe = qkv[:, nq:nq + cfg.kv_lora], qkv[:, nq + cfg.kv_lora:]
            else:
                q = F.linear(hidden, self.w_q).view(T, H, cfg.q_head)
                kv_a = F.linear(hidden, self.w_kv_a)
                kv_c, k_pe = kv_a[:, :cfg.kv_lora], kv_a[:, cfg.kv_lora:]
            q_abs = torch.bmm(q[..., :cfg.qk_nope].transpose(0, 1), self.w_uk).transpose(0, 1)  # [T, H, 512] (strided view)
            q_full = mla_decode_prologue(q_abs, q[..., cfg.qk_nope:], kv_c, k_pe, st.cos_sin, st.block_table, st.cache_seqlens,
                                         st.s_q, kv_cache, kv_scale=kv_scale, kv_norm_weight=self.kv_norm_w, eps=cfg.rms_eps,
                                         kv_dtype=self.kv_dtype)
        else:
            q = F.linear(hidden, self.w_q).view(T, H, cfg.q_head)
            kv_a = F.linear(hidden, self.w_kv_a)
            kv_c = rmsnorm_ref(kv_a[:, :cfg.kv_lora], self.kv_norm_w, cfg.rms_eps)
            k_pe = _rope_hf(kv_a[:, cfg.kv_lora:], st.cos_sin, st.positions)
            q_pe = _rope_hf(q[..., cfg.qk_nope:], st.cos_sin, st.positions)
            q_abs = torch.bmm(q[..., :cfg.qk_nope].transpose(0, 1), self.w_uk).transpose(0, 1)
            q_full = torch.cat([q_abs, q_pe], dim=-1)
            row = torch.cat([kv_c, k_pe], dim=-1)
            flat = kv_cache.view(-1, kv_cache.shape[-1])
            if self.kv_dtype == "native":
                flat[st.slots] = row
            else:
                from .kv_quant import quantize_kv
                kvq, sc = quantize_kv(row, self.kv_dtype)
                flat[st.slots] = kvq
                kv_scale.view(-1)[st.slots] = sc
        o = self._attend(q_full, kv_cache, kv_scale, st)                       # [T, H, 512]
        self.last_o = o                                                         # (kept for accuracy evaluation)
        y = torch.bmm(o.transpose(0, 1), self.w_uv).transpose(0, 1).reshape(T, H * cfg.v_head)
        return F.linear(y, self.w_o)


# ----------------------------------------------------------------------------------------------------------------------
def alloc_cache(num_blocks, page, kv_dtype="native", dtype=torch.bfloat16, device="cuda", fill_random=True, seed=0):
    """Paged latent KV cache of one layer: native [nblk, page, 1, 576] or (uint8 [nblk, page, 640], fp32 scale [nblk, page])."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    if kind_id(kv_dtype) == 0:
        c = torch.randn(num_blocks, page, 1, 576, generator=g, device=device, dtype=dtype) if fill_random else \
            torch.zeros(num_blocks, page, 1, 576, device=device, dtype=dtype)
        return c, None
    from .kv_quant import quantize_kv
    x = torch.randn(num_blocks, page, 576, generator=g, device=device, dtype=dtype)
    return quantize_kv(x, kv_dtype)


def make_state(cfg, b, s_k, page, s_q=1, device="cuda", attn="legacy", scatter=True, seed=0):
    """Page table for b sequences of length s_k (the last s_q tokens are the new ones) + DecodeState."""
    max_blocks = -(-s_k // page)
    nblk = b * max_blocks
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    perm = torch.randperm(nblk, generator=g, device=device) if scatter else torch.arange(nblk, device=device)
    bt = perm.view(b, max_blocks).to(torch.int32).contiguous()
    lens = torch.full((b,), s_k, dtype=torch.int32, device=device)
    st = DecodeState(cfg, bt, lens, s_q, max_pos=s_k + 8, attn=attn)
    st.refresh(page)
    return st, nblk


def capture_layer_graph(layers, caches, scales, st, hidden_static, out_static=None):
    """Capture one decode step through all `layers` (chained: layer l+1 consumes layer l's output) in a CUDA graph, metadata and
    per-step bookkeeping included. `hidden_static` is the (static) input buffer; the returned graph writes `out_static`."""
    def step():
        st.refresh(st.page)
        h = hidden_static
        for l, layer in enumerate(layers):
            h = layer(h, caches[l], st, scales[l] if scales else None)
        return h

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = step()
    return g, out
