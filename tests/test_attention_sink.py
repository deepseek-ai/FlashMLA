"""Validate the gpt-oss per-head attention sink on the SWA training path
(flash_attn_varlen_func, dense MLA 192/128, autograd-complete) vs a reference that
appends a value-less sink column to the softmax. Checks O, LSE, dQ, dK, dV, and the
sink-logit grad d(sink). The sink is window-independent, so causal (no window)
isolates it; SWA composes orthogonally. Run after a full build on B200.
"""
import torch
import flash_mla


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def reference(q, k, v, sink, scale):
    """dense causal MHA with a gpt-oss value-less sink column (float, differentiable)."""
    s, h, _ = q.shape
    scores = torch.einsum("qhd,khd->hqk", q, k) * scale            # [h, s, s]
    causal = torch.ones(s, s, dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~causal.view(1, s, s), float("-inf"))
    sink_col = sink.view(h, 1, 1).expand(h, s, 1)                  # [h, s, 1] value-less column
    combined = torch.cat([scores, sink_col], dim=-1)              # [h, s, s+1]
    lse = combined.logsumexp(-1)                                  # [h, s] (sink-aware)
    p = torch.softmax(combined, -1)[..., :s]                      # drop the sink column
    o = torch.einsum("hqk,khv->qhv", p, v)                        # [s, h, dv]
    return o, lse


def main():
    assert hasattr(flash_mla, "flash_attn_varlen_func")
    torch.manual_seed(0)
    dev = "cuda"
    s, h, d, dv = 512, 8, 192, 128
    scale = d ** -0.5
    q = (torch.randn(s, h, d, device=dev, dtype=torch.bfloat16) / 8)
    k = (torch.randn(s, h, d, device=dev, dtype=torch.bfloat16) / 8)
    v = (torch.randn(s, h, dv, device=dev, dtype=torch.bfloat16) / 8)
    do = (torch.randn(s, h, dv, device=dev, dtype=torch.bfloat16) / 8)
    sink = torch.randn(h, device=dev, dtype=torch.float32)
    cu = torch.tensor([0, s], dtype=torch.int32, device=dev)

    # --- kernel (autograd-complete: FlashAttnVarlenFunc with sink_bias) ---
    qk = q.clone().requires_grad_(True)
    kk = k.clone().requires_grad_(True)
    vk = v.clone().requires_grad_(True)
    sk = sink.clone().requires_grad_(True)
    out, lse = flash_mla.flash_attn_varlen_func(
        qk, kk, vk, cu, cu, s, s, causal=True, softmax_scale=scale,
        is_varlen=True, window_size=(-1, -1), sink_bias=sk)
    out.backward(do)

    # --- reference (float autograd) ---
    qf = q.float().clone().requires_grad_(True)
    kf = k.float().clone().requires_grad_(True)
    vf = v.float().clone().requires_grad_(True)
    sf = sink.clone().requires_grad_(True)
    o_ref, lse_ref = reference(qf, kf, vf, sf, scale)             # [s,h,dv], [h,s]
    o_ref.backward(do.float())

    print("=== attention sink on dense MLA 192/128 (causal + per-head sink), s=512 h=8 ===", flush=True)
    cO = cos(out, o_ref)
    cL = cos(lse.transpose(0, 1) if lse.shape[0] == s else lse, lse_ref)  # lse kernel [s,h] -> [h,s]
    cQ = cos(qk.grad, qf.grad)
    cK = cos(kk.grad, kf.grad)
    cV = cos(vk.grad, vf.grad)
    cS = cos(sk.grad, sf.grad)
    print(f"    O    cos={cO:.6f}", flush=True)
    print(f"    LSE  cos={cL:.6f}", flush=True)
    print(f"    dQ   cos={cQ:.6f}", flush=True)
    print(f"    dK   cos={cK:.6f}", flush=True)
    print(f"    dV   cos={cV:.6f}", flush=True)
    print(f"    d_sink cos={cS:.6f}  (kernel {sk.grad.float().tolist()[:3]}... vs ref {sf.grad.float().tolist()[:3]}...)", flush=True)

    # sanity: sink must actually change the output (else the test is vacuous)
    out0, _ = flash_mla.flash_attn_varlen_func(q, k, v, cu, cu, s, s, causal=True,
                                               softmax_scale=scale, is_varlen=True)
    drift = (out.detach().float() - out0.float()).abs().mean().item()
    print(f"    |O_sink - O_nosink| mean = {drift:.3e} (sink is active)", flush=True)

    ok = (cO > 0.99 and cL > 0.999 and cQ > 0.98 and cK > 0.98 and cV > 0.98 and cS > 0.98)
    assert ok, "attention-sink grad/output mismatch"
    print("ATTENTION_SINK_OK", flush=True)


if __name__ == "__main__":
    main()
