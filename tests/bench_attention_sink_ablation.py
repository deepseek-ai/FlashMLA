"""Ablation: attention-sink overhead (w/ vs w/o) on the SWA paths -- FWD + BWD (training,
dense MLA 192/128 prefill via flash_attn_varlen_func) and INFERENCE (dense MLA decode via
flash_mla_with_kvcache). The sink is an exact O/LSE renormalization, so the expected
overhead is tiny; this quantifies it (and shows in-kernel fusion is unnecessary). Run
after a full build on B200."""
import torch
import triton
import flash_mla

DEV = "cuda"


def bench_train(s, h, do_bwd, with_sink):
    d, dv = 192, 128
    scale = d ** -0.5
    q = (torch.randn(s, h, d, device=DEV, dtype=torch.bfloat16) / 8).requires_grad_(do_bwd)
    k = (torch.randn(s, h, d, device=DEV, dtype=torch.bfloat16) / 8).requires_grad_(do_bwd)
    v = (torch.randn(s, h, dv, device=DEV, dtype=torch.bfloat16) / 8).requires_grad_(do_bwd)
    do = torch.randn(s, h, dv, device=DEV, dtype=torch.bfloat16) / 8
    cu = torch.tensor([0, s], dtype=torch.int32, device=DEV)
    sink = torch.randn(h, device=DEV, dtype=torch.float32).requires_grad_(do_bwd) if with_sink else None

    def fwd():
        return flash_mla.flash_attn_varlen_func(q, k, v, cu, cu, s, s, causal=True,
                                                softmax_scale=scale, is_varlen=True, sink_bias=sink)

    if not do_bwd:
        fn = lambda: fwd()[0]
    else:
        def fn():
            o, _ = fwd()
            o.backward(do, retain_graph=False)
            for t in (q, k, v):
                t.grad = None
            if sink is not None:
                sink.grad = None
    t = triton.testing.do_bench(fn, warmup=5, rep=20)
    del q, k, v, do; torch.cuda.empty_cache()
    return t


def bench_decode(b, s_k, h_q, with_sink, d=576, dv=512, block_size=64):
    scale = d ** -0.5
    cache_seqlens = torch.full((b,), s_k, dtype=torch.int32, device=DEV)
    max_pad = ((s_k + block_size - 1) // block_size) * block_size
    q = torch.randn(b, 1, h_q, d, device=DEV, dtype=torch.bfloat16) / 10
    n_per = max_pad // block_size
    block_table = torch.arange(b * n_per, dtype=torch.int32, device=DEV).view(b, n_per)
    blocked_k = torch.randn(b * n_per, block_size, 1, d, device=DEV, dtype=torch.bfloat16) / 10
    meta, num_splits = flash_mla.get_mla_metadata()
    sink = torch.randn(h_q, device=DEV, dtype=torch.float32) if with_sink else None

    def fn():
        return flash_mla.flash_mla_with_kvcache(
            q, blocked_k, block_table, cache_seqlens, dv, meta, num_splits,
            softmax_scale=scale, causal=True, attn_sink=sink)
    t = triton.testing.do_bench(fn, warmup=5, rep=20)
    del q, blocked_k, block_table, cache_seqlens; torch.cuda.empty_cache()
    return t


def row(name, t0, t1):
    ov = (t1 / t0 - 1.0) * 100.0 if t0 > 0 else float("nan")
    print(f"  {name:>22}  w/o={t0:8.3f}ms  w/={t1:8.3f}ms  overhead={ov:+6.2f}%", flush=True)


if __name__ == "__main__":
    print("# Attention-sink ablation (w/ vs w/o), B200. sink = gpt-oss per-head scalar, "
          "CUTLASS in-kernel fwd + CUDA d_sink (no torch).", flush=True)
    print("## TRAINING dense MLA 192/128 prefill (flash_attn_varlen_func), h=8, causal", flush=True)
    for s in [8192, 16384, 32768, 65536]:
        try:
            row(f"FWD s={s}", bench_train(s, 8, False, False), bench_train(s, 8, False, True))
        except Exception as e:
            print(f"  FWD s={s} FAIL: {repr(e)[:60]}", flush=True)
    for s in [8192, 16384, 32768, 65536]:
        try:
            row(f"FWD+BWD s={s}", bench_train(s, 8, True, False), bench_train(s, 8, True, True))
        except Exception as e:
            print(f"  BWD s={s} FAIL: {repr(e)[:60]}", flush=True)
    print("## INFERENCE dense MLA decode (flash_mla_with_kvcache), b=64 h_q=128 s_q=1, causal", flush=True)
    for s_k in [8192, 32768, 131072]:
        try:
            row(f"DECODE s_k={s_k}", bench_decode(64, s_k, 128, False), bench_decode(64, s_k, 128, True))
        except Exception as e:
            print(f"  DECODE s_k={s_k} FAIL: {repr(e)[:60]}", flush=True)
    print("ATTENTION_SINK_ABLATION_DONE", flush=True)
