"""Torch reference for block-sparse MLA attention (fwd + autograd bwd).

The validation oracle for the block-sparse KV-outer bwd kernel (Phase 2 of
docs/MSA_BLOCK_SPARSE_KV_OUTER_BWD_DESIGN.md). Defines the contract:

  q   [s_q, h_q, d_qk]            (MLA; rope packed in d_qk)
  kv  [s_kv, 1,  d_qk]            (V == kv[:, 0, :d_v])
  q2k_blocks [num_q_blocks, topk] int32  KV-block ids per Q-block (-1 = pad)
  q_block_size, kv_block_size     block granularities (Q-block x K-block sparsity)

Query qi (Q-block = qi // q_block_size) attends key kj (K-block = kj // kv_block_size)
iff that K-block id is in q2k_blocks[Q-block] AND (causal) kj <= qi + (s_kv - s_q).
dQ/dK/dV come from autograd over the masked-softmax forward = ground truth.
"""
import torch


def build_block_mask(s_q, s_kv, q2k_blocks, q_block_size, kv_block_size, causal, device):
    num_q_blocks, topk = q2k_blocks.shape
    num_kv_blocks = (s_kv + kv_block_size - 1) // kv_block_size
    # per-(q-block, k-block) membership from the selection
    sel = torch.zeros(num_q_blocks, num_kv_blocks, dtype=torch.bool, device=device)
    qb = torch.arange(num_q_blocks, device=device).view(-1, 1)
    blk = q2k_blocks.to(device)
    valid = (blk >= 0) & (blk < num_kv_blocks)
    sel[qb.expand_as(blk)[valid], blk[valid]] = True
    # expand to token granularity
    qtok_blk = (torch.arange(s_q, device=device) // q_block_size)        # [s_q]
    ktok_blk = (torch.arange(s_kv, device=device) // kv_block_size)      # [s_kv]
    mask = sel[qtok_blk][:, ktok_blk]                                    # [s_q, s_kv]
    if causal:
        offset = s_kv - s_q
        qi = torch.arange(s_q, device=device).view(-1, 1)
        kj = torch.arange(s_kv, device=device).view(1, -1)
        mask = mask & (kj <= qi + offset)
    return mask  # [s_q, s_kv] bool


def block_sparse_mla_fwd(q, kv, q2k_blocks, q_block_size, kv_block_size, d_v, sm_scale, causal=True):
    s_q, h_q, d_qk = q.shape
    s_kv = kv.shape[0]
    k = kv[:, 0, :]                       # [s_kv, d_qk]
    v = kv[:, 0, :d_v].float()            # [s_kv, d_v]
    scores = torch.einsum("qhd,kd->hqk", q.float(), k.float()) * sm_scale   # [h_q, s_q, s_kv]
    mask = build_block_mask(s_q, s_kv, q2k_blocks, q_block_size, kv_block_size, causal, q.device)
    scores = scores.masked_fill(~mask.view(1, s_q, s_kv), float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)                 # [h_q, s_q]
    p = torch.softmax(scores, dim=-1)                     # [h_q, s_q, s_kv]
    # rows with no valid key -> softmax gives nan; zero them (lonely query)
    p = torch.nan_to_num(p, nan=0.0)
    o = torch.einsum("hqk,kv->qhv", p, v)                 # [s_q, h_q, d_v]
    return o, lse


if __name__ == "__main__":
    # sanity: full selection (every Q-block selects all K-blocks) == dense causal MLA
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    s_q = s_kv = 512
    h_q, d_qk, d_v = 8, 192, 128
    qbs, kbs = 64, 128
    nqb = (s_q + qbs - 1) // qbs
    nkb = (s_kv + kbs - 1) // kbs
    q = (torch.randn(s_q, h_q, d_qk, device=dev) / 10)
    kv = (torch.randn(s_kv, 1, d_qk, device=dev) / 10)
    sm = d_qk ** -0.5
    full = torch.arange(nkb, device=dev).view(1, nkb).expand(nqb, nkb).contiguous().int()
    o_full, lse_full = block_sparse_mla_fwd(q, kv, full, qbs, kbs, d_v, sm, causal=True)
    # dense causal reference
    k = kv[:, 0, :]; v = kv[:, 0, :d_v].float()
    sc = torch.einsum("qhd,kd->hqk", q.float(), k.float()) * sm
    qi = torch.arange(s_q, device=dev).view(-1, 1); kj = torch.arange(s_kv, device=dev).view(1, -1)
    sc = sc.masked_fill(~(kj <= qi).view(1, s_q, s_kv), float("-inf"))
    o_dense = torch.einsum("hqk,kv->qhv", torch.softmax(sc, -1), v)
    err = (o_full - o_dense).abs().max().item()
    print(f"full-selection vs dense causal: max_abs_err={err:.2e}")
    assert err < 1e-4, "block-sparse ref (full selection) must equal dense causal"
    print("BLOCK_SPARSE_MLA_REF_OK", flush=True)
