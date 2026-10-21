"""Ascend sparse prefill must preserve both BF16 lanes and RN-even rounding."""
import pytest
import torch


@pytest.fixture
def sparse_fwd():
    pytest.importorskip('torch_npu')
    if not torch.npu.is_available():
        pytest.skip('requires an Ascend NPU and an in-place FlashMLA build')
    from flash_mla import flash_mla_sparse_fwd
    return flash_mla_sparse_fwd


@pytest.mark.parametrize('topk', [128, 256])
def test_paired_bf16_lanes_and_ties_to_even(sparse_fwd, topk):
    # A uniform softmax makes output the exact midpoint of two adjacent BF16
    # values. Adjacent columns alternate between even and odd mantissa ties.
    lower = torch.tensor([1.0, 1.0078125, -1.0, -1.0078125], dtype=torch.float32).repeat(128)
    upper = torch.tensor([1.0078125, 1.015625, -1.0078125, -1.015625], dtype=torch.float32).repeat(128)
    kv = torch.stack([lower, upper]).repeat(topk // 2, 1).unsqueeze(1).to(torch.bfloat16)
    q = torch.zeros((17, 64, 512), dtype=torch.bfloat16)
    indices = torch.arange(topk, dtype=torch.int32).view(1, 1, -1).expand(17, 1, -1).contiguous()
    output, _, _ = sparse_fwd(q.npu(), kv.npu(), indices.npu(), sm_scale=0.5)
    torch.npu.synchronize()
    expected = ((lower + upper) / 2).to(torch.bfloat16).view(1, 1, 512).expand_as(q)
    assert output.dtype == torch.bfloat16
    assert torch.equal(output.cpu(), expected)


@pytest.mark.parametrize('query_length', [1, 17, 65])
@pytest.mark.parametrize('variable_topk', [False, True])
def test_sparse_prefill_matches_cpu_reference(sparse_fwd, query_length, variable_topk):
    generator = torch.Generator().manual_seed(42)
    q = (torch.randn((query_length, 64, 512), generator=generator) * 0.1).to(torch.bfloat16)
    kv = (torch.randn((256, 1, 512), generator=generator) * 0.1).to(torch.bfloat16)
    indices = torch.arange(128, dtype=torch.int32).view(1, 1, -1).expand(query_length, 1, -1).contiguous()
    lengths = 1 + torch.arange(query_length, dtype=torch.int32) % 128 if variable_topk else None
    scores = q.float() @ kv[:128, 0].float().T * 0.5
    if lengths is not None:
        scores.masked_fill_(torch.arange(128).view(1, 1, -1) >= lengths.view(-1, 1, 1), -torch.inf)
    expected = scores.softmax(-1) @ kv[:128, 0].float()
    output, maximum, lse = sparse_fwd(q.npu(), kv.npu(), indices.npu(), sm_scale=0.5,
                                    topk_length=lengths.npu() if lengths is not None else None)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu().float(), expected, atol=8e-4, rtol=3.01 / 128)
    torch.testing.assert_close(maximum.cpu(), scores.amax(-1), atol=1e-6, rtol=2.01 / 65536)
    torch.testing.assert_close(lse.cpu(), scores.logsumexp(-1), atol=1e-5, rtol=2.01 / 65536)
