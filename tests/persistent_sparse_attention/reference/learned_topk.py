"""Pure PyTorch reference for learned TopK with canonical cutoff ties."""

import torch


TOPK = 1024


def learned_topk_reference(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk: int = TOPK,
) -> torch.Tensor:
    """Select score-descending indices, resolving equal scores by index."""
    if logits.ndim != 2 or logits.dtype != torch.float32:
        raise ValueError("logits must be float32 [rows, columns]")
    if (
        row_starts.ndim != 1
        or row_ends.ndim != 1
        or row_starts.dtype != torch.int32
        or row_ends.dtype != torch.int32
    ):
        raise ValueError("row bounds must be one-dimensional int32 tensors")
    if row_starts.numel() != logits.shape[0] or row_ends.numel() != logits.shape[0]:
        raise ValueError("row count mismatch")
    if topk <= 0:
        raise ValueError("topk must be positive")

    output = torch.full(
        (logits.shape[0], topk),
        -1,
        dtype=torch.int32,
        device=logits.device,
    )
    for row in range(logits.shape[0]):
        start = int(row_starts[row])
        end = int(row_ends[row])
        if start < 0 or end < start or end > logits.shape[1]:
            raise ValueError("row bounds are outside the logits surface")
        length = end - start
        if length <= topk:
            output[row, :length] = torch.arange(
                length, dtype=torch.int32, device=logits.device
            )
            continue
        scores = logits[row, start:end]
        # Stable descending sort retains original relative-index order for ties.
        order = torch.argsort(scores, descending=True, stable=True)
        output[row] = order[:topk].to(torch.int32)
    return output
