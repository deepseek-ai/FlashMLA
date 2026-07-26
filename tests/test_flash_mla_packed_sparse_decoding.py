"""SM90 correctness coverage for packed selected-KV sparse decode."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import flash_mla


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))
import bench_flash_mla_packed_sparse_decode as bench  # noqa: E402


CASES = (
    # Boundary tile, full tile, H=64.
    (1, 1, 64, 64, (64,)),
    # Multi-query and a partial final tile, H=64 tuned scheduler.
    (2, 3, 64, 576, (511, 449)),
    # L20X full top-k B*S_q=16 fallback to the faster native scheduler.
    (2, 8, 64, 2048, (2048, 2048)),
    # Full production top-k, H=128 native scheduler.
    (1, 4, 128, 2048, (2048,)),
    # Partial final tiles and heterogeneous requests, H=128.
    (2, 2, 128, 256, (193, 129)),
)


@torch.inference_mode()
def run_case(
    batch_size: int,
    query_len: int,
    num_heads: int,
    topk: int,
    active_lengths: tuple[int, ...],
    seed: int,
) -> None:
    q, kv, indices, _ = bench.make_case(
        batch_size=batch_size,
        query_len=query_len,
        num_heads=num_heads,
        topk=topk,
        seed=seed,
    )
    topk_length = torch.tensor(
        active_lengths,
        dtype=torch.int32,
        device=q.device,
    )
    positions = torch.arange(topk, device=q.device)
    suffix = positions.view(1, 1, topk) >= topk_length.view(
        batch_size, 1, 1
    )
    indices = indices.masked_fill(suffix, -1)
    scale = q.shape[-1] ** -0.5

    native_scheduler, _ = flash_mla.get_mla_metadata()
    native_out, native_lse = bench.run_native(
        q,
        kv,
        indices,
        native_scheduler,
        scale,
    )

    workspace_bytes = flash_mla.get_packed_kv_workspace_size(
        batch_size,
        query_len,
        topk,
    )
    storage = torch.empty(
        workspace_bytes,
        dtype=torch.uint8,
        device=q.device,
    )
    workspace = flash_mla.pack_selected_kv(
        kv,
        indices,
        topk_length,
        storage,
    )
    assert workspace.numel() == workspace_bytes
    assert workspace.data_ptr() == storage.data_ptr()
    bench.check_packed_layout(
        kv,
        indices,
        topk_length,
        workspace,
    )

    packed_scheduler = flash_mla.get_packed_mla_metadata(
        batch_size=batch_size,
        query_len=query_len,
        num_heads=num_heads,
        topk=topk,
        topk_length=active_lengths,
        device=q.device,
    )
    packed_out, packed_lse = bench.run_packed(
        q,
        workspace,
        topk_length,
        packed_scheduler,
        scale,
    )

    torch.testing.assert_close(
        packed_out,
        native_out,
        rtol=0.01,
        atol=5e-4,
    )
    torch.testing.assert_close(
        packed_lse,
        native_lse,
        rtol=1e-4,
        atol=1e-4,
    )


def main() -> None:
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("packed sparse decode tests require SM90")
    for seed, case in enumerate(CASES):
        print(
            "testing packed sparse decode "
            f"B={case[0]} S_q={case[1]} H={case[2]} "
            f"K={case[3]} active={case[4]}"
        )
        run_case(*case, seed)
    print(f"passed {len(CASES)} packed sparse-decode cases")


if __name__ == "__main__":
    main()
