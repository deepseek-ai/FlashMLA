#!/usr/bin/env python3
"""Correctness and latency matrix for SM90 packed sparse decode."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch

import flash_mla


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import quant  # noqa: E402


RECORD_BYTES = 656
TILE_SIZE = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=(1, 2, 4, 7, 8),
    )
    parser.add_argument(
        "--query-lengths",
        type=int,
        nargs="+",
        default=(1, 2, 3, 4, 8, 16),
    )
    parser.add_argument(
        "--head-counts",
        type=int,
        nargs="+",
        default=(64, 128),
    )
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--flush-mib", type=int, default=512)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("packed_sparse_decode_latency.csv"),
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def clone_scheduler(
    source: flash_mla.FlashMLASchedMeta,
) -> flash_mla.FlashMLASchedMeta:
    result = flash_mla.FlashMLASchedMeta()
    # Intentionally share the exact tensor objects with the native path.
    result.tile_scheduler_metadata = source.tile_scheduler_metadata
    result.num_splits = source.num_splits
    return result


def make_case(
    *,
    batch_size: int,
    query_len: int,
    num_heads: int,
    topk: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    tokens_per_request = max(topk * 2, 4096)
    num_source_tokens = batch_size * tokens_per_request
    num_blocks = (num_source_tokens + TILE_SIZE - 1) // TILE_SIZE

    source = torch.randn(
        (num_blocks, TILE_SIZE, 1, 576),
        dtype=torch.bfloat16,
        device="cuda",
    ).clamp_(-1.0, 1.0)
    kv = quant.quantize_k_cache(
        source,
        quant.FP8KVCacheLayout.V32_FP8Sparse,
    )
    q = torch.randn(
        (batch_size, query_len, num_heads, 576),
        dtype=torch.bfloat16,
        device="cuda",
    ).clamp_(-1.0, 1.0)

    rows = []
    for batch_idx in range(batch_size):
        scores = torch.rand(
            (query_len, tokens_per_request),
            generator=generator,
        )
        local = scores.topk(topk, dim=-1, sorted=False).indices
        rows.append(local + batch_idx * tokens_per_request)
    indices = torch.stack(rows).to(device="cuda", dtype=torch.int32)
    topk_length = torch.full(
        (batch_size,),
        topk,
        dtype=torch.int32,
        device="cuda",
    )
    return q, kv, indices.contiguous(), topk_length


def check_packed_layout(
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    packed: torch.Tensor,
) -> None:
    batch_size, query_len, topk = indices.shape
    pages = batch_size * query_len * topk // TILE_SIZE
    unpacked = (
        packed.reshape(pages, RECORD_BYTES // 16, TILE_SIZE, 16)
        .permute(0, 2, 1, 3)
        .reshape(batch_size * query_len, topk, RECORD_BYTES)
    )
    source_records = (
        kv.contiguous()
        .view(torch.uint8)
        .reshape(-1, RECORD_BYTES)
    )
    expected = source_records.index_select(
        0,
        indices.clamp_min(0).reshape(-1).long(),
    ).reshape_as(unpacked)
    positions = torch.arange(topk, device=indices.device)
    invalid = positions.view(1, -1) >= (
        topk_length.repeat_interleave(query_len).view(-1, 1)
    )
    expected = expected.masked_fill(invalid.unsqueeze(-1), 0)
    torch.testing.assert_close(unpacked, expected, rtol=0, atol=0)


def run_native(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scheduler: flash_mla.FlashMLASchedMeta,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return flash_mla.flash_mla_with_kvcache(
        q,
        kv,
        None,
        None,
        512,
        scheduler,
        None,
        scale,
        False,
        True,
        indices,
    )


def run_packed(
    q: torch.Tensor,
    packed: torch.Tensor,
    topk_length: torch.Tensor,
    scheduler: flash_mla.FlashMLASchedMeta,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return flash_mla.flash_mla_with_packed_kvcache(
        q,
        packed,
        512,
        scheduler,
        scale,
        topk_length,
    )


def time_cuda(
    fn: Callable[[], object],
    *,
    warmup: int,
    repeats: int,
    before: Callable[[], object] | None = None,
) -> float:
    for _ in range(warmup):
        if before is not None:
            before()
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        if before is not None:
            before()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(samples)


def benchmark_case(
    *,
    batch_size: int,
    query_len: int,
    num_heads: int,
    topk: int,
    warmup: int,
    repeats: int,
    flush: torch.Tensor | None,
    seed: int,
) -> dict[str, float | int | str]:
    q, kv, indices, topk_length = make_case(
        batch_size=batch_size,
        query_len=query_len,
        num_heads=num_heads,
        topk=topk,
        seed=seed,
    )
    scale = q.shape[-1] ** -0.5

    native_scheduler, _ = flash_mla.get_mla_metadata()
    native_out, native_lse = run_native(
        q,
        kv,
        indices,
        native_scheduler,
        scale,
    )
    same_scheduler = clone_scheduler(native_scheduler)
    if num_heads == 64:
        integrated_scheduler = flash_mla.get_packed_mla_metadata(
            batch_size=batch_size,
            query_len=query_len,
            num_heads=num_heads,
            topk=topk,
            topk_length=topk,
            device=q.device,
        )
        scheduler_policy = (
            "native-fallback"
            if integrated_scheduler.tile_scheduler_metadata is None
            else "l20x-h64-tuned"
        )
    else:
        integrated_scheduler = same_scheduler
        scheduler_policy = "native"

    workspace = flash_mla.pack_selected_kv(
        kv,
        indices,
        topk_length,
    )
    check_packed_layout(kv, indices, topk_length, workspace)
    packed_out, packed_lse = run_packed(
        q,
        workspace,
        topk_length,
        integrated_scheduler,
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

    def flush_l2() -> None:
        if flush is not None:
            flush.add_(1)

    def pack() -> torch.Tensor:
        return flash_mla.pack_selected_kv(
            kv,
            indices,
            topk_length,
            workspace,
        )

    native_us = time_cuda(
        lambda: run_native(
            q, kv, indices, native_scheduler, scale
        ),
        warmup=warmup,
        repeats=repeats,
        before=flush_l2,
    )
    packing_us = time_cuda(
        pack,
        warmup=warmup,
        repeats=repeats,
        before=flush_l2,
    )
    packed_us = time_cuda(
        lambda: run_packed(
            q,
            workspace,
            topk_length,
            integrated_scheduler,
            scale,
        ),
        warmup=warmup,
        repeats=repeats,
        before=lambda: (flush_l2(), pack()),
    )

    if num_heads == 64:
        same_scheduler_us = time_cuda(
            lambda: run_packed(
                q,
                workspace,
                topk_length,
                same_scheduler,
                scale,
            ),
            warmup=warmup,
            repeats=repeats,
            before=lambda: (flush_l2(), pack()),
        )
    else:
        same_scheduler_us = packed_us

    return {
        "batch_size": batch_size,
        "query_len": query_len,
        "num_heads": num_heads,
        "topk": topk,
        "scheduler_policy": scheduler_policy,
        "native_attention_us": native_us,
        "packing_us": packing_us,
        "packed_attention_us": packed_us,
        "same_scheduler_packed_attention_us": same_scheduler_us,
        "unhidden_packed_total_us": packing_us + packed_us,
        "attention_speedup": native_us / packed_us,
        "same_scheduler_attention_speedup":
            native_us / same_scheduler_us,
        "unhidden_speedup": native_us / (packing_us + packed_us),
    }


def main() -> None:
    args = parse_args()
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("packed sparse decode benchmark requires SM90")
    if args.topk < 1 or args.topk % TILE_SIZE:
        raise ValueError("--topk must be positive and divisible by 64")
    if args.warmup < 1 or args.repeats < 1:
        raise ValueError("--warmup and --repeats must be positive")

    flush = (
        torch.empty(
            args.flush_mib * 1024 * 1024,
            dtype=torch.uint8,
            device="cuda",
        )
        if args.flush_mib
        else None
    )
    rows = []
    axes = itertools.product(
        args.batch_sizes,
        args.query_lengths,
        args.head_counts,
    )
    for case_idx, (batch_size, query_len, num_heads) in enumerate(axes):
        row = benchmark_case(
            batch_size=batch_size,
            query_len=query_len,
            num_heads=num_heads,
            topk=args.topk,
            warmup=args.warmup,
            repeats=args.repeats,
            flush=flush,
            seed=args.seed + case_idx,
        )
        rows.append(row)
        print(
            f"B={batch_size} S_q={query_len} H={num_heads}: "
            f"native={row['native_attention_us']:.3f}us "
            f"pack={row['packing_us']:.3f}us "
            f"packed={row['packed_attention_us']:.3f}us "
            f"speedup={row['attention_speedup']:.3f}x"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for metric in (
        "attention_speedup",
        "same_scheduler_attention_speedup",
        "unhidden_speedup",
    ):
        values = [float(row[metric]) for row in rows]
        geomean = math.exp(
            sum(math.log(value) for value in values) / len(values)
        )
        wins = sum(value > 1.0 for value in values)
        print(f"{metric}: {geomean:.3f}x, wins={wins}/{len(values)}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
