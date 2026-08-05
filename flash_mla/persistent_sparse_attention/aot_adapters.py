"""Python wrappers for build-time-instantiated persistent pipeline adapters."""

from __future__ import annotations

from typing import Any

from .ops import _extension


def persistent_attention_frontend(*args: Any, **kwargs: Any) -> Any:
    """Run the P0/P1/F1'/P4 persistent fused frontend."""
    return _extension()._persistent_attention_frontend(*args, **kwargs)


def index_logits_out(*args: Any, **kwargs: Any) -> Any:
    """Run the caller-owned sparse-index logits adapter."""
    return _extension()._index_logits_out(*args, **kwargs)


def wqb_ready_out(*args: Any, **kwargs: Any) -> Any:
    """Run WqB and release-publish output chunks as they complete."""
    return _extension()._wqb_ready_out(*args, **kwargs)


def wqb_ready_packed_out(*args: Any, **kwargs: Any) -> Any:
    """Run the packed-scale WqB ready producer."""
    return _extension()._wqb_ready_packed_out(*args, **kwargs)


__all__ = [
    "index_logits_out",
    "persistent_attention_frontend",
    "wqb_ready_out",
    "wqb_ready_packed_out",
]
