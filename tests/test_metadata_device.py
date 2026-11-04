import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


@pytest.fixture
def interface(monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'flash_mla/flash_mla_interface.py'
    spec = importlib.util.spec_from_file_location('flash_mla_interface_device_test', path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def inputs(device):
    return (torch.empty((1, 1, 64, 512), device=device),
            torch.empty((1, 1, 1, 528), device=device),
            torch.empty((1, 1, 64), dtype=torch.int32, device=device))


def test_cached_metadata_stays_on_query_device(interface, monkeypatch):
    calls = []

    def backend(q, kv, indices, topk_length, sink, metadata, splits, *args):
        calls.append((q.device, None if metadata is None else metadata.device))
        return q, q, torch.empty(1, device=q.device), torch.empty(1, device=q.device)

    monkeypatch.setattr(interface, '_backend', types.SimpleNamespace(sparse_decode_fwd=backend))
    metadata, _ = interface.get_mla_metadata()
    q, kv, indices = inputs('cpu')
    interface.flash_mla_with_kvcache(q, kv, None, None, 512, metadata, indices=indices)
    interface.flash_mla_with_kvcache(q, kv, None, None, 512, metadata, indices=indices)
    assert calls == [(torch.device('cpu'), None), (torch.device('cpu'), torch.device('cpu'))]

    q, kv, indices = inputs('meta')
    with pytest.raises(ValueError, match='separate FlashMLASchedMeta'):
        interface.flash_mla_with_kvcache(q, kv, None, None, 512, metadata, indices=indices)
    assert len(calls) == 2


def test_split_count_device_is_checked(interface, monkeypatch):
    def backend(q, kv, indices, topk_length, sink, metadata, splits, *args):
        return q, q, torch.empty(1, device=q.device), torch.empty(1, device=q.device)

    monkeypatch.setattr(interface, '_backend', types.SimpleNamespace(sparse_decode_fwd=backend))
    metadata, _ = interface.get_mla_metadata()
    q, kv, indices = inputs('cpu')
    interface.flash_mla_with_kvcache(q, kv, None, None, 512, metadata, indices=indices)
    metadata.num_splits = torch.empty(1, device='meta')
    with pytest.raises(ValueError, match='Cached num_splits'):
        interface.flash_mla_with_kvcache(q, kv, None, None, 512, metadata, indices=indices)


def test_no_cached_tensors_can_be_reused(interface, monkeypatch):
    calls = []

    def backend(q, kv, indices, topk_length, sink, metadata, splits, *args):
        calls.append((metadata, splits))
        return q, q, None, None

    monkeypatch.setattr(interface, '_backend', types.SimpleNamespace(sparse_decode_fwd=backend))
    metadata, _ = interface.get_mla_metadata()
    for device in ('cpu', 'meta'):
        q, kv, indices = inputs(device)
        interface.flash_mla_with_kvcache(q, kv, None, None, 512, metadata,
                                        indices=indices, enable_batch_invariant=True)
    assert calls == [(None, None), (None, None)]
