# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for exact UND cache ownership, bounds and inference contracts."""

import pytest
import torch

from vllm_omni.diffusion.models.pwm_wam.exact_cache import PwmExactUndCache

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


def _entry(layers=2, tokens=3):
    return [[(torch.randn(tokens, 4), torch.randn(tokens, 1, 2), torch.randn(tokens, 1, 2))] for _ in range(layers)]


def _bytes(entry):
    storage = {
        value.untyped_storage().data_ptr(): value.untyped_storage().nbytes() for layer in entry for value in layer[0]
    }
    return sum(storage.values())


def _cache(*, max_entries=2, max_bytes=1024):
    return PwmExactUndCache(num_layers=2, max_entries=max_entries, max_bytes=max_bytes)


@torch.inference_mode()
def test_borrow_has_independent_list_structure_and_shared_readonly_tensors():
    cache = _cache()
    entry = _entry()
    assert cache.borrow(("model", "text")) is None
    assert cache.commit(("model", "text"), entry)
    first = cache.borrow(("model", "text"))
    second = cache.borrow(("model", "text"))
    assert first is not second and first is not entry
    for original, left, right in zip(entry, first, second, strict=True):
        assert left is not right and left is not original
        assert all(a is b is c for a, b, c in zip(original[0], left[0], right[0], strict=True))
    first[0].clear()
    first[1].append(first[1][0])
    entry.clear()
    third = cache.borrow(("model", "text"))
    assert len(third) == 2 and all(len(layer) == 1 for layer in third)
    assert all(len(layer) == 1 for layer in second)
    assert cache.stats()["hits"] == 3 and cache.stats()["misses"] == 1


@torch.inference_mode()
def test_lru_hits_update_recency_and_entry_limit_evicts_oldest():
    cache = _cache(max_entries=2)
    entries = [_entry() for _ in range(3)]
    for key, entry in zip(("a", "b"), entries, strict=False):
        cache.commit((key,), entry)
    cache.borrow(("a",))
    cache.commit(("c",), entries[2])
    assert cache.borrow(("b",)) is None
    assert cache.borrow(("a",)) is not None
    assert cache.borrow(("c",)) is not None
    assert cache.stats()["entries"] == 2
    assert cache.stats()["evictions"] == 1


@torch.inference_mode()
def test_byte_budget_evicts_even_when_entry_count_allows_more():
    first, second = _entry(), _entry()
    cache = _cache(max_entries=8, max_bytes=_bytes(first))
    cache.commit(("first",), first)
    cache.commit(("second",), second)
    assert cache.borrow(("first",)) is None
    assert cache.borrow(("second",)) is not None
    assert cache.stats()["bytes"] == _bytes(second)
    assert cache.stats()["evictions"] == 1


@torch.inference_mode()
def test_oversized_entry_does_not_evict_or_replace_existing_key():
    first = _entry()
    cache = _cache(max_bytes=_bytes(first))
    cache.commit(("fixed",), first)
    assert not cache.commit(("fixed",), _entry(tokens=6))
    borrowed = cache.borrow(("fixed",))
    assert borrowed[0][0][0] is first[0][0][0]
    assert cache.stats()["oversized_rejections"] == 1
    assert cache.stats()["evictions"] == 0
    assert cache.stats()["bytes"] == _bytes(first)


@torch.inference_mode()
def test_storage_accounting_includes_full_view_allocation_and_shared_entries():
    backing = torch.randn(100)
    triple = (backing[:12].reshape(3, 4), backing[12:18].reshape(3, 1, 2), backing[18:24].reshape(3, 1, 2))
    entry = [[triple], [triple]]
    cache = _cache(max_entries=2, max_bytes=backing.untyped_storage().nbytes())
    cache.commit(("a",), entry)
    cache.commit(("b",), entry)
    assert cache.stats()["entries"] == 2
    assert cache.stats()["bytes"] == backing.untyped_storage().nbytes()
    # Removing one alias must not subtract a storage still owned by another.
    replacement = _entry(tokens=1)
    cache.commit(("a",), replacement)
    assert cache.borrow(("b",)) is None  # full backing + replacement exceeds budget
    assert cache.stats()["bytes"] == _bytes(replacement)
    assert cache.stats()["evictions"] == 1


@torch.inference_mode()
def test_key_isolation_includes_model_text_layout_and_numeric_config():
    cache = _cache(max_entries=8, max_bytes=4096)
    base = ("weights-sha", (1, 2, 3), (4, 8), "bf16", "flash3")
    cache.commit(base, _entry())
    for other in (
        ("other-sha", *base[1:]),
        (base[0], (1, 2, 4), *base[2:]),
        (*base[:2], (8, 4), *base[3:]),
        (*base[:3], "fp32", base[4]),
        (*base[:4], "sdpa"),
    ):
        assert cache.borrow(other) is None
    assert cache.borrow(base) is not None


@pytest.mark.parametrize("invalid", [[], [[]], [[], []], [[()], [()]]])
@torch.inference_mode()
def test_partial_or_malformed_cache_is_rejected_without_mutation(invalid):
    cache = _cache()
    cache.commit(("good",), _entry())
    original_stats = cache.stats()
    with pytest.raises(ValueError, match="requires"):
        cache.commit(("partial",), invalid)
    assert cache.stats() == original_stats
    assert cache.borrow(("good",)) is not None


def test_grad_enabled_access_and_autograd_tensors_are_rejected():
    cache = _cache()
    with torch.inference_mode():
        cache.commit(("good",), _entry())
    with pytest.raises(ValueError, match="inference_mode or no_grad"):
        cache.borrow(("good",))
    entry = _entry()
    entry[0] = [(torch.randn(3, 4, requires_grad=True), entry[0][0][1], entry[0][0][2])]
    with torch.no_grad(), pytest.raises(ValueError, match="autograd"):
        cache.commit(("grad",), entry)


@torch.inference_mode()
def test_finiteness_and_device_synchronization_are_not_hidden_in_commit(monkeypatch):
    cache = _cache()
    entry = _entry()

    def forbidden(*args, **kwargs):
        pytest.fail("Container must not add device reductions, tensor clones or synchronization")

    monkeypatch.setattr(torch, "isfinite", forbidden)
    monkeypatch.setattr(torch.Tensor, "clone", forbidden)
    monkeypatch.setattr(torch.accelerator, "synchronize", forbidden)
    assert cache.commit(("checked-by-runtime",), entry)
    assert cache.borrow(("checked-by-runtime",)) is not None


@torch.inference_mode()
def test_clear_releases_entries_and_optionally_resets_counters():
    cache = _cache()
    cache.commit(("a",), _entry())
    cache.borrow(("a",))
    cache.clear()
    assert cache.stats()["entries"] == cache.stats()["bytes"] == 0
    assert cache.stats()["hits"] == 1
    cache.clear(reset_stats=True)
    assert cache.stats()["hits"] == cache.stats()["commits"] == 0


@pytest.mark.parametrize("key", [(), "text", ([],), ({"request": 1},), (torch.ones(1),)])
@torch.inference_mode()
def test_keys_reject_mutable_metadata_and_tensors(key):
    with pytest.raises(ValueError, match="immutable metadata tuples"):
        _cache().borrow(key)


@pytest.mark.parametrize("name,value", [("num_layers", 0), ("max_entries", False), ("max_bytes", -1)])
def test_budget_configuration_requires_positive_integers(name, value):
    kwargs = dict(num_layers=2, max_entries=4, max_bytes=4096)
    kwargs[name] = value
    with pytest.raises(ValueError, match="positive integer"):
        PwmExactUndCache(**kwargs)
