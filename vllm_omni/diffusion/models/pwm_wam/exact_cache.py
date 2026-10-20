# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded exact UND cache for serialized inference workers.

Keys are supplied by the runtime and must identify every dependency of UND
execution: loaded weights, numerical/backend configuration, text and layout.
This container cannot prove key completeness. Entries hold only UND layer
outputs and raw K/V, never a request, RNG, scheduler or denoising state.

Tensor storage is shared read-only. Borrowed outer/layer lists are independent,
so request-local appends cannot corrupt another borrower. Consumers must not
mutate the tensors in-place. No tensor clone, finite-value reduction or device
synchronization is performed: commit only after the runtime's correctness and
finite-output checks succeed. The caller serializes access and commits only
fully completed inference, using inference_mode or no_grad.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import Tensor

UndLayer = tuple[Tensor, Tensor, Tensor]
UndCache = list[list[UndLayer]]
CacheKey = tuple[object, ...]
StorageKey = tuple[str, int]


@dataclass(frozen=True)
class _Entry:
    layers: tuple[UndLayer, ...]
    storages: dict[StorageKey, int]


def _check_key(key: CacheKey) -> None:
    def immutable(value):
        if isinstance(value, tuple):
            return all(immutable(item) for item in value)
        return value is None or isinstance(value, str | bytes | int | float | bool | torch.dtype | torch.device)

    if not isinstance(key, tuple) or not key or not immutable(key):
        raise ValueError("PWM exact UND cache keys must be nonempty immutable metadata tuples without tensors")


def _require_inference() -> None:
    if torch.is_grad_enabled():
        raise ValueError("PWM exact UND cache requires inference_mode or no_grad")


class PwmExactUndCache:
    def __init__(self, *, num_layers: int, max_entries: int, max_bytes: int):
        for name, value in (("num_layers", num_layers), ("max_entries", max_entries), ("max_bytes", max_bytes)):
            if type(value) is not int or value < 1:
                raise ValueError(f"PWM exact UND cache {name} must be a positive integer")
        self.num_layers = num_layers
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries: OrderedDict[CacheKey, _Entry] = OrderedDict()
        self._storage_refs: dict[StorageKey, tuple[int, int]] = {}
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._commits = 0
        self._oversized_rejections = 0

    def borrow(self, key: CacheKey) -> UndCache | None:
        _require_inference()
        _check_key(key)
        entry = self._entries.get(key)
        if entry is None:
            self._misses += 1
            return None
        self._hits += 1
        self._entries.move_to_end(key)
        return [[layer] for layer in entry.layers]

    def _validate_entry(self, cache: UndCache) -> _Entry:
        if not isinstance(cache, list) or len(cache) != self.num_layers:
            raise ValueError("PWM exact UND cache requires every transformer layer to be complete")
        layers = []
        storages = {}
        tensor_spec = None
        for layer in cache:
            if not isinstance(layer, list) or len(layer) != 1 or not isinstance(layer[0], tuple) or len(layer[0]) != 3:
                raise ValueError("PWM exact UND cache requires one (und, raw_ku, vu) tuple per layer")
            und, key, value = layer[0]
            for tensor in (und, key, value):
                if not isinstance(tensor, Tensor) or tensor.device.type == "meta":
                    raise ValueError("PWM exact UND cache requires materialized tensors")
                if tensor.requires_grad or tensor.grad_fn is not None:
                    raise ValueError("PWM exact UND cache rejects tensors attached to autograd")
                if not tensor.is_floating_point() or not tensor.numel():
                    raise ValueError("PWM exact UND cache requires nonempty floating tensors")
                spec = (tensor.device, tensor.dtype)
                if tensor_spec is None:
                    tensor_spec = spec
                elif tensor_spec != spec:
                    raise ValueError("PWM exact UND cache requires matching tensor devices and dtypes")
                storage = tensor.untyped_storage()
                storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
            if und.ndim != 2 or key.ndim != 3 or key.shape != value.shape or und.shape[0] != key.shape[0]:
                raise ValueError("PWM exact UND cache requires UND [tokens,width] and K/V [tokens,heads,dim]")
            layers.append(layer[0])
        return _Entry(tuple(layers), storages)

    def _remove(self, key: CacheKey) -> None:
        entry = self._entries.pop(key)
        for storage, size in entry.storages.items():
            _, references = self._storage_refs[storage]
            if references == 1:
                del self._storage_refs[storage]
                self._bytes -= size
            else:
                self._storage_refs[storage] = (size, references - 1)

    def commit(self, key: CacheKey, cache: UndCache) -> bool:
        """Store a complete entry; oversized entries return False without eviction.

        Byte accounting counts complete underlying allocations, including views,
        and deduplicates storage both within and across entries. Outstanding
        borrowers may keep an evicted allocation alive; those request-owned
        references are outside this cache's resident-byte budget.
        """
        _require_inference()
        _check_key(key)
        entry = self._validate_entry(cache)
        if sum(entry.storages.values()) > self.max_bytes:
            self._oversized_rejections += 1
            return False
        if key in self._entries:
            self._remove(key)
        self._entries[key] = entry
        for storage, size in entry.storages.items():
            _, references = self._storage_refs.get(storage, (size, 0))
            if not references:
                self._bytes += size
            self._storage_refs[storage] = (size, references + 1)
        while len(self._entries) > self.max_entries or self._bytes > self.max_bytes:
            self._remove(next(iter(self._entries)))
            self._evictions += 1
        self._commits += 1
        return True

    def clear(self, *, reset_stats: bool = False) -> None:
        """Release cached references; cumulative counters survive unless reset."""
        self._entries.clear()
        self._storage_refs.clear()
        self._bytes = 0
        if reset_stats:
            self._hits = self._misses = self._evictions = self._commits = self._oversized_rejections = 0

    def stats(self) -> dict[str, int]:
        return dict(
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            commits=self._commits,
            oversized_rejections=self._oversized_rejections,
            entries=len(self._entries),
            bytes=self._bytes,
            max_entries=self.max_entries,
            max_bytes=self.max_bytes,
        )
