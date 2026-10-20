# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded CUDA Graphs for the native PWM ``forward_hidden`` computation.

The first UND-cache fill runs normally. Subsequent calls capture only the
read-only-cache transformer, leaving RNG, embedding, UniPC and decoding outside
capture. Each entry owns its buffers and graph memory pool. Weights and layer
configuration must remain fixed; clear the runner before changing either.

``end_request`` invalidates all constant-source identities while retaining the
bounded graph allocations. The next request always copies its own RoPE,
metadata and UND values before replay: retained storage is never a semantic
cross-request cache. ``clear`` additionally releases captured allocations.
Only serialized inference is supported. Outputs are independent tensors, so a
later conditional/unconditional replay cannot overwrite a live velocity.
"""

from __future__ import annotations

import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .transformer_pwm_wam import PwmWamTransformer, Rope, Streams

TensorTree: TypeAlias = Tensor | tuple["TensorTree", ...] | list["TensorTree"] | None
Cache: TypeAlias = list[list[tuple[Tensor, Tensor, Tensor]]] | None
Source: TypeAlias = tuple[weakref.ReferenceType[Tensor], int | None]


def _leaves(tree: TensorTree) -> list[Tensor]:
    if isinstance(tree, Tensor):
        return [tree]
    if tree is None:
        return []
    return [tensor for item in tree for tensor in _leaves(item)]


def _signature(tree: TensorTree) -> tuple:
    if isinstance(tree, Tensor):
        return (
            "tensor",
            tuple(tree.shape),
            tuple(tree.stride()),
            tree.storage_offset(),
            tree.dtype,
            tree.device,
            tree.layout,
        )
    if tree is None:
        return ("none",)
    return (type(tree).__name__, tuple(_signature(item) for item in tree))


def _buffers(tree: TensorTree) -> TensorTree:
    if isinstance(tree, Tensor):
        # Stable, writable storage; callers may supply strided views.
        with torch.inference_mode(False):
            result = torch.empty_like(tree, memory_format=torch.contiguous_format)
        result.copy_(tree)
        return result
    if tree is None:
        return None
    values = [_buffers(item) for item in tree]
    return tuple(values) if isinstance(tree, tuple) else values


def _copy(target: TensorTree, source: TensorTree) -> None:
    if isinstance(target, Tensor) and isinstance(source, Tensor):
        target.copy_(source)
    elif isinstance(target, list | tuple) and isinstance(source, list | tuple):
        for left, right in zip(target, source, strict=True):
            _copy(left, right)
    elif target is not None or source is not None:
        raise ValueError("PWM CUDA Graph input structure changed")


def _sources(tree: TensorTree) -> list[Source]:
    # Inference tensors have no version counter. PWM treats cached tensors,
    # RoPE and attention metadata as immutable for the lifetime of a request.
    return [(weakref.ref(t), None if t.is_inference() else t._version) for t in _leaves(tree)]


def _same_sources(previous: list[Source], current: list[Source]) -> bool:
    return len(previous) == len(current) and all(
        old() is new() and old_version == new_version
        for (old, old_version), (new, new_version) in zip(previous, current)
    )


@dataclass
class _GraphEntry:
    inputs: tuple
    output: tuple[Tensor, Tensor, Tensor]
    graph: torch.cuda.CUDAGraph
    last_use: torch.cuda.Event
    sources: list[Source]

    def wait(self) -> None:
        # HTTP execution can move between host threads/current CUDA streams.
        torch.cuda.current_stream(self.output[0].device).wait_event(self.last_use)

    def finish(self) -> None:
        self.last_use.record(torch.cuda.current_stream(self.output[0].device))

    def close(self) -> None:
        # Do not release graph-owned memory while replay or output copies run.
        self.last_use.synchronize()


class PwmGraphRunner:
    """Capture a bounded set of shapes for one immutable PWM transformer.

    Call ``end_request()`` in the request's ``finally`` block. Call ``clear()``
    before replacing weights/backends, moving the model, or shutting it down.
    Capture errors propagate with context; they never silently advertise graph
    execution while returning an eager result. UND prefill is intentionally
    eager and counted separately in ``stats``.
    """

    def __init__(self, transformer: PwmWamTransformer, max_entries: int = 4):
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("PWM CUDA Graph max_entries must be a positive integer")
        if transformer.training:
            raise ValueError("PWM CUDA Graph requires an eval-mode transformer")
        self.transformer = transformer
        self.max_entries = max_entries
        self.entries: OrderedDict[tuple, _GraphEntry] = OrderedDict()
        self._owners: list[weakref.ReferenceType[Tensor]] = []
        self._lock = threading.Lock()
        self.stats = {"prefills": 0, "captures": 0, "replays": 0, "evictions": 0}

    def _validate_inputs(self, arguments: tuple) -> None:
        tensors = _leaves(arguments)
        devices = {t.device for t in tensors}
        if len(devices) != 1 or next(iter(devices)).type != "cuda":
            raise ValueError("PWM CUDA Graph requires all inputs on one CUDA device")
        if any(t.layout != torch.strided or t.requires_grad or t.is_conj() or t.is_neg() for t in tensors):
            raise ValueError("PWM CUDA Graph requires strided inference tensors without conjugate/negative views")
        if self.transformer.training:
            raise ValueError("PWM CUDA Graph does not support training")

    def _branch(self, cache: Cache) -> int:
        if cache is None:
            return -1
        owner = cache[0][0][0]
        for index, previous in enumerate(self._owners):
            if previous() is owner:
                return index
        for index, previous in enumerate(self._owners):
            if previous() is None:
                self._owners[index] = weakref.ref(owner)
                return index
        self._owners.append(weakref.ref(owner))
        return len(self._owners) - 1

    def _capture(self, arguments: tuple) -> _GraphEntry:
        device = arguments[0][0].device
        current = torch.cuda.current_stream(device)
        side = torch.cuda.Stream(device=device)
        side.wait_stream(current)
        # Warm up the exact static-buffer path, including compiled variants and
        # library workspaces, outside capture and on a non-default stream.
        try:
            with torch.cuda.stream(side), torch.inference_mode():
                for _ in range(2):
                    self.transformer.forward_hidden(*arguments)
            side.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=side), torch.inference_mode():
                output = self.transformer.forward_hidden(*arguments)
        finally:
            # Keep allocation lifetimes ordered even if warmup/capture fails.
            current.wait_stream(side)
        return _GraphEntry(arguments, output, graph, torch.cuda.Event(), [])

    @torch.inference_mode()
    def run(
        self,
        streams: Streams,
        rope: tuple[Rope, Rope, Rope],
        action_rope: tuple[Rope, Rope, Rope],
        und_cache: Cache = None,
        attention_metadata: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
    ) -> Streams:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("PWM CUDA Graph supports one serialized inference call at a time")
        try:
            arguments = (streams, rope, action_rope, und_cache, attention_metadata)
            self._validate_inputs(arguments)
            if und_cache is not None:
                if not und_cache or any(len(layer) not in (0, 1) for layer in und_cache):
                    raise ValueError("PWM CUDA Graph requires one UND cache slot per transformer layer")
                filled = [bool(layer) for layer in und_cache]
                if not any(filled):
                    self.stats["prefills"] += 1
                    return self.transformer.forward_hidden(*arguments)
                if not all(filled):
                    raise ValueError("PWM CUDA Graph cannot capture a partially populated UND cache")
            precision = (
                torch.is_autocast_enabled("cuda"),
                torch.get_autocast_dtype("cuda"),
                torch.get_float32_matmul_precision(),
                torch.backends.cudnn.allow_tf32,
                torch.are_deterministic_algorithms_enabled(),
            )
            key = (self._branch(und_cache), precision, _signature(arguments))
            source = _sources(arguments[1:])
            entry = self.entries.get(key)
            if entry is None:
                if len(self.entries) >= self.max_entries:
                    _, previous = self.entries.popitem(last=False)
                    previous.close()
                    self.stats["evictions"] += 1
                buffers = _buffers(arguments)
                try:
                    entry = self._capture(buffers)
                except RuntimeError as error:
                    raise RuntimeError(
                        "PWM forward_hidden CUDA Graph capture failed; disable the graph option to use the baseline"
                    ) from error
                entry.sources = source
                self.entries[key] = entry
                self.stats["captures"] += 1
            else:
                entry.wait()
                _copy(entry.inputs[0], streams)
                if not _same_sources(entry.sources, source):
                    _copy(entry.inputs[1:], arguments[1:])
                    entry.sources = source
                self.entries.move_to_end(key)
            entry.graph.replay()
            # Replay outputs alias static graph storage. Copy before another
            # branch/request can replay; decoding may retain the previous result.
            result = tuple(t.clone() for t in entry.output)
            entry.finish()
            self.stats["replays"] += 1
            return result
        finally:
            self._lock.release()

    def end_request(self) -> None:
        """Release semantic ownership; keep bounded allocations for later shapes."""
        with self._lock:
            self._owners.clear()
            for entry in self.entries.values():
                entry.sources.clear()

    def clear(self) -> None:
        """Wait for outstanding uses and release every capture and static buffer."""
        with self._lock:
            for entry in self.entries.values():
                entry.close()
            self.entries.clear()
            self._owners.clear()
