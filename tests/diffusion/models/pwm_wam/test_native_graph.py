# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Graph state ownership and refresh contracts, without requiring a GPU.

The CPU replay double replaces CUDA capture only. Static-buffer creation,
keys, cache-prefill policy, source invalidation, output ownership and eviction
exercise the production runner. Actual CUDA/compiler capture needs hardware
validation and is not established by these lifecycle tests.
"""

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.pwm_wam.graph import PwmGraphRunner

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


class _Transformer(nn.Module):
    def forward_hidden(self, streams, rope, action_rope, cache=None, metadata=None):
        if cache is None:
            und = streams[0] * 2
        else:
            if not cache[0]:
                cache[0].append((streams[0] * 2, streams[0] + 1, streams[0] + 2))
            und = cache[0][0][0]
        bias = und.sum() + sum(t.sum() for pair in rope + action_rope for t in pair)
        if cache is not None:
            bias = bias + cache[0][0][1].sum() + cache[0][0][2].sum()
        if metadata is not None:
            bias = bias + sum(t.sum() for t in metadata)
        return und, streams[1] + bias, streams[2] - bias


class _Replay:
    def __init__(self, transformer, inputs):
        self.transformer = transformer
        self.inputs = inputs
        self.output = tuple(torch.empty_like(t) for t in transformer.forward_hidden(*inputs))
        self.sources = []
        self.graph = self
        self.closed = False

    def replay(self):
        for target, source in zip(self.output, self.transformer.forward_hidden(*self.inputs)):
            target.copy_(source)

    def wait(self):
        assert not self.closed

    def finish(self):
        pass

    def close(self):
        self.closed = True


def _inputs(size=3, text=1.0):
    streams = (torch.full((2, 4), text), torch.ones(size, 4), torch.ones(2, 2))
    rope = tuple((torch.ones(n, 4), torch.zeros(n, 4)) for n in (2, size, 2))
    action_rope = tuple((torch.ones(n, 4), torch.zeros(n, 4)) for n in (2, size, 2))
    metadata = tuple(torch.tensor([0, n], dtype=torch.int32) for n in (2, size, 2, size + 4))
    return streams, rope, action_rope, metadata


def _runner(monkeypatch, max_entries=4):
    transformer = _Transformer().eval()
    runner = PwmGraphRunner(transformer, max_entries=max_entries)
    monkeypatch.setattr(runner, "_validate_inputs", lambda arguments: None)
    monkeypatch.setattr(runner, "_capture", lambda inputs: _Replay(transformer, inputs))
    return transformer, runner


def _assert_output(expected, actual):
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right)


def test_graph_replays_changing_latents_and_refreshes_mutated_metadata(monkeypatch):
    transformer, runner = _runner(monkeypatch)
    streams, rope, action_rope, metadata = _inputs()
    first = runner.run(streams, rope, action_rope, attention_metadata=metadata)
    first_saved = tuple(t.clone() for t in first)
    streams = tuple(t + 3 for t in streams)
    # Same objects and shapes, changed values: normal tensor version counters
    # must invalidate the static constant buffers.
    rope[1][0].add_(2)
    metadata[0].add_(1)
    actual = runner.run(streams, rope, action_rope, attention_metadata=metadata)
    expected = transformer.forward_hidden(streams, rope, action_rope, None, metadata)
    _assert_output(expected, actual)
    _assert_output(first_saved, first)
    assert runner.stats == {"prefills": 0, "captures": 1, "replays": 2, "evictions": 0}


def test_graph_prefill_is_not_captured_and_request_end_refreshes_und(monkeypatch):
    transformer, runner = _runner(monkeypatch)
    for text in (1.0, 9.0):
        streams, rope, action_rope, metadata = _inputs(text=text)
        cache = [[]]
        runner.run(streams, rope, action_rope, cache, metadata)
        assert len(cache[0]) == 1
        actual = runner.run(streams, rope, action_rope, cache, metadata)
        expected = transformer.forward_hidden(streams, rope, action_rope, cache, metadata)
        _assert_output(expected, actual)
        runner.end_request()
        assert all(not entry.sources for entry in runner.entries.values())
    assert runner.stats == {"prefills": 2, "captures": 1, "replays": 2, "evictions": 0}


def test_graph_equal_shape_cfg_branches_have_separate_und_buffers(monkeypatch):
    transformer, runner = _runner(monkeypatch)
    branches = []
    for text in (2.0, 7.0):
        streams, rope, action_rope, metadata = _inputs(text=text)
        cache = [[]]
        runner.run(streams, rope, action_rope, cache, metadata)
        branches.append((streams, rope, action_rope, cache, metadata))
    saved = []
    for arguments in branches + branches:
        actual = runner.run(*arguments)
        _assert_output(transformer.forward_hidden(*arguments), actual)
        saved.append(actual)
    assert runner.stats["captures"] == 2
    assert runner.stats["replays"] == 4
    _assert_output(saved[0], saved[2])
    assert not torch.equal(saved[0][1], saved[1][1])


def test_graph_new_shape_evicts_at_limit_and_clear_releases_entries(monkeypatch):
    _, runner = _runner(monkeypatch, max_entries=1)
    streams, rope, action_rope, metadata = _inputs()
    runner.run(streams, rope, action_rope, attention_metadata=metadata)
    previous = next(iter(runner.entries.values()))
    streams, rope, action_rope, metadata = _inputs(size=5)
    runner.run(streams, rope, action_rope, attention_metadata=metadata)
    assert previous.closed
    assert len(runner.entries) == 1
    assert runner.stats["evictions"] == 1
    remaining = next(iter(runner.entries.values()))
    runner.clear()
    assert remaining.closed
    assert not runner.entries


def test_graph_rejects_partial_cache_and_concurrent_execution(monkeypatch):
    _, runner = _runner(monkeypatch)
    streams, rope, action_rope, metadata = _inputs()
    partial = [[(streams[0], streams[0], streams[0])], []]
    with pytest.raises(ValueError, match="partially populated"):
        runner.run(streams, rope, action_rope, partial, metadata)
    assert runner._lock.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError, match="serialized"):
            runner.run(streams, rope, action_rope, attention_metadata=metadata)
    finally:
        runner._lock.release()


def test_graph_capture_error_is_explicit_and_not_cached(monkeypatch):
    _, runner = _runner(monkeypatch)
    streams, rope, action_rope, metadata = _inputs()

    def fail(inputs):
        raise RuntimeError("unsupported capture")

    monkeypatch.setattr(runner, "_capture", fail)
    with pytest.raises(RuntimeError, match="capture failed") as error:
        runner.run(streams, rope, action_rope, attention_metadata=metadata)
    assert "unsupported capture" in str(error.value.__cause__)
    assert not runner.entries
    assert runner.stats["replays"] == 0


@pytest.mark.parametrize("max_entries", [0, -1, True, 1.5])
def test_graph_rejects_invalid_bound(max_entries):
    with pytest.raises(ValueError, match="positive integer"):
        PwmGraphRunner(_Transformer().eval(), max_entries=max_entries)


def test_graph_rejects_cpu_and_training():
    with pytest.raises(ValueError, match="eval-mode"):
        PwmGraphRunner(_Transformer())
    runner = PwmGraphRunner(_Transformer().eval())
    streams, rope, action_rope, metadata = _inputs()
    with pytest.raises(ValueError, match="one CUDA device"):
        runner.run(streams, rope, action_rope, attention_metadata=metadata)
