# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise complete SP layers with concurrent CPU collectives.

Only communication and FA3 kernels are replaced: real projections, both RoPE
paths, local VIDEO residuals/MLPs, replicated ACTION, UND caches and final
normalization execute on every rank. This validates layout and arithmetic;
CUDA/NCCL execution and its numerical behavior require separate GPU checks.
"""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier, local

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.pwm_wam import sequence_parallel as sp
from vllm_omni.diffusion.models.pwm_wam import transformer_pwm_wam as model

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


class _Collective:
    def __init__(self, size):
        self.barrier = Barrier(size, timeout=15)
        self.values = [None] * size


class _ThreadGroup:
    ulysses_world_size = 1
    ring_world_size = 1

    def __init__(self, collective, rank):
        self.collective = collective
        self.world_size = self.allgather_world_size = len(collective.values)
        self.rank_in_group = rank
        self.gathers = 0

    def all_gather(self, value, dim=0):
        assert dim == 0 and value.is_contiguous()
        self.collective.values[self.rank_in_group] = value.clone()
        self.collective.barrier.wait()
        result = torch.cat(self.collective.values, dim=0)
        # Readers finish before any rank replaces a slot in the next gather.
        self.collective.barrier.wait()
        self.gathers += 1
        return result


class _TinyTransformer(model.PwmWamTransformer):
    def __init__(self, backend):
        nn.Module.__init__(self)
        self.sequence_parallel = None
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        stack = self.language_model.model
        stack.layers = nn.ModuleList(
            model.PwmThreeTowerLayer(
                width=32, intermediate=48, action_width=16, action_intermediate=24, heads=4, kv_heads=2, dim=8
            )
            for _ in range(2)
        )
        for layer in stack.layers:
            layer.attention_backend = backend
        stack.norm = model.PwmRMSNorm(32)
        stack.norm_moe_gen = model.PwmRMSNorm(32)
        stack.norm_moe_gen_action = model.PwmRMSNorm(16)
        self.eval()


def _fake_fa3(q, k, v, cu_q, cu_k, causal, provider, max_seqlen_q=0, max_seqlen_k=0):
    assert provider == "flash3"
    assert cu_q.tolist() == [0, q.shape[0]]
    assert cu_k.tolist() == [0, k.shape[0]]
    assert max_seqlen_q == q.shape[0]
    assert max_seqlen_k == k.shape[0]
    return model.attention(q, k, v, causal=causal).reshape(q.shape)


def _record_layers(transformer):
    records = []

    def record(_module, _args, output):
        records.append(tuple(value.clone() for value in output))

    for layer in transformer.language_model.model.layers:
        layer.register_forward_hook(record)
    return records


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("backend", ["sdpa", "flash3"])
@pytest.mark.parametrize("cache_und", [False, True])
@torch.inference_mode()
def test_full_transformer_sp_matches_global_reference_after_each_layer(monkeypatch, world_size, backend, cache_und):
    torch.manual_seed(1020)
    monkeypatch.setattr(model, "pwm_fa3_attention", _fake_fa3)
    reference = _TinyTransformer(backend)
    replicas = [deepcopy(reference) for _ in range(world_size)]
    expected_layers = _record_layers(reference)
    local_layers = [_record_layers(replica) for replica in replicas]
    streams = (torch.randn(3, 32), torch.randn(12, 32), torch.randn(5, 16))
    # A second step changes both generated modalities and keeps UND immutable.
    steps = [streams, (streams[0], streams[1] + 0.4, streams[2] - 0.7)]

    def positions():
        angles = [torch.randn(count, 4).repeat(1, 2) for count in (3, 12, 5)]
        return tuple((angle.cos(), angle.sin()) for angle in angles)

    rope, action_rope = positions(), positions()
    metadata = tuple(torch.tensor([0, count], dtype=torch.int32) for count in (3, 12, 5, 20))
    cache = [[], []] if cache_und else None
    expected = [reference.forward_hidden(values, rope, action_rope, cache, metadata) for values in steps]

    collective = _Collective(world_size)
    groups = [_ThreadGroup(collective, rank) for rank in range(world_size)]
    current = local()
    monkeypatch.setattr(sp, "_get_sp_group", lambda: current.group)
    for replica, group in zip(replicas, groups, strict=True):
        current.group = group
        replica.sequence_parallel = sp.PwmVideoSequenceParallel()
        for layer in replica.language_model.model.layers:
            layer.sequence_parallel = replica.sequence_parallel

    @torch.inference_mode()
    def run(rank):
        current.group = groups[rank]
        replica = replicas[rank]
        cache = [[], []] if cache_und else None
        local_metadata = replica.sequence_parallel.attention_metadata(3, 12, 5, device="cpu")
        try:
            outputs = [replica.forward_hidden(values, rope, action_rope, cache, local_metadata) for values in steps]
            return outputs, cache
        except BaseException:
            # Fail peers promptly if one rank fails before its next collective.
            collective.barrier.abort()
            raise

    with ThreadPoolExecutor(max_workers=world_size) as pool:
        futures = [pool.submit(run, rank) for rank in range(world_size)]
        actual = [future.result(timeout=45) for future in futures]

    for rank, (outputs, cache) in enumerate(actual):
        # Every rank returns the global VIDEO and replicated ACTION streams.
        for observed, wanted in zip(outputs, expected, strict=True):
            for value, target in zip(observed, wanted, strict=True):
                torch.testing.assert_close(value, target, atol=2e-6, rtol=2e-5)
        assert len(local_layers[rank]) == len(expected_layers) == 4
        for observed, wanted in zip(local_layers[rank], expected_layers, strict=True):
            torch.testing.assert_close(observed[0], wanted[0], atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(observed[1], wanted[1].chunk(world_size)[rank], atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(observed[2], wanted[2], atol=2e-6, rtol=2e-5)
        # Two K/V gathers and one final hidden gather on each of two steps.
        assert groups[rank].gathers == 6
        if cache_und:
            assert [len(layer) for layer in cache] == [1, 1]
