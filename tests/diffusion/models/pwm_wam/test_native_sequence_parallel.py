# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU fake-group checks for VIDEO SP layout, RoPE and compiled collectives."""

import pytest
import torch

from vllm_omni.diffusion.models.pwm_wam import sequence_parallel as sp
from vllm_omni.diffusion.models.pwm_wam.transformer_pwm_wam import attention, rotate

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


class _FakeGroup:
    ulysses_world_size = 1
    ring_world_size = 1

    def __init__(self, world_size, rank):
        self.world_size = self.allgather_world_size = world_size
        self.rank_in_group = rank
        self.expected_global = None
        self.gathers = 0
        self.broadcasts = 0

    def all_gather(self, value, dim=0):
        assert dim == 0
        assert value.is_contiguous()
        expected = self.expected_global.chunk(self.world_size, dim=0)[self.rank_in_group]
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
        self.gathers += 1
        return self.expected_global.clone()

    def broadcast(self, value, src=0):
        assert src == 0
        self.broadcasts += 1
        return value


def _helper(monkeypatch, world_size, rank):
    group = _FakeGroup(world_size, rank)
    monkeypatch.setattr(sp, "_get_sp_group", lambda: group)
    return sp.PwmVideoSequenceParallel(), group


@pytest.mark.parametrize("world_size", [2, 4])
def test_video_sharding_preserves_und_action_and_local_query_metadata(monkeypatch, world_size):
    video = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8)
    und, action = torch.randn(3, 8), torch.randn(2, 4)
    rope = (video.cos(), video.sin())
    for rank in range(world_size):
        helper, _ = _helper(monkeypatch, world_size, rank)
        streams = helper.shard_streams((und, video, action))
        assert streams[0] is und and streams[2] is action
        torch.testing.assert_close(streams[1], video.chunk(world_size)[rank], rtol=0, atol=0)
        local_rope = helper.video_query_rope(rope)
        for observed, wanted in zip(local_rope, rope, strict=True):
            torch.testing.assert_close(observed, wanted.chunk(world_size)[rank], rtol=0, atol=0)
        metadata = helper.attention_metadata(3, 16, 2, device="cpu")
        assert [value.tolist() for value in metadata] == [[0, 3], [0, 16 // world_size], [0, 2], [0, 21]]


@pytest.mark.parametrize("world_size", [2, 4])
@torch.inference_mode()
def test_local_queries_and_global_kv_match_full_attention_for_video_and_action(monkeypatch, world_size):
    torch.manual_seed(1020)
    qv, qa = torch.randn(16, 4, 8), torch.randn(2, 4, 8)
    keys = (torch.randn(3, 2, 8), torch.randn(16, 2, 8), torch.randn(2, 2, 8))
    values = tuple(torch.randn_like(key) for key in keys)
    rope = tuple((torch.randn(n, 8).cos(), torch.randn(n, 8).sin()) for n in (3, 16, 2))
    action_rope = tuple((torch.randn(n, 8).cos(), torch.randn(n, 8).sin()) for n in (3, 16, 2))
    global_video_keys = torch.cat([rotate(key, positions) for key, positions in zip(keys, rope, strict=True)])
    global_action_keys = torch.cat([rotate(key, positions) for key, positions in zip(keys, action_rope, strict=True)])
    expected_video = attention(rotate(qv, rope[1]), global_video_keys, torch.cat(values))
    expected_action = attention(rotate(qa, action_rope[2]), global_action_keys, torch.cat(values))
    outputs = []
    for rank in range(world_size):
        helper, group = _helper(monkeypatch, world_size, rank)
        group.expected_global = torch.stack((keys[1], values[1]), dim=1)
        gathered_k, gathered_v = helper.gather_video_kv(
            keys[1].chunk(world_size)[rank], values[1].chunk(world_size)[rank]
        )
        assert gathered_k.shape == keys[1].shape
        assert group.gathers == 1  # K and V share one collective.
        all_keys = (keys[0], gathered_k, keys[2])
        all_values = torch.cat((values[0], gathered_v, values[2]))
        local_q = rotate(qv.chunk(world_size)[rank], helper.video_query_rope(rope[1]))
        video_keys = torch.cat([rotate(key, positions) for key, positions in zip(all_keys, rope, strict=True)])
        outputs.append(attention(local_q, video_keys, all_values))
        action_keys = torch.cat([rotate(key, positions) for key, positions in zip(all_keys, action_rope, strict=True)])
        actual_action = attention(rotate(qa, action_rope[2]), action_keys, all_values)
        torch.testing.assert_close(actual_action, expected_action, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(outputs), expected_video, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("world_size", [2, 4])
def test_final_hidden_gather_restores_global_token_order(monkeypatch, world_size):
    video = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8)
    und, action = torch.randn(3, 8), torch.randn(2, 4)
    for rank in range(world_size):
        helper, group = _helper(monkeypatch, world_size, rank)
        group.expected_global = video
        result = helper.gather_streams((und, video.chunk(world_size)[rank], action))
        assert result[0] is und and result[2] is action
        torch.testing.assert_close(result[1], video, rtol=0, atol=0)
        assert group.gathers == 1
        latent = torch.randn(9)
        assert helper.synchronize_latent(latent) is latent
        assert group.broadcasts == 1


@pytest.mark.parametrize("world_size", [2, 4])
def test_all_gather_helper_is_fullgraph_capturable_on_cpu(monkeypatch, world_size):
    torch._dynamo.reset()
    helper, group = _helper(monkeypatch, world_size, 1)
    key, value = torch.randn(16, 2, 8), torch.randn(16, 2, 8)
    group.expected_global = torch.stack((key, value), dim=1)
    graphs = []

    def backend(graph, _):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(helper.gather_video_kv, backend=backend, fullgraph=True)
    try:
        actual_key, actual_value = compiled(key.chunk(world_size)[1], value.chunk(world_size)[1])
        torch.testing.assert_close(actual_key, key, rtol=0, atol=0)
        torch.testing.assert_close(actual_value, value, rtol=0, atol=0)
        assert len(graphs) == 1
        assert any("pwm_sp_all_gather" in str(node.target) for node in graphs[0].graph.nodes)
        assert group.gathers == 1
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("world_size", [2, 4])
def test_uneven_tokens_fail_before_any_collective(monkeypatch, world_size):
    helper, group = _helper(monkeypatch, world_size, 0)
    for operation in (
        lambda: helper.shard_video(torch.randn(7, 8)),
        lambda: helper.video_query_rope((torch.randn(7, 8), torch.randn(7, 8))),
        lambda: helper.attention_metadata(3, 7, 2, device="cpu"),
    ):
        with pytest.raises(ValueError, match="divisible"):
            operation()
    assert group.gathers == 0


@pytest.mark.parametrize(
    "field,value", [("world_size", 3), ("ring_world_size", 2), ("ulysses_world_size", 2), ("allgather_world_size", 1)]
)
def test_invalid_parallel_topology_is_rejected(monkeypatch, field, value):
    group = _FakeGroup(2, 0)
    setattr(group, field, value)
    monkeypatch.setattr(sp, "_get_sp_group", lambda: group)
    with pytest.raises(ValueError, match="PWM VIDEO SP"):
        sp.PwmVideoSequenceParallel()


def test_kv_shape_mismatch_fails_without_collective(monkeypatch):
    helper, group = _helper(monkeypatch, 2, 0)
    with pytest.raises(ValueError, match="matching"):
        helper.gather_video_kv(torch.randn(8, 2, 8), torch.randn(7, 2, 8))
    assert group.gathers == 0
