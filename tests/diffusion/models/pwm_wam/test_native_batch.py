# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Verify packed sample boundaries and independent sampler state on CPU.

A segmented SDPA kernel substitutes for FA3 only. The production embedding,
three-tower layers, positional contract, decoding, batching and UniPC sampler
remain active. These tests establish semantic isolation, not FA3 performance
or CUDA numerical tolerances; hardware comparison is a separate requirement.
"""

from dataclasses import replace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.pwm_wam import transformer_pwm_wam as model
from vllm_omni.diffusion.models.pwm_wam.batch import _packed_metadata, sample_batch
from vllm_omni.diffusion.models.pwm_wam.request import PwmRequest
from vllm_omni.diffusion.models.pwm_wam.runtime import PwmWamRuntime

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


class _Stack(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 32)
        self.layers = nn.ModuleList(
            [
                model.PwmThreeTowerLayer(
                    width=32, intermediate=48, action_width=16, action_intermediate=24, heads=2, kv_heads=1, dim=128
                )
            ]
        )
        self.norm = model.PwmRMSNorm(32)
        self.norm_moe_gen = model.PwmRMSNorm(32)
        self.norm_moe_gen_action = model.PwmRMSNorm(16)
        for layer in self.layers:
            layer.attention_backend = "flash3"


class _Language(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Stack()


class _Transformer(model.PwmWamTransformer):
    def __init__(self):
        nn.Module.__init__(self)
        self.language_model = _Language()
        self.vae2llm = nn.Linear(192, 32)
        self.llm2vae = nn.Linear(32, 192)
        self.time_embedder = model.PwmTimeEmbedding(32)
        self.action_time_embedder = model.PwmTimeEmbedding(16)
        self.action2llm = model.PwmDomainLinear(128, 16)
        self.llm2action = model.PwmDomainLinear(16, 128)
        self.action_modality_embed = nn.Parameter(torch.zeros(16))


class _Runtime:
    sample = PwmWamRuntime.sample

    def __init__(self, cache_und):
        self.transformer = _Transformer().eval()
        self.device = "cpu"
        self.attention_backend = "flash3"
        self.cache_und = cache_und
        self.graph_runner = None
        self.cfg_parallel = None
        self.approx_cache_threshold = 0.0

    def text_ids(self, caption):
        return torch.tensor([1, *(ord(char) % 29 + 2 for char in caption), 2], dtype=torch.long)


def _segmented_attention(q, k, v, cu_q, cu_k, causal, provider, max_seqlen_q=0, max_seqlen_k=0):
    q_lengths, k_lengths = cu_q.diff(), cu_k.diff()
    assert max_seqlen_q == q_lengths.max().item()
    assert max_seqlen_k == k_lengths.max().item()
    outputs = []
    for q_start, q_end, k_start, k_end in zip(cu_q[:-1], cu_q[1:], cu_k[:-1], cu_k[1:]):
        query = q[q_start:q_end]
        value = model.attention(query, k[k_start:k_end], v[k_start:k_end], causal=causal, backend="sdpa")
        outputs.append(value.reshape(query.shape))
    return torch.cat(outputs)


def _request(seed, caption="ab", **changes):
    request = PwmRequest(
        caption=caption,
        negative_prompt="",
        pixels=torch.zeros(3, 1, 16, 16, dtype=torch.uint8),
        domain="test",
        normalization_id=0,
        projection_id=0,
        mask=[True] * 128,
        video_fps=4.0,
        action_fps=2.0,
        action_length=2,
        video_frames=5,
        seam=None,
        seed=seed,
        steps=3,
        guidance=1.5,
        shift=5.0,
        normalize_cfg=False,
        guidance_interval=None,
        return_video=False,
    )
    return replace(request, **changes)


def test_batch_metadata_preserves_each_samples_three_tower_key_order():
    cu_u, cu_v, cu_a, cu_all, order, maximum = _packed_metadata((2, 3, 1), 2, torch.device("cpu"))
    assert cu_u.tolist() == [0, 2, 4]
    assert cu_v.tolist() == [0, 3, 6]
    assert cu_a.tolist() == [0, 1, 2]
    assert cu_all.tolist() == [0, 6, 12]
    assert maximum == (2, 3, 1, 6)
    # Input order U0,U1,V0,V1,A0,A1 becomes [U0,V0,A0],[U1,V1,A1].
    assert order.tolist() == [0, 1, 4, 5, 6, 10, 2, 3, 7, 8, 9, 11]


@pytest.mark.parametrize("cache_und", [False, True])
@pytest.mark.parametrize(
    "guidance,normalize_cfg,interval", [(1.0, False, None), (1.5, False, None), (1.5, True, (0.0, 900.0))]
)
@torch.inference_mode()
def test_batch_matches_independent_requests_and_reordered_neighbors(
    monkeypatch, cache_und, guidance, normalize_cfg, interval
):
    monkeypatch.setattr(model, "pwm_fa3_attention", _segmented_attention)
    torch.manual_seed(1020)
    runtime = _Runtime(cache_und)
    options = dict(guidance=guidance, normalize_cfg=normalize_cfg, guidance_interval=interval)
    requests = [_request(42, "ab", **options), _request(43, "cd", projection_id=1, **options)]
    clean = [torch.randn(48, 2, 2, 2), torch.randn(48, 2, 2, 2)]
    expected = [runtime.sample(request, latent) for request, latent in zip(requests, clean)]
    actual = sample_batch(runtime, requests, clean)
    reversed_outputs = sample_batch(runtime, list(reversed(requests)), list(reversed(clean)))
    for reference, packed, reordered in zip(expected, actual, reversed(reversed_outputs)):
        for left, right, changed_order in zip(reference, packed, reordered):
            torch.testing.assert_close(left, right, atol=2e-4, rtol=2e-5)
            torch.testing.assert_close(right, changed_order, atol=2e-4, rtol=2e-5)
    assert not torch.equal(actual[0][1], actual[1][1])


@pytest.mark.parametrize("changed", [dict(steps=5), dict(guidance=1), dict(seam=0.5), dict(action_fps=3.0)])
def test_batch_rejects_mixed_sampling_and_layout(changed):
    runtime = _Runtime(cache_und=False)
    requests = [_request(42), _request(43, **changed)]
    with pytest.raises(ValueError, match="requires matching"):
        sample_batch(runtime, requests, [torch.zeros(48, 2, 2, 2)] * 2)


def test_batch_rejects_unequal_branch_lengths_and_unsupported_paths():
    runtime = _Runtime(cache_und=False)
    clean = [torch.zeros(48, 2, 2, 2)] * 2
    with pytest.raises(ValueError, match="equal token lengths"):
        sample_batch(runtime, [_request(42, "a"), _request(43, "long")], clean)
    requests = [_request(42), _request(43)]
    runtime.attention_backend = "sdpa"
    with pytest.raises(ValueError, match="Flash Attention 3"):
        sample_batch(runtime, requests, clean)
    runtime.attention_backend = "flash3"
    runtime.graph_runner = object()
    with pytest.raises(ValueError, match="CUDA Graphs"):
        sample_batch(runtime, requests, clean)


def test_layer_rejects_packed_metadata_on_sdpa():
    layer = model.PwmThreeTowerLayer(
        width=32, intermediate=48, action_width=16, action_intermediate=24, heads=2, kv_heads=1, dim=128
    ).eval()
    streams = (torch.ones(4, 32), torch.ones(6, 32), torch.ones(2, 16))
    rope = tuple((torch.ones(n, 128), torch.zeros(n, 128)) for n in (4, 6, 2))
    metadata = _packed_metadata((2, 3, 1), 2, torch.device("cpu"))
    with pytest.raises(ValueError, match="require Flash Attention 3"):
        layer(streams, rope, rope, attention_metadata=metadata)


@pytest.mark.parametrize("batch_size", [1, 4])
@torch.inference_mode()
def test_batch_supported_sizes_keep_independent_outputs(monkeypatch, batch_size):
    monkeypatch.setattr(model, "pwm_fa3_attention", _segmented_attention)
    torch.manual_seed(1030)
    runtime = _Runtime(cache_und=True)
    requests = [_request(42 + index, "ab", projection_id=index) for index in range(batch_size)]
    clean = [torch.randn(48, 2, 2, 2) for _ in requests]
    expected = [runtime.sample(request, latent) for request, latent in zip(requests, clean)]
    actual = sample_batch(runtime, requests, clean)
    for reference, packed in zip(expected, actual):
        for left, right in zip(reference, packed):
            torch.testing.assert_close(left, right, atol=2e-4, rtol=2e-5)
