# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""UND is causal-only: cache must preserve changing VIDEO/ACTION outputs."""

import pytest
import torch

from vllm_omni.diffusion.models.pwm_wam.transformer_pwm_wam import PwmThreeTowerLayer

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


@torch.inference_mode()
def test_request_local_und_cache_matches_uncached_changing_latents():
    torch.manual_seed(42)
    layer = PwmThreeTowerLayer(
        width=32, intermediate=48, action_width=16, action_intermediate=24, heads=4, kv_heads=2, dim=8
    ).eval()
    lengths = (5, 8, 3)
    rope = tuple((torch.randn(n, 8).cos(), torch.randn(n, 8).sin()) for n in lengths)
    action_rope = tuple((torch.randn(n, 8).cos(), torch.randn(n, 8).sin()) for n in lengths)
    cache = []
    text = torch.randn(5, 32)
    for _ in range(3):
        streams = (text, torch.randn(8, 32), torch.randn(3, 16))
        expected = layer(streams, rope, action_rope)
        actual = layer(streams, rope, action_rope, cache)
        for a, b in zip(expected, actual):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert len(cache) == 1
    other_streams = (torch.randn(5, 32), torch.randn(8, 32), torch.randn(3, 16))
    other_cache = []
    expected = layer(other_streams, rope, action_rope)
    actual = layer(other_streams, rope, action_rope, other_cache)
    for a, b in zip(expected, actual):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert not torch.equal(cache[0][0], other_cache[0][0])
