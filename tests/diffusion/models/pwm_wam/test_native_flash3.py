# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FA3 must remain compilable without changing the eager attention kernel."""

import os

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.models.pwm_wam.transformer_pwm_wam import attention

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": "H100"}, num_cards=1)]


@pytest.mark.parametrize("causal", [False, True])
@torch.inference_mode()
def test_flash3_fullgraph_matches_direct_kernel(causal):
    if os.environ.get("PWM_FA3_TEST") != "1":
        pytest.skip("Set PWM_FA3_TEST=1 on Hopper to test bundled FA3")
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    generator = torch.Generator(device="cuda").manual_seed(1020)
    q, k, v = [
        torch.randn(129, heads, 128, generator=generator, device="cuda", dtype=torch.bfloat16) for heads in (32, 8, 8)
    ]
    cu = torch.tensor([0, 129], dtype=torch.int32, device="cuda")
    expected = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=129,
        max_seqlen_k=129,
        causal=causal,
        softmax_scale=128**-0.5,
        fa_version=3,
        num_splits=1,
    ).flatten(1)
    compiled = torch.compile(attention, fullgraph=True, dynamic=False)
    actual = compiled(q, k, v, causal=causal, backend="flash3")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    reused = compiled(q, k, v, causal=causal, backend="flash3", metadata=(cu, cu))
    torch.testing.assert_close(reused, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_compiled_und_cache_is_request_local():
    if os.environ.get("PWM_FA3_TEST") != "1":
        pytest.skip("Set PWM_FA3_TEST=1 on Hopper to test bundled FA3")
    from vllm_omni.diffusion.models.pwm_wam.transformer_pwm_wam import PwmThreeTowerLayer

    torch.manual_seed(1020)
    layer = (
        PwmThreeTowerLayer(
            width=256,
            intermediate=384,
            action_width=128,
            action_intermediate=256,
            heads=4,
            kv_heads=2,
            dim=128,
        )
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    layer.attention_backend = "flash3"
    compiled = torch.compile(layer, fullgraph=True, dynamic=False)
    rope = tuple(
        (
            torch.ones(n, 128, device="cuda", dtype=torch.bfloat16),
            torch.zeros(n, 128, device="cuda", dtype=torch.bfloat16),
        )
        for n in (17, 32, 8)
    )
    metadata = tuple(torch.tensor([0, n], dtype=torch.int32, device="cuda") for n in (17, 32, 8, 57))
    for _ in range(2):
        cache = []
        text = torch.randn(17, 256, device="cuda", dtype=torch.bfloat16)
        for _ in range(3):
            streams = (
                text,
                torch.randn(32, 256, device="cuda", dtype=torch.bfloat16),
                torch.randn(8, 128, device="cuda", dtype=torch.bfloat16),
            )
            expected = compiled(streams, rope, rope)
            actual = compiled(streams, rope, rope, cache, metadata)
            for a, b in zip(expected, actual):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            assert len(cache) == 1
