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
