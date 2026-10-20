# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU qualification of GQA descales and fullgraph FP8 attention preparation."""

import json

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_omni.diffusion.models.pwm_wam import attention_fp8 as module

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


def _inputs(q_tokens=7, k_tokens=13):
    generator = torch.Generator().manual_seed(1020)
    q = torch.randn(q_tokens, 32, 128, generator=generator).to(torch.bfloat16) * 0.1
    k = torch.randn(k_tokens, 8, 128, generator=generator).to(torch.bfloat16) * 0.1
    v = torch.randn(k_tokens, 8, 128, generator=generator).to(torch.bfloat16)
    return q, k, v


def test_scales_follow_adjacent_gqa_groups_and_zero_groups():
    value = torch.ones(5, 32, 128, dtype=torch.bfloat16)
    value[:, :4] = 0
    value[:, 4:8] *= 10
    value[:, 8] *= 100
    quantized, scale = module.quantize_fp8_gqa_groups(value, 8)
    assert scale.shape == (1, 8)
    assert scale.dtype == torch.float32
    assert quantized.dtype == torch.float8_e4m3fn
    assert quantized.is_contiguous() and scale.is_contiguous()
    assert scale[0, 0] == 1
    torch.testing.assert_close(scale[0, 1:3], torch.tensor([10, 100]) / 448, atol=0, rtol=0)
    reconstructed = quantized.float() * scale.repeat_interleave(4, dim=1)[:, :, None]
    assert not reconstructed[:, :4].any()
    torch.testing.assert_close(reconstructed[:, 4:8], value[:, 4:8].float(), rtol=1e-6, atol=0)
    # Head 8 sets the scale shared with heads 9..11; the next KV group stays independent.
    torch.testing.assert_close(scale[0, 3:], torch.full((5,), 1 / 448), rtol=0, atol=0)


@pytest.mark.parametrize(
    "length,expected", [(1, 1), (2560, 1), (2561, 2), (5120, 2), (6967, 4), (19320, 8), (75600, 32)]
)
def test_guarded_split_policy(length, expected):
    assert module.resolve_fp8_attention_splits(length) == expected
    assert module.resolve_fp8_attention_splits(length, 1) == 1
    info = module.fp8_attention_info(length)
    assert info["resolved_num_splits"] == expected
    assert info["softmax_scale"] == 128**-0.5
    json.dumps(info, allow_nan=False)


@pytest.mark.parametrize("length,splits", [(0, 0), (1, -1), (1, 257), (1, True), (1000000, 0)])
def test_reject_invalid_split_configuration(length, splits):
    with pytest.raises(ValueError):
        module.resolve_fp8_attention_splits(length, splits)


def test_full_preparation_routes_descales_with_unchanged_softmax_math(monkeypatch):
    q, k, v = _inputs()
    originals = [value.clone() for value in (q, k, v)]
    calls = []

    def cpu_kernel(qq, kk, vv, cu_q, cu_k, qs, ks, vs, num_splits):
        calls.append(num_splits)
        assert [value.dtype for value in (qq, kk, vv)] == [torch.float8_e4m3fn] * 3
        for scale in (qs, ks, vs):
            assert scale.shape == (1, 8)
        assert cu_q.tolist() == [0, len(q)]
        assert cu_k.tolist() == [0, len(k)]
        reconstructed_q = qq.float() * qs.repeat_interleave(4, dim=1)[:, :, None]
        reconstructed_k = kk.float() * ks[:, :, None]
        reconstructed_v = vv.float() * vs[:, :, None]
        result = F.scaled_dot_product_attention(
            reconstructed_q.transpose(0, 1)[None],
            reconstructed_k.repeat_interleave(4, dim=1).transpose(0, 1)[None],
            reconstructed_v.repeat_interleave(4, dim=1).transpose(0, 1)[None],
            scale=128**-0.5,
        )
        return result[0].transpose(0, 1).to(torch.bfloat16)

    monkeypatch.setattr(module, "pwm_fp8_attention", cpu_kernel)
    actual = module.fp8_attention(q, k, v)
    reference = (
        F.scaled_dot_product_attention(
            q.float().transpose(0, 1)[None],
            k.float().repeat_interleave(4, dim=1).transpose(0, 1)[None],
            v.float().repeat_interleave(4, dim=1).transpose(0, 1)[None],
            scale=128**-0.5,
        )[0]
        .transpose(0, 1)
        .flatten(1)
    )
    assert actual.dtype == torch.bfloat16 and actual.shape == (len(q), 4096)
    assert calls == [1]
    relative_rmse = (actual.float() - reference).square().mean().sqrt() / reference.square().mean().sqrt()
    assert relative_rmse < 0.05  # Synthetic numerical check, not an end-to-end acceptance threshold.
    for value, original in zip((q, k, v), originals, strict=True):
        torch.testing.assert_close(value, original, atol=0, rtol=0)


def test_real_cpu_execution_is_rejected():
    with pytest.raises(ValueError, match="CUDA"):
        module.fp8_attention(*_inputs())


def test_fake_checks_shapes_scales_metadata_and_output():
    with FakeTensorMode():
        q = torch.empty(7, 32, 128, dtype=torch.float8_e4m3fn)
        k = torch.empty(13, 8, 128, dtype=torch.float8_e4m3fn)
        cumulative = torch.empty(2, dtype=torch.int32)
        scale = torch.empty(1, 8)
        result = module.pwm_fp8_attention(q, k, k, cumulative, cumulative, scale, scale, scale, 1)
        assert result.shape == q.shape and result.dtype == torch.bfloat16
        with pytest.raises(ValueError, match="descales"):
            module.pwm_fp8_attention(q, k, k, cumulative, cumulative, torch.empty(1, 32), scale, scale, 1)
        with pytest.raises(ValueError, match="B1"):
            module.pwm_fp8_attention(q, k, k, torch.empty(3, dtype=torch.int32), cumulative, scale, scale, scale, 1)
        with pytest.raises(ValueError, match="Q32/KV8"):
            module.pwm_fp8_attention(q[:, :16], k, k, cumulative, cumulative, scale, scale, scale, 1)


def test_entire_quantization_and_custom_op_exports_as_one_graph():
    class Attention(nn.Module):
        def forward(self, q, k, v):
            return module.fp8_attention(q, k, v)

    exported = torch.export.export(Attention(), _inputs())
    nodes = [str(node.target) for node in exported.graph.nodes]
    assert "vllm_omni.pwm_fp8_attention.default" in nodes
    assert any("amax" in node for node in nodes)
