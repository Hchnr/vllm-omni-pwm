# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for selective VIDEO quantization, routing and failure isolation."""

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_omni.diffusion.models.pwm_wam.optimization_linear import (
    PwmFP8Linear,
    install_video_fp8,
    pwm_fp8_scaled_mm,
    quantize_fp8_per_tensor,
)
from vllm_omni.diffusion.models.pwm_wam.transformer_pwm_wam import PwmThreeTowerLayer

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


def _transformer():
    model = nn.Module()
    model.language_model = nn.Module()
    model.language_model.model = nn.Module()
    model.language_model.model.layers = nn.ModuleList(
        [
            PwmThreeTowerLayer(
                width=32, intermediate=64, action_width=16, action_intermediate=32, heads=4, kv_heads=2, dim=8
            )
            for _ in range(2)
        ]
    )
    return model.to(dtype=torch.bfloat16).eval()


def test_fp8_zero_scale_and_reconstruction():
    zero, scale = quantize_fp8_per_tensor(torch.zeros(16, 32, dtype=torch.bfloat16))
    assert zero.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(scale, torch.ones(1), atol=0, rtol=0)
    assert not zero.float().any()
    generator = torch.Generator().manual_seed(1020)
    source = torch.randn(32, 64, generator=generator).to(torch.bfloat16)
    quantized, scale = quantize_fp8_per_tensor(source)
    reconstructed = quantized.float() * scale
    assert torch.isfinite(reconstructed).all()
    # E4M3 nearest rounding has at most ~1/16 relative error for normal values;
    # this data-dependent bound permits subnormals without assuming exactness.
    torch.testing.assert_close(reconstructed, source.float(), rtol=0.063, atol=float(scale) / 256)
    linear = PwmFP8Linear(source)
    assert linear.weight_fp8.dtype == torch.float8_e4m3fn
    assert list(linear.parameters()) == []
    assert linear.conversion_metrics["weight_rmse"] > 0
    assert linear.conversion_metrics["fp8_weight_and_scale_bytes"] < linear.conversion_metrics["bf16_weight_bytes"]


@pytest.mark.parametrize("scope,per_layer", [("mlp", 3), ("attention", 4), ("all", 7)])
def test_install_only_selected_video_linears(scope, per_layer):
    model = _transformer()
    before = dict(model.named_modules())
    report = install_video_fp8(model, scope)
    changed = {name for name, module in model.named_modules() if before.get(name) is not module}
    assert len(changed) == 2 * per_layer
    assert changed == {record["name"] for record in report["modules"]}
    assert report["module_count"] == len(changed)
    assert report["activation_quantization_in_request_timing"] is True
    assert report["use_fast_accum"] is False
    for name in changed:
        assert "action" not in name
        assert "mlp_moe_gen." in name or name.endswith("_moe_gen")
        assert isinstance(model.get_submodule(name), PwmFP8Linear)
    for index, layer in enumerate(model.language_model.model.layers):
        assert layer.self_attn.q_proj is before[f"language_model.model.layers.{index}.self_attn.q_proj"]
        assert (
            layer.action_block_moe_gen_action
            is before[f"language_model.model.layers.{index}.action_block_moe_gen_action"]
        )
        assert layer.mlp is before[f"language_model.model.layers.{index}.mlp"]


def test_reject_double_install_and_invalid_scope():
    model = _transformer()
    with pytest.raises(ValueError, match="scope"):
        install_video_fp8(model, "everything")
    install_video_fp8(model, "mlp")
    with pytest.raises(ValueError, match="unquantized"):
        install_video_fp8(model, "mlp")


@pytest.mark.parametrize("failure", ["fused", "nonfinite", "misaligned", "bias", "float32", "compiled"])
def test_failed_install_does_not_partially_replace_modules(failure):
    model = _transformer()
    last = model.language_model.model.layers[-1]
    if failure == "fused":
        last.mlp_moe_gen.gate_up_proj = nn.Linear(32, 128, bias=False)
    elif failure == "nonfinite":
        with torch.no_grad():
            last.mlp_moe_gen.down_proj.weight[0, 0] = float("nan")
    elif failure == "misaligned":
        last.mlp_moe_gen.up_proj = nn.Linear(32, 63, bias=False).to(torch.bfloat16)
    elif failure == "bias":
        last.mlp_moe_gen.up_proj = nn.Linear(32, 64).to(torch.bfloat16)
    elif failure == "float32":
        last.mlp_moe_gen.up_proj = last.mlp_moe_gen.up_proj.float()
    else:
        last._orig_mod = nn.Identity()
    before = dict(model.named_modules())
    with pytest.raises(ValueError):
        install_video_fp8(model, "mlp")
    assert dict(model.named_modules()) == before


def test_cpu_forward_does_not_silently_dequantize():
    linear = PwmFP8Linear(torch.zeros(32, 16, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="CUDA"):
        linear(torch.zeros(7, 16, dtype=torch.bfloat16))
    linear.train()
    with pytest.raises(ValueError, match="inference-only"):
        linear(torch.zeros(7, 16, dtype=torch.bfloat16))


def test_scaled_mm_fake_shape_dtype_and_alignment():
    with FakeTensorMode():
        x = torch.empty(16, 32, dtype=torch.float8_e4m3fn)
        weight = torch.empty(64, 32, dtype=torch.float8_e4m3fn)
        scale = torch.ones(1)
        output = pwm_fp8_scaled_mm(x, weight, scale, scale)
        assert output.shape == (16, 64)
        assert output.dtype == torch.bfloat16
        with pytest.raises(ValueError, match="divisible by 16"):
            pwm_fp8_scaled_mm(x[:7], weight, scale, scale)


@pytest.mark.parametrize("scope,per_layer", [("mlp", 2), ("attention", 2), ("all", 4)])
def test_install_quantizes_fused_video_projections_once(scope, per_layer):
    model = _transformer()
    for layer in model.language_model.model.layers:
        layer.configure_fusions(gate_up=True, qkv=True)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    report = install_video_fp8(model, scope)
    assert report["module_count"] == 2 * per_layer
    for record in report["modules"]:
        linear = model.get_submodule(record["name"])
        source = before[record["name"] + ".weight"]
        expected, scale = quantize_fp8_per_tensor(source)
        torch.testing.assert_close(linear.weight_fp8.float(), expected.float(), atol=0, rtol=0)
        torch.testing.assert_close(linear.weight_scale, scale, atol=0, rtol=0)
        assert any(part in record["name"] for part in ("gate_up_proj", "down_proj", "qkv_proj", "o_proj_moe_gen"))
