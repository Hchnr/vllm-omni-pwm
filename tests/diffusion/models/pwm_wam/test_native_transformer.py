# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native three-tower layer compared against the independent Cosmos reference."""

import importlib.util
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]
helpers = pytest.importorskip("cosmos_framework.model.generator.mot.small_action_optimization_test")
reference = pytest.importorskip("cosmos_framework.model.generator.mot.small_action_tower")

spec = importlib.util.spec_from_file_location(
    "pwm_native", Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/pwm_wam/transformer_pwm_wam.py"
)
assert spec is not None and spec.loader is not None
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def test_native_three_tower_matches_reference():
    torch.manual_seed(1019)
    expected_layer, shape = helpers.make_layer()
    actual_layer = native.PwmThreeTowerLayer(
        width=64, intermediate=128, action_width=32, action_intermediate=64, heads=4, kv_heads=2, dim=16
    )
    actual_layer.load_state_dict(expected_layer.state_dict(), strict=True)
    expected_layer.eval()
    actual_layer.eval()
    with torch.inference_mode():
        for _ in range(3):
            pack = helpers.make_pack([(torch.randn(3, 64), torch.randn(5, 64), torch.randn(2, 32))], shape)
            expected = reference.small_layer_forward(expected_layer, pack)[0]
            streams = tuple(pack[key] for key in ("causal_seq", "full_only_seq", "action_seq"))
            actual = actual_layer(streams, pack["_small_rope"], pack["_small_action_rope"])
            for key, tensor in zip(("causal_seq", "full_only_seq", "action_seq"), actual):
                torch.testing.assert_close(tensor, expected[key], rtol=2e-5, atol=2e-6)
