# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run in the optional Cosmos environment; no Omni dependency is imported."""

import importlib.util
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]

pytest.importorskip("cosmos_framework")
from cosmos_framework.model.generator.mot.inference_small_action_memory import SmallActionMemoryState
from cosmos_framework.model.generator.mot.small_action_optimization_test import make_layer, make_pack
from cosmos_framework.model.generator.mot.small_action_tower import small_layer_forward

spec = importlib.util.spec_from_file_location(
    "pwm_cache", Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/pwm_wam/und_cache.py"
)
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


@pytest.mark.parametrize("compiled", [False, True])
def test_projected_cache_matches_uncached_with_changing_noisy_streams(compiled):
    torch.manual_seed(19)
    layer, shape = make_layer()
    layer.eval()
    run = torch.compile(cache._forward, backend="eager", fullgraph=True) if compiled else cache._forward
    memory = SmallActionMemoryState(1)
    und = [torch.randn(2, 64), torch.randn(3, 64)]
    with torch.inference_mode():
        for _ in range(3):
            samples = [(u, torch.randn(4, 64), torch.randn(2, 32)) for u in und]
            pack = make_pack(samples, shape)
            expected = small_layer_forward(layer, pack)[0]
            actual, _, stored = run(layer, pack, None, None, memory_value=memory.read_for_layer(0))
            if stored is not None:
                memory.write_for_layer(0, stored)
            for key in ("causal_seq", "full_only_seq", "action_seq"):
                torch.testing.assert_close(actual[key], expected[key], rtol=2e-5, atol=2e-6)


def test_projected_cache_rejects_training():
    layer, shape = make_layer()
    pack = make_pack([(torch.randn(2, 64), torch.randn(3, 64), torch.randn(2, 32))], shape)
    with pytest.raises(ValueError, match="eval"):
        cache._forward(layer, pack, None, None)
