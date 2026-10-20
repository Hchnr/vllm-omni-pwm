# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reject silently ignored optimization switches before loading weights."""

import pytest

from vllm_omni.diffusion.models.pwm_wam.optimization_config import PwmOptimizationConfig

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


@pytest.mark.parametrize(
    "options",
    [
        [],
        {"fuse_qkv": 1},
        {"cuda_graph": "false"},
        {"fp8_scope": "action"},
        {"graph_max_entries": 0},
        {"graph_max_entries": True},
        {"fuse_qvk": True},
        {"vae_fastpath": "rms"},
    ],
)
def test_reject_unsupported_or_mistyped_options(options):
    with pytest.raises(ValueError):
        PwmOptimizationConfig.from_mapping(options)


def test_explicit_false_stays_disabled():
    assert PwmOptimizationConfig.from_mapping({"fuse_qkv": False}) == PwmOptimizationConfig()
