# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA regression for CPU-scalar UniPC checkpoint compatibility.

Set PWM_COSMOS_ROOT to run against the independent reference solver source.
"""

import importlib.util
import os
from pathlib import Path

import pytest
import torch

from tests.helpers.mark import hardware_marks

pytest.importorskip("vllm")
from vllm_omni.diffusion.models.schedulers.scheduling_flow_unipc_multistep import FlowUniPCMultistepScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]


@pytest.mark.parametrize("steps", [5, 10, 35])
def test_cuda_unipc_matches_reference_cpu_coefficients(steps):
    root = os.environ.get("PWM_COSMOS_ROOT")
    if not root:
        pytest.skip("Set PWM_COSMOS_ROOT for the independent reference comparison")
    path = Path(root) / "cosmos_framework/model/generator/diffusion/samplers/fm_solvers_unipc.py"
    spec = importlib.util.spec_from_file_location("reference_unipc", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = module.FlowUniPCMultistepScheduler()
    actual = FlowUniPCMultistepScheduler(use_cpu_coefficients=True)
    for scheduler in (reference, actual):
        scheduler.set_timesteps(steps, device="cuda", shift=5.0)
    torch.testing.assert_close(actual.timesteps, reference.timesteps, rtol=0, atol=0)
    expected = torch.linspace(-3, 3, 512, device="cuda").reshape(1, -1)
    observed = expected.clone()
    for timestep in reference.timesteps:
        velocity = (expected[0].sin() * 0.13).to(torch.bfloat16)
        expected = reference.step(velocity, timestep, expected, return_dict=False)[0]
        observed = actual.step(velocity, timestep, observed, return_dict=False)[0]
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
