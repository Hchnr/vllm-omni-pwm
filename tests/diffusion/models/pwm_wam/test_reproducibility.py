# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validation controls must seed all RNGs and report their effective settings."""

import random

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.pwm_wam.reproducibility import configure_validation

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


def test_workspace_must_be_configured_before_validation(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(ValueError, match="before startup"):
        configure_validation(42)


def test_repeat_seed_and_actual_settings(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    old = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    try:
        settings = configure_validation(42)
        first = (random.random(), np.random.rand(), torch.rand(4))
        configure_validation(42)
        second = (random.random(), np.random.rand(), torch.rand(4))
        assert first[:2] == second[:2]
        torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
        configure_validation(43)
        assert not torch.equal(first[2], torch.rand(4))
        assert settings["deterministic_algorithms"] is True
        assert settings["deterministic_warn_only"] is False
        assert settings["cudnn_benchmark"] is False
        assert settings["matmul_tf32"] is False
    finally:
        torch.use_deterministic_algorithms(old[0], warn_only=old[1])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = old[2:4]
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old[4:6]
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
