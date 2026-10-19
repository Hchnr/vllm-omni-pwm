# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reject unaccepted native configurations before touching a checkpoint."""

import pytest
import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.pwm_wam.pipeline_pwm_wam_native import PwmWamNativePipeline

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


@pytest.mark.parametrize(
    "model_config,message",
    [
        ({"pwm_attention_backend": "flash3"}, "SDPA only"),
        ({"pwm_compile": True}, "compilation"),
        ({"pwm_precision": "float32"}, "BF16 only"),
        ({"pwm_reuse_und": True}, "Legacy UND cache"),
    ],
)
def test_unaccepted_configuration_rejected_before_weight_loading(model_config, message):
    config = OmniDiffusionConfig(model="/nonexistent/checkpoint", enforce_eager=True, model_config=model_config)
    with pytest.raises(ValueError, match=message):
        PwmWamNativePipeline(od_config=config)


def test_generic_dtype_cannot_silently_override_native_precision():
    config = OmniDiffusionConfig(model="/nonexistent/checkpoint", enforce_eager=True, dtype=torch.float32)
    with pytest.raises(ValueError, match="BF16 only"):
        PwmWamNativePipeline(od_config=config)


def test_experimental_engine_forwards_explicit_backend_options(monkeypatch):
    from vllm_omni.diffusion.models.pwm_wam import pipeline_pwm_wam_native as pipeline

    captured = {}

    class RuntimeStub(torch.nn.Module):
        def __init__(self, checkpoint, **kwargs):
            super().__init__()
            captured.update(kwargs)

    monkeypatch.setattr(pipeline, "PwmWamRuntime", RuntimeStub)
    config = OmniDiffusionConfig(
        model="/nonexistent/checkpoint",
        enforce_eager=True,
        model_config={
            "pwm_experimental_backend": True,
            "pwm_compile": True,
            "pwm_attention_backend": "flash3",
            "pwm_native_und_cache": True,
            "pwm_vae_path": "vae",
            "pwm_tokenizer_path": "tokenizer",
        },
    )
    PwmWamNativePipeline(od_config=config)
    assert captured["compile_layers"] is True
    assert captured["cache_und"] is True
    assert captured["attention_backend"] == "flash3"
