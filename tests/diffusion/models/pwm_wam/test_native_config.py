# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reject unaccepted native configurations before touching a checkpoint."""

import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.pwm_wam.pipeline_pwm_wam_native import PwmWamNativePipeline

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


@pytest.mark.parametrize(
    "model_config,message",
    [
        ({"pwm_attention_backend": "flash3"}, "SDPA only"),
        ({"pwm_compile": True}, "compilation"),
        ({"pwm_precision": "float32"}, "BF16 only"),
        ({"pwm_reuse_und": True}, "not yet implemented"),
    ],
)
def test_unaccepted_configuration_rejected_before_weight_loading(model_config, message):
    config = OmniDiffusionConfig(model="/nonexistent/checkpoint", enforce_eager=True, model_config=model_config)
    with pytest.raises(ValueError, match=message):
        PwmWamNativePipeline(od_config=config)
