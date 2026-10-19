# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Preserve the checkpoint's Wan2.2 VAE normalization arithmetic."""

from types import MethodType

import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm


def _checkpoint_rms_forward(self, value):
    # The original checkpoint runtime normalizes in the input dtype. Recent
    # Diffusers upcasts this reduction to FP32, changing BF16 conditioning.
    normalized = F.normalize(value, dim=1 if self.channel_first else -1)
    return normalized * self.scale * self.gamma + self.bias


def preserve_checkpoint_rms(vae):
    count = 0
    for module in vae.modules():
        if isinstance(module, WanRMS_norm):
            module.forward = MethodType(_checkpoint_rms_forward, module)
            count += 1
    if count == 0:
        raise ValueError("Expected Wan RMSNorm modules in PWM VAE")
    return count
