# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PWM VAE fastpath keeps checkpoint norms and temporal encode/decode semantics."""

from types import MethodType

import pytest
import torch
from diffusers import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm

from vllm_omni.diffusion.models.pwm_wam.vae import (
    _checkpoint_rms_forward,
    install_pwm_vae_fastpath,
    preserve_checkpoint_rms,
)

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


def _small_vae():
    torch.manual_seed(1020)
    return AutoencoderKLWan(
        base_dim=4,
        decoder_base_dim=4,
        z_dim=4,
        dim_mult=[1, 2],
        num_res_blocks=1,
        temperal_downsample=[False],
        latents_mean=[0.0] * 4,
        latents_std=[1.0] * 4,
        scale_factor_temporal=1,
        scale_factor_spatial=2,
    ).eval()


@pytest.mark.parametrize("scope", ["encoder", "decoder", "all"])
@torch.inference_mode()
def test_fastpath_preserves_norms_and_full_small_vae(scope):
    vae = _small_vae()
    count = preserve_checkpoint_rms(vae)
    before = {name: module.forward for name, module in vae.named_modules()}
    generator = torch.Generator().manual_seed(1020)
    pixels = torch.randn(1, 3, 5, 8, 8, generator=generator)
    expected_latent = vae.encode(pixels).latent_dist.mode()
    expected_video = vae.decode(expected_latent).sample
    report = install_pwm_vae_fastpath(vae, scope=scope)
    assert report["norm_modules_preserved"] == count
    assert report["conv_count"] > 0
    assert (
        all(
            name.startswith((scope, "quant_conv" if scope == "encoder" else "post_quant_conv"))
            for name in report["modules"]
        )
        if scope != "all"
        else True
    )
    for name, module in vae.named_modules():
        if isinstance(module, WanRMS_norm):
            assert module.forward.__func__ is _checkpoint_rms_forward
            assert module.forward == before[name]
    assert install_pwm_vae_fastpath(vae, scope=scope) is report
    actual_latent = vae.encode(pixels).latent_dist.mode()
    actual_video = vae.decode(actual_latent).sample
    torch.testing.assert_close(actual_latent, expected_latent, rtol=0, atol=0)
    torch.testing.assert_close(actual_video, expected_video, rtol=0, atol=0)


def test_requires_checkpoint_norm_and_validates_before_mutation():
    vae = _small_vae()
    with pytest.raises(ValueError, match="preserve_checkpoint_rms"):
        install_pwm_vae_fastpath(vae)
    preserve_checkpoint_rms(vae)
    vae.post_quant_conv.forward = MethodType(lambda self, x, cache_x=None: x, vae.post_quant_conv)
    before = {name: module.forward for name, module in vae.named_modules()}
    with pytest.raises(ValueError, match="custom forward"):
        install_pwm_vae_fastpath(vae)
    assert {name: module.forward for name, module in vae.named_modules()} == before


def test_preserved_bf16_rms_matches_checkpoint_arithmetic():
    vae = _small_vae().to(torch.bfloat16)
    preserve_checkpoint_rms(vae)
    norm = next(module for module in vae.modules() if isinstance(module, WanRMS_norm))
    generator = torch.Generator().manual_seed(43)
    value = torch.randn(2, norm.gamma.shape[0], 3, 4, 4, generator=generator).to(torch.bfloat16)
    expected = torch.nn.functional.normalize(value, dim=1) * norm.scale * norm.gamma + norm.bias
    install_pwm_vae_fastpath(vae)
    torch.testing.assert_close(norm(value), expected, rtol=0, atol=0)
