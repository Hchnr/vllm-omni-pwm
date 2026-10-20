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


def install_pwm_vae_fastpath(vae, *, scope: str = "all") -> dict[str, object]:
    """Opt into Wan convolution/nearest-upsample primitives, preserving PWM RMS.

    The generic Wan installer also replaces/inlines RMSNorm with FP32 reduction,
    which is incompatible with this checkpoint's BF16 normalization contract.
    Only two independently usable forwards are installed here. Encoder/decoder
    blocks keep their original cache ownership, ordering and norm calls. The
    shared causal-convolution primitive probes alternate cuDNN padding once per
    shape and falls back when its numerical guard declines the optimization.

    Install after ``preserve_checkpoint_rms``. This opt-in is independent from
    the generic VAE fast-path flags; it does not enable channels-last, fused
    norm/SiLU, changed precision, decoder preallocation or spatial parallelism.
    """
    from diffusers import AutoencoderKLWan
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d, WanUpsample

    if scope not in ("encoder", "decoder", "all"):
        raise ValueError("PWM VAE fastpath scope must be encoder, decoder, or all")
    if not isinstance(vae, AutoencoderKLWan):
        raise ValueError("PWM VAE fastpath requires a diffusers AutoencoderKLWan")
    existing = getattr(vae, "_pwm_vae_fastpath_report", None)
    if existing is not None:
        if existing["scope"] != scope:
            raise ValueError("PWM VAE fastpath is already installed with a different scope")
        return existing
    for name in ("_vllm_omni_wan_fastpath_report", "_vllm_omni_wan_encode_fastpath_report"):
        if getattr(getattr(vae, name, None), "installed", False):
            raise ValueError("PWM VAE fastpath cannot be combined with generic Wan norm fastpaths")
    norms = [module for module in vae.modules() if isinstance(module, WanRMS_norm)]
    if not norms or any(getattr(module.forward, "__func__", None) is not _checkpoint_rms_forward for module in norms):
        raise ValueError("Run preserve_checkpoint_rms before installing the PWM VAE fastpath")

    # Import only for explicit opt-in. The shared implementations retain their
    # own CPU/unsupported-shape fallbacks and do not mutate the norm modules.
    from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath.forwards import (
        causal_conv_forward,
        upsample_forward,
    )

    components = []
    if scope in ("encoder", "all"):
        components.extend(("encoder", "quant_conv"))
    if scope in ("decoder", "all"):
        components.extend(("decoder", "post_quant_conv"))
    selected = []
    for component in components:
        for relative_name, module in getattr(vae, component).named_modules():
            if type(module) is WanCausalConv3d:
                replacement = causal_conv_forward
            elif type(module) is WanUpsample:
                if module.mode not in ("nearest", "nearest-exact"):
                    raise ValueError("PWM VAE fastpath requires nearest-mode WanUpsample")
                replacement = upsample_forward
            else:
                continue
            path = component + ("." + relative_name if relative_name else "")
            current = module.forward
            if not isinstance(current, MethodType) or current.__self__ is not module:
                raise ValueError(f"PWM VAE fastpath refuses an unrecognized forward: {path}")
            if current.__func__ is not type(module).forward:
                raise ValueError(f"PWM VAE fastpath refuses to overwrite a custom forward: {path}")
            selected.append((path, module, replacement))
    if not selected:
        raise ValueError("PWM VAE fastpath found no supported convolution or upsample modules")
    for _, module, replacement in selected:
        module.forward = MethodType(replacement, module)
    report = {
        "scope": scope,
        "installed": True,
        "normalization": "checkpoint_input_dtype_unchanged",
        "norm_modules_preserved": len(norms),
        "conv_count": sum(type(module) is WanCausalConv3d for _, module, _ in selected),
        "upsample_count": sum(type(module) is WanUpsample for _, module, _ in selected),
        "modules": [name for name, _, _ in selected],
        "shared_primitives": ["causal_conv_forward", "upsample_forward"],
        "first_shape_probe_in_warmup": True,
    }
    vae._pwm_vae_fastpath_report = report
    return report
