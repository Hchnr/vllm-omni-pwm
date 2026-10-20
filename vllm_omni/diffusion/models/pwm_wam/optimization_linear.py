# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Experimental VIDEO-only FP8 linears for native PWM inference.

Weights use one E4M3 scale per matrix (default) or per output row, calculated
once during installation. Activations use matching dynamic per-tensor or
per-token-row scales on each forward. The scaled GEMM requests FP32
accumulation (``use_fast_accum=False``) and returns BF16. Quantization and row
padding are part of request latency, not a free preprocessing step. Pure Torch
activation quantization stays visible to Inductor; only the scaled GEMM is an
opaque custom op. This is not FP8 attention: Q/K/V passed to FA3 remain BF16.

Install after HF loading and before compiling layers. Checkpoint loading and
saving through the transformed module are intentionally unsupported. ACTION,
UND, norms, embeddings, output adapters and the VAE remain unchanged. VIDEO
QKV/gate-up fusion may be installed first; quantize its packed matrices once.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8_DTYPE).max
_ALIGNMENT = 16


def quantize_fp8_per_tensor(value: Tensor) -> tuple[Tensor, Tensor]:
    """Return E4M3 values and a FP32 dequantization scale (CPU-testable).

    No data-dependent host synchronization is used. A zero tensor gets scale
    one. Nonfinite inputs are not silently replaced; load-time weight checks
    reject them and runtime output validation remains responsible for requests.
    """
    full = value.float()
    maximum = full.abs().amax()
    scale = torch.where(maximum > 0, maximum / _FP8_MAX, torch.ones_like(maximum)).reshape(1)
    quantized = (full / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE)
    return quantized, scale


def quantize_fp8_per_row(value: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize a matrix independently per row, returning FP32 scales [rows, 1].

    Weight rows correspond to output features; activation rows to tokens. Zero
    rows receive scale one, including the alignment padding added in forward.
    """
    if value.ndim != 2 or any(size < 1 for size in value.shape):
        raise ValueError("PWM FP8 row quantization requires a nonempty matrix")
    full = value.float()
    maximum = full.abs().amax(dim=1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / _FP8_MAX, torch.ones_like(maximum))
    quantized = (full / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE)
    return quantized, scale


def _validate_scaling(scaling: str) -> None:
    if scaling not in ("tensor", "row"):
        raise ValueError("PWM FP8 scaling must be tensor or row")


def _validate_mm_shape(x: Tensor, weight: Tensor) -> None:
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("PWM FP8 GEMM requires [rows, in_features] and [out_features, in_features]")
    if any(size < 1 or size % _ALIGNMENT for size in (*x.shape, weight.shape[0])):
        raise ValueError("PWM FP8 GEMM requires positive dimensions divisible by 16")


def _scaled_mm_weight_scale(x: Tensor, weight: Tensor, input_scale: Tensor, weight_scale: Tensor) -> Tensor:
    for scale in (input_scale, weight_scale):
        if scale.dtype != torch.float32 or scale.device != x.device or not scale.is_contiguous():
            raise ValueError("PWM FP8 GEMM requires contiguous FP32 scales on the input device")
    if input_scale.shape == (1,) and weight_scale.shape == (1,):
        return weight_scale
    if input_scale.shape == (x.shape[0], 1) and weight_scale.shape == (weight.shape[0], 1):
        # _scaled_mm consumes [M, 1] for A and [1, N] for column-major B.
        return weight_scale.t()
    raise ValueError("PWM FP8 scales must both be [1], or activation [M, 1] and weight [N, 1]")


@torch.library.custom_op("vllm_omni::pwm_fp8_scaled_mm", mutates_args=())
def pwm_fp8_scaled_mm(x: Tensor, weight: Tensor, input_scale: Tensor, weight_scale: Tensor) -> Tensor:
    """CUDA-only E4M3 GEMM with BF16 output and no fast accumulation."""
    _validate_mm_shape(x, weight)
    if x.device.type != "cuda" or weight.device != x.device:
        raise ValueError("PWM FP8 execution requires CUDA inputs and weights on the same device")
    if x.dtype != _FP8_DTYPE or weight.dtype != _FP8_DTYPE:
        raise ValueError("PWM FP8 GEMM requires E4M3 input and weight tensors")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("PWM FP8 GEMM requires contiguous row-major inputs and weights")
    mm_weight_scale = _scaled_mm_weight_scale(x, weight, input_scale, weight_scale)
    # A is row major; B must be column major for cuBLASLt's FP8 scaled GEMM.
    return torch._scaled_mm(
        x,
        weight.t(),
        input_scale,
        mm_weight_scale,
        out_dtype=torch.bfloat16,
        use_fast_accum=False,
    )


@pwm_fp8_scaled_mm.register_fake
def _pwm_fp8_scaled_mm_fake(x: Tensor, weight: Tensor, input_scale: Tensor, weight_scale: Tensor) -> Tensor:
    _validate_mm_shape(x, weight)
    _scaled_mm_weight_scale(x, weight, input_scale, weight_scale)
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.bfloat16)


class PwmFP8Linear(nn.Module):
    """Frozen bias-free linear; quantized weights are buffers, never parameters.

    Construction on CPU supports deterministic quantization/routing tests.
    Forward requires CUDA and BF16 activations. M is zero-padded to a multiple
    of 16; K/N must already be aligned. Numerical comparison must include this
    activation quantization, not only the weight reconstruction statistics.
    """

    def __init__(self, weight: Tensor, *, scaling: str = "tensor"):
        super().__init__()
        _validate_scaling(scaling)
        self.scaling = scaling
        if weight.ndim != 2 or any(size < 1 or size % _ALIGNMENT for size in weight.shape):
            raise ValueError("PWM FP8 linear in/out features must be positive multiples of 16")
        if weight.dtype != torch.bfloat16 or weight.device.type not in ("cpu", "cuda"):
            raise ValueError("PWM FP8 conversion requires loaded BF16 weights on CPU or CUDA")
        started = time.perf_counter()
        if not torch.isfinite(weight).all().item():
            raise ValueError("PWM FP8 conversion rejects nonfinite weights")
        with torch.no_grad():
            quantizer = quantize_fp8_per_tensor if scaling == "tensor" else quantize_fp8_per_row
            quantized, scale = quantizer(weight.detach())
            delta = quantized.float() * scale - weight.float()
            max_abs = delta.abs().amax().item()
            rmse = delta.square().mean().sqrt().item()
            rms = weight.float().square().mean().sqrt().item()
            scale_value = scale.item() if scaling == "tensor" else None
            scale_min = scale_value if scaling == "tensor" else scale.amin().item()
            scale_max = scale_value if scaling == "tensor" else scale.amax().item()
        self.register_buffer("weight_fp8", quantized.contiguous())
        self.register_buffer("weight_scale", scale)
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.conversion_metrics = {
            "shape": list(weight.shape),
            "scale": scale_value,
            "scaling": scaling,
            "scale_shape": list(scale.shape),
            "scale_min": scale_min,
            "scale_max": scale_max,
            "weight_max_abs_error": max_abs,
            "weight_rmse": rmse,
            "weight_relative_rmse": rmse / rms if rms else 0.0,
            # Scalar reads above synchronize startup quantization/diagnostics.
            "conversion_and_diagnostics_seconds": time.perf_counter() - started,
            "bf16_weight_bytes": weight.numel() * weight.element_size(),
            "fp8_weight_and_scale_bytes": quantized.numel() * quantized.element_size() + scale.numel() * 4,
        }
        self.eval()

    @classmethod
    def from_linear(cls, linear: nn.Linear, *, scaling: str = "tensor") -> PwmFP8Linear:
        if type(linear) is not nn.Linear or linear.bias is not None:
            raise ValueError("PWM FP8 conversion supports ordinary bias-free nn.Linear only")
        return cls(linear.weight, scaling=scaling)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            raise ValueError("PWM FP8 linear is inference-only; call eval()")
        if x.device.type != "cuda" or x.device != self.weight_fp8.device:
            raise ValueError("PWM FP8 execution requires CUDA input on the weight device")
        if x.dtype != torch.bfloat16 or x.ndim < 2 or x.shape[-1] != self.in_features:
            raise ValueError("PWM FP8 linear requires BF16 input [..., in_features]")
        output_shape = (*x.shape[:-1], self.out_features)
        flat = x.reshape(-1, self.in_features)
        rows = flat.shape[0]
        if rows == 0:
            raise ValueError("PWM FP8 linear does not accept empty activations")
        padding = (-rows) % _ALIGNMENT
        if padding:
            flat = F.pad(flat, (0, 0, 0, padding))
        quantizer = quantize_fp8_per_tensor if self.scaling == "tensor" else quantize_fp8_per_row
        quantized, input_scale = quantizer(flat)
        result = pwm_fp8_scaled_mm(quantized.contiguous(), self.weight_fp8, input_scale, self.weight_scale)
        return result[:rows].reshape(output_shape)


def install_video_fp8(transformer: nn.Module, scope: str = "mlp", *, scaling: str = "tensor") -> dict[str, object]:
    """Replace selected VIDEO linears, with validation before any mutation.

    ``mlp`` converts gate/up/down in ``mlp_moe_gen``; ``attention`` converts
    Q/K/V/O in ``self_attn`` with suffix ``_moe_gen``; ``all`` selects both.
    Already-fused gate/up and QKV matrices are supported and count as one
    quantized linear each. Configure fusion before installing FP8.
    ``scaling="row"`` uses one scale per weight output row and activation token;
    the default ``"tensor"`` retains the original per-matrix quantization.
    The returned report describes one-time weight error/cost. Request-time
    activation quantization/padding cost requires measured end-to-end ablation.
    """
    _validate_scaling(scaling)
    if scope not in ("mlp", "attention", "all"):
        raise ValueError("PWM VIDEO FP8 scope must be mlp, attention, or all")
    try:
        layers = transformer.language_model.model.layers
    except AttributeError as exc:
        raise ValueError("PWM VIDEO FP8 requires the native three-tower layer layout") from exc
    if not isinstance(layers, nn.ModuleList) or not len(layers):
        raise ValueError("PWM VIDEO FP8 requires a nonempty, uncompiled ModuleList")
    selected: list[tuple[str, nn.Module, str, nn.Linear]] = []
    for index, layer in enumerate(layers):
        if hasattr(layer, "_orig_mod"):
            raise ValueError("Install PWM VIDEO FP8 before torch.compile")
        for owner_name, leaves in (
            ("mlp_moe_gen", ("gate_proj", "up_proj", "down_proj") if scope in ("mlp", "all") else ()),
            (
                "self_attn",
                ("q_proj_moe_gen", "k_proj_moe_gen", "v_proj_moe_gen", "o_proj_moe_gen")
                if scope in ("attention", "all")
                else (),
            ),
        ):
            if not leaves:
                continue
            owner = getattr(layer, owner_name, None)
            if not isinstance(owner, nn.Module):
                raise ValueError(f"Missing native VIDEO module: layer {index}.{owner_name}")
            fused_name = "gate_up_proj" if owner_name == "mlp_moe_gen" else "qkv_proj_moe_gen"
            if getattr(owner, fused_name, None) is not None:
                original_names = leaves[:-1]
                if any(getattr(owner, name, None) is not None for name in original_names):
                    raise ValueError("PWM VIDEO FP8 rejects ambiguous fused and unfused projection ownership")
                leaves = (fused_name, leaves[-1])
            for leaf in leaves:
                linear = getattr(owner, leaf, None)
                path = f"language_model.model.layers.{index}.{owner_name}.{leaf}"
                if type(linear) is not nn.Linear or linear.bias is not None:
                    raise ValueError(f"PWM VIDEO FP8 requires an unquantized bias-free nn.Linear: {path}")
                if linear.weight.dtype != torch.bfloat16:
                    raise ValueError(f"PWM VIDEO FP8 requires loaded BF16 weights: {path}")
                if any(size < 1 or size % _ALIGNMENT for size in linear.weight.shape):
                    raise ValueError(f"PWM VIDEO FP8 dimensions must be positive multiples of 16: {path}")
                selected.append((path, owner, leaf, linear))
    # Build every replacement before mutation; a later invalid/nonfinite weight
    # cannot leave an installed mixture of FP8 and BF16 linears after failure.
    replacements = [
        (path, owner, leaf, PwmFP8Linear.from_linear(linear, scaling=scaling)) for path, owner, leaf, linear in selected
    ]
    for _, owner, leaf, replacement in replacements:
        setattr(owner, leaf, replacement)
    modules = [{"name": path, **replacement.conversion_metrics} for path, _, _, replacement in replacements]
    return {
        "scope": scope,
        "format": "float8_e4m3fn",
        "scaling": scaling,
        "weight_scaling": "static_per_tensor" if scaling == "tensor" else "static_per_output_row",
        "activation_scaling": "dynamic_per_tensor" if scaling == "tensor" else "dynamic_per_token_row",
        "output_dtype": "bfloat16",
        "use_fast_accum": False,
        "activation_quantization_in_request_timing": True,
        "activation_row_alignment": _ALIGNMENT,
        "activation_row_padding": "zero-pad M to 16, then slice output",
        "activation_cost_per_linear_call": "amax reduction, scale/cast to E4M3, optional row padding and output slice",
        "module_count": len(modules),
        "conversion_and_diagnostics_seconds": math.fsum(
            module["conversion_and_diagnostics_seconds"] for module in modules
        ),
        "modules": modules,
    }
