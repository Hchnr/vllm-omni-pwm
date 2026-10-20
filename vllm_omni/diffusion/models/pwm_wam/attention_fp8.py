# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Experimental SM90 bundled-FA3 FP8 attention for PWM's VIDEO queries.

The FA3 descale contract is [batch, KV heads] for Q, K and V, including GQA.
Thus each group of four Q heads shares a scale; K and V each have eight
independent scales. Scales are dynamic over tokens, head dimensions and (for
Q) the four heads in a GQA group. No rotation or softmax-scale adjustment is
introduced. E4M3 Q/K/V produce BF16 output.

This prototype supports one unpadded sequence, Q32/KV8 and D128, noncausal
attention only. Quantization and copies remain in request timing. Pure Torch
quantization is visible to Inductor; the bundled FA3 call has a custom-op fake
implementation for fullgraph capture. The caller opts in only for VIDEO.

Long single-split FP8 reductions have a known constant-V accuracy issue in
some FA3 versions. The default follows the framework's conservative policy:
next power of two of ceil(KV tokens / 2560). Explicit num_splits=1 is a
numerical diagnostic, not the default. Compare BF16 at the same split count
when attributing any speed or accuracy change to FP8.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from torch import Tensor

_FP8 = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8).max
_QUERY_HEADS = 32
_KV_HEADS = 8
_HEAD_DIM = 128


def resolve_fp8_attention_splits(max_seqlen_k: int, num_splits: int = 0) -> int:
    """Resolve zero to the same guarded split policy as FlashAttentionImpl."""
    if isinstance(max_seqlen_k, bool) or not isinstance(max_seqlen_k, int) or max_seqlen_k < 1:
        raise ValueError("PWM FP8 attention requires a positive KV sequence length")
    if isinstance(num_splits, bool) or not isinstance(num_splits, int) or not 0 <= num_splits <= 256:
        raise ValueError("PWM FP8 attention num_splits must be an integer in [0, 256]")
    if num_splits:
        return num_splits
    required = (max_seqlen_k + 2559) // 2560
    result = 1 << (required - 1).bit_length()
    if result > 256:
        raise ValueError("PWM FP8 attention length exceeds the guarded FA3 split limit")
    return result


def fp8_attention_info(max_seqlen_k: int, num_splits: int = 0) -> dict[str, object]:
    """Report an immutable shape-specific kernel/quantization configuration."""
    return {
        "backend": "bundled_fa3",
        "qkv_dtype": "float8_e4m3fn",
        "output_dtype": "bfloat16",
        "device_family": "SM90",
        "batch_size": 1,
        "query_heads": _QUERY_HEADS,
        "kv_heads": _KV_HEADS,
        "head_dim": _HEAD_DIM,
        "q_scale_granularity": "per_GQA_group_of_4_query_heads",
        "kv_scale_granularity": "per_KV_head",
        "descale_shape": [1, _KV_HEADS],
        "max_seqlen_k": max_seqlen_k,
        "requested_num_splits": num_splits,
        "resolved_num_splits": resolve_fp8_attention_splits(max_seqlen_k, num_splits),
        "softmax_scale": _HEAD_DIM**-0.5,
        "causal": False,
        "quantization_and_copy_cost_in_request_timing": True,
    }


def quantize_fp8_gqa_groups(value: Tensor, kv_heads: int) -> tuple[Tensor, Tensor]:
    """CPU-testable [tokens, heads, dim] E4M3 quantization with [1, kv_heads] scale.

    Adjacent query heads share a KV head in FA3's GQA ordering. This function
    groups that head axis without interleaving different KV groups. Passing
    heads == kv_heads gives ordinary per-head K/V quantization.
    """
    if value.ndim != 3 or any(size < 1 for size in value.shape) or not value.is_floating_point():
        raise ValueError("PWM FP8 group quantization requires nonempty floating [tokens, heads, dim]")
    if isinstance(kv_heads, bool) or not isinstance(kv_heads, int) or kv_heads < 1 or value.shape[1] % kv_heads:
        raise ValueError("PWM FP8 query heads must be divisible by positive KV head count")
    groups = value.shape[1] // kv_heads
    full = value.float().reshape(value.shape[0], kv_heads, groups, value.shape[2])
    maximum = full.abs().amax(dim=(0, 2, 3), keepdim=True)
    scale = torch.where(maximum > 0, maximum / _FP8_MAX, torch.ones_like(maximum))
    quantized = (full / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8)
    return quantized.reshape(value.shape).contiguous(), scale.reshape(1, kv_heads).contiguous()


def _validate_attention_shapes(q: Tensor, k: Tensor, v: Tensor) -> None:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("PWM FP8 attention requires [tokens, heads, dim] inputs")
    if q.shape[1:] != (_QUERY_HEADS, _HEAD_DIM) or k.shape[1:] != (_KV_HEADS, _HEAD_DIM) or v.shape != k.shape:
        raise ValueError("PWM FP8 attention supports only Q32/KV8, D128, with matching K/V layouts")
    if q.shape[0] < 1 or k.shape[0] < 1:
        raise ValueError("PWM FP8 attention requires positive Q/KV lengths")
    if q.device != k.device or q.device != v.device:
        raise ValueError("PWM FP8 attention Q/K/V must share a device")


def _validate_metadata(q: Tensor, cu_q: Tensor, cu_k: Tensor) -> None:
    for cumulative in (cu_q, cu_k):
        if cumulative.shape != (2,) or cumulative.dtype != torch.int32 or cumulative.device != q.device:
            raise ValueError("PWM FP8 attention requires B1 int32 cu_seqlens [2] on the Q device")
        if not cumulative.is_contiguous():
            raise ValueError("PWM FP8 attention requires contiguous cu_seqlens")


def _validate_quantized(
    q: Tensor, k: Tensor, v: Tensor, cu_q: Tensor, cu_k: Tensor, q_scale: Tensor, k_scale: Tensor, v_scale: Tensor
) -> None:
    _validate_attention_shapes(q, k, v)
    _validate_metadata(q, cu_q, cu_k)
    if any(value.dtype != _FP8 or not value.is_contiguous() for value in (q, k, v)):
        raise ValueError("PWM FP8 attention kernel requires contiguous E4M3 Q/K/V")
    for scale in (q_scale, k_scale, v_scale):
        if scale.shape != (1, _KV_HEADS) or scale.dtype != torch.float32 or scale.device != q.device:
            raise ValueError("PWM FP8 attention requires FP32 [1, 8] descales on the Q device")
        if not scale.is_contiguous():
            raise ValueError("PWM FP8 attention requires contiguous descales")


@lru_cache(maxsize=16)
def _require_sm90(device_index: int) -> None:
    from vllm.platforms import current_platform
    from vllm.vllm_flash_attn.flash_attn_interface import FA3_AVAILABLE, FA3_UNAVAILABLE_REASON

    capability = current_platform.get_device_capability(device_index)
    if capability is None or capability.major != 9:
        raise ValueError("PWM FP8 attention requires SM90 CUDA hardware (H100/H200)")
    if not FA3_AVAILABLE:
        raise ValueError(f"PWM FP8 attention requires bundled FA3: {FA3_UNAVAILABLE_REASON}")


@torch.library.custom_op("vllm_omni::pwm_fp8_attention", mutates_args=())
def pwm_fp8_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_q: Tensor,
    cu_k: Tensor,
    q_scale: Tensor,
    k_scale: Tensor,
    v_scale: Tensor,
    num_splits: int,
) -> Tensor:
    """Opaque bundled-FA3 kernel; all quantization is outside this operation."""
    _validate_quantized(q, k, v, cu_q, cu_k, q_scale, k_scale, v_scale)
    if q.device.type != "cuda":
        raise ValueError("PWM FP8 attention execution requires CUDA")
    _require_sm90(q.device.index)
    if not 1 <= num_splits <= 256:
        raise ValueError("PWM FP8 attention kernel requires resolved num_splits in [1, 256]")
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    result = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=q.shape[0],
        max_seqlen_k=k.shape[0],
        q_descale=q_scale,
        k_descale=k_scale,
        v_descale=v_scale,
        softmax_scale=_HEAD_DIM**-0.5,
        causal=False,
        num_splits=num_splits,
        fa_version=3,
    )
    if result.dtype != torch.bfloat16 or result.shape != q.shape:
        raise ValueError("Unexpected FP8 FA3 output dtype or shape")
    return result


@pwm_fp8_attention.register_fake
def _pwm_fp8_attention_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_q: Tensor,
    cu_k: Tensor,
    q_scale: Tensor,
    k_scale: Tensor,
    v_scale: Tensor,
    num_splits: int,
) -> Tensor:
    _validate_quantized(q, k, v, cu_q, cu_k, q_scale, k_scale, v_scale)
    if not 1 <= num_splits <= 256:
        raise ValueError("PWM FP8 attention kernel requires resolved num_splits in [1, 256]")
    return q.new_empty(q.shape, dtype=torch.bfloat16)


def fp8_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    metadata: tuple[Tensor, Tensor] | None = None,
    num_splits: int = 0,
) -> Tensor:
    """Quantize BF16 Q/K/V and return flattened BF16 VIDEO attention output.

    Reuse request-local metadata where available; each B1 cumulative length is
    [0, actual_token_count]. The caller owns that metadata contract, as in the
    existing native FA3 path. Packed batches, padding and SP are unsupported.
    """
    _validate_attention_shapes(q, k, v)
    if any(value.dtype != torch.bfloat16 for value in (q, k, v)):
        raise ValueError("PWM FP8 attention expects BF16 Q/K/V before quantization")
    if metadata is None:
        metadata = (
            torch.tensor([0, q.shape[0]], device=q.device, dtype=torch.int32),
            torch.tensor([0, k.shape[0]], device=q.device, dtype=torch.int32),
        )
    if len(metadata) != 2:
        raise ValueError("PWM FP8 attention requires Q/K cumulative lengths")
    _validate_metadata(q, *metadata)
    split_count = resolve_fp8_attention_splits(k.shape[0], num_splits)
    qq, qs = quantize_fp8_gqa_groups(q, _KV_HEADS)
    kk, ks = quantize_fp8_gqa_groups(k, _KV_HEADS)
    vv, vs = quantize_fp8_gqa_groups(v, _KV_HEADS)
    return pwm_fp8_attention(qq, kk, vv, *metadata, qs, ks, vs, split_count).flatten(1)
