# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Strict, opt-in native PWM optimization configuration."""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PwmOptimizationConfig:
    fuse_gate_up: bool = False
    fuse_qkv: bool = False
    fp8_scope: str = "none"
    fp8_scaling: str = "tensor"
    cuda_graph: bool = False
    graph_max_entries: int = 4
    vae_fastpath: str = "none"
    load_mode: str = "serial"
    approx_cache_threshold: float = 0.0
    cross_request_und_cache_bytes: int = 0
    fp8_attention: bool = False
    video_attention_splits: int = 0

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, dict):
            raise ValueError("pwm_optimization must be a dictionary")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown PWM optimization options: {sorted(unknown)}")
        result = cls(**value)
        for name in ("fuse_gate_up", "fuse_qkv", "cuda_graph", "fp8_attention"):
            if type(getattr(result, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if result.fp8_scope not in ("none", "mlp", "attention", "all"):
            raise ValueError("fp8_scope must be none, mlp, attention or all")
        if result.fp8_scaling not in ("tensor", "row"):
            raise ValueError("fp8_scaling must be tensor or row")
        if result.vae_fastpath not in ("none", "encoder", "decoder", "all"):
            raise ValueError("vae_fastpath must be none, encoder, decoder or all")
        if result.load_mode not in ("serial", "async_h2d"):
            raise ValueError("load_mode must be serial or async_h2d")
        if (
            type(result.approx_cache_threshold) not in (float, int)
            or not math.isfinite(result.approx_cache_threshold)
            or result.approx_cache_threshold < 0
        ):
            raise ValueError("approx_cache_threshold must be a finite nonnegative number")
        if result.approx_cache_threshold and result.cuda_graph:
            raise ValueError("Approximate velocity cache cannot combine with CUDA Graph")
        if type(result.cross_request_und_cache_bytes) is not int or result.cross_request_und_cache_bytes < 0:
            raise ValueError("cross_request_und_cache_bytes must be a nonnegative integer")
        if type(result.video_attention_splits) is not int or result.video_attention_splits not in (0, 1, 2, 4, 8, 16):
            raise ValueError("video_attention_splits must be 0(auto),1,2,4,8 or 16")
        if type(result.graph_max_entries) is not int or not 1 <= result.graph_max_entries <= 16:
            raise ValueError("graph_max_entries must be an integer in [1,16]")
        return result

    def runtime_kwargs(self):
        return asdict(self)
