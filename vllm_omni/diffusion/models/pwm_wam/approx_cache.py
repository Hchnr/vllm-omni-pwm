# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Experimental PWM joint-latent gate with zero-order velocity reuse.

This is not SeaCache, TeaCache or cache-dit. Both raw VIDEO and ACTION latent
streams must remain close to the last *computed* step before the entire cached
velocity may be reused. In particular, no post-final-RMSNorm hidden state is
misinterpreted as a transformer block residual. A small input delta does not
establish any output-error bound: full trajectories still require validation.

Create one cache per request. Its two CFG branches own separate snapshots and
velocities. The prompt/conditioning/sampler contract must stay fixed within
that request. The gate's FP32 reductions, device-to-host decision, copies and
velocity reuse all belong in request timing. Call around an ordinary eager or
compiled velocity function; graph capture, CFG parallelism and request batches
must be explicitly rejected by the caller until separately validated.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor


def _compatible(left: Tensor, right: Tensor) -> bool:
    return left.shape == right.shape and left.dtype == right.dtype and left.device == right.device


def _relative_rms_change(value: Tensor, previous: Tensor) -> Tensor:
    difference = (value.float() - previous.float()).square().mean().sqrt()
    reference = previous.float().square().mean().sqrt().clamp_min(1e-12)
    return difference / reference


@dataclass
class _BranchState:
    video: Tensor | None = None
    action: Tensor | None = None
    velocity: Tensor | None = None
    last_step: int = -1
    computed: int = 0
    consecutive_skips: int = 0
    records: list[dict[str, object]] = field(default_factory=list)


class PwmVelocityCache:
    """Conservative opt-in gate; a zero threshold disables skipping."""

    def __init__(
        self,
        *,
        video_threshold: float,
        action_threshold: float | None = None,
        warmup_steps: int = 2,
        max_consecutive_skips: int = 1,
    ):
        if action_threshold is None:
            action_threshold = video_threshold
        if any(not math.isfinite(value) or value < 0 for value in (video_threshold, action_threshold)):
            raise ValueError("PWM velocity-cache thresholds must be finite and nonnegative")
        if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 1:
            raise ValueError("PWM velocity cache requires at least one warmup computation")
        if (
            isinstance(max_consecutive_skips, bool)
            or not isinstance(max_consecutive_skips, int)
            or max_consecutive_skips < 1
        ):
            raise ValueError("PWM velocity cache requires a positive maximum consecutive skip count")
        self.video_threshold = video_threshold
        self.action_threshold = action_threshold
        self.warmup_steps = warmup_steps
        self.max_consecutive_skips = max_consecutive_skips
        self._branches: dict[int, _BranchState] = {}
        self._total_steps: int | None = None

    @torch.inference_mode()
    def run(
        self,
        compute: Callable[[], Tensor],
        video: Tensor,
        action: Tensor,
        *,
        branch: int,
        step: int,
        total_steps: int,
    ) -> Tensor:
        """Compute or reuse a flattened [VIDEO; ACTION] velocity for one branch."""
        if isinstance(branch, bool) or not isinstance(branch, int) or branch not in (0, 1):
            raise ValueError("PWM velocity cache supports CFG branch 0 or 1")
        if (
            isinstance(step, bool)
            or isinstance(total_steps, bool)
            or not isinstance(step, int)
            or not isinstance(total_steps, int)
            or not 0 <= step < total_steps
        ):
            raise ValueError("PWM velocity cache requires 0 <= step < total_steps")
        if any(not value.is_floating_point() or not value.numel() for value in (video, action)):
            raise ValueError("PWM velocity cache requires nonempty floating latent tensors")
        if video.device != action.device:
            raise ValueError("PWM velocity-cache latent tensors must share a device")
        state = self._branches.get(branch, _BranchState())
        if step <= state.last_step or self._total_steps not in (None, total_steps):
            raise ValueError("Create a fresh PWM velocity cache for each request")

        video_change = action_change = None
        if state.velocity is None:
            reason = "first_call"
        elif self.video_threshold == 0 or self.action_threshold == 0:
            reason = "disabled"
        elif step < self.warmup_steps or state.computed < self.warmup_steps:
            reason = "warmup"
        elif step == total_steps - 1:
            reason = "final_step"
        elif state.consecutive_skips >= self.max_consecutive_skips:
            reason = "skip_limit"
        elif not _compatible(video, state.video) or not _compatible(action, state.action):
            reason = "layout_changed"
        else:
            # Stack diagnostics before the single device-to-host gate decision.
            video_change, action_change = torch.stack(
                (_relative_rms_change(video, state.video), _relative_rms_change(action, state.action))
            ).tolist()
            if not math.isfinite(video_change) or not math.isfinite(action_change):
                reason = "nonfinite_gate"
                video_change = video_change if math.isfinite(video_change) else None
                action_change = action_change if math.isfinite(action_change) else None
            elif video_change < self.video_threshold and action_change < self.action_threshold:
                reason = "reused"
            else:
                reason = "latent_delta"

        if reason == "reused":
            # The caller owns this output and may perform in-place CFG math.
            result = state.velocity.clone()
            state.consecutive_skips += 1
        else:
            result = compute()
            if (
                not isinstance(result, Tensor)
                or result.ndim != 1
                or result.numel() != video.numel() + action.numel()
                or not result.is_floating_point()
                or result.device != video.device
            ):
                raise ValueError("PWM velocity cache requires a flattened floating [VIDEO; ACTION] velocity")
            # Store owned snapshots. Both exact and skipped outputs remain safe
            # if the scheduler updates its input or result tensors in place.
            state.video = video.detach().clone()
            state.action = action.detach().clone()
            state.velocity = result.detach().clone()
            state.computed += 1
            state.consecutive_skips = 0
        state.last_step = step
        state.records.append(
            {"step": step, "decision": reason, "video_relative_rms": video_change, "action_relative_rms": action_change}
        )
        self._branches[branch] = state
        self._total_steps = total_steps
        return result

    def report(self) -> dict[str, object]:
        branches = {}
        for branch, state in self._branches.items():
            count = len(state.records)
            skipped = count - state.computed
            tensors = (state.video, state.action, state.velocity)
            branches[str(branch)] = {
                "computed_calls": state.computed,
                "reused_calls": skipped,
                "reuse_ratio": skipped / count if count else 0.0,
                "state_bytes": sum(value.numel() * value.element_size() for value in tensors if value is not None),
                "records": [dict(record) for record in state.records],
            }
        return {
            "algorithm": "pwm_joint_latent_zero_order_velocity_reuse_experimental",
            "video_threshold": self.video_threshold,
            "action_threshold": self.action_threshold,
            "warmup_steps": self.warmup_steps,
            "max_consecutive_skips": self.max_consecutive_skips,
            "final_step_always_computed": True,
            "gate_and_copy_cost_in_request_timing": True,
            "branches": branches,
        }
