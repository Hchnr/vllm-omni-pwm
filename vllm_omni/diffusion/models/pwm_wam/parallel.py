# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native WAM CFG2 using the vLLM-Omni worker-owned CFG process group.

No process, device, scheduler or persistent request state is owned here. Every
rank receives the same request. Rank 0 predicts the conditional branch and rank
1 predicts the unconditional branch; both combine the gathered predictions and
advance their request-local UniPC scheduler. Initial latents are broadcast from
rank 0 so conditioning encode differences cannot desynchronize the trajectories.
"""

from collections.abc import Callable
from typing import Protocol

import torch
from torch import Tensor


class _CfgGroup(Protocol):
    world_size: int
    rank_in_group: int

    def all_gather(self, value: Tensor, *, separate_tensors: bool) -> list[Tensor]: ...

    def broadcast(self, value: Tensor, src: int = 0) -> Tensor: ...


class PwmCfgParallel:
    def __init__(self, group: _CfgGroup | None = None):
        if group is None:
            from vllm_omni.diffusion.distributed.parallel_state import get_cfg_group, is_cfg_group_initialized

            if not is_cfg_group_initialized():
                raise ValueError("PWM CFG2 requires an initialized vLLM-Omni CFG process group")
            group = get_cfg_group()
        if group.world_size != 2 or group.rank_in_group not in (0, 1):
            raise ValueError("PWM CFG2 requires exactly two CFG ranks")
        self.group = group
        self.rank = group.rank_in_group

    def synchronize_latent(self, latent: Tensor) -> Tensor:
        """Establish the same noisy/conditioned starting tensor on both ranks."""
        return self.group.broadcast(latent, src=0)

    def predict(
        self,
        velocity: Callable[[int], Tensor],
        *,
        needs_cfg: bool,
        guidance: float,
        normalize: bool,
    ) -> Tensor:
        if not needs_cfg:
            # Like the upstream CFG mixin, execute the positive branch on both
            # ranks outside the guidance interval. No collective is needed.
            return velocity(0)
        local_prediction = velocity(self.rank)
        predictions = self.group.all_gather(local_prediction, separate_tensors=True)
        if len(predictions) != 2:
            raise RuntimeError("PWM CFG2 expected two gathered predictions")
        conditional, unconditional = predictions
        predicted = unconditional + guidance * (conditional - unconditional)
        if normalize:
            # Preserve PWM's global norm, epsilon and clipping. The generic
            # CFG mixin normalizes along the last axis without this clipping.
            predicted = predicted * (torch.norm(conditional) / (torch.norm(predicted) + 1e-8)).clamp(0, 1)
        return predicted
