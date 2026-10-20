# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Experimental VIDEO token sharding with replicated UND/ACTION streams.

Each SP rank owns a contiguous VIDEO token interval. Layer residuals and VIDEO
MLPs stay local; normalized VIDEO K/V are gathered before either attention path
applies its own global RoPE. Only VIDEO queries use sliced RoPE. Final VIDEO
hidden states are gathered for the unchanged output decoder.

This is all-gather-KV sequence parallelism, not Ulysses: use allgather_degree
2 or 4 with other parallel degrees equal to one. No worker or device ownership
is created here. Uneven sequence lengths are rejected rather than padded.
"""

from __future__ import annotations

import torch
from torch import Tensor

Rope = tuple[Tensor, Tensor]
Streams = tuple[Tensor, Tensor, Tensor]


def _get_sp_group():
    from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

    return get_sp_group()


@torch.library.custom_op("vllm_omni::pwm_sp_all_gather", mutates_args=())
def pwm_sp_all_gather(value: Tensor, world_size: int) -> Tensor:
    """Opaque inference collective with a known rank-ordered output shape.

    GroupCoordinator.all_gather uses all_gather_into_tensor and newly allocated
    output storage. Keeping its process-group Python code behind this boundary
    permits fullgraph layer compilation without tracing distributed setup.
    """
    group = _get_sp_group()
    if world_size not in (2, 4) or group.world_size != world_size:
        raise ValueError("PWM VIDEO SP collective requires the configured two/four-rank group")
    if value.ndim < 2 or value.shape[0] < 1:
        raise ValueError("PWM VIDEO SP all-gather requires nonempty token-major tensors")
    result = group.all_gather(value.contiguous(), dim=0)
    if not isinstance(result, Tensor):
        raise RuntimeError("PWM VIDEO SP expected one concatenated all-gather tensor")
    return result


@pwm_sp_all_gather.register_fake
def _pwm_sp_all_gather_fake(value: Tensor, world_size: int) -> Tensor:
    return value.new_empty((value.shape[0] * world_size, *value.shape[1:]))


class PwmVideoSequenceParallel:
    """Request-independent shape/communication helpers for a worker-owned group."""

    def __init__(self):
        group = _get_sp_group()
        self.world_size = group.world_size
        self.rank = group.rank_in_group
        if self.world_size not in (2, 4) or self.rank not in range(self.world_size):
            raise ValueError("PWM VIDEO SP supports exactly two or four ranks")
        if group.ulysses_world_size != 1 or group.ring_world_size != 1:
            raise ValueError("PWM VIDEO SP requires pure all-gather parallelism, without Ulysses/ring")
        if group.allgather_world_size != self.world_size:
            raise ValueError("PWM VIDEO SP requires allgather_degree equal to the SP world size")

    def _local_length(self, video_tokens: int) -> int:
        if video_tokens < 1 or video_tokens % self.world_size:
            raise ValueError("PWM VIDEO token count must be positive and divisible by the SP degree")
        return video_tokens // self.world_size

    def shard_video(self, video: Tensor) -> Tensor:
        if video.ndim != 2:
            raise ValueError("PWM VIDEO hidden states must have shape [tokens, width]")
        local_length = self._local_length(video.shape[0])
        return video.narrow(0, self.rank * local_length, local_length).contiguous()

    def shard_streams(self, streams: Streams) -> Streams:
        return streams[0], self.shard_video(streams[1]), streams[2]

    def video_query_rope(self, global_rope: Rope) -> Rope:
        cos, sin = global_rope
        if cos.ndim != 2 or cos.shape != sin.shape:
            raise ValueError("PWM VIDEO RoPE requires matching [tokens, head_dim] tensors")
        local_length = self._local_length(cos.shape[0])
        start = self.rank * local_length
        return cos.narrow(0, start, local_length), sin.narrow(0, start, local_length)

    def gather_video_kv(self, local_key: Tensor, local_value: Tensor) -> tuple[Tensor, Tensor]:
        """Gather normalized, unrotated VIDEO keys and values in one collective."""
        if local_key.ndim != 3 or local_key.shape != local_value.shape:
            raise ValueError("PWM VIDEO K/V require matching [local_tokens, kv_heads, head_dim] tensors")
        if local_key.dtype != local_value.dtype or local_key.device != local_value.device:
            raise ValueError("PWM VIDEO K/V must have matching dtype and device")
        packed = torch.stack((local_key, local_value), dim=1)
        gathered = pwm_sp_all_gather(packed, self.world_size)
        return gathered[:, 0], gathered[:, 1]

    def gather_video_hidden(self, local_video: Tensor) -> Tensor:
        if local_video.ndim != 2:
            raise ValueError("PWM VIDEO hidden states must have shape [local_tokens, width]")
        return pwm_sp_all_gather(local_video, self.world_size)

    def gather_streams(self, streams: Streams) -> Streams:
        return streams[0], self.gather_video_hidden(streams[1]), streams[2]

    def attention_metadata(
        self, text_tokens: int, video_tokens: int, action_tokens: int, *, device: torch.device | str
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Keep local VIDEO Q lengths distinct from global joint K/V lengths."""
        if text_tokens < 1 or action_tokens < 1:
            raise ValueError("PWM VIDEO SP requires nonempty UND and ACTION streams")
        lengths = (
            text_tokens,
            self._local_length(video_tokens),
            action_tokens,
            text_tokens + video_tokens + action_tokens,
        )
        return tuple(torch.tensor([0, length], dtype=torch.int32, device=device) for length in lengths)

    def synchronize_latent(self, latent: Tensor) -> Tensor:
        """Optional initial-latent broadcast, called outside the compiled layer."""
        group = _get_sp_group()
        if group.world_size != self.world_size or group.rank_in_group != self.rank:
            raise RuntimeError("PWM VIDEO SP process group changed after helper initialization")
        return group.broadcast(latent, src=0)
