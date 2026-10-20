# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fixed-shape FA3 WAM batching with per-request noise and UniPC history.

This is a computation prototype, not an admission scheduler. Requests must have
matching sampling/layout contracts and token lengths within each CFG branch.
Embedding and output projection remain per request; transformer layers receive
packed streams. A KV permutation groups each request's three towers before
varlen attention so no sample can read another sample's keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from vllm_omni.diffusion.models.schedulers.scheduling_flow_unipc_multistep import FlowUniPCMultistepScheduler

from .positions import position_ids, rotary
from .request import PwmRequest
from .transformer_pwm_wam import AttentionMetadata, Rope

if TYPE_CHECKING:
    from .runtime import PwmWamRuntime


@dataclass
class _State:
    latent: Tensor
    video_shape: tuple[int, ...]
    video_size: int
    action_length: int
    projection: Tensor
    video_indices: Tensor
    action_indices: Tensor
    scheduler: FlowUniPCMultistepScheduler
    generator: torch.Generator


@dataclass
class _Branch:
    ids: list[Tensor]
    rope: tuple[Rope, Rope, Rope]
    action_rope: tuple[Rope, Rope, Rope]
    cache: list[list[tuple[Tensor, Tensor, Tensor]]] | None
    metadata: AttentionMetadata
    lengths: tuple[int, int, int]


def _packed_metadata(lengths: tuple[int, int, int], batch_size: int, device: torch.device) -> AttentionMetadata:
    """Describe fixed-length samples and tower-major to sample-major KV order."""
    maximum = (*lengths, sum(lengths))
    cumulative = tuple(torch.arange(batch_size + 1, device=device, dtype=torch.int32) * n for n in maximum)
    offsets = (0, batch_size * lengths[0], batch_size * (lengths[0] + lengths[1]))
    order = torch.cat(
        [
            torch.arange(offset + index * size, offset + (index + 1) * size, device="cpu")
            for index in range(batch_size)
            for offset, size in zip(offsets, lengths)
        ]
    ).to(device)
    return (*cumulative, order, maximum)


def _validate(runtime: PwmWamRuntime, requests: list[PwmRequest], clean: list[Tensor]) -> None:
    if not requests or len(requests) != len(clean) or len(requests) > 4:
        raise ValueError("Fixed-shape PWM batching requires 1-4 requests and one clean latent per request")
    if runtime.attention_backend not in ("flash3", "flash3_nv"):
        raise ValueError("Fixed-shape PWM batching requires Flash Attention 3")
    if runtime.graph_runner is not None:
        raise ValueError("Fixed-shape PWM batching does not yet support CUDA Graphs")
    if runtime.cfg_parallel is not None:
        raise ValueError("Fixed-shape PWM batching does not yet support distributed CFG")
    fields = (
        "steps",
        "guidance",
        "shift",
        "normalize_cfg",
        "guidance_interval",
        "video_fps",
        "action_fps",
        "action_length",
        "video_frames",
        "seam",
        "return_video",
    )
    reference = requests[0]
    shape = clean[0].shape
    if len(shape) != 4 or shape[0] != 48 or shape[1] != (reference.video_frames - 1) // 4 + 1:
        raise ValueError("PWM batch clean latents must match the request's 48-channel video layout")
    device = torch.device(runtime.device)
    if clean[0].device.type != device.type or device.index is not None and clean[0].device.index != device.index:
        raise ValueError("PWM batch clean latents must be on the runtime device")
    for request, latent in zip(requests, clean):
        for name in fields:
            if getattr(request, name) != getattr(reference, name):
                raise ValueError(f"Fixed-shape PWM batching requires matching {name}")
        if request.pixels.shape != reference.pixels.shape:
            raise ValueError("Fixed-shape PWM batching requires matching observation layouts")
        if latent.shape != shape or latent.device != clean[0].device or latent.dtype != torch.float32:
            raise ValueError("Fixed-shape PWM batching requires equal FP32 clean latent shapes/devices")


def _make_state(request: PwmRequest, clean: Tensor) -> _State:
    device = clean.device
    # Reset NumPy independently for VIDEO and ACTION, exactly as B1 sampling.
    video = torch.from_numpy(np.random.RandomState(request.seed).standard_normal(tuple(clean.shape)).astype(np.float32))
    video = video.to(device)
    video[:, :1] = clean[:, :1]
    action = torch.from_numpy(
        np.random.RandomState(request.seed).standard_normal((request.action_length, 128)).astype(np.float32)
    ).to(device)
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=False,
        use_cpu_coefficients=True,
        use_legacy_corrector=True,
    )
    scheduler.set_timesteps(request.steps, device=device, shift=request.shift)
    return _State(
        latent=torch.cat((video.flatten(), action.flatten())),
        video_shape=tuple(video.shape),
        video_size=video.numel(),
        action_length=request.action_length,
        projection=torch.full((request.action_length,), request.projection_id, dtype=torch.long, device=device),
        video_indices=torch.arange(1, video.shape[1], device=device),
        action_indices=torch.arange(request.action_length, device=device),
        scheduler=scheduler,
        generator=torch.Generator(device=device).manual_seed(request.seed),
    )


def _make_branch(runtime: PwmWamRuntime, requests: list[PwmRequest], states: list[_State], negative: bool) -> _Branch:
    ids = [runtime.text_ids(request.negative_prompt if negative else request.caption) for request in requests]
    if len({len(tokens) for tokens in ids}) != 1:
        raise ValueError("Fixed-shape PWM batching requires equal token lengths within each CFG branch")
    request, state = requests[0], states[0]
    _, frames, height, width = state.video_shape
    grid = (frames, (height + 1) // 2, (width + 1) // 2)
    positions = position_ids(
        len(ids[0]), grid, request.action_length, request.video_fps, request.action_fps, seam=request.seam
    )
    positions = tuple(position.to(state.latent.device) for position in positions)
    rope = tuple(rotary(position, dtype=torch.bfloat16) for position in positions)
    action_rope = tuple(rotary(position, dtype=torch.bfloat16, action=True) for position in positions)
    count = len(requests)
    rope = tuple(tuple(torch.cat([part] * count) for part in pair) for pair in rope)
    action_rope = tuple(tuple(torch.cat([part] * count) for part in pair) for pair in action_rope)
    lengths = (len(ids[0]), grid[0] * grid[1] * grid[2], request.action_length)
    return _Branch(
        ids=ids,
        rope=rope,
        action_rope=action_rope,
        cache=[[] for _ in runtime.transformer.language_model.model.layers] if runtime.cache_und else None,
        metadata=_packed_metadata(lengths, count, state.latent.device),
        lengths=lengths,
    )


def _velocity(runtime: PwmWamRuntime, states: list[_State], branch: _Branch, timestep: Tensor) -> list[Tensor]:
    embedded = [
        runtime.transformer.encode_inputs(
            ids,
            state.latent[: state.video_size].reshape(state.video_shape),
            state.latent[state.video_size :].reshape(state.action_length, 128),
            timestep.reshape(1),
            state.projection,
            state.video_indices,
            state.action_indices,
        )
        for ids, state in zip(branch.ids, states)
    ]
    packed = tuple(torch.cat([streams[index] for streams in embedded]) for index in range(3))
    hidden = runtime.transformer.forward_hidden(packed, branch.rope, branch.action_rope, branch.cache, branch.metadata)
    velocities = []
    for index, state in enumerate(states):
        streams = tuple(tensor[index * length : (index + 1) * length] for tensor, length in zip(hidden, branch.lengths))
        video, action = runtime.transformer.decode_outputs(
            streams, state.video_shape, state.projection, state.video_indices
        )
        velocities.append(torch.cat((video.flatten(), action.flatten())))
    return velocities


@torch.inference_mode()
def sample_batch(
    runtime: PwmWamRuntime, requests: list[PwmRequest], clean: list[Tensor]
) -> list[tuple[Tensor, Tensor]]:
    """Return per-request VIDEO latents and normalized ACTION tensors.

    Normalization/masking, VAE decoding and serving remain the caller's concern.
    Every request owns its generator, noise and multistep solver history. Batch
    members can differ in seed, caption, observation and projection/domain;
    sampling parameters, layouts and branch token lengths must match.
    """
    _validate(runtime, requests, clean)
    states = [_make_state(request, latent) for request, latent in zip(requests, clean)]
    branches = [_make_branch(runtime, requests, states, negative) for negative in (False, True)]
    contract = requests[0]
    for timestep in states[0].scheduler.timesteps:
        conditional = _velocity(runtime, states, branches[0], timestep)
        needs_cfg = contract.guidance != 1
        if contract.guidance_interval is not None:
            lo, hi = contract.guidance_interval
            needs_cfg = needs_cfg and lo < timestep.item() < hi
        unconditional = _velocity(runtime, states, branches[1], timestep) if needs_cfg else None
        for index, (request, state) in enumerate(zip(requests, states)):
            predicted = conditional[index]
            if unconditional is not None:
                predicted = unconditional[index] + request.guidance * (conditional[index] - unconditional[index])
                if request.normalize_cfg:
                    ratio = torch.norm(conditional[index]) / (torch.norm(predicted) + 1e-8)
                    predicted = predicted * ratio.clamp(0, 1)
            state.latent = state.scheduler.step(
                model_output=predicted,
                timestep=timestep,
                sample=state.latent[None],
                generator=state.generator,
                return_dict=False,
            )[0].squeeze(0)
    return [
        (
            state.latent[: state.video_size].reshape(state.video_shape),
            state.latent[state.video_size :].reshape(state.action_length, 128),
        )
        for state in states
    ]
