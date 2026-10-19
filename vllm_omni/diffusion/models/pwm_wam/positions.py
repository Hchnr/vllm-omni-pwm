# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Position contracts for non-autoregressive Action128 WAM requests."""

import math

import torch
from torch import Tensor


def position_ids(
    text_length: int,
    video_grid: tuple[int, int, int],
    action_length: int,
    video_fps: float,
    action_fps: float,
    *,
    seam: float | None,
    temporal_margin: int = 15000,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build CPU FP32 coordinates before transfer, as in the training packer.

    seam is the actual horizontal canvas boundary in units of 32 pixels;
    a single camera has no seam. ACTION starts at frame zero and uses its own
    FPS. VIDEO latent-time has compression factor four.
    """
    if text_length < 0 or action_length < 1 or any(size < 1 for size in video_grid):
        raise ValueError("Invalid WAM sequence dimensions")
    if any(not math.isfinite(fps) or fps <= 0 for fps in (video_fps, action_fps)):
        raise ValueError("FPS must be finite and positive")
    if seam is not None and (not math.isfinite(seam) or seam <= 0):
        raise ValueError("Seam must be finite and positive")
    frames, height, width = video_grid
    text = torch.arange(text_length, dtype=torch.float32)[None].expand(3, -1)
    start = text_length + (temporal_margin if text_length else 0)
    video_t = torch.arange(frames, dtype=torch.float32) / (video_fps / 4) * 6 + start
    rows = torch.arange(height, dtype=torch.float32)
    if seam is not None:
        rows = seam - rows
    columns = torch.arange(width, dtype=torch.float32)
    video = torch.stack(tuple(grid.flatten() for grid in torch.meshgrid(video_t, rows, columns, indexing="ij")))
    action = torch.zeros((3, action_length), dtype=torch.float32)
    action[0] = torch.arange(action_length, dtype=torch.float32) / action_fps * 6 + start
    return text, video, action


def rotary(positions: Tensor, *, dtype: torch.dtype, action: bool = False) -> tuple[Tensor, Tensor]:
    """Keep the reference VIDEO and ACTION inverse-frequency formulas distinct."""
    exponent = torch.arange(0, 128, 2, device=positions.device, dtype=torch.float32) / 128
    inverse = 5000000 ** (-exponent) if action else 1.0 / (5000000**exponent)
    frequencies = positions.float()[..., None] * inverse
    mixed = frequencies[0].clone()
    for axis in (1, 2):
        mixed[:, axis:60:3] = frequencies[axis, :, axis:60:3]
    angles = torch.cat((mixed, mixed), -1)
    return angles.cos().to(dtype), angles.sin().to(dtype)
