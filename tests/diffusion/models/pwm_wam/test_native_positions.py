# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent reference checks for time, seam, and three-axis RoPE."""

import importlib.util
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]
reference = pytest.importorskip("cosmos_framework.data.generator.sequence_packing.mrope")
action_reference = pytest.importorskip("cosmos_framework.model.generator.mot.small_action_tower")
spec = importlib.util.spec_from_file_location(
    "native_positions", Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/pwm_wam/positions.py"
)
assert spec and spec.loader
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


@pytest.mark.parametrize(
    "text_length,seam,video_fps,action_fps", [(298, 15, 20, 20), (0, None, 24, 10), (17, 7.5, 30, 25)]
)
def test_positions_match_reference(text_length, seam, video_fps, action_fps):
    actual = native.position_ids(text_length, (11, 30, 20), 40, video_fps, action_fps, seam=seam)
    text, _ = reference.get_3d_mrope_ids_text_tokens(text_length, temporal_offset=0, use_float_positions=True)
    offset = text_length + (15000 if text_length else 0)
    video, _ = reference.get_3d_mrope_ids_vae_tokens(
        grid_t=11,
        grid_h=30,
        grid_w=20,
        temporal_offset=offset,
        reset_spatial_indices=True,
        fps=video_fps,
        base_fps=24,
        temporal_compression_factor=4,
    )
    if seam is not None:
        video[1] = seam - video[1]
    action, _ = reference.get_3d_mrope_ids_vae_tokens(
        grid_t=40,
        grid_h=1,
        grid_w=1,
        temporal_offset=offset,
        reset_spatial_indices=True,
        fps=action_fps,
        base_fps=24,
        temporal_compression_factor=1,
        base_temporal_compression_factor=4,
        start_frame_offset=0,
    )
    for got, expected in zip(actual, (text, video, action)):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        for a, b in zip(
            native.rotary(got, dtype=torch.float32, action=True), action_reference.action_rope(got, torch.float32, 128)
        ):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
