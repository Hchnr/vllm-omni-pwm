# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Public WAM request contract and invalid-request checks without GPU/model loads."""

import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]
spec = importlib.util.spec_from_file_location(
    "native_request", Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/pwm_wam/request.py"
)
assert spec and spec.loader
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


@pytest.fixture
def contract(tmp_path):
    image = tmp_path / "observation.png"
    Image.new("RGB", (32, 48), color=(100, 50, 0)).save(image)
    registry = dict(domains=["source"], domain_projection_ids=[7], domain_embodiments=[0], embodiments=["robot"])
    raw = dict(
        model_mode="wam",
        prompt=json.dumps(dict(task_title="pick", command="lift", task_plan=["reach", "lift"])),
        vision_path=str(image),
        seed=42,
        fps=20,
        action_chunk_size=40,
        num_frames=41,
        extra={
            "action128": dict(
                action_domain="source",
                action_dim_mask=[True] * 10 + [False] * 118,
                selected_roles=["main", "left_wrist"],
                camera_layout=[["main", "top", 32, 24], ["left_wrist", "bottom", 32, 24]],
            )
        },
    )
    return raw, {"registry": registry}


def test_preserves_caption_plan_and_projection_routing(contract):
    raw, spec = contract
    result = native.prepare_request(raw, spec)
    caption = json.loads(result.caption)
    assert caption["task_plan"] == ["reach", "lift"]
    assert caption["resolution"] == {"W": 32, "H": 48}
    assert result.projection_id == 7 and result.normalization_id == 0
    assert result.seam == 0.75 and result.video_frames == 41
    assert result.action_fps == 20 and result.action_length == 40


def test_resizing_updates_actual_camera_seam(contract):
    raw, spec = contract
    raw["image_size"] = 96
    result = native.prepare_request(raw, spec)
    assert result.pixels.shape == (3, 1, 96, 64)
    assert result.seam == 1.5
    assert raw["extra"]["action128"]["camera_layout"][0][3] == 24


@pytest.mark.parametrize("change", ["mask", "horizon", "layout", "frames", "mode"])
def test_rejects_invalid_contract(contract, change):
    raw, spec = contract
    if change == "mask":
        raw["extra"]["action128"]["action_dim_mask"] = [False] * 128
    if change == "horizon":
        raw["action_chunk_size"] = 41
    if change == "layout":
        raw["extra"]["action128"]["camera_layout"][0][3] = 20
    if change == "frames":
        raw["num_frames"] = 40
    if change == "mode":
        raw["model_mode"] = "inverse_dynamics"
    with pytest.raises(ValueError):
        native.prepare_request(raw, spec)


@pytest.mark.parametrize("field", ["fps", "action_chunk_size", "image_size"])
def test_explicit_zero_is_not_silently_defaulted(contract, field):
    raw, spec = contract
    raw[field] = 0
    with pytest.raises(ValueError):
        native.prepare_request(raw, spec)


def test_zero_action_fps_rejected(contract):
    raw, spec = contract
    raw["extra"]["action128"]["action_fps"] = 0
    with pytest.raises(ValueError):
        native.prepare_request(raw, spec)
