# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Velocity cache gates both streams and isolates CFG branches and requests."""

import json

import pytest
import torch

from vllm_omni.diffusion.models.pwm_wam.approx_cache import PwmVelocityCache

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


def _inputs():
    return torch.ones(2, 3), torch.ones(2, 8)


def _compute(video, action, offset=0):
    return torch.cat((video.flatten(), action.flatten())) * 2 + offset


def test_joint_gate_uses_action_threshold_and_bounds_skips():
    cache = PwmVelocityCache(video_threshold=0.1, action_threshold=0.01, warmup_steps=1)
    video, action = _inputs()
    first = cache.run(lambda: _compute(video, action), video, action, branch=0, step=0, total_steps=5)
    reused = cache.run(lambda: pytest.fail("expected reuse"), video, action, branch=0, step=1, total_steps=5)
    torch.testing.assert_close(reused, first, rtol=0, atol=0)
    cache.run(lambda: _compute(video, action), video, action, branch=0, step=2, total_steps=5)
    action = action + 0.02
    cache.run(lambda: _compute(video, action), video, action, branch=0, step=3, total_steps=5)
    cache.run(lambda: _compute(video, action), video, action, branch=0, step=4, total_steps=5)
    report = cache.report()["branches"]["0"]
    assert [row["decision"] for row in report["records"]] == [
        "first_call",
        "reused",
        "skip_limit",
        "latent_delta",
        "final_step",
    ]
    assert report["computed_calls"] == 4
    assert report["reused_calls"] == 1
    assert report["reuse_ratio"] == pytest.approx(0.2)
    assert report["state_bytes"] > 0
    json.dumps(cache.report(), allow_nan=False)


@torch.inference_mode()
def test_gate_compares_last_computed_snapshot_and_copies_mutable_values():
    cache = PwmVelocityCache(video_threshold=0.1, warmup_steps=1, max_consecutive_skips=4)
    video, action = _inputs()
    first = cache.run(lambda: _compute(video, action), video, action, branch=0, step=0, total_steps=5)
    expected = first.clone()
    first.add_(100)
    video.add_(0.06)
    reused = cache.run(lambda: pytest.fail("expected reuse"), video, action, branch=0, step=1, total_steps=5)
    torch.testing.assert_close(reused, expected, rtol=0, atol=0)
    reused.add_(100)
    twice = cache.run(lambda: pytest.fail("expected reuse"), video, action, branch=0, step=2, total_steps=5)
    torch.testing.assert_close(twice, expected, rtol=0, atol=0)
    video.add_(0.06)
    cache.run(lambda: _compute(video, action), video, action, branch=0, step=3, total_steps=5)
    record = cache.report()["branches"]["0"]["records"][-1]
    assert record["decision"] == "latent_delta"
    assert record["video_relative_rms"] == pytest.approx(0.12, abs=1e-6)


def test_warmup_branch_and_request_isolation():
    cache = PwmVelocityCache(video_threshold=0.1)
    video, action = _inputs()
    for branch in (0, 1):
        for step in (0, 1):
            cache.run(lambda: _compute(video, action, branch), video, action, branch=branch, step=step, total_steps=6)
        assert cache.report()["branches"][str(branch)]["records"][-1]["decision"] == "warmup"
        actual = cache.run(lambda: pytest.fail("expected reuse"), video, action, branch=branch, step=2, total_steps=6)
        torch.testing.assert_close(actual, _compute(video, action, branch), atol=0, rtol=0)
    with pytest.raises(ValueError, match="fresh"):
        cache.run(lambda: _compute(video, action), video, action, branch=0, step=0, total_steps=6)
    other = PwmVelocityCache(video_threshold=0.1, warmup_steps=1)
    other.run(lambda: _compute(video, action), video, action, branch=0, step=0, total_steps=6)
    assert other.report()["branches"]["0"]["computed_calls"] == 1
    assert other.report()["branches"]["0"]["reused_calls"] == 0


def test_zero_threshold_and_shape_change_force_computation():
    video, action = _inputs()
    cache = PwmVelocityCache(video_threshold=0, warmup_steps=1)
    for step in range(4):
        cache.run(lambda: _compute(video, action), video, action, branch=0, step=step, total_steps=4)
    assert cache.report()["branches"]["0"]["reused_calls"] == 0
    cache = PwmVelocityCache(video_threshold=0.1, warmup_steps=1)
    cache.run(lambda: _compute(video, action), video, action, branch=0, step=0, total_steps=4)
    video = video[:1]
    cache.run(lambda: _compute(video, action), video, action, branch=0, step=1, total_steps=4)
    assert cache.report()["branches"]["0"]["records"][-1]["decision"] == "layout_changed"


def test_nonfinite_gate_cannot_reuse_and_report_is_json():
    video, action = _inputs()
    cache = PwmVelocityCache(video_threshold=0.1, warmup_steps=1)
    cache.run(lambda: _compute(video, action), video, action, branch=0, step=0, total_steps=4)
    video = video * float("nan")
    cache.run(lambda: _compute(video, action), video, action, branch=0, step=1, total_steps=4)
    assert cache.report()["branches"]["0"]["records"][-1]["decision"] == "nonfinite_gate"
    json.dumps(cache.report(), allow_nan=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"video_threshold": -1},
        {"video_threshold": float("nan")},
        {"video_threshold": 0.1, "warmup_steps": 0},
        {"video_threshold": 0.1, "max_consecutive_skips": 0},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        PwmVelocityCache(**kwargs)
