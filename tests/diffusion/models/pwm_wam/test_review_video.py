# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Video metrics distinguish observed frames from generated-frame errors."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


@pytest.fixture
def review(monkeypatch):
    pytest.importorskip("skimage")
    directory = Path(__file__).resolve().parents[4] / "tools/pwm_wam"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("pwm_video_review", directory / "review_video.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_metrics_exclude_observation(review):
    reference = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    candidate = reference.copy()
    candidate[0] = 255
    metrics = review.video_metrics(reference, candidate)
    assert metrics["generated_rmse_0_1"] == 0
    assert metrics["generated_psnr_db"] is None
    assert metrics["generated_ssim_gray_320"] == 1
    assert metrics["max_pixel_error_0_255"] == 255
    assert metrics["motion_delta_mae_0_1"] == 0.5
    assert not metrics["identical"]
    candidate[1:] = 255
    assert review.video_metrics(reference, candidate)["generated_rmse_0_1"] == 1


def test_empty_or_mismatched_video_rejected(review):
    frames = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="match"):
        review.video_metrics(frames, frames[:2])
    with pytest.raises(ValueError, match="nonempty"):
        review.video_metrics(frames[:0], frames[:0])
