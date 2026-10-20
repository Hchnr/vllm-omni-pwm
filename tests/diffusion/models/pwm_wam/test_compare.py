# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ensure the acceptance checker cannot hide errors in padded action slots."""

import importlib.util
from pathlib import Path

import pytest

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]

spec = importlib.util.spec_from_file_location(
    "pwm_compare", Path(__file__).resolve().parents[4] / "tools/pwm_wam/compare.py"
)
assert spec is not None and spec.loader is not None
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def sample(value=1.0):
    output = {
        "action": [[value, 0]],
        "normalized_action": [[value, 0]],
        "seed": 42,
        "action_domain": "test",
        "action_dim_mask": [True, False],
        "action_fps": 20,
        "action_units": "raw_physical_or_source_units",
        "action_timestamps_seconds": [0.0],
        "conditioning_caption_sha256": "same",
    }
    return {"runs": [{"request_index": 0, "seconds": 1.0, "output": output}]}


def test_identical_outputs_pass():
    assert comparison.compare(sample(), sample())["passed"]


def test_valid_dimension_error_fails_without_padding_dilution():
    assert not comparison.compare(sample(), sample(1.011))["passed"]


def test_inactive_nonzero_rejected():
    actual = sample()
    actual["runs"][0]["output"]["action"][0][1] = 1
    with pytest.raises(ValueError, match="Inactive"):
        comparison.compare(sample(), actual)


def test_nonfinite_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        comparison.compare(sample(), sample(float("nan")))


def test_changed_seed_rejected():
    actual = sample()
    actual["runs"][0]["output"]["seed"] = 43
    with pytest.raises(ValueError, match="seed"):
        comparison.compare(sample(), actual)


def test_missing_request_rejected():
    expected = sample()
    expected["runs"].append({**expected["runs"][0], "request_index": 1})
    with pytest.raises(ValueError, match="Missing"):
        comparison.compare(expected, sample())


def test_eager_reference_cannot_qualify_deployment_performance():
    result = {"args": {"mode": "worker", "attention_backend": "sdpa", "compile": False, "precision": "bfloat16"}}
    assert not comparison.optimized_reference(result)


def test_optimized_reference_requires_compile_and_original_cache_policy():
    args = {
        "mode": "worker",
        "attention_backend": "flash3",
        "compile": True,
        "precision": "bfloat16",
        "reuse_und": False,
        "projected_und": False,
    }
    assert comparison.optimized_reference({"args": args})
    assert not comparison.optimized_reference({"args": {**args, "compile": False}})
    assert not comparison.optimized_reference({"args": {**args, "reuse_und": True}})
    assert not comparison.optimized_reference({"args": {**args, "mode": "engine"}})
