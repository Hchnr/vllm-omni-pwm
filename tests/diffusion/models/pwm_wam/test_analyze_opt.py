# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Optimization reports must not mix workload boundaries or hide batch members."""

from copy import deepcopy

import pytest

from tools.pwm_wam.analyze_opt import compare, compare_batch

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


def _output(seed=42):
    action = [[0.1] + [0.0] * 127, [0.2] + [0.0] * 127]
    return dict(
        action=action,
        normalized_action=deepcopy(action),
        action_domain="test",
        action_dim_mask=[True] + [False] * 127,
        action_fps=20.0,
        action_timestamps_seconds=[0.0, 0.05],
        action_units="raw_physical_or_source_units",
        seed=seed,
        conditioning_caption_sha256="caption",
    )


def _run():
    return dict(
        boundary="synchronized native runtime whole request; excludes engine/HTTP",
        input_sha256=["input"],
        asset_sha256=["observation"],
        checkpoint="/models/test",
        checkpoint_metadata_sha256={"config.json": "config", "model.safetensors.index.json": "index"},
        batch_size=1,
        environment={"CUDA_VISIBLE_DEVICES": "0"},
        packages={"torch": "test"},
        options={"attention_backend": "flash3", "fp8_scope": "none"},
        runs=[dict(request_index=0, repetition=0, seconds=2.0, output=_output())],
    )


def test_b1_matched_comparison_reports_explicit_option_changes():
    before, after = _run(), _run()
    after["runs"][0]["seconds"] = 1.0
    after["options"]["fp8_scope"] = "video_mlp"
    result = compare(before, after)
    assert result["results"][0]["saved_ms"] == 1000
    assert result["results"][0]["latency_reduction_percent"] == 50
    assert result["changes"]["options"]["fp8_scope"]["candidate"] == "video_mlp"
    assert "proxy" in result["rate_scope"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("boundary", "HTTP"),
        ("input_sha256", ["different"]),
        ("asset_sha256", ["changed"]),
        ("checkpoint", "/other"),
        ("checkpoint_metadata_sha256", {"config.json": "other"}),
        ("environment", {"CUDA_VISIBLE_DEVICES": "1"}),
        ("batch_size", 2),
    ],
)
def test_b1_rejects_mismatched_workloads_and_resources(field, value):
    before, after = _run(), _run()
    after[field] = value
    with pytest.raises(ValueError):
        compare(before, after)


@pytest.mark.parametrize(
    "field,value",
    [
        ("action_units", "other_units"),
        ("conditioning_caption_sha256", "other_caption"),
        ("seed", 43),
    ],
)
def test_b1_rejects_changed_output_contract(field, value):
    before, after = _run(), _run()
    after["runs"][0]["output"][field] = value
    with pytest.raises(ValueError, match="Contract changed"):
        compare(before, after)


def test_b1_checks_baseline_repeats_and_forbids_broadcasting():
    before, after = _run(), _run()
    before["runs"].append(deepcopy(before["runs"][0]))
    before["runs"][1]["output"]["action"][0][0] = float("nan")
    with pytest.raises(ValueError, match="Invalid or inactive"):
        compare(before, after)
    before = _run()
    after["runs"][0]["output"]["normalized_action"] = [[0.1] + [0.0] * 127]
    with pytest.raises(ValueError, match="shape"):
        compare(before, after)


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_b1_rejects_invalid_durations(seconds):
    before, after = _run(), _run()
    after["runs"][0]["seconds"] = seconds
    with pytest.raises(ValueError, match="duration"):
        compare(before, after)


def _batch():
    data = _run()
    data["batch_size"] = 2
    outputs = [_output(42), _output(43)]
    data["runs"][0]["seconds"] = 3
    data["runs"][0]["batch_outputs"] = outputs
    data["serial_isolation_references"] = [deepcopy(outputs)]
    return data


def test_batch_counts_all_members_and_uses_batch_wall_time():
    result = compare_batch(_batch())
    assert result["completed_requests"] == 2
    assert result["requests_per_timed_second"] == pytest.approx(2 / 3)
    assert result["results"][0]["batch_p50_seconds"] == 3
    assert [member["seed"] for member in result["results"][0]["members"]] == [42, 43]
    assert "latency_reduction_percent" not in result["results"][0]


@pytest.mark.parametrize("damage", ["seed", "nan", "inactive", "shape", "missing"])
def test_batch_audits_second_member_not_just_primary_output(damage):
    data = _batch()
    output = data["runs"][0]["batch_outputs"][1]
    if damage == "seed":
        output["seed"] = 99
    elif damage == "nan":
        output["action"][0][0] = float("nan")
    elif damage == "inactive":
        output["normalized_action"][0][1] = 1
    elif damage == "shape":
        output["action"] = [output["action"][0]]
    else:
        data["runs"][0]["batch_outputs"].pop()
    with pytest.raises(ValueError):
        compare_batch(data)


def test_batch_requires_isolated_references_and_checks_primary_duplicate():
    data = _batch()
    data.pop("serial_isolation_references")
    with pytest.raises(ValueError, match="isolation references"):
        compare_batch(data)
    data = _batch()
    data["runs"][0]["output"]["action"][0][0] = 2
    with pytest.raises(ValueError, match="Primary output"):
        compare_batch(data)
