# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Screen matched B1 runs or audit every member of a fixed computation batch."""

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

CONTRACT_FIELDS = (
    "action_domain",
    "action_dim_mask",
    "action_fps",
    "action_timestamps_seconds",
    "action_units",
    "seed",
    "conditioning_caption_sha256",
)
ACTION_FIELDS = ("normalized_action", "action")


def _checked_output(output):
    for field in (*CONTRACT_FIELDS, *ACTION_FIELDS):
        if field not in output:
            raise ValueError(f"Missing action contract field: {field}")
    mask = output["action_dim_mask"]
    if not isinstance(mask, list) or len(mask) != 128 or not all(type(v) is bool for v in mask) or not any(mask):
        raise ValueError("Action mask must contain 128 booleans with an active slot")
    mask = np.asarray(mask, dtype=bool)
    if not math.isfinite(output["action_fps"]) or output["action_fps"] <= 0:
        raise ValueError("Invalid action FPS")
    arrays = {key: np.asarray(output[key], dtype=np.float64) for key in ACTION_FIELDS}
    shape = arrays["action"].shape
    if len(shape) != 2 or shape[0] < 1 or shape[1] != 128:
        raise ValueError("Action shape must be [nonempty horizon, 128]")
    times = np.asarray(output["action_timestamps_seconds"], dtype=np.float64)
    if times.shape != (shape[0],) or not np.isfinite(times).all():
        raise ValueError("Action timestamps do not match the action horizon")
    for key, values in arrays.items():
        if values.shape != shape:
            raise ValueError(f"Action shape differs: {key}")
        if not np.isfinite(values).all() or np.any(values[:, ~mask] != 0):
            raise ValueError(f"Invalid or inactive action: {key}")
    return mask, arrays


def _difference(reference, actual):
    mask, expected = _checked_output(reference)
    _, values = _checked_output(actual)
    for field in CONTRACT_FIELDS:
        if actual[field] != reference[field]:
            raise ValueError(f"Contract changed: {field}")
    if values["action"].shape != expected["action"].shape:
        raise ValueError("Action shapes differ; broadcasting is forbidden")
    return {key: (values[key] - expected[key])[:, mask] for key in ACTION_FIELDS}


def _metrics(differences):
    result = {}
    for key in ACTION_FIELDS:
        values = np.stack([difference[key] for difference in differences])
        result[key] = dict(
            max_abs=float(np.abs(values).max()),
            rmse=float(np.sqrt(np.mean(values**2))),
            p95_abs=float(np.quantile(np.abs(values), 0.95)),
            p99_abs=float(np.quantile(np.abs(values), 0.99)),
        )
    return result


def _groups(data):
    count = len(data["input_sha256"])
    groups = {index: [] for index in range(count)}
    for row in data["runs"]:
        index = row["request_index"]
        if index not in groups or not math.isfinite(row["seconds"]) or row["seconds"] <= 0:
            raise ValueError("Invalid request index or non-positive/non-finite request duration")
        if row.get("repetition", 0) < 0 or row.get("status", "success") != "success":
            raise ValueError("Warmup or failed requests cannot enter successful-run screening metrics")
        groups[index].append(row)
    if not groups or any(not rows for rows in groups.values()):
        raise ValueError("Every declared input requires measured runs")
    return groups


def _devices(data):
    visible = data["environment"]["CUDA_VISIBLE_DEVICES"]
    devices = visible.split(",") if isinstance(visible, str) else []
    if not devices or len(set(devices)) != len(devices) or any(x not in ("0", "1", "2", "3") for x in devices):
        raise ValueError("Expected explicit distinct physical GPUs from 0,1,2,3")
    return devices


def _changes(left, right):
    return {
        key: {"baseline": left.get(key), "candidate": right.get(key)}
        for key in sorted(left.keys() | right.keys())
        if left.get(key) != right.get(key)
    }


def compare(baseline, candidate):
    if baseline.get("batch_size", 1) != 1 or candidate.get("batch_size", 1) != 1:
        raise ValueError("Ordinary comparison requires B1; use compare_batch for fixed batches")
    for field in ("boundary", "input_sha256", "asset_sha256", "checkpoint", "checkpoint_metadata_sha256"):
        if field not in baseline or field not in candidate or baseline[field] != candidate[field]:
            raise ValueError(f"Matched-workload manifest differs or is missing: {field}")
    if _devices(baseline) != _devices(candidate):
        raise ValueError("GPU count/order/physical mapping differ")
    aa, bb = _groups(baseline), _groups(candidate)
    rows = []
    for index, base_rows in aa.items():
        candidate_rows = bb[index]
        reference = base_rows[0]["output"]
        repeat_differences = [_difference(reference, row["output"]) for row in base_rows]
        differences = [_difference(reference, row["output"]) for row in candidate_rows]
        a = statistics.median(row["seconds"] for row in base_rows)
        b = statistics.median(row["seconds"] for row in candidate_rows)
        rows.append(
            dict(
                request_index=index,
                baseline_p50_seconds=a,
                candidate_p50_seconds=b,
                saved_ms=(a - b) * 1000,
                latency_reduction_percent=(1 - b / a) * 100,
                serial_rate_gain_percent=(a / b - 1) * 100,
                numerical=_metrics(differences),
                baseline_repeat_variation=_metrics(repeat_differences),
                baseline_n=len(base_rows),
                candidate_n=len(candidate_rows),
            )
        )
    return dict(
        results=rows,
        changes={
            "options": _changes(baseline.get("options", {}), candidate.get("options", {})),
            "packages": _changes(baseline.get("packages", {}), candidate.get("packages", {})),
            "environment": _changes(baseline["environment"], candidate["environment"]),
            "revision": {"baseline": baseline.get("revision"), "candidate": candidate.get("revision")},
        },
        scope="Matched B1 per-input p50 screening; no independent-block intervals or causal attribution",
        rate_scope="serial_rate_gain is an inverse-median latency proxy, not measured service throughput",
        provenance_limit="Metadata/path matching does not re-audit weight, VAE or tokenizer file contents",
        quality="Numeric diagnostics only; video and known C0 warnings require separate review",
    )


def compare_batch(data):
    """Validate every batch member against its own isolated request output."""
    size = data.get("batch_size", 1)
    if size not in (2, 4):
        raise ValueError("Batch comparison requires B2 or B4")
    groups = _groups(data)
    references = data.get("serial_isolation_references")
    if not isinstance(references, list) or len(references) != len(groups):
        raise ValueError("Missing serial isolation references for all inputs")
    devices = _devices(data)
    results = []
    total_seconds = 0.0
    total_completed = 0
    for index, rows in groups.items():
        reference = references[index]
        if len(reference) != size:
            raise ValueError(f"Missing isolated batch members for input {index}")
        member_differences = [[] for _ in range(size)]
        for row in rows:
            outputs = row.get("batch_outputs", [])
            if len(outputs) != size:
                raise ValueError(f"Incomplete batch outputs for input {index}")
            for member, (expected, actual) in enumerate(zip(reference, outputs)):
                member_differences[member].append(_difference(expected, actual))
            if "output" in row:
                duplicate = _difference(outputs[0], row["output"])
                if any(np.any(value != 0) for value in duplicate.values()):
                    raise ValueError("Primary output does not match the first batch member")
        elapsed = sum(row["seconds"] for row in rows)
        completed = size * len(rows)
        total_seconds += elapsed
        total_completed += completed
        results.append(
            dict(
                request_index=index,
                batch_size=size,
                measured_batches=len(rows),
                completed_requests=completed,
                batch_p50_seconds=statistics.median(row["seconds"] for row in rows),
                batch_p95_seconds=float(np.quantile([row["seconds"] for row in rows], 0.95)),
                timed_seconds=elapsed,
                requests_per_timed_second=completed / elapsed,
                members=[
                    dict(member=member, seed=reference[member]["seed"], numerical=_metrics(differences))
                    for member, differences in enumerate(member_differences)
                ],
            )
        )
    return dict(
        results=results,
        completed_requests=total_completed,
        timed_seconds=total_seconds,
        requests_per_timed_second=total_completed / total_seconds,
        requests_per_timed_second_per_gpu=total_completed / total_seconds / len(devices),
        boundary=data.get("boundary"),
        options=data.get("options", {}),
        scope="Fixed-batch runtime computation: completed requests / sum of timed batch intervals",
        timing_limit="Excludes between-run gaps, arrival/queue time and HTTP; not sustained service capacity",
        quality="All members checked against isolated seed/contract; numerical diagnostics do not establish quality",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path, nargs="?")
    parser.add_argument(
        "--batch", action="store_true", help="Audit one batch run including serial isolation references"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch:
        if args.candidate is not None:
            parser.error("--batch takes exactly one run file")
        result = compare_batch(json.loads(args.baseline.read_text()))
    else:
        if args.candidate is None:
            parser.error("B1 comparison requires baseline and candidate run files")
        result = compare(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
