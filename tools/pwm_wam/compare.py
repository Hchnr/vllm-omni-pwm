# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare complete action trajectories and semantic metadata, then report timing.

Thresholds are fixed before experiments: normalized max-abs <= .05, RMSE <= .01.
Physical-unit errors are reported separately because domains have different units.
"""

import argparse
import json
import math
import statistics
from pathlib import Path


def errors(reference, actual):
    if len(reference) != len(actual) or any(len(a) != len(b) for a, b in zip(reference, actual)):
        raise ValueError("Action shape mismatch")
    differences = [float(b) - float(a) for ra, rb in zip(reference, actual) for a, b in zip(ra, rb)]
    if not differences or not all(math.isfinite(d) for d in differences):
        raise ValueError("Empty or non-finite action output")
    return {
        "max_abs": max(map(abs, differences)),
        "rmse": math.sqrt(sum(d * d for d in differences) / len(differences)),
    }


def compare(reference, actual):
    references = {}
    for run in reference["runs"]:
        references.setdefault(run["request_index"], run["output"])
    checks = []
    for run in actual["runs"]:
        expected = references[run["request_index"]]
        observed = run["output"]
        for field in (
            "seed",
            "action_domain",
            "action_dim_mask",
            "action_fps",
            "action_units",
            "action_timestamps_seconds",
            "conditioning_caption_sha256",
        ):
            if expected[field] != observed[field]:
                raise ValueError(f"Request {run['request_index']}: {field} mismatch")
        # Score only valid dimensions; padding zeros must not dilute RMSE.
        mask = expected["action_dim_mask"]

        def select(rows):
            return [[x for x, valid in zip(row, mask) if valid] for row in rows]

        normalized = errors(select(expected["normalized_action"]), select(observed["normalized_action"]))
        physical = errors(select(expected["action"]), select(observed["action"]))
        for key in ("action", "normalized_action"):
            if any(x != 0 for row in observed[key] for x, valid in zip(row, mask) if not valid):
                raise ValueError("Inactive action dimensions must be exactly zero")
        checks.append(
            {
                "request_index": run["request_index"],
                "normalized": normalized,
                "physical": physical,
                "passed": normalized["max_abs"] <= 0.05 and normalized["rmse"] <= 0.01,
            }
        )
    return {
        "passed": bool(checks) and all(c["passed"] for c in checks),
        "checks": checks,
        "baseline_median_seconds": statistics.median(r["seconds"] for r in reference["runs"]),
        "candidate_median_seconds": statistics.median(r["seconds"] for r in actual["runs"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    candidate = json.loads(args.candidate.read_text())
    result = compare(reference, candidate)
    result["reference"] = str(args.reference)
    result["candidate"] = str(args.candidate)
    keys = ("precision", "attention_backend", "steps", "warmup", "repeats", "input")
    ra, ca = reference.get("args", {}), candidate.get("args", {})
    comparable = all(key in ra and ra[key] == ca.get(key) for key in keys)
    comparable = comparable and ra.get("warmup", 0) >= 2 and ra.get("repeats", 0) >= 3
    comparable = comparable and reference.get("cuda_visible_devices") == candidate.get("cuda_visible_devices")
    result["timing_comparable"] = comparable
    result["speedup"] = (result["baseline_median_seconds"] / result["candidate_median_seconds"]) if comparable else None
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
