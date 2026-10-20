# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare matching native optimization runs without pooling unequal shapes."""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np


def compare(baseline, candidate):
    if baseline["input_sha256"] != candidate["input_sha256"]:
        raise ValueError("Input hashes differ; compare only matched workloads")
    rows = []
    for index in sorted({r["request_index"] for r in baseline["runs"]}):
        aa = [r for r in baseline["runs"] if r["request_index"] == index]
        bb = [r for r in candidate["runs"] if r["request_index"] == index]
        if not bb:
            raise ValueError(f"Missing candidate input {index}")
        a = statistics.median(r["seconds"] for r in aa)
        b = statistics.median(r["seconds"] for r in bb)
        mask = np.asarray(aa[0]["output"]["action_dim_mask"], dtype=bool)
        reference = aa[0]["output"]
        numerical = {}
        for key in ("normalized_action", "action"):
            diffs = []
            for r in bb:
                actual = r["output"]
                for field in ("action_domain", "action_dim_mask", "action_fps", "action_timestamps_seconds", "seed"):
                    if actual[field] != reference[field]:
                        raise ValueError(f"Contract changed: {field}, input={index}")
                value = np.asarray(actual[key])
                if not np.isfinite(value).all() or np.any(value[:, ~mask] != 0):
                    raise ValueError(f"Invalid or inactive action, input={index}")
                diffs.append((value - np.asarray(reference[key]))[:, mask])
            diff = np.stack(diffs)
            numerical[key] = dict(
                max_abs=float(np.abs(diff).max()),
                rmse=float(np.sqrt(np.mean(diff**2))),
                p95_abs=float(np.quantile(np.abs(diff), 0.95)),
                p99_abs=float(np.quantile(np.abs(diff), 0.99)),
            )
        rows.append(
            dict(
                request_index=index,
                baseline_p50_seconds=a,
                candidate_p50_seconds=b,
                saved_ms=(a - b) * 1000,
                latency_reduction_percent=(1 - b / a) * 100,
                serial_rate_gain_percent=(a / b - 1) * 100,
                numerical=numerical,
                baseline_n=len(aa),
                candidate_n=len(bb),
            )
        )
    return dict(
        results=rows,
        scope="Per-input p50; these are within-run screening comparisons, not independent-block confidence intervals",
        quality="Numeric diagnostics only; video and known C0 warnings require separate review",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
