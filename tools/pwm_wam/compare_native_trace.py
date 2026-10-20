# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Locate the first divergent network boundary in reference/native traces."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("reference", type=Path)
parser.add_argument("native", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
results = []
for path in sorted(args.native.glob("network_*.safetensors")):
    reference = load_file(str(args.reference / path.name))
    actual = load_file(str(path))
    root = path.stem + "."
    mapping = {
        "input.video": root + "kwargs.packed_seq.vision.tokens.0",
        "input.action": root + "kwargs.packed_seq.action.tokens.0",
        "input.text_ids": root + "kwargs.packed_seq.text_ids",
        "velocity.video": root + "output.preds_vision.0",
        "velocity.action": root + "output.preds_action.0",
    }
    checks = {}
    for key, refkey in mapping.items():
        a, b = actual[key], reference[refkey]
        if b.ndim == a.ndim + 1 and b.shape[0] == 1:
            b = b[0]
        if a.shape != b.shape:
            raise ValueError(f"{path.name} {key}: {a.shape} != {b.shape}")
        a = a.to(b.dtype)
        error = a.float() - b.float()
        checks[key] = dict(
            exact=torch.equal(a, b), max_abs=error.abs().max().item(), rmse=error.square().mean().sqrt().item()
        )
    results.append(dict(call=path.stem, checks=checks))
args.output.write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
