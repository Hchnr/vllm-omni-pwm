# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Replay real reference layer fixtures in the independent Omni environment."""

import argparse
import importlib.util
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--fixture", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
source = Path(__file__).resolve().parents[2] / "vllm_omni/diffusion/models/pwm_wam/transformer_pwm_wam.py"
spec = importlib.util.spec_from_file_location("native_pwm", source)
assert spec and spec.loader
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
results = []
with torch.inference_mode():
    for layer_id in (0, 35):
        fixture = load_file(str(args.fixture / f"layer_{layer_id:02d}.safetensors"), device="cuda")
        layer = native.PwmThreeTowerLayer().to(device="cuda", dtype=fixture["input.causal_seq"].dtype).eval()
        prefix = f"model.net.language_model.model.layers.{layer_id}."
        weights = {}
        for shard in sorted({v for k, v in index.items() if k.startswith(prefix)}):
            with safe_open(args.checkpoint / shard, framework="pt", device="cpu") as f:
                weights.update(
                    {
                        k.removeprefix(prefix): f.get_tensor(k)
                        for k, v in index.items()
                        if v == shard and k.startswith(prefix)
                    }
                )
        layer.load_state_dict(weights, strict=True)
        names = ("causal_seq", "full_only_seq", "action_seq")
        ropes = [
            tuple(tuple(fixture[f"{name}.{s}.{c}"] for c in range(2)) for s in range(3))
            for name in ("_small_rope", "_small_action_rope")
        ]
        actual = layer(tuple(fixture["input." + name] for name in names), *ropes)
        for name, value in zip(names, actual):
            error = value.float() - fixture["output." + name].float()
            results.append(
                dict(
                    layer=layer_id,
                    stream=name,
                    max_abs=error.abs().max().item(),
                    rmse=error.square().mean().sqrt().item(),
                    shape=list(value.shape),
                )
            )
        del layer, weights, fixture, actual
args.output.write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
