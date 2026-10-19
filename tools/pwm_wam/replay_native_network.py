# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validate complete native embeddings, 36 layers, and velocity projections."""

import argparse
import importlib.util
import json
from pathlib import Path

import torch
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
results = {}


def compare(name, actual, expected):
    difference = actual.float() - expected.float()
    results[name] = dict(
        max_abs=difference.abs().max().item(),
        rmse=difference.square().mean().sqrt().item(),
        shape=list(actual.shape),
        exact=torch.equal(actual, expected),
    )
    print(name, results[name], flush=True)


with torch.inference_mode():
    model = native.PwmWamTransformer.from_pretrained(args.checkpoint)
    print("Loaded native checkpoint", flush=True)
    tensors = load_file(str(args.fixture / "network.safetensors"), device="cuda")
    first = load_file(str(args.fixture / "layer_00.safetensors"), device="cuda")
    projection = load_file(str(args.fixture / "action2llm.safetensors"), device="cuda")["action2llm.inputs.1"]
    prefix = "network.kwargs.packed_seq."
    video = tensors[prefix + "vision.tokens.0"].squeeze(0)
    video_indices = tensors[prefix + "vision.noisy_frame_indexes.0"]
    streams = model.encode_inputs(
        tensors[prefix + "text_ids"],
        video,
        tensors[prefix + "action.tokens.0"],
        tensors[prefix + "action.timesteps"][:1],
        projection,
        video_indices,
        tensors[prefix + "action.noisy_frame_indexes.0"],
    )
    for name, tensor in zip(("causal_seq", "full_only_seq", "action_seq"), streams):
        compare("embedding." + name, tensor, first["input." + name])
    ropes = [
        tuple(tuple(first[f"{name}.{s}.{c}"] for c in range(2)) for s in range(3))
        for name in ("_small_rope", "_small_action_rope")
    ]
    hidden = model.forward_hidden(streams, *ropes)
    velocity, action = model.decode_outputs(hidden, tuple(video.shape), projection, video_indices)
    compare("velocity.video", velocity, tensors["network.output.preds_vision.0"].squeeze(0))
    compare("velocity.action", action, tensors["network.output.preds_action.0"])
args.output.write_text(json.dumps(results, indent=2) + "\n")
