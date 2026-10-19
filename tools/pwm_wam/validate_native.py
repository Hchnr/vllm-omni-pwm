# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run complete native WAM requests without a Cosmos runtime dependency."""

import argparse
import json
import time
from pathlib import Path

import torch

from vllm_omni.diffusion.models.pwm_wam.runtime import PwmWamRuntime

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--vae", required=True)
parser.add_argument("--tokenizer", required=True)
parser.add_argument("--input", type=Path, nargs="+", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps", type=int, default=5)
parser.add_argument("--attention-backend", choices=("sdpa", "flash3"), default="sdpa")
parser.add_argument("--compile", action="store_true")
parser.add_argument("--trace", type=Path)
args = parser.parse_args()
started = time.perf_counter()
model = PwmWamRuntime(
    args.checkpoint,
    vae_path=args.vae,
    tokenizer_path=args.tokenizer,
    attention_backend=args.attention_backend,
    compile_layers=args.compile,
)
result = {"startup_seconds": time.perf_counter() - started, "runs": []}
if args.trace:
    from safetensors.torch import save_file

    args.trace.mkdir(parents=True, exist_ok=True)
    original_encode = model.transformer.encode_inputs
    original_decode = model.transformer.decode_outputs
    captured = {}
    trace_index = 0

    def encode(*values, **kwargs):
        streams = original_encode(*values, **kwargs)
        for name, value in zip(
            ("text_ids", "video", "action", "timestep", "projection", "video_indices", "action_indices"), values
        ):
            captured["input." + name] = value.detach().cpu().contiguous()
        for name, value in zip(("und", "video", "action"), streams):
            captured["embedding." + name] = value.detach().cpu().contiguous()
        return streams

    def decode(*values, **kwargs):
        global trace_index
        outputs = original_decode(*values, **kwargs)
        for name, value in zip(("video", "action"), outputs):
            captured["velocity." + name] = value.detach().cpu().contiguous()
        save_file(captured, str(args.trace / f"network_{trace_index:02d}.safetensors"))
        captured.clear()
        trace_index += 1
        return outputs

    model.transformer.encode_inputs = encode
    model.transformer.decode_outputs = decode
print("Native model ready", flush=True)
for i, path in enumerate(args.input):
    raw = json.loads(path.read_text())
    raw["num_steps"] = args.steps
    started = time.perf_counter()
    with torch.inference_mode():
        output, video, fps = model(raw)
    result["runs"].append(dict(request_index=i, seconds=time.perf_counter() - started, output=output))
    args.output.write_text(json.dumps(result) + "\n")
    print("Finished request", i, "seconds", result["runs"][-1]["seconds"], flush=True)
