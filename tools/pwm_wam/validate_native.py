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
