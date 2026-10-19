# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reproducible full-request validation through the worker or Omni engine.

Run worker mode with any Python; engine mode requires the Omni environment.
Every measured call has a fresh request and seed; outputs are retained for parity.
"""

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("worker", "engine"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cosmos-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-config", type=Path)
    parser.add_argument("--attention-backend", choices=("flash3", "sdpa"), default="flash3")
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--reuse-und", action="store_true")
    parser.add_argument("--projected-und", action="store_true")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    requests = [json.loads(p.read_text()) for p in args.input]
    for request in requests:
        request["num_steps"] = args.steps
    result = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "runs": [],
    }
    result["args"]["input"] = [str(p) for p in args.input]
    result["input_sha256"] = [hashlib.sha256(p.read_bytes()).hexdigest() for p in args.input]
    result["fork_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    result["cosmos_revision"] = subprocess.check_output(
        ["git", "-C", str(args.cosmos_root), "rev-parse", "HEAD"], text=True
    ).strip()
    result["gpu"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv"], text=True
    )
    started = time.perf_counter()
    if args.mode == "worker":
        command = [
            str(args.python),
            "-u",
            str(Path(__file__).resolve().parents[2] / "vllm_omni/diffusion/models/pwm_wam/worker.py"),
            "--checkpoint",
            str(args.checkpoint),
            "--precision",
            args.precision,
            "--attention-backend",
            args.attention_backend,
        ]
        if args.reference_config:
            command += ["--reference-config", str(args.reference_config)]
        if args.compile:
            command += ["--compile"]
        if args.reuse_und:
            command += ["--reuse-und"]
        if args.projected_und:
            command += ["--projected-und"]
        env = dict(
            os.environ, PYTHONPATH=str(args.cosmos_root), COSMOS_TRAINING="0", LD_LIBRARY_PATH="", OMP_NUM_THREADS="1"
        )
        log = args.output.with_suffix(".log").open("w")
        process = subprocess.Popen(
            command,
            env=env,
            cwd=args.cosmos_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            bufsize=1,
        )
        ready = json.loads(process.stdout.readline())
        assert ready["ready"], ready
        result["runtime"] = ready

        def run(request, index):
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            response = json.loads(process.stdout.readline())
            if "error" in response:
                raise RuntimeError(response)
            return response["result"]

        def close():
            process.stdin.close()
            process.wait(timeout=30)
            log.close()
    else:
        from vllm_omni.diffusion.data import OmniDiffusionConfig
        from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        engine = DiffusionEngine(
            OmniDiffusionConfig(
                model=str(args.checkpoint),
                model_class_name="PwmWamPipeline",
                enforce_eager=True,
                model_config={
                    "pwm_python": str(args.python),
                    "pwm_cosmos_root": str(args.cosmos_root),
                    "pwm_compile": args.compile,
                    "pwm_reuse_und": args.reuse_und,
                    "pwm_projected_und": args.projected_und,
                    "pwm_precision": args.precision,
                    "pwm_attention_backend": args.attention_backend,
                },
            )
        )
        loop = asyncio.new_event_loop()

        def run(request, index):
            params = OmniDiffusionSamplingParams(
                seed=request["seed"],
                num_inference_steps=request["num_steps"],
                guidance_scale=request["guidance"],
                extra_args={"pwm_request": request},
            )
            outputs = loop.run_until_complete(
                engine.step(
                    OmniDiffusionRequest(prompt=request["prompt"], sampling_params=params, request_id=f"pwm-{index}")
                )
            )
            if len(outputs) != 1:
                raise RuntimeError(f"Expected one output, received {len(outputs)}")
            mm = outputs[0].multimodal_output
            return {"action": mm["actions"], **mm["metadata"]["actions"]}

        def close():
            engine.close()
            loop.close()

    result["startup_seconds"] = time.perf_counter() - started
    try:
        for repetition in range(args.warmup + args.repeats):
            for index, request in enumerate(requests):
                started = time.perf_counter()
                output = run(request, repetition * len(requests) + index)
                elapsed = time.perf_counter() - started
                if repetition >= args.warmup:
                    result["runs"].append({"request_index": index, "seconds": elapsed, "output": output})
                    args.output.write_text(json.dumps(result, indent=2) + "\n")
        latencies = [r["seconds"] for r in result["runs"]]
        result["median_seconds"] = statistics.median(latencies)
        result["chunks_per_second"] = len(latencies) / sum(latencies)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))
    finally:
        close()


if __name__ == "__main__":
    main()
