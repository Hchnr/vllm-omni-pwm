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
import math
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
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--native-und-cache", action="store_true")
    parser.add_argument("--vae", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--attention-backend", choices=("flash3", "sdpa"), default="flash3")
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--reuse-und", action="store_true")
    parser.add_argument("--projected-und", action="store_true")
    parser.add_argument("--steps", type=int, help="Override steps; otherwise preserve each request")
    parser.add_argument("--check-invalid-request", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.native and (args.mode != "engine" or not args.vae or not args.tokenizer):
        parser.error("--native requires engine mode, --vae, and --tokenizer")
    if args.native and (args.reuse_und or args.projected_und or args.precision != "bfloat16"):
        parser.error("Native validation currently requires BF16 without UND reuse")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    requests = [json.loads(p.read_text()) for p in args.input]
    for request in requests:
        if args.steps is not None:
            request["num_steps"] = args.steps
        request.setdefault("num_steps", 30)
    result = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "runs": [],
    }
    result["args"]["input"] = [str(p) for p in args.input]
    result["input_sha256"] = [hashlib.sha256(p.read_bytes()).hexdigest() for p in args.input]
    result["fork_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    diff = subprocess.check_output(["git", "diff", "HEAD", "--"], text=True)
    result["fork_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    result["fork_diff_sha256"] = hashlib.sha256(diff.encode()).hexdigest()
    result["cosmos_revision"] = subprocess.check_output(
        ["git", "-C", str(args.cosmos_root), "rev-parse", "HEAD"], text=True
    ).strip()
    result["gpu"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv"], text=True
    )
    result["input_asset_sha256"] = [
        hashlib.sha256(Path(request["vision_path"]).read_bytes()).hexdigest() for request in requests
    ]

    def gpu_snapshot():
        return subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv"], text=True
        )

    result["gpu_processes_before"] = gpu_snapshot()
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
                model_class_name="PwmWamNativePipeline" if args.native else "PwmWamPipeline",
                enforce_eager=True,
                model_config={
                    "pwm_python": str(args.python),
                    "pwm_cosmos_root": str(args.cosmos_root),
                    "pwm_compile": args.compile,
                    "pwm_native_und_cache": args.native_und_cache,
                    "pwm_reuse_und": args.reuse_und,
                    "pwm_projected_und": args.projected_und,
                    "pwm_precision": args.precision,
                    "pwm_attention_backend": args.attention_backend,
                    "pwm_vae_path": str(args.vae) if args.vae else None,
                    "pwm_tokenizer_path": str(args.tokenizer) if args.tokenizer else None,
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

            async def execute():
                final = []
                req = OmniDiffusionRequest(prompt=request["prompt"], sampling_params=params, request_id=f"pwm-{index}")
                async for output_batch in engine.step_streaming(req):
                    final = output_batch
                return final

            outputs = loop.run_until_complete(execute())
            if len(outputs) != 1:
                raise RuntimeError(f"Expected one output, received {len(outputs)}")
            mm = outputs[0].multimodal_output
            output = {"action": mm["actions"], **mm["metadata"]["actions"]}
            if request.get("extra", {}).get("action128", {}).get("return_video", True):
                video = outputs[0].images[0]
                output["video"] = {
                    "shape": list(video.shape),
                    "dtype": str(video.dtype),
                    "sha256": hashlib.sha256(video.tobytes()).hexdigest(),
                    "std": float(video.std()),
                    "fps": mm["fps"],
                }
            return output

        def close():
            engine.close()
            loop.close()

    result["startup_seconds"] = time.perf_counter() - started
    try:
        if args.check_invalid_request:
            if args.mode != "engine":
                raise ValueError("--check-invalid-request requires engine mode")
            from vllm_omni.errors import OmniClientError

            invalid = json.loads(json.dumps(requests[0]))
            invalid["extra"]["action128"]["action_dim_mask"] = [True]
            try:
                run(invalid, "invalid")
            except OmniClientError as exc:
                if "action_dim_mask" not in str(exc):
                    raise
                result["invalid_request_rejected"] = True
            else:
                raise AssertionError("Invalid action mask was accepted")
        for repetition in range(args.warmup + args.repeats):
            for index, request in enumerate(requests):
                started = time.perf_counter()
                output = run(request, repetition * len(requests) + index)
                elapsed = time.perf_counter() - started
                if repetition >= args.warmup:
                    result["runs"].append({"request_index": index, "seconds": elapsed, "output": output})
                    args.output.write_text(json.dumps(result, indent=2) + "\n")
        latencies = [r["seconds"] for r in result["runs"]]
        result["gpu_processes_after"] = gpu_snapshot()
        result["per_request_latency"] = {}
        for index in range(len(requests)):
            values = sorted(r["seconds"] for r in result["runs"] if r["request_index"] == index)
            result["per_request_latency"][str(index)] = {
                "count": len(values),
                "p50_seconds": statistics.median(values),
                "p95_seconds": values[max(0, math.ceil(0.95 * len(values)) - 1)],
            }
        result["median_seconds"] = statistics.median(latencies)
        result["chunks_per_second"] = len(latencies) / sum(latencies)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))
    finally:
        close()


if __name__ == "__main__":
    main()
