# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Measure a native PWM runtime candidate; engine/HTTP qualification is separate.

All outputs retain action values, input/asset hashes, timings and the actual
constructor configuration. CUDA profiling is performed after formal measurement.
Run with CUDA_VISIBLE_DEVICES set to physical GPU(s) 0--3 only.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/pwm-wam-hf"))
    parser.add_argument("--vae", default="/share/project/eai_pwm/models/Cosmos3-Nano/vae")
    parser.add_argument("--tokenizer", default="/share/project/eai_pwm/models/Cosmos3-Nano-VLM")
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--options", type=Path, help="JSON constructor overrides; baseline uses FA3+compile+UND")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or any(x not in ("0", "1", "2", "3") for x in visible.split(",")):
        parser.error("Set explicit physical CUDA_VISIBLE_DEVICES using only 0,1,2,3")
    if args.warmup < 2 or args.repeats < 1:
        parser.error("At least two warmup requests and one measured repeat are required")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch

    from vllm_omni.diffusion.models.pwm_wam.runtime import PwmWamRuntime

    args.output.parent.mkdir(parents=True, exist_ok=True)
    diff = subprocess.check_output(["git", "diff", "HEAD", "--"], text=True)
    args.output.with_suffix(".diff").write_text(diff)
    options = dict(attention_backend="flash3", compile_layers=True, cache_und=True, validation_deterministic=True)
    if args.options:
        options.update(json.loads(args.options.read_text()))
    options.update(vae_path=args.vae, tokenizer_path=args.tokenizer)
    requests = [json.loads(p.read_text()) for p in args.input]
    record = dict(
        boundary="synchronized native runtime whole request; excludes engine/HTTP",
        options=options,
        checkpoint=str(args.checkpoint.resolve()),
        input=[str(p.resolve()) for p in args.input],
        input_sha256=[sha256(p) for p in args.input],
        asset_sha256=[sha256(r["vision_path"]) for r in requests],
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dirty=subprocess.check_output(["git", "status", "--porcelain"], text=True),
        diff_sha256=hashlib.sha256(diff.encode()).hexdigest(),
        command=__import__("sys").argv,
        environment={
            k: os.environ.get(k)
            for k in ("CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS")
        },
        packages={p: importlib.metadata.version(p) for p in ("torch", "vllm", "diffusers", "transformers")},
        gpu=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,clocks.sm,temperature.gpu,memory.used",
                "--format=csv",
                "-i",
                visible,
            ],
            text=True,
        ),
        started_at=time.time(),
        runs=[],
        warmup=[],
    )
    config = args.checkpoint / "config.json"
    index = args.checkpoint / "model.safetensors.index.json"
    record["checkpoint_metadata_sha256"] = {p.name: sha256(p) for p in (config, index)}
    before_load = time.perf_counter()
    model = PwmWamRuntime(args.checkpoint, **options)
    torch.accelerator.synchronize()
    record["load_seconds"] = time.perf_counter() - before_load
    if hasattr(model, "optimization_info"):
        record["optimization_info"] = model.optimization_info
    print(f"Loaded in {record['load_seconds']:.3f}s", flush=True)
    torch.accelerator.reset_peak_memory_stats()
    for repeat in range(args.warmup + args.repeats):
        for i, request in enumerate(requests):
            torch.accelerator.synchronize()
            started = time.perf_counter()
            result, frames, fps = model(request)
            torch.accelerator.synchronize()
            seconds = time.perf_counter() - started
            row = dict(request_index=i, repetition=repeat - args.warmup, seconds=seconds, output=result)
            if repeat < args.warmup:
                record["warmup"].append(dict(request_index=i, seconds=seconds))
            else:
                if frames is not None:
                    row["video"] = dict(shape=list(frames.shape), sha256=hashlib.sha256(frames.tobytes()).hexdigest())
                    if args.save_video and repeat == args.warmup:
                        import imageio.v2 as imageio

                        stem = args.output.parent / (args.output.stem + f"_i{i}")
                        np.save(stem.with_suffix(".npy"), frames, allow_pickle=False)
                        imageio.mimwrite(stem.with_suffix(".mp4"), frames, fps=fps, macro_block_size=1)
                        row["video"].update(
                            raw_path=str(stem.with_suffix(".npy")), review_path=str(stem.with_suffix(".mp4"))
                        )
                record["runs"].append(row)
            record["allocated_peak_bytes"] = torch.accelerator.max_memory_allocated()
            record["reserved_peak_bytes"] = torch.accelerator.max_memory_reserved()
            args.output.write_text(json.dumps(record, indent=2) + "\n")
            print(f"repeat={repeat - args.warmup} input={i} seconds={seconds:.6f}", flush=True)
    if args.profile:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            model(requests[0])
            torch.accelerator.synchronize()
        prof.export_chrome_trace(str(args.output.with_suffix(".trace.json")))
        args.output.with_suffix(".profile.txt").write_text(
            prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=60)
        )
    record["finished_at"] = time.time()
    args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
