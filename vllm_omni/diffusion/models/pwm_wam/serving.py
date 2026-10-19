# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Local Cosmos-compatible HTTP facade for the native Omni WAM worker.

Run with ``python -m vllm_omni.diffusion.models.pwm_wam.serving --help``.
The engine remains the sole owner of model execution; no Cosmos dependency.
"""

import argparse
import asyncio
import hashlib
import json
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path


class NativeService:
    def __init__(self, checkpoint, vae, tokenizer, output_dir, attention_backend="sdpa"):
        from vllm_omni.diffusion.data import OmniDiffusionConfig
        from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        config_bytes = (Path(checkpoint) / "config.json").read_bytes()
        self.spec = json.loads(config_bytes)["model"]["config"]["action128_spec"]
        self.info = {
            "model_name": "action128",
            "backend": "PwmWamNativePipeline",
            "checkpoint": str(Path(checkpoint).resolve()),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "attention_backend": attention_backend,
            "contract_sha256": hashlib.sha256(json.dumps(self.spec, sort_keys=True).encode()).hexdigest(),
            "precision": "bfloat16",
            "compile": False,
            "world_size": 1,
            "max_concurrent_requests": 1,
            "output_dir": str(self.output_dir),
        }
        self.engine = DiffusionEngine(
            OmniDiffusionConfig(
                model=str(checkpoint),
                model_class_name="PwmWamNativePipeline",
                enforce_eager=True,
                model_config={
                    "pwm_vae_path": str(vae),
                    "pwm_tokenizer_path": str(tokenizer),
                    "pwm_attention_backend": attention_backend,
                },
            )
        )
        self.loop = asyncio.new_event_loop()

    def close(self):
        self.engine.close()
        self.loop.close()

    def predict(self, raw):
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        # Cosmos overrides serialize unset fields as null. Resolve only supported
        # defaults, leaving explicit false/zero for validation by the worker.
        request = {key: value for key, value in raw.items() if value is not None}
        request.setdefault("seed", 0)
        request.setdefault("num_steps", 30)
        request.setdefault("guidance", 1.0)
        request.setdefault("name", "action128_wam")
        params = OmniDiffusionSamplingParams(
            seed=request["seed"],
            num_inference_steps=request["num_steps"],
            guidance_scale=request["guidance"],
            extra_args={"pwm_request": request},
        )
        request_id = uuid.uuid4().hex

        async def execute():
            final = []
            req = OmniDiffusionRequest(prompt=request.get("prompt", ""), sampling_params=params, request_id=request_id)
            async for batch in self.engine.step_streaming(req):
                final = batch
            return final

        started = time.perf_counter()
        outputs = self.loop.run_until_complete(execute())
        if len(outputs) != 1:
            raise RuntimeError("Native WAM worker did not return exactly one output")
        output = outputs[0]
        mm = output.multimodal_output
        metadata = dict(mm["metadata"]["actions"])
        files = []
        # Unique server-owned directories avoid overwrite and path traversal from
        # user-provided sample names. The original client accepts any first part.
        directory = self.output_dir / request_id
        directory.mkdir()
        if request.get("extra", {}).get("action128", {}).get("return_video", True):
            import imageio.v3 as iio

            video_path = directory / "vision.mp4"
            iio.imwrite(video_path, output.images[0], fps=mm["fps"], codec="libx264")
            files.append(str(video_path.relative_to(self.output_dir)))
        metadata["elapsed_seconds"] = time.perf_counter() - started
        response = {
            "args": request,
            "status": "success",
            "outputs": [{"content": {"action": mm["actions"], "action128": metadata}, "files": files}],
        }
        (directory / "sample_args.json").write_text(json.dumps(request) + "\n")
        (directory / "sample_outputs.json").write_text(json.dumps(response) + "\n")
        return response


def create_app(service):
    from fastapi import FastAPI, HTTPException
    from fastapi.staticfiles import StaticFiles

    from vllm_omni.errors import OmniClientError

    @asynccontextmanager
    async def lifespan(app):
        yield
        service.close()

    app = FastAPI(title="Native Omni PWM WAM", lifespan=lifespan)
    lock = threading.Lock()
    app.mount("/outputs", StaticFiles(directory=service.output_dir), name="outputs")

    @app.get("/healthz")
    def healthz():
        return {"status": "ready"}

    @app.get("/models")
    def models():
        return [service.info["model_name"]]

    @app.get("/info")
    def info():
        return {**service.info, "models": models()}

    @app.get("/domains")
    def domains():
        return {
            "domains": service.spec["registry"]["domains"],
            "action_dim": 128,
            "contract_sha256": service.info["contract_sha256"],
        }

    @app.post("/generate")
    def generate(request: dict):
        if request.get("model") not in (None, "", service.info["model_name"]):
            raise HTTPException(status_code=404, detail="Unknown model")
        if not lock.acquire(blocking=False):
            raise HTTPException(status_code=503, detail="Model is busy", headers={"Retry-After": "1"})
        try:
            try:
                return service.predict(request)
            except (OmniClientError, ValueError, TypeError, KeyError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            # Sync routes finish even when the HTTP caller disconnects: never
            # release model ownership while a worker request is still running.
            lock.release()

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attention-backend", choices=("sdpa", "flash3"), default="sdpa")
    parser.add_argument("--warmup-input", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8019)
    args = parser.parse_args()
    service = NativeService(args.checkpoint, args.vae, args.tokenizer, args.output_dir, args.attention_backend)
    try:
        if args.warmup_input:
            raw = json.loads(args.warmup_input.read_text())
            service.info["warmup"] = {
                "input_sha256": hashlib.sha256(args.warmup_input.read_bytes()).hexdigest(),
                "completed": False,
            }
            for _ in range(2):
                service.predict(raw)
            service.info["warmup"]["completed"] = True
        import uvicorn

        uvicorn.run(create_app(service), host=args.host, port=args.port, workers=1)
    finally:
        # The application lifespan normally closes it; cover startup exceptions.
        if not service.loop.is_closed():
            service.close()


if __name__ == "__main__":
    main()
