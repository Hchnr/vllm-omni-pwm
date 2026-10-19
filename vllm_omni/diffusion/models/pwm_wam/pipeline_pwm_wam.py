# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PWM Action128 WAM compatibility pipeline with an isolated Cosmos runtime."""

import json
import os
import select
import subprocess
import threading
import weakref
from collections.abc import Iterable
from pathlib import Path

from torch import Tensor, nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch


def _stop_worker(process: subprocess.Popen[str]) -> None:
    if process.stdin is not None:
        process.stdin.close()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    if process.stdout is not None:
        process.stdout.close()


class PwmWamPipeline(nn.Module):
    """Run a complete non-AR WAM request; output is an Action128 payload.

    No step batching, CFG parallelism, or generic compilation is advertised.
    The worker owns the three towers, VAE, solver, and request-local RNG state.
    """

    dummy_run_num_frames = 0
    _dit_modules = []
    _vae_modules = []
    _encoder_modules = []

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        if od_config.parallel_config.world_size != 1:
            raise ValueError("PWM compatibility runtime supports world_size=1 only")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible or len(visible.split(",")) != 1:
            raise ValueError("Set CUDA_VISIBLE_DEVICES to exactly one GPU for PWM")
        config = od_config.model_config
        python = Path(config["pwm_python"]).resolve()
        cosmos_root = Path(config["pwm_cosmos_root"]).resolve()
        checkpoint = Path(od_config.model).resolve()
        if not python.is_file() or not (cosmos_root / "cosmos_framework").is_dir():
            raise ValueError("pwm_python and pwm_cosmos_root must identify an installed Cosmos runtime")
        if not (checkpoint / "model.safetensors.index.json").is_file():
            raise ValueError("PWM requires a local sharded HF checkpoint")
        command = [str(python), "-u", str(Path(__file__).with_name("worker.py")), "--checkpoint", str(checkpoint)]
        for key in ("attention_backend", "precision"):
            if f"pwm_{key}" in config:
                command.extend(["--" + key.replace("_", "-"), str(config[f"pwm_{key}"])])
        if config.get("pwm_compile", False):
            command.append("--compile")
        if config.get("pwm_projected_und", False):
            command.append("--projected-und")
        if config.get("pwm_reuse_und", False):
            command.append("--reuse-und")
        env = dict(os.environ)
        # The child is a separate single-rank process group, not another Omni rank.
        for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            env.pop(key, None)
        env.update(PYTHONPATH=str(cosmos_root), COSMOS_TRAINING="0", LD_LIBRARY_PATH="", OMP_NUM_THREADS="1")
        self._timeout = float(config.get("pwm_timeout", 1800))
        self._lock = threading.Lock()
        self._process = subprocess.Popen(
            command, cwd=cosmos_root, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )
        self._finalizer = weakref.finalize(self, _stop_worker, self._process)
        try:
            ready = self._receive()
            if not ready.get("ready"):
                raise RuntimeError(f"PWM worker startup failed: {ready}")
        except BaseException:
            self.close()
            raise

    def _receive(self):
        if not select.select([self._process.stdout], [], [], self._timeout)[0]:
            self.close()
            raise TimeoutError("PWM worker timed out; worker has been stopped")
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError(f"PWM worker exited (code {self._process.poll()}); inspect worker stderr")
        return json.loads(line)

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        # The isolated worker completed strict HF loading before its ready reply.
        # There are no parent-process parameters or duplicate weight sources.
        if next(iter(weights), None) is not None:
            raise ValueError("PWM weights must be loaded by the isolated worker")
        return set()

    def close(self) -> None:
        self._finalizer()

    def forward(self, req: DiffusionRequestBatch, **kwargs) -> DiffusionOutput:
        if len(req.prompts) != 1:
            raise ValueError("PWM currently accepts exactly one request per forward")
        params = req.sampling_params
        if params.num_outputs_per_prompt != 1:
            raise ValueError("PWM requires num_outputs_per_prompt=1")
        raw = params.extra_args.get("pwm_request")
        if not isinstance(raw, dict):
            raise ValueError("Provide the Cosmos WAM request in extra_args.pwm_request")
        raw = dict(raw)
        # The sampling object is authoritative for scheduler-visible parameters.
        raw["seed"] = params.seed
        if params.num_inference_steps is not None:
            raw["num_steps"] = params.num_inference_steps
        raw["guidance"] = params.guidance_scale
        with self._lock:
            if self._process.poll() is not None:
                raise RuntimeError("PWM worker is not running")
            self._process.stdin.write(json.dumps(raw, allow_nan=False) + "\n")
            self._process.stdin.flush()
            response = self._receive()
        if "error" in response:
            raise ValueError(response["error"])
        result = response["result"]
        actions = result.pop("action")
        return DiffusionOutput(output={"payload": {"actions": actions}, "metadata": {"actions": result}})
