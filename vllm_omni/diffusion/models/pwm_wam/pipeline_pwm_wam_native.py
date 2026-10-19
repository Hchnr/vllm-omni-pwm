# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native single-GPU Action128 pipeline executed inside the Omni worker."""

from collections.abc import Iterable
from pathlib import Path

from torch import Tensor, nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.errors import OmniClientError

from .runtime import PwmWamRuntime


class PwmWamNativePipeline(nn.Module):
    dummy_run_num_frames = 0
    _dit_modules = ["runtime.transformer"]
    _vae_modules = ["runtime.vae"]
    _encoder_modules = []
    _resident_modules = []

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        if od_config.parallel_config.world_size != 1:
            raise ValueError("Native PWM currently supports one GPU per worker")
        if od_config.step_execution or od_config.max_num_seqs != 1:
            raise ValueError("Native PWM requires request execution with max_num_seqs=1")
        if not od_config.enforce_eager:
            raise ValueError("Native PWM requires enforce_eager=True; compilation is not accepted")
        if (
            od_config.enable_sleep_mode
            or od_config.enable_cpu_offload
            or od_config.enable_layerwise_offload
            or od_config.enable_distributed_layerwise_offload
            or od_config.diffusion_offload_config
        ):
            raise ValueError("Native PWM sleep/offload is not yet supported")
        if od_config.quantization_config is not None or od_config.lora_path:
            raise ValueError("Native PWM quantization and LoRA are not yet supported")
        if od_config.cache_backend not in (None, "none"):
            raise ValueError("Native PWM does not support generic diffusion caches")
        config = od_config.model_config
        if config.get("pwm_attention_backend", "sdpa") != "sdpa":
            raise ValueError("Native PWM currently accepts SDPA only; FA3 failed full-trajectory parity")
        if config.get("pwm_compile", False):
            raise ValueError("Native PWM compilation has not passed eager-reference acceptance")
        if config.get("pwm_precision", "bfloat16") != "bfloat16":
            raise ValueError("Native PWM currently supports BF16 only")
        if config.get("pwm_reuse_und") or config.get("pwm_projected_und"):
            raise ValueError("Native PWM UND cache is not yet implemented")
        self.runtime = PwmWamRuntime(
            Path(od_config.model),
            vae_path=config["pwm_vae_path"],
            tokenizer_path=config["pwm_tokenizer_path"],
            attention_backend=config.get("pwm_attention_backend", "sdpa"),
            compile_layers=config.get("pwm_compile", False),
            cache_und=config.get("pwm_native_und_cache", False),
        )
        self._loaded_parameters = set(dict(self.named_parameters()))

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        # Both native HF and VAE loaders have completed strict shape/key checks.
        # No duplicate ComponentSource is declared to the generic loader.
        if next(iter(weights), None) is not None:
            raise ValueError("Native PWM uses its explicit HF/VAE checkpoint loaders")
        return self._loaded_parameters

    def forward(self, req: DiffusionRequestBatch, **kwargs) -> DiffusionOutput:
        if len(req.prompts) != 1 or req.sampling_params.num_outputs_per_prompt != 1:
            raise OmniClientError("Native PWM requires one request/output per forward")
        params = req.sampling_params
        raw = params.extra_args.get("pwm_request")
        if not isinstance(raw, dict):
            raise OmniClientError("Provide extra_args.pwm_request")
        raw = dict(raw)
        raw["seed"] = params.seed
        if params.num_inference_steps is not None:
            raw["num_steps"] = params.num_inference_steps
        raw["guidance"] = params.guidance_scale
        try:
            result, video, fps = self.runtime(raw)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            raise OmniClientError(str(exc)) from exc
        payload = {"actions": result.pop("action")}
        metadata = {"actions": result}
        if video is not None:
            payload["video"] = [video]
            metadata["video"] = {"fps": fps}
        return DiffusionOutput(output={"payload": payload, "metadata": metadata})
