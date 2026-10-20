# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native Action128 pipeline with single-GPU and experimental CFG2 execution."""

from collections.abc import Iterable
from pathlib import Path

from torch import Tensor, bfloat16, nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.errors import OmniClientError

from .optimization_config import PwmOptimizationConfig
from .runtime import PwmWamRuntime


class PwmWamNativePipeline(nn.Module):
    dummy_run_num_frames = 0
    _dit_modules = ["runtime.transformer"]
    _vae_modules = ["runtime.vae"]
    _encoder_modules = []
    _resident_modules = []

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        cfg_size = od_config.parallel_config.cfg_parallel_size
        sp_size = od_config.parallel_config.allgather_degree
        if (
            cfg_size not in (1, 2)
            or sp_size not in (1, 2, 4)
            or cfg_size > 1
            and sp_size > 1
            or od_config.parallel_config.world_size != cfg_size * sp_size
        ):
            raise ValueError("Native PWM supports single GPU, CFG2 or pure VIDEO all-gather SP2/SP4")
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
        experimental = config.get("pwm_experimental_backend", False)
        if cfg_size == 2 and not experimental:
            raise ValueError("PWM CFG2 requires pwm_experimental_backend")
        if sp_size > 1 and not experimental:
            raise ValueError("PWM VIDEO SP requires pwm_experimental_backend")
        if not experimental and config.get("pwm_attention_backend", "sdpa") != "sdpa":
            raise ValueError("Native PWM defaults to SDPA only; FA3 requires pwm_experimental_backend")
        if not experimental and config.get("pwm_compile", False):
            raise ValueError("Native PWM compilation requires pwm_experimental_backend")
        if od_config.dtype != bfloat16 or config.get("pwm_precision", "bfloat16") != "bfloat16":
            raise ValueError("Native PWM currently supports BF16 only")
        if config.get("pwm_reuse_und") or config.get("pwm_projected_und"):
            raise ValueError("Legacy UND cache flags are not supported; use pwm_native_und_cache")
        optimization = PwmOptimizationConfig.from_mapping(config.get("pwm_optimization", {}))
        if optimization != PwmOptimizationConfig() and not experimental:
            raise ValueError("PWM optimization options require pwm_experimental_backend")
        if cfg_size == 2 and (optimization.cuda_graph or optimization.fp8_scope != "none"):
            raise ValueError("PWM CFG2 currently requires CUDA Graph and FP8 to be disabled")
        self.runtime = PwmWamRuntime(
            Path(od_config.model),
            vae_path=config["pwm_vae_path"],
            tokenizer_path=config["pwm_tokenizer_path"],
            attention_backend=config.get("pwm_attention_backend", "sdpa"),
            compile_layers=config.get("pwm_compile", False),
            cache_und=config.get("pwm_native_und_cache", False),
            validation_deterministic=config.get("pwm_validation_deterministic", False),
            cfg_parallel=cfg_size == 2,
            sequence_parallel=sp_size > 1,
            **optimization.runtime_kwargs(),
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
