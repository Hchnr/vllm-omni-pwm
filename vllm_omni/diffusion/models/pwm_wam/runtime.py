# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native PWM WAM execution: no Cosmos imports or subprocess inference."""

import hashlib
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLWan
from torch import nn
from transformers import AutoTokenizer

from vllm_omni.diffusion.models.schedulers.scheduling_flow_unipc_multistep import FlowUniPCMultistepScheduler

from .parallel import PwmCfgParallel
from .positions import position_ids, rotary
from .reproducibility import configure_validation
from .request import PwmRequest, prepare_request
from .transformer_pwm_wam import PwmWamTransformer
from .vae import preserve_checkpoint_rms


class PwmWamRuntime(nn.Module):
    def __init__(
        self,
        checkpoint: Path,
        *,
        vae_path: str,
        tokenizer_path: str,
        device="cuda",
        attention_backend="sdpa",
        compile_layers=False,
        cache_und=False,
        validation_deterministic=False,
        fuse_gate_up=False,
        fuse_qkv=False,
        fp8_scope="none",
        cuda_graph=False,
        graph_max_entries=4,
        vae_fastpath="none",
        cfg_parallel=False,
    ):
        super().__init__()
        if attention_backend not in ("sdpa", "flash3", "flash3_nv"):
            raise ValueError("Native PWM supports sdpa, flash3 or flash3_nv")
        # Match the reference runtime's precision contract. vLLM's CUDA
        # platform disables cuDNN SDPA globally; restore this backend before
        # execution in this dedicated model worker.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_cudnn_sdp(True)
        self.compile_layers = compile_layers
        self.attention_backend = attention_backend
        self.cache_und = cache_und
        self.validation_deterministic = validation_deterministic
        self.cfg_parallel = PwmCfgParallel() if cfg_parallel else None
        if validation_deterministic:
            configure_validation(0)
        self.device = device
        config = json.loads((checkpoint / "config.json").read_text())["model"]["config"]
        self.spec = config["action128_spec"]
        self.transformer = PwmWamTransformer.from_pretrained(checkpoint, device=device)
        self.optimization_info = dict(
            fuse_gate_up=fuse_gate_up,
            fuse_qkv=fuse_qkv,
            fp8_scope=fp8_scope,
            cuda_graph=cuda_graph,
            graph_max_entries=graph_max_entries,
            vae_fastpath=vae_fastpath,
            cfg_parallel_size=2 if cfg_parallel else 1,
        )
        if fuse_gate_up or fuse_qkv:
            self.transformer.configure_fusions(gate_up=fuse_gate_up, qkv=fuse_qkv)
        if fp8_scope != "none":
            from .optimization_linear import install_video_fp8

            self.optimization_info["fp8"] = install_video_fp8(self.transformer, scope=fp8_scope)
        for layer in self.transformer.language_model.model.layers:
            layer.attention_backend = attention_backend
        if compile_layers:
            for index, layer in enumerate(self.transformer.language_model.model.layers):
                self.transformer.language_model.model.layers[index] = torch.compile(
                    layer, fullgraph=True, dynamic=False
                )
        self.graph_runner = None
        if cuda_graph:
            from .graph import PwmGraphRunner

            self.graph_runner = PwmGraphRunner(self.transformer, max_entries=graph_max_entries)
        contract = dict(
            action128_spec=self.spec,
            vision_spatial_position_mode="action128_seam",
            spatial_coordinate_version="pixel_seam_patch_corner_v1",
        )
        expected_digest = hashlib.blake2b(json.dumps(contract, sort_keys=True).encode(), digest_size=16).digest()
        actual_digest = bytes(self.transformer.action_normalization.contract_digest.cpu().tolist())
        if actual_digest != expected_digest:
            raise ValueError("HF Action128 normalization contract digest mismatch")
        normalizer = self.transformer.action_normalization
        for name in ("scale", "bias", "normalization_mask"):
            dtype = torch.bool if name == "normalization_mask" else torch.float32
            value = torch.tensor([entry[name] for entry in self.spec["transforms"]], dtype=dtype, device=device)
            if value.shape != getattr(normalizer, name).shape:
                raise ValueError("Action128 normalization shape mismatch")
            setattr(normalizer, name, value)
        if not torch.isfinite(normalizer.scale).all() or not torch.isfinite(normalizer.bias).all():
            raise ValueError("Non-finite Action128 transforms")
        if (normalizer.scale[normalizer.normalization_mask] <= 0).any():
            raise ValueError("Action128 normalization scale must be positive")
        vae, loading_info = AutoencoderKLWan.from_pretrained(
            vae_path, torch_dtype=torch.bfloat16, local_files_only=True, output_loading_info=True
        )
        if any(loading_info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Incomplete PWM VAE checkpoint: {loading_info}")
        self.vae = vae.to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        preserve_checkpoint_rms(self.vae)
        if vae_fastpath != "none":
            from .vae import install_pwm_vae_fastpath

            self.optimization_info["vae"] = install_pwm_vae_fastpath(self.vae, scope=vae_fastpath)
        self.eval()

    def text_ids(self, caption: str):
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": caption}],
            tokenize=True,
            add_generation_prompt=True,
            add_vision_id=False,
            return_dict=False,
        )
        if self.tokenizer.bos_token_id is not None:
            ids = [self.tokenizer.bos_token_id] + ids
        ids = ids + [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|vision_start|>")]
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def vae_scale(self):
        mean = torch.tensor(self.vae.config.latents_mean, device=self.device, dtype=torch.bfloat16).view(1, 48, 1, 1, 1)
        inverse = 1 / torch.tensor(self.vae.config.latents_std, device=self.device, dtype=torch.bfloat16).view(
            1, 48, 1, 1, 1
        )
        return mean, inverse

    def encode_observation(self, request: PwmRequest):
        pixels = request.pixels[None].to(device=self.device, dtype=torch.float32) / 127.5 - 1
        mean, inverse = self.vae_scale()
        encoded = self.vae.encode(pixels.to(torch.bfloat16)).latent_dist.mode()
        first = ((encoded - mean) * inverse)[0].float()
        clean = first.new_zeros((48, (request.video_frames - 1) // 4 + 1, *first.shape[-2:]))
        clean[:, :1] = first
        return clean

    def sample(self, request: PwmRequest, clean: torch.Tensor):
        # NumPy RandomState, independently reset for each modality, matches the
        # architecture-invariant reference noise contract. No global RNG mutation.
        video = torch.from_numpy(
            np.random.RandomState(request.seed).standard_normal(tuple(clean.shape)).astype(np.float32)
        ).to(self.device)
        video[:, :1] = clean[:, :1]
        action = torch.from_numpy(
            np.random.RandomState(request.seed).standard_normal((request.action_length, 128)).astype(np.float32)
        ).to(self.device)
        latent = torch.cat((video.flatten(), action.flatten()))
        if self.cfg_parallel is not None:
            latent = self.cfg_parallel.synchronize_latent(latent)
        video_size = video.numel()
        video_indices = torch.arange(1, video.shape[1], device=self.device)
        action_indices = torch.arange(request.action_length, device=self.device)
        projection = torch.full((request.action_length,), request.projection_id, dtype=torch.long, device=self.device)
        branches = []
        for caption in (request.caption, request.negative_prompt):
            ids = self.text_ids(caption)
            grid = (video.shape[1], (video.shape[2] + 1) // 2, (video.shape[3] + 1) // 2)
            positions = position_ids(
                len(ids), grid, request.action_length, request.video_fps, request.action_fps, seam=request.seam
            )
            positions = tuple(p.to(self.device) for p in positions)
            ropes = tuple(rotary(p, dtype=torch.bfloat16) for p in positions)
            action_ropes = tuple(rotary(p, dtype=torch.bfloat16, action=True) for p in positions)
            # Per-request, per-CFG-branch ownership; never reused across calls.
            cache = [[] for _ in self.transformer.language_model.model.layers] if self.cache_und else None
            attention_metadata = None
            if self.attention_backend in ("flash3", "flash3_nv"):
                lengths = (len(ids), grid[0] * grid[1] * grid[2], request.action_length)
                attention_metadata = tuple(
                    torch.tensor([0, n], dtype=torch.int32, device=self.device) for n in (*lengths, sum(lengths))
                )
            branches.append((ids, ropes, action_ropes, cache, attention_metadata))
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=1.0,
            use_dynamic_shifting=False,
            use_cpu_coefficients=True,
            use_legacy_corrector=True,
        )
        scheduler.set_timesteps(request.steps, device=self.device, shift=request.shift)
        generator = torch.Generator(device=self.device).manual_seed(request.seed)
        for timestep in scheduler.timesteps:

            def velocity(branch):
                ids, rope, action_rope, cache, attention_metadata = branch
                streams = self.transformer.encode_inputs(
                    ids,
                    latent[:video_size].reshape(video.shape),
                    latent[video_size:].reshape(action.shape),
                    timestep.reshape(1),
                    projection,
                    video_indices,
                    action_indices,
                )
                forward_hidden = self.transformer.forward_hidden if self.graph_runner is None else self.graph_runner.run
                hidden = forward_hidden(streams, rope, action_rope, cache, attention_metadata)
                vv, va = self.transformer.decode_outputs(hidden, tuple(video.shape), projection, video_indices)
                return torch.cat((vv.flatten(), va.flatten()))

            needs_cfg = request.guidance != 1
            if request.guidance_interval is not None:
                lo, hi = request.guidance_interval
                needs_cfg = needs_cfg and lo < timestep.item() < hi
            if self.cfg_parallel is not None:
                predicted = self.cfg_parallel.predict(
                    lambda branch_index: velocity(branches[branch_index]),
                    needs_cfg=needs_cfg,
                    guidance=request.guidance,
                    normalize=request.normalize_cfg,
                )
            elif needs_cfg:
                conditional = velocity(branches[0])
                unconditional = velocity(branches[1])
                predicted = unconditional + request.guidance * (conditional - unconditional)
                if request.normalize_cfg:
                    predicted = predicted * (torch.norm(conditional) / (torch.norm(predicted) + 1e-8)).clamp(0, 1)
            else:
                predicted = velocity(branches[0])
            latent = scheduler.step(
                model_output=predicted, timestep=timestep, sample=latent[None], generator=generator, return_dict=False
            )[0].squeeze(0)
        return latent[:video_size].reshape(video.shape), latent[video_size:].reshape(action.shape)

    @torch.inference_mode()
    def forward(self, raw: dict):
        started = time.perf_counter()
        request = prepare_request(raw, self.spec)
        reproducibility = configure_validation(request.seed) if self.validation_deterministic else None
        clean = self.encode_observation(request)
        # Two CFG branches and cache fill/reuse specialize each input shape.
        # Match the reference capacity, scoped to this serialized worker call.
        limits = (
            torch._dynamo.config.patch(recompile_limit=512, accumulated_recompile_limit=4096)
            if self.compile_layers
            else nullcontext()
        )
        try:
            with limits:
                video, normalized = self.sample(request, clean)
        finally:
            if self.graph_runner is not None:
                self.graph_runner.end_request()
        normalized = normalized.float()
        normalizer = self.transformer.action_normalization
        domain = request.normalization_id
        physical = normalized.clone()
        transform_mask = normalizer.normalization_mask[domain]
        physical[:, transform_mask] = (
            physical[:, transform_mask] - normalizer.bias[domain, transform_mask]
        ) / normalizer.scale[domain, transform_mask]
        mask = torch.tensor(request.mask, device=self.device, dtype=torch.bool)
        normalized = normalized.masked_fill(~mask, 0)
        physical = physical.masked_fill(~mask, 0)
        if not torch.isfinite(normalized).all() or not torch.isfinite(physical).all():
            raise RuntimeError("Non-finite native WAM actions")
        result = dict(
            action=physical.cpu().tolist(),
            normalized_action=normalized.cpu().tolist(),
            action_domain=request.domain,
            action_dim_mask=request.mask,
            action_fps=request.action_fps,
            action_timestamps_seconds=(
                torch.arange(request.action_length, dtype=torch.float64) / request.action_fps
            ).tolist(),
            action_units="raw_physical_or_source_units",
            seed=request.seed,
            conditioning_caption_sha256=hashlib.sha256(request.caption.encode()).hexdigest(),
            elapsed_seconds=time.perf_counter() - started,
        )
        frames = None
        if reproducibility is not None:
            result["reproducibility"] = reproducibility
        if request.return_video:
            mean, inverse = self.vae_scale()
            decoded = self.vae.decode((video[None].to(torch.bfloat16) / inverse) + mean, return_dict=False)[0]
            frames = (
                (((decoded[0].float().permute(1, 2, 3, 0) + 1) / 2).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            )
        return result, frames, request.video_fps
