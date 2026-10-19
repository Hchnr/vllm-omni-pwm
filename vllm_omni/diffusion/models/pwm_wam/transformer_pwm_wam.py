# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Standalone PWM three-tower transformer; no Cosmos runtime imports.

The native parameter tree deliberately matches the exported HF checkpoint. This
keeps every ACTION parameter and source-domain buffer auditable without reusing
the upstream shared-GEN weight mapping. Initial execution supports one sample.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import Tensor, nn

Rope = tuple[Tensor, Tensor]
Streams = tuple[Tensor, Tensor, Tensor]


class PwmRMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        value = x.float()
        value = value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * value.to(x.dtype)


def rotate(x: Tensor, rope: Rope) -> Tensor:
    left, right = x.chunk(2, dim=-1)
    return x * rope[0][:, None] + torch.cat((-right, left), dim=-1) * rope[1][:, None]


def attention(q: Tensor, k: Tensor, v: Tensor, *, causal: bool = False, backend: str = "sdpa") -> Tensor:
    if backend == "flash3":
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        cu_q = torch.tensor([0, q.shape[0]], dtype=torch.int32, device=q.device)
        cu_k = torch.tensor([0, k.shape[0]], dtype=torch.int32, device=k.device)
        result = flash_attn_varlen_func(
            q=q.contiguous(),
            k=k.contiguous(),
            v=v.contiguous(),
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=q.shape[0],
            max_seqlen_k=k.shape[0],
            causal=causal,
            softmax_scale=q.shape[-1] ** -0.5,
            fa_version=3,
            num_splits=1,
        )
        return result.flatten(1)
    if backend != "sdpa":
        raise ValueError(f"Unsupported native PWM attention backend: {backend}")
    mask = torch.ones((q.shape[0], k.shape[0]), dtype=torch.bool, device=q.device)
    if causal:
        mask = mask.tril()
    result = F.scaled_dot_product_attention(
        q.transpose(0, 1)[None],
        k.transpose(0, 1)[None],
        v.transpose(0, 1)[None],
        attn_mask=mask[None, None],
        enable_gqa=True,
        scale=q.shape[-1] ** -0.5,
    )
    return result[0].transpose(0, 1).flatten(1)


class PwmMLP(nn.Module):
    def __init__(self, width: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(width, intermediate, bias=False)
        self.up_proj = nn.Linear(width, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class PwmAttentionWeights(nn.Module):
    def __init__(self, width: int, *, video: bool = False, heads: int = 32, kv_heads: int = 8, dim: int = 128):
        super().__init__()
        self.heads, self.kv_heads, self.dim = heads, kv_heads, dim
        for suffix in ("", "_moe_gen") if video else ("",):
            for name, output in (("q_proj", heads * dim), ("k_proj", kv_heads * dim), ("v_proj", kv_heads * dim)):
                setattr(self, name + suffix, nn.Linear(width, output, bias=False))
            setattr(self, "o_proj" + suffix, nn.Linear(heads * dim, width, bias=False))
            setattr(self, "q_norm" + suffix, PwmRMSNorm(dim))
            setattr(self, "k_norm" + suffix, PwmRMSNorm(dim))

    def project(self, x: Tensor, suffix: str = "") -> tuple[Tensor, Tensor, Tensor]:
        q = getattr(self, "q_proj" + suffix)(x).view(-1, self.heads, self.dim)
        k = getattr(self, "k_proj" + suffix)(x).view(-1, self.kv_heads, self.dim)
        v = getattr(self, "v_proj" + suffix)(x).view(-1, self.kv_heads, self.dim)
        return getattr(self, "q_norm" + suffix)(q), getattr(self, "k_norm" + suffix)(k), v


class PwmActionBlock(PwmAttentionWeights):
    def __init__(self, width: int = 1280, intermediate: int = 3840, **kwargs):
        super().__init__(width, **kwargs)
        self.input_layernorm = PwmRMSNorm(width)
        self.post_attention_layernorm = PwmRMSNorm(width)
        self.gate_proj = nn.Linear(width, intermediate, bias=False)
        self.up_proj = nn.Linear(width, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, width, bias=False)

    def mlp(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class PwmThreeTowerLayer(nn.Module):
    def __init__(
        self,
        width: int = 4096,
        intermediate: int = 12288,
        action_width: int = 1280,
        action_intermediate: int = 3840,
        heads: int = 32,
        kv_heads: int = 8,
        dim: int = 128,
    ):
        super().__init__()
        self.self_attn = PwmAttentionWeights(width, video=True, heads=heads, kv_heads=kv_heads, dim=dim)
        self.attention_backend = "sdpa"
        self.input_layernorm = PwmRMSNorm(width)
        self.input_layernorm_moe_gen = PwmRMSNorm(width)
        self.post_attention_layernorm = PwmRMSNorm(width)
        self.post_attention_layernorm_moe_gen = PwmRMSNorm(width)
        self.mlp = PwmMLP(width, intermediate)
        self.mlp_moe_gen = PwmMLP(width, intermediate)
        self.action_block_moe_gen_action = PwmActionBlock(
            action_width, action_intermediate, heads=heads, kv_heads=kv_heads, dim=dim
        )

    def forward(self, streams: Streams, rope: tuple[Rope, Rope, Rope], action_rope: tuple[Rope, Rope, Rope]) -> Streams:
        und, video, action = streams
        attn, act = self.self_attn, self.action_block_moe_gen_action
        qu, ku, vu = attn.project(self.input_layernorm(und))
        qv, kv, vv = attn.project(self.input_layernorm_moe_gen(video), "_moe_gen")
        qa, ka, va = act.project(act.input_layernorm(action))
        keys, values = (ku, kv, ka), (vu, vv, va)
        yu = attention(rotate(qu, rope[0]), rotate(ku, rope[0]), vu, causal=True, backend=self.attention_backend)
        yv = attention(
            rotate(qv, rope[1]),
            torch.cat([rotate(k, r) for k, r in zip(keys, rope)]),
            torch.cat(values),
            backend=self.attention_backend,
        )
        ya = attention(
            rotate(qa, action_rope[2]),
            torch.cat([rotate(k, r) for k, r in zip(keys, action_rope)]),
            torch.cat(values),
            backend=self.attention_backend,
        )
        action = action + act.o_proj(ya)
        action = action + act.mlp(act.post_attention_layernorm(action))
        und = und + attn.o_proj(yu)
        und = und + self.mlp(self.post_attention_layernorm(und))
        video = video + attn.o_proj_moe_gen(yv)
        video = video + self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(video))
        return und, video, action


class PwmDomainLinear(nn.Module):
    def __init__(self, input_width: int, output_width: int):
        super().__init__()
        self.input_width, self.output_width = input_width, output_width
        self.fc = nn.Embedding(64, input_width * output_width)
        self.bias = nn.Embedding(64, output_width)

    def forward(self, x: Tensor, domain_ids: Tensor) -> Tensor:
        weights = self.fc(domain_ids).view(-1, self.input_width, self.output_width)
        return torch.bmm(x[:, None], weights).squeeze(1) + self.bias(domain_ids)


class PwmTimeEmbedding(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))
        frequencies = torch.exp(-math.log(10000) * torch.arange(128, dtype=torch.float32, device="cpu") / 128)
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, time: Tensor) -> Tensor:
        phase = time.float()[:, None] * self.frequencies.to(time.device)[None]
        encoded = torch.cat((phase.cos(), phase.sin()), dim=-1)
        # Reference uses CUDA FP32 autocast even with BF16 stored weights.
        # Explicit casts preserve that contract across PyTorch environments.
        first, last = self.mlp[0], self.mlp[2]
        hidden = F.silu(F.linear(encoded, first.weight.float(), first.bias.float()))
        return F.linear(hidden, last.weight.float(), last.bias.float())


class PwmTextStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(151936, 4096)
        self.layers = nn.ModuleList([PwmThreeTowerLayer() for _ in range(36)])
        self.norm = PwmRMSNorm(4096)
        self.norm_moe_gen = PwmRMSNorm(4096)
        self.norm_moe_gen_action = PwmRMSNorm(1280)


class PwmLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = PwmTextStack()
        # Retained for strict checkpoint coverage; WAM does not emit text logits.
        self.lm_head = nn.Linear(4096, 151936, bias=False)


class PwmNormalizationBuffers(nn.Module):
    def __init__(self, num_domains: int):
        super().__init__()
        self.register_buffer("scale", torch.empty(num_domains, 128, dtype=torch.float32))
        self.register_buffer("bias", torch.empty(num_domains, 128, dtype=torch.float32))
        self.register_buffer("normalization_mask", torch.empty(num_domains, 128, dtype=torch.bool))
        self.register_buffer("contract_digest", torch.empty(16, dtype=torch.uint8))


class PwmWamTransformer(nn.Module):
    def __init__(self, num_domains: int):
        super().__init__()
        self.language_model = PwmLanguageModel()
        self.vae2llm = nn.Linear(192, 4096)
        self.llm2vae = nn.Linear(4096, 192)
        self.time_embedder = PwmTimeEmbedding(4096)
        self.action_time_embedder = PwmTimeEmbedding(1280)
        self.action2llm = PwmDomainLinear(128, 1280)
        self.llm2action = PwmDomainLinear(1280, 128)
        self.action_modality_embed = nn.Parameter(torch.zeros(1280))
        self.action_normalization = PwmNormalizationBuffers(num_domains)

    @classmethod
    def from_pretrained(cls, path: Path, *, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        config = json.loads((path / "config.json").read_text())["model"]["config"]
        if config["tower_structure"] != "three_tower" or config["action_tower_preset"] != "width_only_1b":
            raise ValueError("Native PWM requires Nano width_only_1b three-tower weights")
        with torch.device("meta"):
            model = cls(len(config["action128_spec"]["registry"]["domains"]))
        expected = model.state_dict()
        index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
        source = {key.removeprefix("model.net."): shard for key, shard in index.items()}
        if set(expected) != set(source):
            raise ValueError(
                f"HF tensor mismatch: missing={set(expected) - set(source)}, extra={set(source) - set(expected)}"
            )
        parameters = set(dict(model.named_parameters()))
        for shard in sorted(set(source.values())):
            with safe_open(path / shard, framework="pt", device="cpu") as tensors:
                for name, filename in source.items():
                    if filename != shard:
                        continue
                    value = tensors.get_tensor("model.net." + name)
                    if value.shape != expected[name].shape:
                        raise ValueError(f"HF shape mismatch for {name}: {value.shape} != {expected[name].shape}")
                    target_dtype = dtype if name in parameters else expected[name].dtype
                    value = value.to(device=device, dtype=target_dtype)
                    owner, _, leaf = name.rpartition(".")
                    module = model.get_submodule(owner) if owner else model
                    setattr(module, leaf, nn.Parameter(value, requires_grad=False) if name in parameters else value)
        return model.eval()

    def encode_inputs(
        self,
        text_ids: Tensor,
        video_latents: Tensor,
        actions: Tensor,
        timestep: Tensor,
        projection_ids: Tensor,
        noisy_video_frames: Tensor,
        noisy_action_frames: Tensor,
    ) -> Streams:
        """Embed one packed WAM sample, preserving the reference operation order.

        Video layout is C,T,H,W; timestep is the sampler's unscaled scalar.
        Caller supplies noisy indices explicitly so conditioning never gets a
        timestep embedding. Projection IDs are not normalization-domain IDs.
        """
        channels, frames, height, width = video_latents.shape
        if channels != 48 or actions.ndim != 2 or actions.shape[1] != 128:
            raise ValueError("PWM requires 48-channel video latents and Action128")
        if timestep.numel() != 1:
            raise ValueError("Native WAM currently supports one timestep per request")
        dtype = self.vae2llm.weight.dtype
        padded = F.pad(video_latents, (0, width % 2, 0, height % 2))
        hp, wp = (height + 1) // 2, (width + 1) // 2
        patches = padded.reshape(channels, frames, hp, 2, wp, 2)
        patches = patches.permute(1, 2, 4, 3, 5, 0).reshape(-1, 192)
        video = self.vae2llm(patches.to(dtype))
        scaled_time = timestep.reshape(1).float() * 0.001
        video_time = self.time_embedder(scaled_time).to(dtype)
        spatial = torch.arange(hp * wp, device=video.device)
        noisy_tokens = (noisy_video_frames[:, None] * (hp * wp) + spatial).flatten()
        video[noisy_tokens] = video[noisy_tokens] + video_time
        action = self.action2llm(actions.to(dtype), projection_ids)
        action = action + self.action_modality_embed[None]
        action_time = self.action_time_embedder(scaled_time).to(dtype)
        action[noisy_action_frames] = action[noisy_action_frames] + action_time
        return self.language_model.model.embed_tokens(text_ids), video, action

    def decode_outputs(
        self,
        streams: Streams,
        video_shape: tuple[int, int, int, int],
        projection_ids: Tensor,
        noisy_video_frames: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Decode final normalized hidden states into video/action velocities."""
        _, frames, height, width = video_shape
        hp, wp = (height + 1) // 2, (width + 1) // 2
        video = streams[1].reshape(frames, hp * wp, -1)[noisy_video_frames].flatten(0, 1)
        patches = self.llm2vae(video).reshape(-1, hp, wp, 2, 2, 48)
        decoded = patches.permute(5, 0, 1, 3, 2, 4).reshape(48, -1, hp * 2, wp * 2)
        velocity = decoded.new_zeros(video_shape)
        velocity[:, noisy_video_frames] = decoded[:, :, :height, :width]
        return velocity, self.llm2action(streams[2], projection_ids)

    def forward_hidden(
        self, streams: Streams, rope: tuple[Rope, Rope, Rope], action_rope: tuple[Rope, Rope, Rope]
    ) -> Streams:
        for layer in self.language_model.model.layers:
            streams = layer(streams, rope, action_rope)
        stack = self.language_model.model
        return stack.norm(streams[0]), stack.norm_moe_gen(streams[1]), stack.norm_moe_gen_action(streams[2])
