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
# The extended form is emitted only by fixed-shape request batching: the
# permutation packs each sample's UND/VIDEO/ACTION keys, followed by exact
# per-sample maximum lengths. Four-tensor metadata retains the B1 contract.
AttentionMetadata = (
    tuple[Tensor, Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor, Tensor, tuple[int, int, int, int]]
)


def _linear_with_weight(weight: Tensor, *, requires_grad: bool) -> nn.Linear:
    """Build a linear without allocating an unused initialization tensor."""
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False, device="meta", dtype=weight.dtype)
    linear.weight = nn.Parameter(weight, requires_grad=requires_grad)
    return linear


def _pack_linears(owner: nn.Module, names: tuple[str, ...], packed_name: str) -> None:
    """Move projections to one weight allocation, retaining no dense copies."""
    linears = [getattr(owner, name) for name in names]
    first = linears[0].weight
    if first.is_meta:
        raise ValueError("PWM projection fusion requires loaded weights")
    if any(
        linear.bias is not None
        or linear.weight.shape[1] != first.shape[1]
        or linear.weight.dtype != first.dtype
        or linear.weight.device != first.device
        or linear.weight.requires_grad != first.requires_grad
        for linear in linears
    ):
        raise ValueError("PWM fusion requires compatible bias-free projection weights")
    with torch.no_grad():
        weight = torch.cat([linear.weight for linear in linears], dim=0)
    setattr(owner, packed_name, _linear_with_weight(weight, requires_grad=first.requires_grad))
    for name in names:
        delattr(owner, name)


def _unpack_linear(owner: nn.Module, names: tuple[str, ...], sizes: tuple[int, ...], packed_name: str) -> None:
    """Restore original projection names using views of the packed storage."""
    packed = getattr(owner, packed_name)
    for name, weight in zip(names, packed.weight.detach().split(sizes, dim=0), strict=True):
        setattr(owner, name, _linear_with_weight(weight, requires_grad=packed.weight.requires_grad))
    setattr(owner, packed_name, None)


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


@torch.library.custom_op("vllm_omni::pwm_fa3_attention", mutates_args=())
def pwm_fa3_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_q: Tensor,
    cu_k: Tensor,
    causal: bool,
    provider: str,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
) -> Tensor:
    """Keep the inference-only FA3 call opaque to Inductor with known metadata."""
    kwargs = dict(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_seqlen_q or q.shape[0],
        max_seqlen_k=max_seqlen_k or k.shape[0],
        causal=causal,
        softmax_scale=q.shape[-1] ** -0.5,
        num_splits=1,
    )
    if provider == "flash3_nv":
        from flash_attn_3_nv.flash_attn_interface import flash_attn_varlen_func

        return flash_attn_varlen_func(**kwargs, return_attn_probs=True)[0]
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    return flash_attn_varlen_func(**kwargs, fa_version=3)


@pwm_fa3_attention.register_fake
def _pwm_fa3_attention_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_q: Tensor,
    cu_k: Tensor,
    causal: bool,
    provider: str,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
) -> Tensor:
    return torch.empty(q.shape, dtype=q.dtype, device=q.device)


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    causal: bool = False,
    backend: str = "sdpa",
    metadata: tuple[Tensor, Tensor] | None = None,
    max_seqlens: tuple[int, int] | None = None,
) -> Tensor:
    if backend in ("flash3", "flash3_nv"):
        if metadata is None:
            metadata = (
                torch.tensor([0, q.shape[0]], dtype=torch.int32, device=q.device),
                torch.tensor([0, k.shape[0]], dtype=torch.int32, device=k.device),
            )
        maximum = (q.shape[0], k.shape[0]) if max_seqlens is None else max_seqlens
        return pwm_fa3_attention(q, k, v, *metadata, causal, backend, *maximum).flatten(1)
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
        self.gate_up_proj = None

    def configure_gate_up_fusion(self, enabled: bool) -> None:
        if enabled and self.gate_up_proj is None:
            _pack_linears(self, ("gate_proj", "up_proj"), "gate_up_proj")
        elif not enabled and self.gate_up_proj is not None:
            intermediate = self.down_proj.in_features
            _unpack_linear(self, ("gate_proj", "up_proj"), (intermediate, intermediate), "gate_up_proj")

    def forward(self, x: Tensor) -> Tensor:
        if self.gate_up_proj is not None:
            gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
            return self.down_proj(F.silu(gate) * up)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class PwmAttentionWeights(nn.Module):
    def __init__(self, width: int, *, video: bool = False, heads: int = 32, kv_heads: int = 8, dim: int = 128):
        super().__init__()
        self.heads, self.kv_heads, self.dim = heads, kv_heads, dim
        self.qkv_proj_moe_gen = None
        for suffix in ("", "_moe_gen") if video else ("",):
            for name, output in (("q_proj", heads * dim), ("k_proj", kv_heads * dim), ("v_proj", kv_heads * dim)):
                setattr(self, name + suffix, nn.Linear(width, output, bias=False))
            setattr(self, "o_proj" + suffix, nn.Linear(heads * dim, width, bias=False))
            setattr(self, "q_norm" + suffix, PwmRMSNorm(dim))
            setattr(self, "k_norm" + suffix, PwmRMSNorm(dim))

    def configure_video_qkv_fusion(self, enabled: bool) -> None:
        names = ("q_proj_moe_gen", "k_proj_moe_gen", "v_proj_moe_gen")
        if enabled and self.qkv_proj_moe_gen is None:
            _pack_linears(self, names, "qkv_proj_moe_gen")
        elif not enabled and self.qkv_proj_moe_gen is not None:
            sizes = (self.heads * self.dim, self.kv_heads * self.dim, self.kv_heads * self.dim)
            _unpack_linear(self, names, sizes, "qkv_proj_moe_gen")

    def project(self, x: Tensor, suffix: str = "", *, normalize_key: bool = True) -> tuple[Tensor, Tensor, Tensor]:
        if suffix == "_moe_gen" and self.qkv_proj_moe_gen is not None:
            sizes = (self.heads * self.dim, self.kv_heads * self.dim, self.kv_heads * self.dim)
            q, k, v = self.qkv_proj_moe_gen(x).split(sizes, dim=-1)
        else:
            q = getattr(self, "q_proj" + suffix)(x)
            k = getattr(self, "k_proj" + suffix)(x)
            v = getattr(self, "v_proj" + suffix)(x)
        q = q.reshape(-1, self.heads, self.dim)
        k = k.reshape(-1, self.kv_heads, self.dim)
        v = v.reshape(-1, self.kv_heads, self.dim)
        key = getattr(self, "k_norm" + suffix)(k) if normalize_key else k
        return getattr(self, "q_norm" + suffix)(q), key, v


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

    def configure_fusions(self, *, gate_up: bool = False, qkv: bool = False) -> None:
        """Select VIDEO projections before compilation, leaving UND/ACTION intact."""
        self.mlp_moe_gen.configure_gate_up_fusion(gate_up)
        self.self_attn.configure_video_qkv_fusion(qkv)

    def forward(
        self,
        streams: Streams,
        rope: tuple[Rope, Rope, Rope],
        action_rope: tuple[Rope, Rope, Rope],
        und_cache: list[tuple[Tensor, Tensor, Tensor]] | None = None,
        attention_metadata: AttentionMetadata | None = None,
    ) -> Streams:
        und, video, action = streams
        attn, act = self.self_attn, self.action_block_moe_gen_action
        cached = und_cache is not None and len(und_cache) != 0
        if cached:
            und_output, raw_ku, vu = und_cache[0]
        else:
            qu, raw_ku, vu = attn.project(self.input_layernorm(und), normalize_key=False)
        # A normalized cached K forces BF16 materialization between norm/RoPE.
        # Preserve the uncached compiler fusion by retaining the linear output.
        ku = attn.k_norm(raw_ku)
        qv, kv, vv = attn.project(self.input_layernorm_moe_gen(video), "_moe_gen")
        qa, ka, va = act.project(act.input_layernorm(action))
        keys, values = (ku, kv, ka), (vu, vv, va)
        cu = attention_metadata
        packed = cu is not None and len(cu) == 6
        if packed and self.attention_backend not in ("flash3", "flash3_nv"):
            raise ValueError("Packed PWM request batches require Flash Attention 3")
        if cu is not None and len(cu) not in (4, 6):
            raise ValueError("PWM attention metadata requires 4 tensors or the fixed-batch extension")
        maximum = cu[5] if packed else None
        if not cached:
            yu = attention(
                rotate(qu, rope[0]),
                rotate(ku, rope[0]),
                vu,
                causal=True,
                backend=self.attention_backend,
                metadata=None if cu is None else (cu[0], cu[0]),
                max_seqlens=None if maximum is None else (maximum[0], maximum[0]),
            )
        video_keys = torch.cat([rotate(k, r) for k, r in zip(keys, rope)])
        video_values = torch.cat(values)
        if packed:
            video_keys = video_keys.index_select(0, cu[4])
            video_values = video_values.index_select(0, cu[4])
        yv = attention(
            rotate(qv, rope[1]),
            video_keys,
            video_values,
            backend=self.attention_backend,
            metadata=None if cu is None else (cu[1], cu[3]),
            max_seqlens=None if maximum is None else (maximum[1], maximum[3]),
        )
        action_keys = torch.cat([rotate(k, r) for k, r in zip(keys, action_rope)])
        if packed:
            action_keys = action_keys.index_select(0, cu[4])
        ya = attention(
            rotate(qa, action_rope[2]),
            action_keys,
            video_values if packed else torch.cat(values),
            backend=self.attention_backend,
            metadata=None if cu is None else (cu[2], cu[3]),
            max_seqlens=None if maximum is None else (maximum[2], maximum[3]),
        )
        action = action + act.o_proj(ya)
        action = action + act.mlp(act.post_attention_layernorm(action))
        if cached:
            und = und_output
        else:
            und = und + attn.o_proj(yu)
            und = und + self.mlp(self.post_attention_layernorm(und))
            if und_cache is not None:
                und_cache.append((und, raw_ku, vu))
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

    def configure_fusions(self, *, gate_up: bool = False, qkv: bool = False) -> None:
        """Configure loaded weights before compiling layers or serving requests.

        HF loading always validates the original parameter tree first. Fusion
        packs only VIDEO projections and changes their runtime state_dict keys;
        disable both options to recover the original checkpoint layout. Packed
        weights replace the original allocations. Disabling uses split views,
        so neither mode keeps a second dense copy of the weights.
        """
        for layer in self.language_model.model.layers:
            layer.configure_fusions(gate_up=gate_up, qkv=qkv)

    @classmethod
    def from_pretrained(
        cls,
        path: Path,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        load_mode: str = "serial",
    ):
        """Load the original HF parameter tree, optionally batching H2D waits.

        ``async_h2d`` enqueues nonblocking CUDA transfers, retaining each CPU
        source tensor until an explicit synchronization at the shard boundary.
        This bounds pending sources to one shard and guarantees ready weights
        on return, including when the caller's current device differs from the
        requested target. CPU loading never initializes an accelerator.

        ``load_info`` contains host wall times. Safetensors reads are mmap
        views: page faults, dtype conversion and transfer waits can occur in
        ``transfer_enqueue_wall_seconds``; it is not pure GPU copy time.
        """
        import time

        if load_mode not in ("serial", "async_h2d"):
            raise ValueError("PWM load_mode must be serial or async_h2d; shard prefetch is not implemented")
        started = time.perf_counter()
        target_device = torch.device(device)
        async_cuda = load_mode == "async_h2d" and target_device.type == "cuda"
        read_wall = 0.0
        transfer_wall = 0.0
        sync_wall = 0.0
        sync_count = 0
        loaded_bytes = 0
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
        shards = sorted(set(source.values()))
        for shard in shards:
            with safe_open(path / shard, framework="pt", device="cpu") as tensors:
                pending_sources = []
                try:
                    for name, filename in source.items():
                        if filename != shard:
                            continue
                        read_started = time.perf_counter()
                        value = tensors.get_tensor("model.net." + name)
                        read_wall += time.perf_counter() - read_started
                        if value.shape != expected[name].shape:
                            raise ValueError(f"HF shape mismatch for {name}: {value.shape} != {expected[name].shape}")
                        target_dtype = dtype if name in parameters else expected[name].dtype
                        if async_cuda:
                            pending_sources.append(value)
                        transfer_started = time.perf_counter()
                        value = value.to(device=target_device, dtype=target_dtype, non_blocking=async_cuda)
                        transfer_wall += time.perf_counter() - transfer_started
                        loaded_bytes += value.numel() * value.element_size()
                        owner, _, leaf = name.rpartition(".")
                        module = model.get_submodule(owner) if owner else model
                        setattr(module, leaf, nn.Parameter(value, requires_grad=False) if name in parameters else value)
                finally:
                    # Also drain already-enqueued copies if a later key/shape
                    # or transfer fails, before closing the mmap-backed shard.
                    if pending_sources:
                        sync_started = time.perf_counter()
                        torch.accelerator.synchronize(target_device)
                        sync_wall += time.perf_counter() - sync_started
                        sync_count += 1
                        pending_sources.clear()
        model.load_info = dict(
            mode=load_mode,
            device=str(target_device),
            non_blocking_cuda=async_cuda,
            tensor_count=len(source),
            shard_count=len(shards),
            loaded_tensor_bytes=loaded_bytes,
            tensor_read_wall_seconds=read_wall,
            transfer_enqueue_wall_seconds=transfer_wall,
            synchronization_wall_seconds=sync_wall,
            synchronization_count=sync_count,
            max_pending_source_shards=1 if async_cuda else 0,
            load_wall_seconds=time.perf_counter() - started,
        )
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
        self,
        streams: Streams,
        rope: tuple[Rope, Rope, Rope],
        action_rope: tuple[Rope, Rope, Rope],
        und_cache: list[list[tuple[Tensor, Tensor, Tensor]]] | None = None,
        attention_metadata: AttentionMetadata | None = None,
    ) -> Streams:
        for index, layer in enumerate(self.language_model.model.layers):
            streams = layer(
                streams, rope, action_rope, None if und_cache is None else und_cache[index], attention_metadata
            )
        stack = self.language_model.model
        return stack.norm(streams[0]), stack.norm_moe_gen(streams[1]), stack.norm_moe_gen_action(streams[2])
