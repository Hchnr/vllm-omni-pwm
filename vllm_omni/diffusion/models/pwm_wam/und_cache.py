# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-local UND cache that retains projected keys before RMS normalization.

Materializing a BF16 normalized key can change compiled norm/RoPE rounding.
Caching the linear projection keeps normalization inside the consuming graph.
This specialization supports only the width-only Action tower's shared KV geometry.
"""

from types import MethodType

import torch


def _project(module, hidden, heads, kv_heads, head_dim, suffix=""):
    q = getattr(module, "q_proj" + suffix)(hidden).view(-1, heads, head_dim)
    k = getattr(module, "k_proj" + suffix)(hidden).view(-1, kv_heads, head_dim)
    v = getattr(module, "v_proj" + suffix)(hidden).view(-1, kv_heads, head_dim)
    return q, k, v


def _forward(
    self, input, attention_mask, packed_position_embeddings, natten_metadata=None, memory_value=None, gen_only=False
):
    from cosmos_framework.model.generator.mot.small_action_tower import attend, rotate

    if gen_only or self.training or torch.is_grad_enabled():
        raise ValueError("Projected UND caching supports eval inference only")
    attn = self.self_attn
    action = self.action_block_moe_gen_action
    und, video, act = input["causal_seq"], input["full_only_seq"], input["action_seq"]
    heads, kv_heads, dim = attn.num_attention_heads, attn.num_key_value_heads, attn.head_dim
    cached = memory_value is not None and memory_value.und_key is not None
    if cached:
        ku, vu = memory_value.und_key, memory_value.und_value
    else:
        qu, ku, vu = _project(attn, self.input_layernorm(und), heads, kv_heads, dim)
    qv, kv, vv = _project(attn, self.input_layernorm_moe_gen(video), heads, kv_heads, dim, "_moe_gen")
    qa, ka, va = _project(action, action.input_layernorm(act), heads, kv_heads, dim)
    keys = (attn.k_norm(ku), attn.k_norm_moe_gen(kv), action.k_norm(ka))
    rope = input["_small_rope"]
    action_rope = input["_small_action_rope"]
    masks = input["_small_masks"]
    # Keep each consumer's norm/RoPE fusion identical to the uncached graph.
    video_keys = torch.cat([rotate(k, r) for k, r in zip(keys, rope)])
    action_keys = torch.cat([rotate(k, r) for k, r in zip(keys, action_rope)])
    values = torch.cat((vu, vv, va))
    video_attention = attend(rotate(attn.q_norm_moe_gen(qv), rope[1]), video_keys, values, masks[(True, 1)])
    action_attention = attend(rotate(action.q_norm(qa), action_rope[2]), action_keys, values, masks[(True, 2)])
    act = act + action.o_proj(action_attention)
    act = act + action.mlp(action.post_attention_layernorm(act))
    if cached:
        und = memory_value.und_output
    else:
        und_attention = attend(
            rotate(attn.q_norm(qu), rope[0]), rotate(keys[0], rope[0]), vu, masks[(False, 0)], frozen_und=True
        )
        und = und + attn.o_proj(und_attention)
        und = und + self.mlp(self.post_attention_layernorm(und))
    video = video + attn.o_proj_moe_gen(video_attention)
    video = video + self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(video))
    result = dict(input, causal_seq=und, full_only_seq=video, action_seq=act)
    store = (und, und.new_empty(0), ku, vu) if memory_value is not None and not cached else None
    return result, {}, store


def install_projected_und_cache(model) -> None:
    """Install before the first compiled forward; never mutate the Cosmos source."""
    if model.config.action_tower_preset != "width_only_1b":
        raise ValueError("Projected UND cache requires width_only_1b")
    for wrapped in model.net.language_model.model.layers:
        layer = getattr(wrapped, "_orig_mod", wrapped)
        block = layer.action_block_moe_gen_action
        if len(block.adapters):
            raise ValueError("Projected UND cache does not support KV axis adapters")
        layer.forward = MethodType(_forward, layer)
