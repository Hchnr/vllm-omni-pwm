# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VIDEO projection fusion preserves three-tower semantics and weight ownership."""

import copy
import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from vllm_omni.diffusion.models.pwm_wam.transformer_pwm_wam import PwmThreeTowerLayer, PwmWamTransformer

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


def _layer():
    return PwmThreeTowerLayer(
        width=32, intermediate=48, action_width=16, action_intermediate=24, heads=4, kv_heads=2, dim=8
    ).eval()


def _inputs():
    lengths = (5, 8, 3)
    streams = (torch.randn(5, 32), torch.randn(8, 32), torch.randn(3, 16))
    rope = tuple((torch.randn(n, 8).cos(), torch.randn(n, 8).sin()) for n in lengths)
    action_rope = tuple((torch.randn(n, 8).cos(), torch.randn(n, 8).sin()) for n in lengths)
    return streams, rope, action_rope


def _storage_bytes(module):
    allocations = {p.untyped_storage().data_ptr(): p.untyped_storage().nbytes() for p in module.parameters()}
    return sum(allocations.values())


@pytest.fixture
def isolated_dynamo_cache():
    # Each configuration covers three cache states. Do not accumulate those
    # specializations across independently parametrized test cases.
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.mark.parametrize("gate_up,qkv", [(False, False), (True, False), (False, True), (True, True)])
@torch.inference_mode()
def test_video_fusions_preserve_all_streams_and_request_local_und_cache(gate_up, qkv):
    torch.manual_seed(1020)
    reference = _layer()
    fused = copy.deepcopy(reference)
    fused.configure_fusions(gate_up=gate_up, qkv=qkv)
    assert (fused.mlp_moe_gen.gate_up_proj is not None) == gate_up
    assert (fused.self_attn.qkv_proj_moe_gen is not None) == qkv
    assert fused.mlp.gate_up_proj is None
    assert fused.action_block_moe_gen_action.qkv_proj_moe_gen is None
    streams, rope, action_rope = _inputs()
    cache = []
    for _ in range(3):
        # Only UND remains fixed across denoising steps.
        streams = (streams[0], torch.randn_like(streams[1]), torch.randn_like(streams[2]))
        expected = reference(streams, rope, action_rope)
        actual = fused(streams, rope, action_rope, cache)
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    assert len(cache) == 1


@torch.inference_mode()
def test_fusion_roundtrip_recovers_original_keys_values_and_storage_size():
    torch.manual_seed(1020)
    layer = _layer().requires_grad_(False)
    initial = {key: value.clone() for key, value in layer.state_dict().items()}
    expected_bytes = _storage_bytes(layer)
    expected_numel = sum(p.numel() for p in layer.parameters())
    inputs = _inputs()
    expected_output = layer(*inputs)
    for gate_up, qkv in ((True, True), (False, True), (True, False), (False, False), (True, True), (False, False)):
        layer.configure_fusions(gate_up=gate_up, qkv=qkv)
        assert _storage_bytes(layer) == expected_bytes
        assert sum(p.numel() for p in layer.parameters()) == expected_numel
        assert all(not p.requires_grad for p in layer.parameters())
        pointers = [p.data_ptr() for p in layer.parameters()]
        layer.configure_fusions(gate_up=gate_up, qkv=qkv)
        assert [p.data_ptr() for p in layer.parameters()] == pointers
    assert layer.state_dict().keys() == initial.keys()
    for key, value in layer.state_dict().items():
        torch.testing.assert_close(value, initial[key], rtol=0, atol=0)
    for actual, expected in zip(layer(*inputs), expected_output, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("gate_up,qkv", [(True, False), (False, True), (True, True)])
@torch.inference_mode()
def test_fused_layers_capture_as_one_full_graph_on_cpu(gate_up, qkv, isolated_dynamo_cache):
    torch.manual_seed(1020)
    layer = _layer()
    layer.configure_fusions(gate_up=gate_up, qkv=qkv)
    inputs = _inputs()
    expected = layer(*inputs)
    # Eager backend verifies full-graph capture without requiring CUDA or
    # pretending to qualify a GPU Inductor performance result.
    compiled = torch.compile(layer, fullgraph=True, backend="eager")
    for actual, reference in zip(compiled(*inputs), expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    cache = []
    compiled(*inputs, cache)
    for actual, reference in zip(compiled(*inputs, cache), expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


class _TinyTransformer(PwmWamTransformer):
    """Exercise the production HF loader without allocating the full checkpoint."""

    def __init__(self, num_domains):
        nn.Module.__init__(self)
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        self.language_model.model.layers = nn.ModuleList([_layer(), _layer()])


def _write_checkpoint(path):
    torch.manual_seed(1020)
    model = _TinyTransformer(1)
    state = {"model.net." + key: value for key, value in model.state_dict().items()}
    config = {
        "model": {
            "config": {
                "tower_structure": "three_tower",
                "action_tower_preset": "width_only_1b",
                "action128_spec": {"registry": {"domains": ["test"]}},
            }
        }
    }
    (path / "config.json").write_text(json.dumps(config))
    index = {"weight_map": dict.fromkeys(state, "model.safetensors")}
    (path / "model.safetensors.index.json").write_text(json.dumps(index))
    save_file(state, path / "model.safetensors")
    return model, index


def test_hf_loader_validates_original_keys_before_runtime_fusion(tmp_path):
    expected, index = _write_checkpoint(tmp_path)
    loaded = _TinyTransformer.from_pretrained(tmp_path, device="cpu", dtype=torch.float32)
    assert loaded.state_dict().keys() == expected.state_dict().keys()
    loaded.configure_fusions(gate_up=True, qkv=True)
    assert all(layer.mlp_moe_gen.gate_up_proj is not None for layer in loaded.language_model.model.layers)
    loaded.configure_fusions()
    for key, value in loaded.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[key], rtol=0, atol=0)
    del index["weight_map"]["model.net.language_model.model.layers.0.mlp_moe_gen.gate_proj.weight"]
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="HF tensor mismatch"):
        _TinyTransformer.from_pretrained(tmp_path, device="cpu", dtype=torch.float32)


def test_fusion_rejects_meta_weights_before_allocating():
    with torch.device("meta"):
        layer = _layer()
    with pytest.raises(ValueError, match="requires loaded weights"):
        layer.configure_fusions(gate_up=True)
