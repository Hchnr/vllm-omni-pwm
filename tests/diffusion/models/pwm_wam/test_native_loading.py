# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU validation of HF loading modes, dtype contracts and transfer lifetime."""

import json
import weakref

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from vllm_omni.diffusion.models.pwm_wam.transformer_pwm_wam import PwmWamTransformer

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


class _TinyTransformer(PwmWamTransformer):
    def __init__(self, num_domains):
        nn.Module.__init__(self)
        self.projection = nn.Linear(4, 3)
        self.register_buffer("scale", torch.empty(num_domains, 3, dtype=torch.float32))
        self.register_buffer("mask", torch.empty(num_domains, 3, dtype=torch.bool))
        self.register_buffer("digest", torch.empty(16, dtype=torch.uint8))


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(1020)
    reference = _TinyTransformer(2)
    reference.scale.fill_(0.125)
    reference.mask.copy_(torch.tensor([[True, False, True], [False, True, False]]))
    reference.digest.copy_(torch.arange(16, dtype=torch.uint8))
    state = reference.state_dict()
    config = {
        "model": {
            "config": {
                "tower_structure": "three_tower",
                "action_tower_preset": "width_only_1b",
                "action128_spec": {"registry": {"domains": ["a", "b"]}},
            }
        }
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    # Explicit ordering exercises a failed second tensor after a successful
    # copy in the same shard, independent of Module.state_dict buffer ordering.
    first = ["projection.weight", "projection.bias"]
    second = ["scale", "mask", "digest"]
    weight_map = {}
    for shard, keys in (("a.safetensors", first), ("b.safetensors", second)):
        values = {"model.net." + key: state[key] for key in keys}
        save_file(values, tmp_path / shard)
        weight_map.update(dict.fromkeys(values, shard))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return tmp_path, reference, weight_map


@pytest.mark.parametrize("mode", ["serial", "async_h2d"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cpu_load_modes_preserve_keys_values_and_buffer_dtypes(checkpoint, monkeypatch, mode, dtype):
    path, reference, _ = checkpoint

    def no_accelerator(*args, **kwargs):
        pytest.fail("CPU loading must not synchronize or initialize an accelerator")

    monkeypatch.setattr(torch.accelerator, "synchronize", no_accelerator)
    loaded = _TinyTransformer.from_pretrained(path, device="cpu", dtype=dtype, load_mode=mode)
    parameter_keys = set(dict(reference.named_parameters()))
    assert loaded.state_dict().keys() == reference.state_dict().keys()
    for key, value in loaded.state_dict().items():
        expected = reference.state_dict()[key]
        expected = expected.to(dtype) if key in parameter_keys else expected
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    assert not loaded.training
    info = loaded.load_info
    assert info["mode"] == mode
    assert info["non_blocking_cuda"] is False
    assert info["tensor_count"] == 5
    assert info["shard_count"] == 2
    assert info["synchronization_count"] == 0
    assert info["max_pending_source_shards"] == 0
    assert info["loaded_tensor_bytes"] == sum(t.numel() * t.element_size() for t in loaded.state_dict().values())
    assert info["load_wall_seconds"] >= info["tensor_read_wall_seconds"] + info["transfer_enqueue_wall_seconds"]


def _simulate_cuda_transfers(monkeypatch):
    """Test enqueue/sync control flow on CPU; this does not validate CUDA DMA."""
    original_to = torch.Tensor.to
    events = []
    pending = []

    def transfer(value, *args, **kwargs):
        target = kwargs.get("device")
        if target is not None and torch.device(target).type == "cuda":
            events.append(("transfer", kwargs.get("non_blocking")))
            pending.append(weakref.ref(value))
            kwargs["device"] = "cpu"
        return original_to(value, *args, **kwargs)

    def synchronize(device):
        assert device == torch.device("cuda:3")
        assert pending and all(source() is not None for source in pending)
        events.append(("synchronize", len(pending)))
        pending.clear()

    monkeypatch.setattr(torch.Tensor, "to", transfer)
    monkeypatch.setattr(torch.accelerator, "synchronize", synchronize)
    return events


def test_async_enqueues_copies_and_drains_each_shard_before_return_cpu_simulation(checkpoint, monkeypatch):
    path, _, _ = checkpoint
    events = _simulate_cuda_transfers(monkeypatch)
    loaded = _TinyTransformer.from_pretrained(path, device="cuda:3", load_mode="async_h2d")
    assert events == [
        ("transfer", True),
        ("transfer", True),
        ("synchronize", 2),
        ("transfer", True),
        ("transfer", True),
        ("transfer", True),
        ("synchronize", 3),
    ]
    assert loaded.load_info["synchronization_count"] == 2
    assert loaded.load_info["max_pending_source_shards"] == 1
    assert loaded.load_info["non_blocking_cuda"] is True


@pytest.mark.parametrize("mode", ["serial", "async_h2d"])
def test_missing_extra_and_shape_errors_are_not_weakened(checkpoint, mode):
    path, _, weight_map = checkpoint
    index_path = path / "model.safetensors.index.json"
    for invalid in (dict(weight_map, extra="a.safetensors"), dict(list(weight_map.items())[1:])):
        index_path.write_text(json.dumps({"weight_map": invalid}))
        with pytest.raises(ValueError, match="HF tensor mismatch"):
            _TinyTransformer.from_pretrained(path, device="cpu", load_mode=mode)
    index_path.write_text(json.dumps({"weight_map": weight_map}))
    save_file(
        {"model.net.projection.weight": torch.ones(3, 4), "model.net.projection.bias": torch.ones(4)},
        path / "a.safetensors",
    )
    with pytest.raises(ValueError, match="HF shape mismatch for projection.bias"):
        _TinyTransformer.from_pretrained(path, device="cpu", load_mode=mode)


def test_async_drains_prior_copies_when_a_later_shape_check_fails_cpu_simulation(checkpoint, monkeypatch):
    path, _, _ = checkpoint
    save_file(
        {"model.net.projection.weight": torch.ones(3, 4), "model.net.projection.bias": torch.ones(4)},
        path / "a.safetensors",
    )
    events = _simulate_cuda_transfers(monkeypatch)
    with pytest.raises(ValueError, match="HF shape mismatch for projection.bias"):
        _TinyTransformer.from_pretrained(path, device="cuda:3", load_mode="async_h2d")
    assert events == [("transfer", True), ("synchronize", 1)]


def test_unimplemented_modes_fail_before_checkpoint_access(tmp_path):
    for mode in ("prefetch", "typo"):
        with pytest.raises(ValueError, match="load_mode must be serial or async_h2d"):
            _TinyTransformer.from_pretrained(tmp_path / "does-not-exist", device="cpu", load_mode=mode)
