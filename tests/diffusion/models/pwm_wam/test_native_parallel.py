# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for CFG2 branch ownership, normalization and runtime routing."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.pwm_wam.parallel import PwmCfgParallel

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


class _FakeGroup:
    world_size = 2

    def __init__(self, rank, predictions):
        self.rank_in_group = rank
        self.predictions = predictions
        self.gather_count = 0
        self.broadcast_count = 0

    def all_gather(self, value, *, separate_tensors):
        assert separate_tensors
        torch.testing.assert_close(value, self.predictions[self.rank_in_group], rtol=0, atol=0)
        self.gather_count += 1
        return self.predictions

    def broadcast(self, value, src=0):
        assert src == 0
        self.broadcast_count += 1
        return value


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("normalize", [False, True])
def test_cfg2_computes_only_owned_branch_and_preserves_pwm_normalization(rank, normalize):
    predictions = [torch.tensor([1.0, 4.0, -2.0]), torch.tensor([-1.0, 0.25, 3.0])]
    group = _FakeGroup(rank, predictions)
    parallel = PwmCfgParallel(group)
    calls = []

    def velocity(index):
        calls.append(index)
        return predictions[index]

    observed = parallel.predict(velocity, needs_cfg=True, guidance=1.5, normalize=normalize)
    expected = predictions[1] + 1.5 * (predictions[0] - predictions[1])
    if normalize:
        expected *= (torch.norm(predictions[0]) / (torch.norm(expected) + 1e-8)).clamp(0, 1)
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    assert calls == [rank]
    assert group.gather_count == 1


@pytest.mark.parametrize("rank", [0, 1])
def test_cfg_interval_disabled_uses_conditional_without_collective(rank):
    predictions = [torch.ones(4), torch.zeros(4)]
    group = _FakeGroup(rank, predictions)
    parallel = PwmCfgParallel(group)
    calls = []

    def velocity(index):
        calls.append(index)
        return predictions[index]

    observed = parallel.predict(velocity, needs_cfg=False, guidance=4.0, normalize=True)
    torch.testing.assert_close(observed, predictions[0], rtol=0, atol=0)
    assert calls == [0]
    assert group.gather_count == 0


def test_zero_predictions_remain_finite_with_normalization():
    group = _FakeGroup(0, [torch.zeros(4), torch.zeros(4)])
    actual = PwmCfgParallel(group).predict(lambda _: torch.zeros(4), needs_cfg=True, guidance=2.0, normalize=True)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual) == 0


def test_cfg2_synchronizes_initial_latent_through_owned_group():
    expected = torch.arange(4, dtype=torch.float32)

    class BroadcastGroup(_FakeGroup):
        def broadcast(self, value, src=0):
            super().broadcast(value, src=src)
            return value.copy_(expected)

    group = BroadcastGroup(1, [])
    actual = PwmCfgParallel(group).synchronize_latent(torch.zeros(4))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert group.broadcast_count == 1


@pytest.mark.parametrize("world_size,rank", [(1, 0), (4, 0), (2, 2)])
def test_invalid_cfg_group_is_rejected(world_size, rank):
    with pytest.raises(ValueError, match="exactly two CFG ranks"):
        PwmCfgParallel(SimpleNamespace(world_size=world_size, rank_in_group=rank))


def test_runtime_cfg2_matches_sequential_unipc_and_halves_branch_calls():
    from vllm_omni.diffusion.models.pwm_wam.runtime import PwmWamRuntime

    clean = torch.zeros(48, 2, 2, 2)
    prediction_size = clean.numel() + 2 * 128
    predictions = [torch.full((prediction_size,), 0.125), torch.full((prediction_size,), -0.0625)]
    request = SimpleNamespace(
        seed=42,
        action_length=2,
        projection_id=0,
        caption="positive",
        negative_prompt="negative",
        video_fps=4,
        action_fps=2,
        seam=None,
        steps=2,
        shift=5.0,
        guidance=1.5,
        guidance_interval=None,
        normalize_cfg=True,
    )

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def encode_inputs(self, ids, *args):
            branch = ids[0].item()
            self.calls.append(branch)
            return branch

        def forward_hidden(self, branch, *args):
            return branch

        def decode_outputs(self, branch, *args):
            prediction = predictions[branch]
            return prediction[: clean.numel()].reshape(clean.shape), prediction[clean.numel() :].reshape(2, 128)

    def runtime(parallel):
        result = PwmWamRuntime.__new__(PwmWamRuntime)
        nn.Module.__init__(result)
        result.device = "cpu"
        result.cache_und = False
        result.attention_backend = "sdpa"
        result.cfg_parallel = parallel
        result.graph_runner = None
        result.approx_cache_threshold = 0.0
        result.transformer = Transformer()
        result.text_ids = lambda caption: torch.tensor([0 if caption == "positive" else 1])
        return result

    reference = runtime(None)
    expected = reference.sample(request, clean)
    assert reference.transformer.calls == [0, 1, 0, 1]
    for rank in (0, 1):
        group = _FakeGroup(rank, predictions)
        candidate = runtime(PwmCfgParallel(group))
        observed = candidate.sample(request, clean)
        assert candidate.transformer.calls == [rank, rank]
        assert group.gather_count == request.steps
        assert group.broadcast_count == 1
        for actual, wanted in zip(observed, expected, strict=True):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


def test_pipeline_routes_cfg2_and_rejects_other_parallel_topologies(monkeypatch):
    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
    from vllm_omni.diffusion.models.pwm_wam import pipeline_pwm_wam_native as pipeline

    captured = {}

    class RuntimeStub(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            captured.update(kwargs)

    monkeypatch.setattr(pipeline, "PwmWamRuntime", RuntimeStub)
    model_config = dict(pwm_experimental_backend=True, pwm_vae_path="vae", pwm_tokenizer_path="tokenizer")
    config = OmniDiffusionConfig(
        model="/nonexistent",
        enforce_eager=True,
        model_config=model_config,
        parallel_config=DiffusionParallelConfig(cfg_parallel_size=2),
    )
    pipeline.PwmWamNativePipeline(od_config=config)
    assert captured["cfg_parallel"] is True
    config = OmniDiffusionConfig(
        model="/nonexistent",
        enforce_eager=True,
        model_config=model_config,
        parallel_config=DiffusionParallelConfig(tensor_parallel_size=2),
    )
    with pytest.raises(ValueError, match="TP/SP/PP"):
        pipeline.PwmWamNativePipeline(od_config=config)
