# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reference-only fixture capture; run in Cosmos env with worker CLI and stdin.

PWM_NATIVE_FIXTURE names an output directory. Stores tensors with safetensors,
never pickle. This is validation tooling, not part of native inference runtime.
"""

import dataclasses
import json
import os
import runpy
from pathlib import Path

import torch
from cosmos_framework.inference.inference import OmniInference
from cosmos_framework.model.generator.mot import small_action_tower
from safetensors.torch import save_file

output = Path(os.environ["PWM_NATIVE_FIXTURE"])
output.mkdir(parents=True, exist_ok=True)
original_create = OmniInference.create
original_forward = small_action_tower.small_layer_forward
calls = 0


def save_tree(name, value):
    tensors = {}

    def encode(item, key):
        if isinstance(item, torch.Tensor):
            tensors[key] = item.detach().cpu().contiguous()
            return {"tensor": key}
        if dataclasses.is_dataclass(item):
            return {
                field.name: encode(getattr(item, field.name), key + "." + field.name)
                for field in dataclasses.fields(item)
            }
        if isinstance(item, dict):
            return {str(k): encode(v, key + "." + str(k)) for k, v in item.items()}
        if isinstance(item, list | tuple):
            return [encode(v, key + "." + str(i)) for i, v in enumerate(item)]
        if item is None or isinstance(item, str | int | float | bool):
            return item
        return {"type": str(type(item))}

    structure = encode(value, name)
    save_file(tensors, str(output / (name + ".safetensors")))
    (output / (name + ".json")).write_text(json.dumps(structure, indent=2) + "\n")


def create(*args, **kwargs):
    inference = original_create(*args, **kwargs)
    net = inference.model.net
    metadata = {
        name: str(parameter.dtype)
        for name, parameter in net.named_parameters()
        if name.startswith(("time_embedder.", "action_time_embedder.", "vae2llm."))
    }
    (output / "dtypes.json").write_text(json.dumps(metadata, indent=2) + "\n")
    seen = set()

    def hook(name):
        def capture(module, inputs, kwargs, result):
            if name not in seen:
                seen.add(name)
                save_tree(name, {"inputs": inputs, "kwargs": kwargs, "output": result})

        return capture

    for name in ("time_embedder", "action_time_embedder", "vae2llm", "action2llm"):
        getattr(net, name).register_forward_hook(hook(name), with_kwargs=True)
    net.register_forward_hook(hook("network"), with_kwargs=True)
    return inference


def forward(layer, pack, **kwargs):
    global calls
    index = calls
    calls += 1
    captured = {}
    if index in (0, 35):
        for name in ("causal_seq", "full_only_seq", "action_seq"):
            captured["input." + name] = pack[name].detach().cpu().contiguous()
        for name in ("_small_rope", "_small_action_rope"):
            for stream, pair in enumerate(pack[name]):
                for component, tensor in enumerate(pair):
                    captured[f"{name}.{stream}.{component}"] = tensor.detach().cpu().contiguous()
    result = original_forward(layer, pack, **kwargs)
    if captured:
        for name in ("causal_seq", "full_only_seq", "action_seq"):
            captured["output." + name] = result[0][name].detach().cpu().contiguous()
        save_file(captured, str(output / f"layer_{index:02d}.safetensors"))
    return result


OmniInference.create = create
small_action_tower.small_layer_forward = forward
runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "vllm_omni/diffusion/models/pwm_wam/worker.py"), run_name="__main__"
)
