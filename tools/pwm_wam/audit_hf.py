# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audit a native PWM HF export without allocating tensors or importing torch."""

import argparse
import hashlib
import json
import struct
from pathlib import Path


def audit(path: Path):
    if (path / "model_index.json").exists():
        raise ValueError("Native PWM HF artifacts must not contain a Diffusers model_index.json")
    config = json.loads((path / "config.json").read_text())["model"]["config"]
    if config["tower_structure"] != "three_tower" or config["action_tower_preset"] != "width_only_1b":
        raise ValueError("Expected the supported three_tower / width_only_1b architecture")
    if not config["action_gen"] or config["sound_gen"]:
        raise ValueError("Expected action generation without sound")
    weight_map = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    headers = {}
    for shard in set(weight_map.values()):
        shard_path = (path / shard).resolve()
        if not shard_path.is_relative_to(path.resolve()):
            raise ValueError("Shard path escapes checkpoint directory")
        with shard_path.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            if header_size > 100_000_000:
                raise ValueError("Unreasonably large safetensors header")
            headers[shard] = json.loads(stream.read(header_size))
    for key, shard in weight_map.items():
        if key not in headers[shard]:
            raise ValueError(f"Index tensor missing from shard: {key}")
    required = [
        f"model.net.action_normalization.{name}" for name in ("scale", "bias", "normalization_mask", "contract_digest")
    ]
    required += [
        f"model.net.language_model.model.layers.{i}.action_block_moe_gen_action.q_proj.weight" for i in range(36)
    ]
    missing = sorted(set(required) - weight_map.keys())
    if missing:
        raise ValueError(f"Missing required PWM weights: {missing}")
    spec = config["action128_spec"]
    return {
        "tensor_count": len(weight_map),
        "shard_count": len(headers),
        "action_block_tensor_count": sum("action_block_moe_gen_action" in k for k in weight_map),
        "normalization_domains": len(spec["registry"]["domains"]),
        "config_sha256": hashlib.sha256((path / "config.json").read_bytes()).hexdigest(),
        "index_sha256": hashlib.sha256((path / "model.safetensors.index.json").read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.checkpoint)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
