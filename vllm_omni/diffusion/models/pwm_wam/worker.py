# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Isolated PWM runtime. Invoked as a file with the Cosmos environment's Python.

The JSONL protocol deliberately transports no Python objects or executable code.
Only this optional worker imports Cosmos; the Omni environment remains independent.
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-config", type=Path)
    parser.add_argument("--attention-backend", choices=("sdpa", "flash3"), default="flash3")
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--reuse-und", action="store_true")
    parser.add_argument("--projected-und", action="store_true")
    args = parser.parse_args()
    if args.projected_und and not args.reuse_und:
        parser.error("--projected-und requires --reuse-und")
    if args.precision == "float32" and args.attention_backend != "sdpa":
        parser.error("float32 requires --attention-backend sdpa")
    # Keep stdout exclusively for protocol records, including during imports.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    os.environ["COSMOS_TRAINING"] = "0"

    import torch
    from cosmos_framework.inference.action_serving import prepare_wam_request
    from cosmos_framework.inference.args import OmniSampleOverrides, OmniSetupOverrides
    from cosmos_framework.inference.inference import OmniInference
    from cosmos_framework.utils import misc
    from omegaconf import OmegaConf

    def emit(message):
        protocol.write(json.dumps(message, allow_nan=False) + "\n")
        protocol.flush()

    # Never inherit the parent engine / torchrun rendezvous. Port zero reserves
    # an ephemeral listener atomically, allowing independent workers to coexist.
    for key in tuple(os.environ):
        if key.startswith("TORCHELASTIC_") or key in ("MASTER_ADDR", "MASTER_PORT", "GROUP_RANK", "ROLE_RANK"):
            os.environ.pop(key, None)
    os.environ.update(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0", LOCAL_WORLD_SIZE="1")
    store = torch.distributed.TCPStore("127.0.0.1", 0, 1, True)
    torch.distributed.init_process_group("nccl", store=store, rank=0, world_size=1)
    with tempfile.TemporaryDirectory(prefix="pwm-wam-") as scratch:
        setup = OmniSetupOverrides(
            checkpoint_path=str(args.checkpoint),
            config_file=str(args.reference_config or args.checkpoint / "config.json"),
            output_dir=Path(scratch),
            experiment_overrides=[
                "model.config.ema.enabled=false",
                "model.config.action_tower_init=checkpoint",
                "model.config.fixed_sequence_length.enabled=false",
                "model.config.action_tower_und_short_tokens=0",
                f"model.config.action_tower_attention_backend={args.attention_backend}",
                f"model.config.precision={args.precision}",
            ],
            use_ema_weights=False,
            use_torch_compile=args.compile,
            compiled_region="language",
            compile_dynamic=False,
            use_cuda_graphs=False,
            guardrails=False,
            dp_shard_size=1,
            dp_replicate_size=1,
            cp_size=1,
            cfgp_size=1,
        ).build_setup(world_size=1)
        model = OmniInference.create(setup).model.eval()
        if model.config.tower_structure != "three_tower" or not model.config.action_gen:
            raise ValueError("PWM runtime requires an Action128 three_tower checkpoint")
        if args.projected_und:
            from und_cache import install_projected_und_cache

            install_projected_und_cache(model)
        spec = OmegaConf.to_container(model.config.action128_spec, resolve=True)
        normalizer = model.net.action_normalization
        normalizer.validate_contract()
        normalizer._reset_transforms()
        emit({"ready": True, "torch": torch.__version__, "precision": args.precision})
        try:
            for line in sys.stdin:
                try:
                    raw = json.loads(line)
                    if raw == {"shutdown": True}:
                        break
                    started = time.perf_counter()
                    request = OmniSampleOverrides.model_validate(raw)
                    with tempfile.TemporaryDirectory(dir=scratch) as request_dir:
                        prepared, batch = prepare_wam_request(
                            request, spec, model.config, Path(request_dir), model.input_video_key
                        )
                        if prepared.return_video:
                            raise ValueError("PWM action output requires extra.action128.return_video=false")
                        batch = misc.to(batch, device="cuda")
                        with torch.inference_mode():
                            samples = model.generate_samples_from_batch(
                                batch,
                                seed=[prepared.seed],
                                num_steps=prepared.num_steps,
                                guidance=prepared.guidance,
                                shift=prepared.shift,
                                guidance_interval=prepared.sample_args["guidance_interval"],
                                normalize_cfg=prepared.sample_args["normalize_cfg"],
                                sigma_max=prepared.sample_args["sigma_max"],
                                has_negative_prompt="neg_ai_caption" in batch,
                                reuse_small_action_metadata=True,
                                reuse_small_action_und=args.reuse_und,
                            )
                            normalized = samples["action"][0].float()
                            domain = spec["registry"]["domains"].index(prepared.action_domain)
                            physical = normalizer.denormalize(normalized, domain)
                            mask = torch.tensor(prepared.action_dim_mask, device=normalized.device, dtype=torch.bool)
                            normalized = normalized.masked_fill(~mask, 0).cpu()
                            physical = physical.masked_fill(~mask, 0).cpu()
                            if not torch.isfinite(normalized).all() or not torch.isfinite(physical).all():
                                raise RuntimeError("Non-finite Action128 output")
                        emit(
                            {
                                "result": {
                                    "action": physical.tolist(),
                                    "normalized_action": normalized.tolist(),
                                    "action_domain": prepared.action_domain,
                                    "action_dim_mask": prepared.action_dim_mask,
                                    "action_fps": prepared.action_fps,
                                    "action_timestamps_seconds": batch["sequence_plan"][0].action_timestamps_seconds,
                                    "action_units": "raw_physical_or_source_units",
                                    "seed": prepared.seed,
                                    "conditioning_caption_sha256": hashlib.sha256(
                                        batch["ai_caption"][0].encode()
                                    ).hexdigest(),
                                    "elapsed_seconds": time.perf_counter() - started,
                                }
                            }
                        )
                except (ValueError, TypeError, KeyError) as exc:
                    traceback.print_exc(file=sys.stderr)
                    emit({"error": str(exc), "error_type": "invalid_request"})
        finally:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
