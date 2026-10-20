# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Probe actual H100 FA3 FP8, including constant-V and split-matched controls."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1", "2", "3"):
        parser.error("Select exactly one physical GPU from 0--3")
    import torch

    from vllm_omni.diffusion.models.pwm_wam.attention_fp8 import fp8_attention, resolve_fp8_attention_splits
    from vllm_omni.diffusion.models.pwm_wam.reproducibility import configure_validation
    from vllm_omni.diffusion.models.pwm_wam.transformer_pwm_wam import pwm_fa3_attention

    configure_validation(42)
    result = dict(scope="Eager microbenchmark including quantization; not model or service latency", probes=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for nq, nk in ((128, 192), (6600, 6967), (19170, 19320)):
            q = torch.randn(nq, 32, 128, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(nk, 8, 128, device="cuda", dtype=torch.bfloat16)
            cuq = torch.tensor([0, nq], device="cuda", dtype=torch.int32)
            cuk = torch.tensor([0, nk], device="cuda", dtype=torch.int32)
            guarded = resolve_fp8_attention_splits(nk, 0)
            for constant in (True, False):
                v = torch.full_like(k, 0.25) if constant else torch.randn_like(k)

                def bf16(splits):
                    return pwm_fa3_attention(q, k, v, cuq, cuk, False, "flash3", nq, nk, splits).flatten(1)

                reference = bf16(1)
                methods = {
                    "bf16_split1": lambda: bf16(1),
                    "bf16_guarded_splits": lambda: bf16(guarded),
                    "fp8_split1": lambda: fp8_attention(q, k, v, metadata=(cuq, cuk), num_splits=1),
                    "fp8_guarded_splits": lambda: fp8_attention(q, k, v, metadata=(cuq, cuk)),
                }
                for name, function in methods.items():
                    row = dict(query_tokens=nq, key_tokens=nk, constant_v=constant, method=name, guarded_splits=guarded)
                    try:
                        for _ in range(3):
                            output = function()
                        times = []
                        for _ in range(20):
                            torch.accelerator.synchronize()
                            started = time.perf_counter()
                            output = function()
                            torch.accelerator.synchronize()
                            times.append((time.perf_counter() - started) * 1000)
                        error = output.float() - reference.float()
                        row.update(
                            p50_ms=statistics.median(times),
                            max_abs=float(error.abs().max()),
                            rmse=float(error.square().mean().sqrt()),
                            finite=bool(torch.isfinite(output).all()),
                            constant_v_max_error=float((output.float() - 0.25).abs().max()) if constant else None,
                        )
                    except RuntimeError as exc:
                        row["error"] = str(exc)
                    result["probes"].append(row)
                    args.output.write_text(json.dumps(result, indent=2) + "\n")
                    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
