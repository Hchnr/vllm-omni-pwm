# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Numerical/video review artifacts; similarity is not a task-quality verdict.

Requires NumPy, Pillow, scikit-image and imageio. Compare raw uint8 frames, not
lossy review MP4s. The output deliberately has no automatic video-quality gate.
"""

import argparse
import html
import json
import math
import os
from pathlib import Path

import numpy as np
from compare import compare
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity


def video_metrics(reference, candidate):
    if reference.shape != candidate.shape or reference.ndim != 4 or reference.shape[-1] != 3:
        raise ValueError("Video frame count, dimensions, and RGB layout must match")
    if not all(reference.shape):
        raise ValueError("Video dimensions must be nonempty")
    if reference.dtype != np.uint8 or candidate.dtype != np.uint8:
        raise ValueError("Compare raw uint8 decoded frames")
    mse, mae, ssim, motion = [], [], [], []
    previous_r = previous_c = None
    max_error = 0
    for r, c in zip(reference, candidate):
        rf, cf = r.astype(np.float32), c.astype(np.float32)
        error = cf - rf
        mse.append(float(np.mean(error**2)))
        mae.append(float(np.mean(np.abs(error))))
        max_error = max(max_error, int(np.abs(error).max()))
        size = (max(8, round(r.shape[1] * 320 / max(r.shape[:2]))), max(8, round(r.shape[0] * 320 / max(r.shape[:2]))))
        rg = np.asarray(Image.fromarray(r).convert("L").resize(size))
        cg = np.asarray(Image.fromarray(c).convert("L").resize(size))
        ssim.append(float(structural_similarity(rg, cg, data_range=255)))
        if previous_r is not None:
            motion.append(float(np.mean(np.abs((cf - previous_c) - (rf - previous_r)))) / 255)
        previous_r, previous_c = rf, cf
    # Exclude the observed first frame from the headline generated-video metric.
    generated = slice(1, None) if len(mse) > 1 else slice(None)
    average_mse = float(np.mean(mse[generated]))
    return {
        "shape": list(reference.shape),
        "generated_mae_0_255": float(np.mean(mae[generated])),
        "generated_rmse_0_1": math.sqrt(average_mse) / 255,
        "generated_psnr_db": 10 * math.log10(255**2 / average_mse) if average_mse else None,
        "generated_ssim_gray_320": float(np.mean(ssim[generated])),
        "max_pixel_error_0_255": max_error,
        "motion_delta_mae_0_1": float(np.mean(motion)) if motion else 0.0,
        "per_frame_mae_0_255": mae,
        "per_frame_ssim_gray_320": ssim,
        "identical": max_error == 0,
        "note": "PSNR null means zero MSE; SSIM uses grayscale resized to max dimension 320. Not a quality verdict.",
    }


def read_frames(run):
    return np.load(run["output"]["video"]["raw_path"], mmap_mode="r", allow_pickle=False)


def action_error(reference, candidate):
    # This also checks domain, units, mask, timestamps, seed and finite values.
    return compare({"runs": [reference]}, {"runs": [candidate]})["checks"][0]


def make_sheet(rows, name, output):
    positions = sorted(set(np.linspace(0, len(rows[0][1]) - 1, 5, dtype=int).tolist()))
    height, width = rows[0][1].shape[1:3]
    cell_w, cell_h = 320, round(height * 320 / width)
    canvas = Image.new("RGB", (len(positions) * cell_w, 32 + len(rows) * (cell_h + 24)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), name, fill="black")
    for row, (label, frames) in enumerate(rows):
        y = 32 + row * (cell_h + 24)
        for col, frame_index in enumerate(positions):
            draw.text((col * cell_w + 4, y + 4), f"{label}: frame {frame_index}", fill="black")
            canvas.paste(Image.fromarray(frames[frame_index]).resize((cell_w, cell_h)), (col * cell_w, y + 24))
    canvas.save(output)
    return positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = [args.reference, *args.candidate]
    datasets = [(path.stem, json.loads(path.read_text())) for path in paths]
    reference = datasets[0][1]
    grouped = {}
    summary = {"comparisons": [], "repeatability": [], "seed_variation": [], "visual_review": "pending"}
    for name, data in datasets:
        if data["input_sha256"] != reference["input_sha256"]:
            raise ValueError("Different request manifest")
        if data["input_asset_sha256"] != reference["input_asset_sha256"]:
            raise ValueError("Different conditioning images")
        groups = {}
        for run in data["runs"]:
            settings = run["output"].get("reproducibility", {})
            if not settings.get("deterministic_algorithms") or settings.get("deterministic_warn_only"):
                raise ValueError("Expected recorded strict deterministic validation settings")
            groups.setdefault(run["request_index"], []).append(run)
        if not groups or any(len(runs) < 2 for runs in groups.values()):
            raise ValueError("Each request requires at least two same-seed repetitions")
        grouped[name] = groups
        for index, runs in groups.items():
            for repeat, run in enumerate(runs[1:], 1):
                summary["repeatability"].append(
                    {
                        "profile": name,
                        "request_index": index,
                        "repeat": repeat,
                        "action": action_error(runs[0], run),
                        "video": video_metrics(read_frames(runs[0]), read_frames(run)),
                    }
                )
    ref_groups = grouped[datasets[0][0]]
    sections = []
    for index, ref_runs in sorted(ref_groups.items()):
        first = ref_runs[0]
        name = Path(reference["args"]["input"][index]).stem
        rows = [(datasets[0][0], read_frames(first))]
        links = []
        for profile, data in datasets:
            run = grouped[profile][index][0]
            if run["output"]["video"]["fps"] != first["output"]["video"]["fps"]:
                raise ValueError("Video FPS differs")
            relative = os.path.relpath(run["output"]["video"]["review_path"], args.output_dir)
            links.append(
                f'<div><p>{html.escape(profile)}</p><video controls muted loop width="320" '
                f'src="{html.escape(relative, quote=True)}"></video></div>'
            )
            if profile == datasets[0][0]:
                continue
            frames = read_frames(run)
            rows.append((profile, frames))
            summary["comparisons"].append(
                {
                    "profile": profile,
                    "request_index": index,
                    "name": name,
                    "action": action_error(first, run),
                    "video": video_metrics(rows[0][1], frames),
                }
            )
        sheet = f"{index:02d}_{name}.jpg"
        positions = make_sheet(rows, name, args.output_dir / sheet)
        sections.append(
            f"<section><h2>{html.escape(name)}</h2><p>Frames: {positions}</p>"
            f'<div style="display:flex;gap:16px">{"".join(links)}</div>'
            f'<a href="{sheet}"><img width="960" src="{sheet}"></a></section>'
        )
    # Reference pairs differing only by seed give a scale for natural variation.
    for i in sorted(ref_groups):
        for j in sorted(ref_groups):
            if j <= i:
                continue
            a = json.loads(Path(reference["args"]["input"][i]).read_text())
            b = json.loads(Path(reference["args"]["input"][j]).read_text())
            for value in (a, b):
                value.pop("name", None)
                value.pop("seed", None)
            if a != b or ref_groups[i][0]["output"]["seed"] == ref_groups[j][0]["output"]["seed"]:
                continue
            ra, rb = ref_groups[i][0], ref_groups[j][0]
            mask = ra["output"]["action_dim_mask"]
            x = np.array(ra["output"]["normalized_action"])[:, mask]
            y = np.array(rb["output"]["normalized_action"])[:, mask]
            summary["seed_variation"].append(
                {
                    "requests": [i, j],
                    "normalized_action_max_abs": float(np.max(np.abs(x - y))),
                    "normalized_action_rmse": float(np.sqrt(np.mean((x - y) ** 2))),
                    "video": video_metrics(read_frames(ra), read_frames(rb)),
                }
            )
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>PWM video review</title>'
        "<h1>PWM deterministic video comparison</h1>"
        "<p>Raw-frame metrics are in metrics.json. Similarity is not task quality. "
        "Videos have independent playback controls; compare the labeled frames below.</p>" + "\n".join(sections)
    )


if __name__ == "__main__":
    main()
