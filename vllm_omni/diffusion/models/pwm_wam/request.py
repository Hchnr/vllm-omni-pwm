# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Standalone non-autoregressive WAM request preparation."""

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor


@dataclass
class PwmRequest:
    caption: str
    negative_prompt: str
    pixels: Tensor
    domain: str
    normalization_id: int
    projection_id: int
    mask: list[bool]
    video_fps: float
    action_fps: float
    action_length: int
    video_frames: int
    seam: float | None
    seed: int
    steps: int
    guidance: float
    shift: float
    normalize_cfg: bool
    guidance_interval: tuple[float, float] | None
    return_video: bool


def canvas_seam(roles, layout, height, width):
    if not isinstance(roles, list) or not 1 <= len(roles) <= 3 or len(set(roles)) != len(roles):
        raise ValueError("selected_roles must contain one to three unique roles")
    if len(roles) == 1:
        if layout is not None:
            raise ValueError("Single-camera layout must be null")
        return None
    locations = ["top", "bottom"] if len(roles) == 2 else ["top", "bottom_left", "bottom_right"]
    if not isinstance(layout, list) or len(layout) != len(roles):
        raise ValueError("camera_layout must match selected_roles")
    for role, location, item in zip(roles, locations, layout):
        if not isinstance(item, list) or len(item) != 4 or item[:2] != [role, location]:
            raise ValueError("Invalid camera_layout role or location")
        if any(type(x) is not int or x <= 0 for x in item[2:]):
            raise ValueError("Camera dimensions must be positive integer pixels")
    seam = layout[0][3]
    if layout[0][2] != width or not 0 < seam < height:
        raise ValueError("Top camera does not match canvas")
    if any(row[3] != height - seam for row in layout[1:]) or sum(row[2] for row in layout[1:]) != width:
        raise ValueError("Bottom cameras do not match canvas")
    return seam / 32


def prepare_request(raw: dict, spec: dict) -> PwmRequest:
    if not isinstance(raw, dict):
        raise ValueError("WAM request must be an object")
    if raw.get("action_path") is not None:
        raise ValueError("Native no-state WAM does not accept action conditioning")
    if raw.get("model_mode") != "wam":
        raise ValueError("Native PWM requires model_mode='wam'")
    for flag in ("autoregressive", "enable_sound", "native_prompt_upsampling"):
        if raw.get(flag):
            raise ValueError(f"Native WAM does not support {flag}")
    if raw.get("num_outputs", 1) != 1 or raw.get("raw_action_dim") not in (None, 128):
        raise ValueError("Native WAM requires one Action128 output per request")
    if raw.get("condition_frame_indexes_vision") not in (None, [0]):
        raise ValueError("Native WAM conditions on vision frame zero only")
    extra = raw.get("extra", {})
    if not isinstance(extra, dict) or not isinstance(extra.get("action128", {}), dict):
        raise ValueError("extra.action128 must be an object")
    ext = copy.deepcopy(extra.get("action128", {}))
    allowed = {
        "action_domain",
        "action_dim_mask",
        "action_fps",
        "selected_roles",
        "camera_layout",
        "task_title",
        "return_video",
    }
    if set(ext) - allowed:
        raise ValueError(f"Unknown Action128 extensions: {set(ext) - allowed}")
    domain = ext.get("action_domain")
    registry = spec["registry"]
    if domain not in registry["domains"]:
        raise ValueError("Unknown Action128 source domain")
    mask = ext.get("action_dim_mask")
    if not isinstance(mask, list) or len(mask) != 128 or any(type(x) is not bool for x in mask) or not any(mask):
        raise ValueError("action_dim_mask must contain 128 booleans and an active slot")
    path = raw.get("vision_path")
    if not path or not Path(path).is_file():
        raise ValueError("Native WAM currently requires a local conditioning image")
    with Image.open(path) as image:
        pixels = torch.from_numpy(np.asarray(image.convert("RGB")).copy()).permute(2, 0, 1)[:, None]
    height, width = pixels.shape[-2:]
    roles, layout = ext.get("selected_roles", ["main"]), ext.get("camera_layout")
    canvas_seam(roles, layout, height, width)
    target_h = raw.get("image_size") or 16 * math.ceil(height / 16)
    if type(target_h) is not int or target_h <= 0 or target_h % 16:
        raise ValueError("image_size must be a positive multiple of 16")
    target_w = max(16, 16 * round(width * target_h / height / 16))
    if (target_h, target_w) != (height, width):
        pixels = (
            F.interpolate(
                pixels.permute(1, 0, 2, 3).float(), size=(target_h, target_w), mode="bilinear", align_corners=False
            )
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(1, 0, 2, 3)
        )
        if layout is not None:
            seam = round(layout[0][3] * target_h / height)
            layout[0][2:] = [target_w, seam]
            if len(layout) == 2:
                layout[1][2:] = [target_w, target_h - seam]
            else:
                left = round(layout[1][2] * target_w / width)
                layout[1][2:] = [left, target_h - seam]
                layout[2][2:] = [target_w - left, target_h - seam]
        height, width = target_h, target_w
    seam = canvas_seam(roles, layout, height, width)
    fps = float(raw.get("fps") or 24)
    action_fps = float(ext.get("action_fps") or fps)
    length = raw.get("action_chunk_size") or 16
    if not math.isfinite(fps) or not 0 < fps <= 30 or not math.isfinite(action_fps) or not 0 < action_fps <= 512:
        raise ValueError("Invalid video or action FPS")
    if type(length) is not int or length < 1 or length / action_fps > 2:
        raise ValueError("Action horizon must be positive and at most two seconds")
    duration = length / action_fps
    frames = 4 * math.ceil(math.ceil(duration * fps - 1e-8) / 4) + 1
    if raw.get("num_frames") not in (None, frames):
        raise ValueError(f"num_frames must equal {frames}")
    seed = raw.get("seed", 0)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be a uint32 integer")
    prompt = raw.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("A nonempty prompt is required")
    try:
        caption = json.loads(prompt)
    except json.JSONDecodeError:
        caption = None
    title = ext.get("task_title") or raw.get("name") or "action128_wam"
    if isinstance(caption, dict):
        title = caption.get("task_title", title)
        prompt = caption.get("command", caption.get("source_task_description", prompt))
    domain_id = registry["domains"].index(domain)
    formatted = dict(
        task_title=title,
        command=prompt,
        embodiment=registry["embodiments"][registry["domain_embodiments"][domain_id]],
        duration=f"{duration:g}s",
        fps=fps,
        action_frequency_hz=action_fps,
        resolution={"W": width, "H": height},
        aspect_ratio=f"{width // math.gcd(width, height)}:{height // math.gcd(width, height)}",
    )
    if layout is not None:
        formatted["camera_layout"] = layout
    draw = (
        int.from_bytes(hashlib.blake2b(json.dumps(seed, sort_keys=True).encode(), digest_size=16).digest(), "big")
        / 2**128
    )
    if draw < 0.9:
        formatted["task_plan"] = []
    if isinstance(caption, dict):
        for key in (
            "embodiment",
            "duration",
            "fps",
            "action_frequency_hz",
            "resolution",
            "aspect_ratio",
            "camera_layout",
        ):
            if key in formatted:
                caption[key] = formatted[key]
            else:
                caption.pop(key, None)
        caption.setdefault("command", prompt)
        caption.setdefault("task_title", title)
    else:
        caption = formatted
    steps, guidance, shift = raw.get("num_steps", 30), float(raw.get("guidance", 1)), float(raw.get("shift", 10))
    if type(steps) is not int or steps < 1 or not math.isfinite(guidance) or not math.isfinite(shift) or shift <= 0:
        raise ValueError("Invalid diffusion sampling parameters")
    interval = raw.get("guidance_interval")
    if interval is not None and (
        len(interval) != 2 or not all(math.isfinite(x) for x in interval) or interval[0] > interval[1]
    ):
        raise ValueError("guidance_interval must be a finite ordered pair")
    return PwmRequest(
        json.dumps(caption, ensure_ascii=False),
        raw.get("negative_prompt") or "",
        pixels,
        domain,
        domain_id,
        registry["domain_projection_ids"][domain_id],
        mask,
        fps,
        action_fps,
        length,
        frames,
        seam,
        seed,
        steps,
        guidance,
        shift,
        bool(raw.get("normalize_cfg", False)),
        tuple(interval) if interval is not None else None,
        ext.get("return_video", True),
    )
