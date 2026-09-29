"""Shared helpers for the ego-dimension diagnostics. Run all scripts from the repo root."""

from __future__ import annotations

import glob
import os

import numpy as np
import torch

from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.eval_det_avoidance import load_model


def load_deployable_model(model_path: str, device):
    """Load a checkpoint, refusing one whose deployable copy would be skipped.

    ``load_model`` takes ``ckpt["model"]``, and a training milestone holds BOTH that
    (the raw optimizer iterates) and ``ema_state_dict`` (the deployable copy). Loading
    the raw weights yields a plausible-looking but wrong verdict with no warning, so
    fail loudly and say how to convert instead.
    """
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "ema_state_dict" in ckpt:
        raise SystemExit(
            f"{model_path} is a training milestone carrying both raw and EMA weights; "
            "load_model would silently use the RAW iterates. Convert it first:\n"
            '  ck = torch.load(src, map_location="cpu", weights_only=False)\n'
            '  state = {k.replace("module.", ""): v for k, v in ck["ema_state_dict"].items()}\n'
            '  torch.save({"model": state}, dst)   # copy args.json next to it'
        )
    return load_model(model_path, device)


def find_scenes(path: str) -> list[str]:
    """Accept a directory (searched recursively), a glob, or a single .npz."""
    if os.path.isdir(path):
        out = sorted(glob.glob(os.path.join(path, "**", "*.npz"), recursive=True))
    else:
        out = sorted(glob.glob(path))
    if not out:
        raise SystemExit(f"no .npz found under {path!r}")
    return out


def to_f32(d: dict) -> dict:
    """Some converters emit float64 for ego_agent_past; the encoder requires float32."""
    return {
        k: (v.float() if torch.is_tensor(v) and v.dtype == torch.float64 else v)
        for k, v in d.items()
    }


def load_scene(path: str, device, goal: tuple[float, float] | None = None) -> dict:
    """Load one scene, optionally overriding the ego-frame goal.

    The override exists because a converter may rewrite ``goal_pose`` onto the ego when
    the recorded run ends stopped; passing the true goal restores the intended scene.
    """
    d = to_f32(load_npz_data(path, device))
    if goal is not None:
        g = d["goal_pose"].clone().reshape(-1)
        g[0], g[1] = goal
        if g.numel() >= 4:  # [x, y, cos, sin] after heading_to_cos_sin
            g[2], g[3] = 1.0, 0.0
        d["goal_pose"] = g.reshape(d["goal_pose"].shape)
    return d


def set_shape(d: dict, wheelbase: float, length: float, width: float) -> dict:
    """Return a copy of the scene with ``ego_shape`` replaced."""
    out = dict(d)
    s = d["ego_shape"].clone().reshape(-1)
    s[0], s[1], s[2] = wheelbase, length, width
    out["ego_shape"] = s.reshape(d["ego_shape"].shape)
    return out


def plan_span(model, margs, d: dict, device) -> float:
    """Straight-line distance from the first to the last predicted ego waypoint.

    This is a coarse "is the model planning to move at all" measure, not a path length:
    at cruise it lands in the tens of metres, and a value near zero means the model is
    planning to stand still.
    """
    from rlvr.autoresearch.tools.eval_det_avoidance import det_inference_batched

    out = det_inference_batched(model, margs, [d], device)
    t = out[0] if not isinstance(out, tuple) else out[0][0]
    t = np.squeeze(t.detach().cpu().numpy())
    return float(np.hypot(t[-1, 0] - t[0, 0], t[-1, 1] - t[0, 1]))


def parse_shape(text: str) -> tuple[float, float, float]:
    parts = [float(x) for x in text.split(",")]
    if len(parts) != 3:
        raise SystemExit(f"expected wheelbase,length,width (got {text!r})")
    return parts[0], parts[1], parts[2]
