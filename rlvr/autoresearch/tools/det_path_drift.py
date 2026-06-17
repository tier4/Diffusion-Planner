"""Track deterministic-trajectory path-length drift vs LoRA-less baseline across epochs.

For each epoch's LoRA, regenerate deterministic trajectories on a scene list and
compare path lengths against the saved baseline in `epoch1_baselines.npz`.
Reports ratio = current_det_path / baseline_det_path per scene, with summary
stats (mean, min, p25, median, p95, max) and count below a threshold.

Usage:
    python -m rlvr.autoresearch.tools.det_path_drift \
        --run_dir /media/.../20260420-xxx_j6_rsft_*/  \
        --model_path /media/.../v4.0/best_model.pth \
        --scenes /media/.../scenes.json \
        --epochs 0 1 3 5 6 7 10 13 15 \
        [--below_threshold 0.7]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config

from preference_optimization.lora_utils import load_lora_checkpoint
from rlvr.autoresearch.tools.calibrate_rb_vs_lane import generate_for_all_scenes
from rlvr.grpo_config import GRPOConfig


def load_baseline_lens(run_dir: Path) -> dict[str, float]:
    bp = run_dir / "epoch1_baselines.npz"
    if not bp.exists():
        raise FileNotFoundError(f"No epoch1_baselines.npz in {run_dir}")
    saved = np.load(bp, allow_pickle=True)
    paths = saved["paths"].tolist()
    trajs = saved["trajectories"]  # (M, T, 4)
    lens = np.linalg.norm(np.diff(trajs[..., :2], axis=-2), axis=-1).sum(axis=-1)
    return {str(p): float(lens[i]) for i, p in enumerate(paths)}


def compute_det_paths(
    model, model_args, scene_paths: list[str], cfg, device
) -> tuple[np.ndarray, list[str]]:
    """Regenerate all trajectories and return traj[0] (det slot) path lengths."""
    torch.manual_seed(42)
    np.random.seed(42)
    trajs, all_data, valid = generate_for_all_scenes(model, model_args, scene_paths, cfg, device)
    lens = []
    for i, p in enumerate(valid):
        xy = trajs[i, 0, :, :2].cpu().numpy()
        lens.append(np.linalg.norm(np.diff(xy, axis=0), axis=-1).sum())
    return np.array(lens), [str(p) for p in valid]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Run directory containing lora_epoch_NNN/ and epoch1_baselines.npz",
    )
    parser.add_argument("--model_path", type=Path, required=True, help="Base v4.0 model path")
    parser.add_argument(
        "--scenes",
        type=str,
        required=True,
        help="JSON list of scene NPZ paths (same set used for training)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=[0, 1, 3, 5, 6, 7, 10, 13, 15, 20],
        help="Epochs to probe. 0 = LoRA-less base model.",
    )
    parser.add_argument(
        "--below_threshold", type=float, default=0.7, help="Count of scenes where ratio < this"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="GRPOConfig JSON. If omitted, loads run_dir/grpo_config.json. "
        "One of --config or a run_dir with grpo_config.json must resolve.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_path = args.config or (args.run_dir / "grpo_config.json")
    if not Path(cfg_path).exists():
        raise FileNotFoundError(
            f"No GRPO config found: pass --config or place grpo_config.json in {args.run_dir}"
        )
    cfg = GRPOConfig.from_json(str(cfg_path))

    # Load baseline map once
    base_map = load_baseline_lens(args.run_dir)
    base_lens_arr = np.array(list(base_map.values()))
    print(
        f"Baseline det path stats ({len(base_map)} scenes): "
        f"mean={base_lens_arr.mean():.2f}, min={base_lens_arr.min():.2f}, "
        f"p25={np.percentile(base_lens_arr, 25):.2f}, "
        f"p95={np.percentile(base_lens_arr, 95):.2f}, max={base_lens_arr.max():.2f}"
    )

    # Build model (base only — LoRA loaded per epoch)
    model_dir = args.model_path.parent
    args_json = model_dir / "args.json"
    if not args_json.exists():
        args_json = model_dir.parent / "args.json"
    model_args = Config(str(args_json))

    with open(args.scenes) as f:
        scene_paths = json.load(f)

    print()
    print(
        f"{'Epoch':>6s}  {'mean':>6s}  {'min':>6s}  {'p25':>6s}  {'med':>6s}  {'p95':>6s}  {'max':>6s}  {'<' + str(args.below_threshold):>6s}  {'det path m':>12s}"
    )
    for ep in args.epochs:
        model = Diffusion_Planner(model_args)
        ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
        state = {k.replace("module.", ""): v for k, v in ckpt.get("model", ckpt).items()}
        model.load_state_dict(state)

        label = "base" if ep == 0 else f"ep{ep}"
        if ep > 0:
            lora_dir = args.run_dir / f"lora_epoch_{ep:03d}"
            if not lora_dir.exists():
                print(f"{label:>6s}  (missing)")
                continue
            model = load_lora_checkpoint(model, str(lora_dir))
        model.to(device).eval()

        det_lens, valid = compute_det_paths(model, model_args, scene_paths, cfg, device)
        base_lens_aligned = np.array([base_map.get(p, 0.0) for p in valid])
        ratios = det_lens / np.clip(base_lens_aligned, 1e-3, None)

        n_below = int((ratios < args.below_threshold).sum())
        print(
            f"{label:>6s}  {ratios.mean():>6.3f}  {ratios.min():>6.3f}  "
            f"{np.percentile(ratios, 25):>6.3f}  {np.median(ratios):>6.3f}  "
            f"{np.percentile(ratios, 95):>6.3f}  {ratios.max():>6.3f}  "
            f"{n_below:>3d}/{len(ratios):>2d}  "
            f"det_mean={det_lens.mean():>6.2f}"
        )


if __name__ == "__main__":
    main()
