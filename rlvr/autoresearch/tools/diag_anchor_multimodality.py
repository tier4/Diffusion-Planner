"""Diagnose anchor-following guidance multimodality.

For each scene, generates one trajectory per commanded anchor prototype
(single anchor_following guidance function, zero initial noise so the ONLY
source of diversity is the guidance itself) plus an unguided deterministic
baseline, then measures how much the commanded anchor actually moves the
output:

  - endpoint displacement vs the unguided deterministic trajectory
  - endpoint spread across anchors (max pairwise distance, lateral std)
  - anchor obedience: fraction of samples whose nearest prototype
    (full-trajectory L2) is the commanded one

Low displacement / spread = guidance is inert (the multimodality failure
this tool exists to quantify). Writes a JSON report and a per-scene overlay
PNG.

Usage:
    python -m rlvr.autoresearch.tools.diag_anchor_multimodality \
        --model_path <best_model.pth> --scenes <scenes.json> \
        --prototypes rlvr/prototypes_k16.npy --output_dir <dir> \
        --num_scenes 8 --anchor_scale 3.0
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from diffusion_planner.model.guidance.composer import GuidanceComposer  # noqa: E402
from diffusion_planner.model.guidance.config import (  # noqa: E402
    GuidanceConfig,
    GuidanceSetConfig,
)

from guidance_gui.generate_samples import generate_samples  # noqa: E402
from rlvr.autoresearch.run_experiment import load_npz_data  # noqa: E402
from rlvr.autoresearch.tools.eval_det_avoidance import load_model  # noqa: E402


def nearest_prototype(traj_xy: np.ndarray, prototypes: np.ndarray) -> int:
    """Index of the prototype closest to traj_xy [T,2] by mean per-step L2."""
    d = np.linalg.norm(prototypes - traj_xy[None], axis=-1).mean(axis=-1)  # (K,)
    return int(d.argmin())


def run_scene(
    model,
    model_args,
    norm_data,
    prototypes_path: str,
    anchor_indices: list[int],
    anchor_scale: float,
    noise_scale: float,
    device,
    extra_params: dict | None = None,
) -> dict:
    """Generate det + one sample per anchor; return trajectories keyed by label."""
    out = {}
    det = generate_samples(
        model=model,
        model_args=model_args,
        data=norm_data,
        noise_scale=0.0,
        n_samples=1,
        composer=None,
        device=device,
    )[0]
    out["det"] = det

    for idx in anchor_indices:
        params = {"prototypes_path": prototypes_path, "anchor_index": idx}
        if extra_params:
            params.update(extra_params)
        cfg = GuidanceSetConfig(
            functions=[
                GuidanceConfig(
                    name="anchor_following",
                    enabled=True,
                    scale=anchor_scale,
                    params=params,
                )
            ],
            global_scale=1.0,
        )
        composer = GuidanceComposer(cfg)
        sample = generate_samples(
            model=model,
            model_args=model_args,
            data=norm_data,
            noise_scale=noise_scale,
            n_samples=1,
            composer=composer,
            device=device,
        )[0]
        out[f"anchor{idx}"] = sample
    return out


def scene_metrics(trajs: dict, anchor_indices: list[int], prototypes: np.ndarray) -> dict:
    det_end = trajs["det"][-1, :2]
    endpoints = []
    disp, obeyed = [], 0
    for idx in anchor_indices:
        xy = trajs[f"anchor{idx}"][:, :2]
        endpoints.append(xy[-1])
        disp.append(float(np.linalg.norm(xy[-1] - det_end)))
        if nearest_prototype(xy, prototypes) == idx:
            obeyed += 1
    endpoints = np.asarray(endpoints)
    pairwise = np.linalg.norm(endpoints[:, None] - endpoints[None], axis=-1)
    return {
        "mean_endpoint_disp_vs_det": float(np.mean(disp)),
        "max_endpoint_disp_vs_det": float(np.max(disp)),
        "max_pairwise_endpoint_dist": float(pairwise.max()),
        "endpoint_lateral_std": float(endpoints[:, 1].std()),
        "anchor_obedience_rate": obeyed / len(anchor_indices),
    }


def plot_scene(trajs: dict, anchor_indices, prototypes, out_png: Path, title: str):
    fig, ax = plt.subplots(figsize=(9, 9))
    cmap = plt.get_cmap("tab20")
    for k, idx in enumerate(anchor_indices):
        ax.plot(
            prototypes[idx, :, 0],
            prototypes[idx, :, 1],
            "--",
            color=cmap(k % 20),
            lw=0.8,
            alpha=0.5,
        )
        xy = trajs[f"anchor{idx}"][:, :2]
        ax.plot(xy[:, 0], xy[:, 1], "-", color=cmap(k % 20), lw=1.6, label=f"anchor{idx}")
    ax.plot(trajs["det"][:, 0], trajs["det"][:, 1], "k-", lw=2.5, label="det (unguided)")
    ax.plot(0, 0, "k*", ms=12)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True)
    p.add_argument("--scenes", required=True, help="JSON list of NPZ paths")
    p.add_argument("--prototypes", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_scenes", type=int, default=8)
    p.add_argument("--anchor_scale", type=float, default=3.0)
    p.add_argument("--noise_scale", type=float, default=0.0)
    p.add_argument(
        "--anchors",
        default=None,
        help="comma-separated anchor indices; default = all prototypes",
    )
    p.add_argument(
        "--extra_params",
        default=None,
        help="JSON dict of extra anchor_following params (for A/B of new knobs)",
    )
    p.add_argument("--tag", default="baseline", help="label for this configuration in the report")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prototypes = np.load(args.prototypes)
    anchor_indices = (
        [int(s) for s in args.anchors.split(",")]
        if args.anchors
        else list(range(prototypes.shape[0]))
    )
    extra_params = json.loads(args.extra_params) if args.extra_params else None

    with open(args.scenes) as f:
        scene_paths = json.load(f)[: args.num_scenes]

    model, model_args = load_model(args.model_path, device)

    report = {
        "tag": args.tag,
        "anchor_scale": args.anchor_scale,
        "noise_scale": args.noise_scale,
        "extra_params": extra_params,
        "scenes": {},
    }
    for sp in scene_paths:
        data = load_npz_data(sp, device)
        norm_data = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        norm_data = model_args.observation_normalizer(norm_data)
        trajs = run_scene(
            model,
            model_args,
            norm_data,
            args.prototypes,
            anchor_indices,
            args.anchor_scale,
            args.noise_scale,
            device,
            extra_params,
        )
        m = scene_metrics(trajs, anchor_indices, prototypes)
        name = Path(sp).stem
        report["scenes"][name] = m
        plot_scene(
            trajs,
            anchor_indices,
            prototypes,
            out_dir / f"{args.tag}_{name}.png",
            f"{args.tag} {name} scale={args.anchor_scale} "
            f"disp={m['mean_endpoint_disp_vs_det']:.2f}m "
            f"obey={m['anchor_obedience_rate']:.0%}",
        )
        print(
            f"{name}: mean_disp={m['mean_endpoint_disp_vs_det']:.2f}m "
            f"max_disp={m['max_endpoint_disp_vs_det']:.2f}m "
            f"spread={m['max_pairwise_endpoint_dist']:.2f}m "
            f"lat_std={m['endpoint_lateral_std']:.2f}m "
            f"obey={m['anchor_obedience_rate']:.0%}"
        )

    vals = report["scenes"].values()
    agg = {k: float(np.mean([v[k] for v in vals])) for k in next(iter(vals)).keys()}
    report["aggregate"] = agg
    with open(out_dir / f"report_{args.tag}.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nAGGREGATE [{args.tag}]: " + " ".join(f"{k}={v:.3f}" for k, v in agg.items()))
    print(f"Report: {out_dir / f'report_{args.tag}.json'}")


if __name__ == "__main__":
    main()
