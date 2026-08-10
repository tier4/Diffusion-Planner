"""Stage 1: Sample DP trajectories and score each scene. Writes incremental CSV."""

import argparse
import csv
import json
import traceback
from pathlib import Path

import numpy as np
from tqdm import tqdm

from human_match_prototype.energy_score import HORIZONS, per_scene_energy_score
from human_match_prototype.route_projection import (
    frenet_energy_scores,
    project_to_route,
    stitch_route_lanes,
    update_qa_after_projection,
)
from human_match_prototype.sampler import TrajectorySampler

DEFAULT_MODEL_DIR = Path("/opt/autoware/mlmodels/diffusion_planner_for_x2")

NAN_FRENET = {f"es_{c}_{h}": float("nan") for h in HORIZONS for c in ("lon", "lat")}


def _load_done_paths(csv_path: Path) -> set[str]:
    """Read an existing CSV and return the set of already-scored npz_path values."""
    if not csv_path.exists():
        return set()
    done = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(row["npz_path"])
    return done


def score_one_scene(
    npz_path: str,
    sampler: TrajectorySampler,
    num_samples: int = 64,
    seed: int = 0,
    temperature: float = 1.0,
) -> dict[str, float]:
    """Score a single scene: Energy Score + route Frenet + QA."""
    result = sampler.sample(npz_path, num_samples=num_samples, seed=seed, temperature=temperature)
    human_xy = result.human_future[:, :2]  # (80, 2)
    samples_xy = result.ego_samples[:, :, :2]  # (N, 80, 2)

    row: dict[str, float] = {"npz_path": npz_path}

    # x-y Energy Score
    row.update(per_scene_energy_score(human_xy, samples_xy))

    # Route projection
    data = np.load(npz_path)
    route_lanes = data["route_lanes"]
    route = stitch_route_lanes(route_lanes)

    if not route.qa.route_valid or len(route.centerline) < 2:
        row.update(NAN_FRENET)
        row.update(route.qa.to_dict())
        return row

    human_s, human_d, human_pd = project_to_route(route, human_xy)
    samples_s, samples_d, samples_pd = project_to_route(route, samples_xy)
    update_qa_after_projection(route, human_s, human_pd, samples_pd)

    if route.qa.route_coverage_insufficient:
        row.update(NAN_FRENET)
        row.update(route.qa.to_dict())
        return row

    human_sd = np.stack([human_s, human_d], axis=-1)  # (80, 2)
    samples_sd = np.stack([samples_s, samples_d], axis=-1)  # (N, 80, 2)
    row.update(frenet_energy_scores(human_sd, samples_sd))
    row.update(route.qa.to_dict())

    return row


def main():
    p = argparse.ArgumentParser(description="Stage 1: Score validation scenes.")
    p.add_argument("--npz_list", required=True, help="JSON path list of NPZs")
    p.add_argument("--output", required=True, help="Output CSV path")
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--resume", action="store_true", help="Skip already-scored scenes, append to output CSV"
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument(
        "--mirror", default=None, help="Local mirror root; paths are resolved relative to this."
    )
    args = p.parse_args()

    model_dir = args.model_dir or DEFAULT_MODEL_DIR
    sampler = TrajectorySampler(
        str(model_dir / "args.json"),
        str(model_dir / "diffusion_planner.onnx"),
        args.device,
    )

    with open(args.npz_list) as f:
        paths = json.load(f)
    if args.limit:
        paths = paths[: args.limit]

    if args.mirror:
        mirror = Path(args.mirror)
        paths = [str(mirror / p.lstrip("/")) for p in paths]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_paths: set[str] = set()
    if args.resume:
        done_paths = _load_done_paths(out_path)
        if done_paths:
            print(f"Resume: {len(done_paths)} scenes already scored, skipping.")

    pending = [p for p in paths if p not in done_paths]
    if not pending:
        print(f"All {len(paths)} scenes already scored.")
        return

    fieldnames = None
    skipped = 0
    mode = "a" if args.resume and done_paths else "w"
    with open(out_path, mode, newline="") as csvfile:
        writer = None
        for path in tqdm(pending, desc="Scoring"):
            try:
                row = score_one_scene(path, sampler, args.num_samples, args.seed, args.temperature)
                if writer is None:
                    fieldnames = list(row.keys())
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    if mode == "w":
                        writer.writeheader()
                writer.writerow(row)
                csvfile.flush()
            except Exception:
                skipped += 1
                print(f"skip {path}")
                traceback.print_exc()

    total_done = len(done_paths) + len(pending) - skipped
    print(
        f"Wrote {len(pending) - skipped} new rows to {out_path} ({skipped} skipped, {total_done} total)"
    )


if __name__ == "__main__":
    main()
