"""Fetch validation NPZs from sakurab and verify route format."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def create_subsample(
    full_list_path: str,
    n: int,
    seed: int,
    output_path: str,
) -> list[str]:
    """Write a JSON path list of n randomly sampled paths."""
    with open(full_list_path) as f:
        full_list = json.load(f)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(full_list), size=min(n, len(full_list)), replace=False)
    subsample = [full_list[i] for i in sorted(indices)]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(subsample, f, indent=2)
    return subsample


def fetch_npzs_local(
    path_list: list[str],
    dest: str,
    host: str = "sakurab",
    bwlimit: int = 10000,
) -> None:
    """rsync NPZ files from host to local dest."""
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    to_fetch = []
    for p in path_list:
        local = dest_path / p.lstrip("/")
        if not local.exists():
            to_fetch.append(p)
    if not to_fetch:
        print(f"All {len(path_list)} NPZs already present locally.")
        return

    listfile = dest_path / ".rsync_fetch_list.txt"
    listfile.write_text("\n".join(p.lstrip("/") for p in to_fetch) + "\n")

    cmd = [
        "rsync",
        "-a",
        "--info=progress2",
        f"--bwlimit={bwlimit}",
        f"--files-from={listfile}",
        f"{host}:/",
        str(dest_path),
    ]
    print(f"Fetching {len(to_fetch)} NPZs from {host}...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"rsync exited with code {result.returncode}")


def verify_route_format(npz_path: str) -> dict:
    """Check route_lanes shape and content from one NPZ."""
    data = np.load(npz_path)
    rl = data["route_lanes"]
    shape_raw = rl.shape
    if rl.ndim == 4:
        rl = rl.squeeze(0)

    n_nonempty = 0
    gaps = []
    prev_end = None
    for seg in rl:
        xy = seg[:, :2]
        if np.allclose(xy, 0.0):
            continue
        nonzero = ~np.all(xy == 0.0, axis=-1)
        if not nonzero.any():
            continue
        last_valid = np.where(nonzero)[0][-1]
        seg_xy = xy[: last_valid + 1]
        if prev_end is not None:
            gap = float(np.linalg.norm(seg_xy[0] - prev_end))
            gaps.append(gap)
        prev_end = seg_xy[-1]
        n_nonempty += 1

    return {
        "shape_raw": shape_raw,
        "shape": rl.shape,
        "n_nonempty_segments": n_nonempty,
        "segment_gaps": gaps,
        "max_gap": max(gaps) if gaps else 0.0,
    }


def main():
    p = argparse.ArgumentParser(description="Fetch and verify validation NPZs.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub_sample = sub.add_parser("subsample", help="Create a random subsample path list.")
    sub_sample.add_argument("--full_list", required=True)
    sub_sample.add_argument("--n", type=int, default=500)
    sub_sample.add_argument("--seed", type=int, default=42)
    sub_sample.add_argument("--output", required=True)

    sub_fetch = sub.add_parser("fetch", help="rsync NPZs from sakurab.")
    sub_fetch.add_argument("--path_list", required=True)
    sub_fetch.add_argument("--dest", default="data/per_scene_eval/mirror")
    sub_fetch.add_argument("--host", default="sakurab")
    sub_fetch.add_argument("--bwlimit", type=int, default=10000)

    sub_verify = sub.add_parser("verify", help="Check route_lanes format on sample NPZs.")
    sub_verify.add_argument("--path_list", required=True)
    sub_verify.add_argument("--mirror", default="data/per_scene_eval/mirror")
    sub_verify.add_argument("--n", type=int, default=10)

    args = p.parse_args()
    if args.cmd == "subsample":
        paths = create_subsample(args.full_list, args.n, args.seed, args.output)
        print(f"Wrote {len(paths)} paths to {args.output}")
    elif args.cmd == "fetch":
        with open(args.path_list) as f:
            paths = json.load(f)
        fetch_npzs_local(paths, args.dest, args.host, args.bwlimit)
    elif args.cmd == "verify":
        with open(args.path_list) as f:
            paths = json.load(f)
        mirror = Path(args.mirror)
        rng = np.random.default_rng(0)
        sample = rng.choice(paths, size=min(args.n, len(paths)), replace=False)
        all_gaps = []
        for p_str in sample:
            local = mirror / p_str.lstrip("/")
            if not local.exists():
                print(f"MISSING: {local}")
                continue
            info = verify_route_format(str(local))
            print(
                f"{Path(p_str).name}: {info['n_nonempty_segments']} segs, "
                f"max_gap={info['max_gap']:.3f}m, shape={info['shape']}"
            )
            all_gaps.extend(info["segment_gaps"])
        if all_gaps:
            print(
                f"\nGap histogram: min={min(all_gaps):.3f} median={np.median(all_gaps):.3f} "
                f"p95={np.percentile(all_gaps, 95):.3f} max={max(all_gaps):.3f}"
            )


if __name__ == "__main__":
    main()
