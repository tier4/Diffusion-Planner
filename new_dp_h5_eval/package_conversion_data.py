"""Package the exact ROS bags and source NPZ sidecars needed for matrix conversion.

This deliberately copies only files referenced by open_loop_matrix.json.  It is
intended for making a portable archive on a machine where ROS 2 is available.
"""
from __future__ import annotations

import argparse, json, shutil
from pathlib import Path


def matrix_paths(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    return [p for values in data.values() for p in values] if isinstance(data, dict) else list(data)


def copy_tree(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise FileNotFoundError(src)
    if dst.exists():
        return
    shutil.copytree(src, dst, symlinks=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("matrix", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--basic-ros-root", type=Path, default=Path("/mnt/storage_rdma/diffusion_planner/rosbags_from_label"))
    ap.add_argument("--override-ros-root", type=Path, default=Path("/mnt/storage_rdma/diffusion_planner/dataset/evaluator_override_dataset/source_rosbags/x2_dev"))
    args = ap.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise SystemExit(f"output already exists: {out} (remove it explicitly if you want to rebuild)")
    (out / "matrices").mkdir(parents=True)
    (out / "npz").mkdir()
    (out / "rosbags").mkdir()
    paths = matrix_paths(args.matrix)
    seen: set[str] = set(); bags: dict[str, str] = {}; records = []
    for text in paths:
        if text in seen:
            continue
        seen.add(text)
        src = Path(text).resolve()
        if "20260814_basic_dataset" in src.parts:
            marker, kind, rosroot = "20260814_basic_dataset", "basic", args.basic_ros_root
        elif "dataset_all" in src.parts:
            marker, kind, rosroot = "dataset_all", "override", args.override_ros_root
        else:
            raise ValueError(f"unsupported dataset path: {src}")
        i = src.parts.index(marker)
        rel = Path(*src.parts[i + 1 :])
        dst_npz = out / "npz" / marker / rel
        dst_npz.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_npz)
        side = src.with_suffix(".json")
        if side.is_file():
            shutil.copy2(side, dst_npz.with_suffix(".json"))
        # The converter resolves a bag as rosbag_root / rel.parent.parent.
        bag_rel = rel.parent.parent
        bag = rosroot / bag_rel
        if not (bag / "log_file_info.json").is_file():
            raise FileNotFoundError(f"ROSBAG metadata not found: {bag} (from {src})")
        bag_key = f"{kind}:{bag_rel.as_posix()}"
        if bag_key not in bags:
            stage_rel = (Path("x2_dev") / bag_rel) if kind == "override" else bag_rel
            dst_bag = out / "rosbags" / kind / stage_rel
            copy_tree(bag, dst_bag)
            # Maps are outside the bag and are required by ml_planner_data.
            map_src = bag.parents[2] / "map"
            if map_src.is_dir():
                map_rel = (Path("x2_dev") / bag_rel.parents[1] / "map") if kind == "override" else (bag_rel.parents[1] / "map")
                copy_tree(map_src, out / "rosbags" / kind / map_rel)
            bags[bag_key] = bag.as_posix()
        records.append({"source_npz": src.as_posix(), "package_npz": (Path("npz") / marker / rel).as_posix(), "kind": kind, "bag": bag.as_posix()})
    shutil.copy2(args.matrix, out / "matrices" / "open_loop_matrix.original.json")
    (out / "manifest.json").write_text(json.dumps({"matrix": args.matrix.resolve().as_posix(), "unique_samples": len(records), "unique_bags": len(bags), "records": records}, indent=2) + "\n")
    print(f"packaged {len(records)} samples from {len(bags)} bags into {out}")


if __name__ == "__main__":
    main()
