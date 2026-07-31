#!/usr/bin/env python3
"""Write site/split tags into NPZ sidecars for one converted dataset tree.

This is an external helper script, not part of the public ``tag_toolkit`` API.
It follows the layout produced by ``dataset/generate_from_labeled.sh``:

    <data_root>/<project>/<map_id>/{manual,auto}/<date>/<time>/...

If ``split_labels.json`` is available, it refines ``manual`` into
``train`` / ``valid``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Mapping


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


sys.path.insert(0, str(_repo_root() / "Diffusion-Planner"))

from tag_toolkit.sidecar import drop_dimension, format_tag, normalize_tags, read_tags, write_tags
from tag_toolkit.source import Source, expand_source

MODE_TOKENS = frozenset({"manual", "auto", "train", "valid"})
# Site is whatever is between project and mode — kept permissive
SITE_RE = re.compile(r"^.+$")


def _default_split_labels(data_root: Path) -> Path | None:
    candidate = data_root / "split_labels.json"
    return candidate if candidate.is_file() else None


def parse_site_split(npz: str | Path) -> tuple[str, str]:
    """Infer ``(site, split)`` from an NPZ path."""
    parts = Path(npz).resolve().parts
    mode_idx = None
    for i, part in enumerate(parts):
        if part in MODE_TOKENS:
            mode_idx = i
            break

    site = "unknown"
    if mode_idx is not None and mode_idx > 0:
        candidate = parts[mode_idx - 1]
        if SITE_RE.match(candidate):
            site = candidate

    split = "unknown"
    if mode_idx is not None:
        mode = parts[mode_idx]
        if mode in ("train", "valid", "auto"):
            split = mode
        elif mode == "manual":
            split = "manual"

    return site, split


def load_split_labels(path: str | Path) -> dict[str, str]:
    """Load ``split_labels.json`` -> ``{route_key_without_mode: split}``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("split_labels.json must be an object")
    out: dict[str, str] = {}
    for key, value in data.items():
        if isinstance(value, dict) and "split" in value:
            out[str(key)] = str(value["split"])
        elif isinstance(value, str):
            out[str(key)] = value
    return out


def _route_key_candidates(npz: Path) -> list[str]:
    parts = list(npz.resolve().parts)
    if parts and parts[-1].endswith(".npz"):
        parts = parts[:-1]
    if parts and parts[-1] == "routes":
        parts = parts[:-1]
    if len(parts) < 4:
        return []
    bag_time, date = parts[-1], parts[-2]
    if len(parts) >= 5 and parts[-3] in MODE_TOKENS:
        mode = parts[-3]
        site = parts[-4]
        proj = parts[-5]
        return [
            f"{proj}/{site}/{date}/{bag_time}",
            f"{proj}/{site}/{mode}/{date}/{bag_time}",
        ]
    if len(parts) >= 4:
        site = parts[-3]
        proj = parts[-4]
        return [f"{proj}/{site}/{date}/{bag_time}"]
    return []


def apply_path_tags(
    source: Source,
    *,
    split_labels: str | Path | Mapping[str, str] | None = None,
) -> int:
    """Write ``site:`` / ``split:`` for every NPZ under *source*."""
    labels: Mapping[str, str] | None
    if split_labels is None:
        labels = None
    elif isinstance(split_labels, Mapping):
        labels = split_labels
    else:
        labels = load_split_labels(split_labels)

    n = 0
    for npz in expand_source(source):
        site, split = parse_site_split(npz)
        if labels is not None:
            for key in _route_key_candidates(npz):
                if key in labels:
                    split = labels[key]
                    break
        desired = [format_tag("site", site), format_tag("split", split)]
        current = drop_dimension(drop_dimension(read_tags(npz), "site"), "split")
        merged = normalize_tags(list(current) + desired)
        if merged != normalize_tags(current):
            write_tags(npz, merged)
            n += 1
    return n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write site/split tags into NPZ sidecars for one dataset root."
    )
    parser.add_argument("data_root", type=Path, help="converted dataset root")
    parser.add_argument(
        "--split-labels",
        type=Path,
        help="optional split_labels.json; defaults to <data_root>/split_labels.json if present",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {data_root}")

    split_labels = (
        args.split_labels.resolve() if args.split_labels else _default_split_labels(data_root)
    )
    updated = apply_path_tags(data_root, split_labels=split_labels)

    if split_labels is None:
        print(f"updated {updated} sidecars from path layout only")
    else:
        print(f"updated {updated} sidecars using split labels: {split_labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
