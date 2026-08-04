"""Build the ``sample_dataset/`` fixture for tag_toolkit tests.

The fixture mirrors a small slice of the real Diffusion Planner dataset layout,
but with empty NPZ placeholders. The script generates synthetic sidecar
JSON so the fixture is self-contained and works without access to the original data.

Re-running this script is safe — it rebuilds the data subdirectories from
scratch.

Source layout used:
- aomi_centerline → proj_a / xxxx_site_a / auto / 2026-06-23 / 10-55-13
- ariake_leftturn → proj_a / xxxx_site_a / auto / 2026-07-07 / 15-16-36
- zeikan_centerline → proj_b / xxxx_site_c / manual / 2026-04-15 / psim_training_bag_0_0

Each route holds 10 contiguous NPZ frames. Per-frame tags are deterministic:
- All frames: site:<map_id>, split:<spec.split_tag>, override_metric:centerline
- Aomi/ariake half (by frame-number parity): also lateral:turn
- Psim half (by frame-number parity): split:manual, the other half: split:valid
- Aomi 2 specific frames (positions 3 and 7): also longitudinal:yield

It writes:
- sample_dataset/<project>/<map_id>/<split>/<date>/<bag_time>/routes/<frame>.npz
- sample_dataset/<project>/<map_id>/<split>/<date>/<bag_time>/routes/<frame>.json
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

# Make ``import tag_toolkit`` work when this script is run directly
# (``python _build.py``) rather than via the test runner. The editable install
# already covers the test-runner path, but this keeps the script runnable in
# isolation. We need the parent of the package directory on sys.path (so
# Python finds the ``tag_toolkit`` subdirectory), not the package dir itself.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@dataclass(frozen=True)
class RouteSpec:
    """One route to materialise under ``DEST_ROOT``."""

    project_id: str
    map_id: str
    split: str  # directory token: "manual" or "auto"
    split_tag: str  # tag written to the sidecar (e.g. "auto", "train", "manual")
    date: str  # ISO date as used in the source paths
    bag_time: str  # bag_time directory
    site_tag: str  # value for site:<site_tag>

    def dest_route_dir(self, dest_root: Path) -> Path:
        return dest_root / self.project_id / self.map_id / self.split / self.date / self.bag_time


# 10 contiguous frames per route, picked from each closed-loop valid list.
# The middle 8-digit prefix is always 00000000 in this dataset.
AOMI_FRAMES = [f"00000000_{n:08d}" for n in range(2997, 3007)]
ARIAKE_FRAMES = [f"00000000_{n:08d}" for n in range(6278, 6288)]
PSIM_FRAMES = [f"00000000_{n:08d}" for n in range(31, 41)]


# Deterministic 2-frame picks for aomi (positions 3 and 7 of the route).
# Kept as a derived constant so changing AOMI_FRAMES still yields a
# self-consistent set.
AOMI_LONGITUDINAL_YIELD = frozenset({AOMI_FRAMES[3], AOMI_FRAMES[7]})


ROUTES: tuple[RouteSpec, ...] = (
    RouteSpec(
        map_id="xxxx_site_a",
        project_id="proj_a",
        split="auto",
        split_tag="auto",
        date="2026-06-23",
        bag_time="10-55-13",
        site_tag="xxxx_site_a",
    ),
    RouteSpec(
        map_id="xxxx_site_a",
        project_id="proj_a",
        split="auto",
        split_tag="train",
        date="2026-07-07",
        bag_time="15-16-36",
        site_tag="xxxx_site_a",
    ),
    RouteSpec(
        map_id="xxxx_site_c",
        project_id="proj_b",
        split="manual",
        split_tag="manual",  # psim frames alternate manual/valid via frame parity below
        date="2026-04-15",
        bag_time="psim_training_bag_0_0",
        site_tag="xxxx_site_c",
    ),
)


# Map route → frame list. The order matches the real closed-loop list.
ROUTE_FRAMES: dict[str, list[str]] = {
    "10-55-13": AOMI_FRAMES,
    "15-16-36": ARIAKE_FRAMES,
    "psim_training_bag_0_0": PSIM_FRAMES,
}

# Which routes get the longitudinal:yield tag on their yield frames.
AOMI_BAG_TIMES = frozenset({"10-55-13"})


def _frame_num(stem: str) -> int:
    """Extract frame number from a stem like ``10-55-13_00000000_00003000``."""
    return int(stem.rsplit("_", 1)[1])


def _split_for_frame(spec: RouteSpec, frame_num: str) -> str:
    """Resolve the per-frame ``split:*`` tag.

    Aomi/ariake use the spec's single ``split_tag``. Psim alternates between
    ``manual`` and ``valid`` by frame-number parity so the fixture covers
    multiple split values without ballooning the route count.
    """
    if spec.bag_time == "psim_training_bag_0_0":
        return "manual" if _frame_num(frame_num) % 2 == 0 else "valid"
    return spec.split_tag


def _build_one_route(spec: RouteSpec, dest_root: Path) -> list[Path]:
    """Materialise one route on disk; return the NPZ paths (frame-level list)."""
    dest = spec.dest_route_dir(dest_root)
    if dest.exists():
        shutil.rmtree(dest)
    routes_dir = dest / "routes"
    routes_dir.mkdir(parents=True)

    npz_paths: list[Path] = []
    for frame_num in ROUTE_FRAMES[spec.bag_time]:
        stem = f"{spec.bag_time}_{frame_num}"
        npz_dest = routes_dir / f"{stem}.npz"
        json_dest = routes_dir / f"{stem}.json"

        # Empty placeholder NPZ (real NPZs are hundreds of MB; tag_toolkit
        # never reads them).
        npz_dest.write_bytes(b"")

        sidecar = {
            "bag_time": spec.bag_time,
            "date": spec.date,
            "project_id": spec.project_id,
            "tags": [],
        }

        # Compute the deterministic tag set per frame.
        tags = [
            f"site:{spec.site_tag}",
            f"split:{_split_for_frame(spec, frame_num)}",
            "override_metric:centerline",
        ]
        if _frame_num(stem) % 2 == 1:
            tags.append("lateral:turn")
        if spec.bag_time in AOMI_BAG_TIMES and frame_num in AOMI_LONGITUDINAL_YIELD:
            tags.append("longitudinal:yield")
        sidecar["tags"] = sorted(set(tags))

        json_dest.write_text(
            json.dumps(sidecar, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        npz_paths.append(npz_dest)

    return npz_paths


def build(dest_root: Path | None = None) -> dict:
    """Build the sample dataset under *dest_root*.

    If *dest_root* is None, defaults to the directory holding this script
    (so ``python _build.py`` rebuilds the checked-in tree).

    Returns a summary dict with frame / route counts.
    """
    if dest_root is None:
        dest_root = Path(__file__).resolve().parent

    all_npz_paths: list[Path] = []
    route_dirs: list[Path] = []
    for spec in ROUTES:
        paths = _build_one_route(spec, dest_root)
        all_npz_paths.extend(paths)
        route_dirs.append(spec.dest_route_dir(dest_root))

    return {
        "frame_count": len(all_npz_paths),
        "route_count": len(route_dirs),
    }


if __name__ == "__main__":
    summary = build()
    print(f"Built sample_dataset: {summary['route_count']} routes, {summary['frame_count']} frames")