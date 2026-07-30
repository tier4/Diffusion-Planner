"""Build the ``sample_dataset/`` fixture for tag_toolkit tests.

The fixture mirrors a small slice of the real Diffusion Planner dataset layout,
but with empty NPZ placeholders. The script tries to copy the real sidecar
JSON verbatim from the source data under ``/mnt/storage_rdma/...`` when
available; when that path is unreachable, it falls back to a minimal
synthetic sidecar so the fixture is self-contained.

The script also writes the pre-built index file and path/route list
artifacts.

Re-running this script is safe — it rebuilds the data subdirectories from
scratch, leaving any list artifacts in place.

Source layout used:
- aomi_centerline → x2_dev / 2231_odaiba_shinagawa_copied_from_xx1 / auto / 2026-06-23 / 10-55-13
- ariake_leftturn → x2_dev / 2231_odaiba_shinagawa_copied_from_xx1 / auto / 2026-07-07 / 15-16-36
- zeikan_centerline → xx1_psim / b_mobility_seed_200_poses_100_jpntaxi_vehicle_8_pilot-auto-v0.49.2 / manual / 2026-04-15 / psim_training_bag_0_0

Each route holds 10 contiguous NPZ frames. Per-frame tags are deterministic:
- All frames: site:<map_id>, split:<spec.split_tag>, override_metric:centerline
- Aomi/ariake half (by frame-number parity): also lateral:turn
- Psim half (by frame-number parity): split:manual, the other half: split:valid
- Aomi 2 specific frames: also longitudinal:yield

It writes:
- sample_dataset/<project>/<map_id>/<split>/<date>/<bag_time>/routes/<frame>.npz
- sample_dataset/<project>/<map_id>/<split>/<date>/<bag_time>/routes/<frame>.json
- sample_dataset/npz_list.json            (frame-level NPZ paths, absolute)
- sample_dataset/route_list.json           (3 route directories, absolute)
- sample_dataset/index.tag                 (pre-built index, pickle)
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# Make ``import tag_toolkit`` work when this script is run directly
# (``python _build.py``) rather than via the test runner. The editable install
# already covers the test-runner path, but this keeps the script runnable in
# isolation. We need the parent of the package directory on sys.path (so
# Python finds the ``tag_toolkit`` subdirectory), not the package dir itself.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tag_toolkit import TagStore

# Read-only source dataset on RDMA. Tests do not write here; they only read
# sidecar JSON to copy into the destination tree. When this path is
# unreachable (typical outside the original cluster), the script falls back
# to a minimal synthetic sidecar per route.
SOURCE_ROOT = Path(
    "/mnt/storage_rdma/diffusion_planner/dataset/20260715_basic_dataset"
)


@dataclass(frozen=True)
class RouteSpec:
    """One route to materialise under ``DEST_ROOT``."""

    project_id: str
    map_id: str
    split: str  # directory token: "manual" or "auto"
    split_tag: str  # tag written to the sidecar (e.g. "auto", "train", "manual")
    date: str  # ISO date as used in the source paths
    bag_time: str  # bag_time directory
    source_routes_dir: Path  # source directory containing real sidecars
    site_tag: str  # value for site:<site_tag>
    longitudinal_yield_frames: frozenset[str]  # frames that also get this tag

    def dest_route_dir(self, dest_root: Path) -> Path:
        return (
            dest_root
            / self.project_id
            / self.map_id
            / self.split
            / self.date
            / self.bag_time
        )


# 10 contiguous frames per route, picked from each closed-loop valid list.
# Source sidecars are at <source_routes_dir>/<bag_time>_00000000_<frame>.json
# The middle 8-digit prefix is always 00000000 in this dataset.
AOMI_FRAMES = [f"00000000_{n:08d}" for n in range(2997, 3007)]
ARIAKE_FRAMES = [f"00000000_{n:08d}" for n in range(6278, 6288)]
PSIM_FRAMES = [f"00000000_{n:08d}" for n in range(31, 41)]


# Deterministic 2-frame picks for aomi (the 4th and 8th frames of the route).
AOMI_LONGITUDINAL_YIELD = frozenset({"00000000_00003000", "00000000_00003004"})


ROUTES: tuple[RouteSpec, ...] = (
    RouteSpec(
        map_id="2231_odaiba_shinagawa_copied_from_xx1",
        project_id="x2_dev",
        split="auto",
        split_tag="auto",
        date="2026-06-23",
        bag_time="10-55-13",
        source_routes_dir=SOURCE_ROOT
        / "x2_dev/2231_odaiba_shinagawa_copied_from_xx1/auto/2026-06-23/10-55-13/routes",
        site_tag="2231_odaiba_shinagawa_copied_from_xx1",
        longitudinal_yield_frames=AOMI_LONGITUDINAL_YIELD,
    ),
    RouteSpec(
        map_id="2231_odaiba_shinagawa_copied_from_xx1",
        project_id="x2_dev",
        split="auto",
        split_tag="train",
        date="2026-07-07",
        bag_time="15-16-36",
        source_routes_dir=SOURCE_ROOT
        / "x2_dev/2231_odaiba_shinagawa_copied_from_xx1/auto/2026-07-07/15-16-36/routes",
        site_tag="2231_odaiba_shinagawa_copied_from_xx1",
        longitudinal_yield_frames=frozenset(),
    ),
    RouteSpec(
        map_id="b_mobility_seed_200_poses_100_jpntaxi_vehicle_8_pilot-auto-v0.49.2",
        project_id="xx1_psim",
        split="manual",
        split_tag="manual",  # psim frames alternate manual/valid via frame parity below
        date="2026-04-15",
        bag_time="psim_training_bag_0_0",
        source_routes_dir=SOURCE_ROOT
        / "xx1_psim/xx1_psim/manual/b_mobility_seed_200_poses_100_jpntaxi_vehicle_8_pilot-auto-v0.49.2/psim_training_bag_0_0/routes",
        site_tag="879_hiratsuka",
        longitudinal_yield_frames=frozenset(),
    ),
)


# Map route → frame list. The order matches the real closed-loop list so that
# npz_list.json is a contiguous slice.
ROUTE_FRAMES: dict[str, list[str]] = {
    "10-55-13": AOMI_FRAMES,
    "15-16-36": ARIAKE_FRAMES,
    "psim_training_bag_0_0": PSIM_FRAMES,
}


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


def _read_or_synthesize_sidecar(spec: RouteSpec, frame_num: str) -> dict:
    """Return the sidecar JSON for a frame, copying from RDMA or synthesizing.

    The real sidecar (when reachable) is preferred because it carries the
    canonical native fields (timestamp, qw/qx/qy/qz, neighbour_ids, etc.) that
    tests occasionally inspect. When unreachable, we emit a minimal stub so
    ``read_tags`` still works.
    """
    stem = f"{spec.bag_time}_{frame_num}"
    sidecar_src = spec.source_routes_dir / f"{stem}.json"
    if sidecar_src.is_file():
        try:
            return json.loads(sidecar_src.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    # Synthetic fallback — minimal fields, no native payload.
    return {
        "bag_time": spec.bag_time,
        "date": spec.date,
        "project_id": spec.project_id,
        "tags": [],
    }


def _build_one_route(spec: RouteSpec, dest_root: Path) -> list[Path]:
    """Materialise one route on disk; return the NPZ paths (frame-level list)."""
    dest = spec.dest_route_dir(dest_root)
    # Remove the route directory only — leave list artifacts and this file
    # alone so the script is idempotent.
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

        sidecar = _read_or_synthesize_sidecar(spec, frame_num)

        # Compute the deterministic tag set per frame.
        tags = [
            f"site:{spec.site_tag}",
            f"split:{_split_for_frame(spec, frame_num)}",
            "override_metric:centerline",
        ]
        if _frame_num(stem) % 2 == 1:
            tags.append("lateral:turn")
        if frame_num in spec.longitudinal_yield_frames:
            tags.append("longitudinal:yield")
        sidecar["tags"] = sorted(set(tags))

        json_dest.write_text(
            json.dumps(sidecar, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        npz_paths.append(npz_dest)

    return npz_paths


def build(dest_root: Path | None = None) -> dict:
    """Build the sample dataset under *dest_root* and emit list/index artifacts.

    If *dest_root* is None, defaults to the directory holding this script
    (so ``python _build.py`` rebuilds the checked-in tree).

    Returns a summary dict with paths to the three generated artifacts and
    the frame / route counts.
    """
    if dest_root is None:
        dest_root = Path(__file__).resolve().parent

    all_npz_paths: list[Path] = []
    route_dirs: list[Path] = []
    for spec in ROUTES:
        paths = _build_one_route(spec, dest_root)
        all_npz_paths.extend(paths)
        route_dirs.append(spec.dest_route_dir(dest_root))

    # Persist absolute NPZ / route lists.
    npz_list = dest_root / "npz_list.json"
    npz_list.write_text(
        json.dumps([str(p) for p in all_npz_paths], indent=1) + "\n",
        encoding="utf-8",
    )

    route_list = dest_root / "route_list.json"
    route_list.write_text(
        json.dumps([str(r) for r in route_dirs], indent=1) + "\n",
        encoding="utf-8",
    )

    # Pre-built index. build_index scans the freshly-written tree, pickles
    # the index to `index.tag`, and returns a TagStore that holds the same
    # index in memory.
    index_path = dest_root / "index.tag"
    TagStore.build_index(dest_root, index_path)

    return {
        "frame_count": len(all_npz_paths),
        "route_count": len(route_dirs),
        "npz_list": npz_list,
        "route_list": route_list,
        "index": index_path,
    }


if __name__ == "__main__":
    summary = build()
    print(
        f"Built sample_dataset: "
        f"{summary['route_count']} routes, "
        f"{summary['frame_count']} frames"
    )
    print(f"  npz_list:   {summary['npz_list']}")
    print(f"  route_list: {summary['route_list']}")
    print(f"  index:      {summary['index']}")