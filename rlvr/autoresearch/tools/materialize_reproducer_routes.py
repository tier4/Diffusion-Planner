"""Materialize contiguous reproducer route datasets from a flat NPZ corpus.

The Perception Reproducer needs each route as a directory of contiguous ``*.npz``
frames with matching pose sidecar ``*.json`` files. Some converted 4-col corpora
are flat NPZ directories whose sidecars still live in the original conversion
tree. This tool matches sidecars by filename stem, groups frames into routes, and
symlinks each route into the workspace.

Route key preference:
1. matched sidecar parent directory name, when sidecars come from ``--sidecar_root``;
2. filename prefix fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

_TWO_INDEX_RE = re.compile(r"(?P<route>.+)_\d{8}_\d{8}$")
_FRAME_RE = re.compile(r"_(?P<frame>\d+)$")


def _frame_index(path: Path) -> int:
    m = _FRAME_RE.search(path.stem)
    if m is None:
        raise ValueError(f"Cannot parse frame index from {path.name!r}")
    return int(m.group("frame"))


def _route_from_stem(path: Path) -> str:
    m = _TWO_INDEX_RE.match(path.stem)
    if m is not None:
        return m.group("route")
    return _FRAME_RE.sub("", path.stem)


def _sidecar_index(sidecar_root: Path | None) -> dict[str, Path]:
    if sidecar_root is None:
        return {}
    return {p.stem: p for p in sidecar_root.rglob("*.json")}


def _resolve_sidecar(npz_path: Path, sidecars_by_stem: dict[str, Path]) -> Path:
    sibling = npz_path.with_suffix(".json")
    if sibling.is_file():
        return sibling
    hit = sidecars_by_stem.get(npz_path.stem)
    if hit is not None:
        return hit
    raise FileNotFoundError(f"No sidecar JSON found for {npz_path}")


def _route_key(npz_path: Path, sidecar_path: Path, sidecar_root: Path | None) -> str:
    if sidecar_root is not None:
        try:
            sidecar_path.relative_to(sidecar_root)
            if sidecar_path.parent.name != "routes":
                return sidecar_path.parent.name
        except ValueError:
            pass
    return _route_from_stem(npz_path)


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._+-]+", "_", name.strip())
    return cleaned or "route"


def _link_or_copy(src: Path, dst: Path, *, copy: bool, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            if dst.resolve() == src.resolve():
                return
            raise FileExistsError(f"{dst} already exists")
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def _assert_contiguous(route: str, frames: list[tuple[int, Path, Path]]) -> None:
    idxs = [f for f, _, _ in frames]
    gaps = [(a, b) for a, b in zip(idxs, idxs[1:]) if b != a + 1]
    if gaps:
        preview = ", ".join(f"{a}->{b}" for a, b in gaps[:8])
        raise ValueError(f"{route}: non-contiguous frame indices ({preview})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--npz_root", type=Path, required=True, help="Flat or nested converted NPZ corpus"
    )
    p.add_argument(
        "--sidecar_root",
        type=Path,
        default=None,
        help="Original conversion tree containing per-frame pose JSON sidecars",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Workspace route output dir. One child directory is created per route.",
    )
    p.add_argument(
        "--route_prefix",
        default="",
        help="Optional prefix for output route directory names, e.g. akebono_",
    )
    p.add_argument("--copy", action="store_true", help="Copy files instead of symlinking")
    p.add_argument("--overwrite", action="store_true", help="Replace existing output links/files")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    npz_paths = sorted(args.npz_root.rglob("*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No .npz files under {args.npz_root}")
    if args.sidecar_root is not None and not args.sidecar_root.is_dir():
        raise NotADirectoryError(args.sidecar_root)

    sidecars_by_stem = _sidecar_index(args.sidecar_root)
    routes: dict[str, list[tuple[int, Path, Path]]] = {}
    for npz_path in npz_paths:
        sidecar_path = _resolve_sidecar(npz_path, sidecars_by_stem)
        key = _route_key(npz_path, sidecar_path, args.sidecar_root)
        routes.setdefault(key, []).append((_frame_index(npz_path), npz_path, sidecar_path))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "npz_root": str(args.npz_root),
        "sidecar_root": str(args.sidecar_root) if args.sidecar_root else None,
        "copy": bool(args.copy),
        "routes": [],
    }

    for key in sorted(routes):
        frames = sorted(routes[key], key=lambda x: x[0])
        _assert_contiguous(key, frames)
        route_name = _safe_name(f"{args.route_prefix}{key}")
        route_dir = args.output_dir / route_name
        route_dir.mkdir(parents=True, exist_ok=True)
        for _, npz_path, sidecar_path in frames:
            _link_or_copy(
                npz_path, route_dir / npz_path.name, copy=args.copy, overwrite=args.overwrite
            )
            _link_or_copy(
                sidecar_path,
                route_dir / f"{npz_path.stem}.json",
                copy=args.copy,
                overwrite=args.overwrite,
            )
        route_manifest = {
            "name": route_name,
            "route_key": key,
            "path": str(route_dir),
            "frames": len(frames),
            "first_frame": frames[0][0],
            "last_frame": frames[-1][0],
        }
        route_meta = None
        first_sidecar_parent = frames[0][2].parent
        cand = first_sidecar_parent / "routes" / f"{key}_sequence_00000000.json"
        if cand.is_file():
            route_meta = cand
            _link_or_copy(
                cand,
                route_dir / "route_metadata.json",
                copy=args.copy,
                overwrite=args.overwrite,
            )
        route_manifest["route_metadata"] = str(route_meta) if route_meta else None
        manifest["routes"].append(route_manifest)
        print(
            f"{route_name}: {len(frames)} frames ({frames[0][0]}..{frames[-1][0]}) -> {route_dir}"
        )

    out_manifest = args.output_dir / "manifest.json"
    out_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {out_manifest}")


if __name__ == "__main__":
    main()
