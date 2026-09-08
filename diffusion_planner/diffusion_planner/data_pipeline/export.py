"""Materialize a query as npz + sidecar json for legacy tools (spec §5b)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from diffusion_planner.data_pipeline.reader import ShardReader
from diffusion_planner.data_pipeline.sidecar import SIDECAR_FIELDS

_SIDECAR_NAMES = [n for n, _ in SIDECAR_FIELDS if n not in ("skip_label", "neighbor_count")]


def _rebuild_sidecar(row: dict) -> dict:
    out = {k: row[k] for k in _SIDECAR_NAMES if row.get(k) is not None}
    if row.get("skip_label") is not None:
        out["skipping_info"] = {"label": row["skip_label"]}
    return out


def export(root: Path, version: str, where: str, out_dir: Path) -> int:
    rd = ShardReader(root, version)
    rows = {r["key"]: r for r in rd.query(where).to_pylist()}
    out_dir = Path(out_dir)
    keys = []
    for key, arrays in rd.iter(where):
        target = out_dir / f"{key}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **arrays)
        sc = _rebuild_sidecar(rows[key])
        if sc:
            (out_dir / f"{key}.json").write_text(json.dumps(sc, indent=2))
        keys.append(key)
    (out_dir / "export_manifest.json").write_text(
        json.dumps(
            {
                "version": rd.version.tag,
                "version_hash": rd.version_hash,
                "where": where,
                "n": len(keys),
                "keys": sorted(keys),
            },
            indent=2,
        )
    )
    return len(keys)
