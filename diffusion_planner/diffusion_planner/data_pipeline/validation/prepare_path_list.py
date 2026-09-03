"""Rewrite a training path list onto the pack host's prefix and validate it (spec §3.2)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath


def _under(path: str, prefix: str) -> bool:
    """True when `path` lies under `prefix` on a path-component boundary."""
    pp, pf = PurePosixPath(path).parts, PurePosixPath(prefix).parts
    return len(pp) >= len(pf) and pp[: len(pf)] == pf


def prepare(in_path: Path, out_path: Path, old_prefix: str, new_prefix: str) -> dict:
    raw = json.loads(Path(in_path).read_text())
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise ValueError(f"{in_path}: expected a JSON list of strings")
    old_prefix = old_prefix.rstrip("/")
    new_prefix = new_prefix.rstrip("/")
    out, seen = [], set()
    for entry in raw:
        if not entry.endswith(".npz"):
            raise ValueError(f"not a .npz path: {entry}")
        if not _under(entry, old_prefix):
            raise ValueError(f"does not start with {old_prefix!r} on a component boundary: {entry}")
        tail = PurePosixPath(entry).parts[len(PurePosixPath(old_prefix).parts) :]
        rewritten = str(PurePosixPath(new_prefix).joinpath(*tail))
        if rewritten in seen:
            raise ValueError(f"duplicate path after rewrite: {rewritten}")
        seen.add(rewritten)
        out.append(rewritten)
    Path(out_path).write_text(json.dumps(out))
    return {
        "entries": len(raw),
        "rewritten": len(out),
        "sha256": hashlib.sha256(Path(out_path).read_bytes()).hexdigest(),
        "out": str(out_path),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="prepare_path_list")
    ap.add_argument("--in", dest="in_path", required=True, type=Path)
    ap.add_argument("--out", dest="out_path", required=True, type=Path)
    ap.add_argument("--old-prefix", required=True)
    ap.add_argument("--new-prefix", required=True)
    a = ap.parse_args(argv)
    try:
        report = prepare(a.in_path, a.out_path, a.old_prefix, a.new_prefix)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
