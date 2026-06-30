"""Small helpers for render output identity and metadata."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path


def path_label(path: str | Path | None) -> str:
    if not path:
        return "none"
    p = Path(path)
    return f"{p.parent.name}/{p.name}"


def slug(text: str, max_len: int = 120) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")
    return (s or "render")[:max_len]


def run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def render_tag(*parts: str | Path | None) -> str:
    labels = [path_label(p) for p in parts if p]
    if not labels:
        labels = ["render"]
    return f"{slug('__'.join(labels), max_len=96)}__{run_stamp()}"


def write_render_meta(out_dir: str | Path, **meta: object) -> None:
    path = Path(out_dir) / "render_meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str))
