# -*- coding: utf-8 -*-
"""SQLite database setup and configuration."""

from __future__ import annotations

import sqlite3
from pathlib import Path

_DB_SUFFIXES = (".tags.db",)

# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS frames (
    path          TEXT PRIMARY KEY,
    route         TEXT NOT NULL,
    sidecar_mtime INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS frames_route ON frames(route);
CREATE INDEX IF NOT EXISTS frames_route_path ON frames(route, path);

CREATE TABLE IF NOT EXISTS tags (
    path TEXT NOT NULL,
    tag  TEXT NOT NULL,
    dim  TEXT NOT NULL,
    val  TEXT NOT NULL,
    PRIMARY KEY (path, tag)
);
CREATE INDEX IF NOT EXISTS tags_tag_path ON tags(tag, path);
CREATE INDEX IF NOT EXISTS tags_path ON tags(path);
CREATE INDEX IF NOT EXISTS tags_dim_val_path ON tags(dim, val, path);
"""


def _init_db(conn: sqlite3.Connection) -> None:
    """Run schema + WAL PRAGMA on a fresh connection."""
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA_SQL)


def _is_db_path(source: str | Path | None) -> bool:
    if source is None:
        return False
    p = Path(source)
    if not p.exists():
        return False
    if any(p.name.endswith(suf) for suf in _DB_SUFFIXES):
        return True
    try:
        conn = sqlite3.connect(str(p))
        conn.execute("SELECT 1 FROM frames LIMIT 1")
        conn.close()
        return True
    except (sqlite3.DatabaseError, sqlite3.OperationalError):
        return False
