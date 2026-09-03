"""Key-sets: the successor of path_list_*.json — coordinates bound to a version hash (spec §5)."""

from __future__ import annotations

import hashlib
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from diffusion_planner.data_pipeline.duck import run_query
from diffusion_planner.data_pipeline.errors import KeysetMismatchError, PlanError
from diffusion_planner.data_pipeline.versioning import DatasetRoot, Version

KEYSET_SCHEMA = pa.schema(
    [("partition_id", pa.string()), ("shard_id", pa.int32()), ("sample_index_in_shard", pa.int32())]
)
_COLS = "partition_id, shard_id, sample_index_in_shard"


def manifest_files(root: DatasetRoot, version: Version) -> list[str]:
    return [
        str(root.manifest_path_for(e.pid, e.data_rev, e.meta_rev))
        for _, e in sorted(version.partitions.items())
    ]


def _write(out_path: Path, table: pa.Table, root: DatasetRoot, tag: str, where: str | None) -> Path:
    if table.num_rows == 0:
        raise PlanError("key-set selection is empty")
    meta = {
        b"dp.version_tag": tag.encode(),
        b"dp.version_hash": root.version_hash(tag).encode(),
        b"dp.created_at": datetime.now(timezone.utc).isoformat().encode(),
    }
    if where is not None:
        meta[b"dp.where"] = where.encode()
    table = table.cast(KEYSET_SCHEMA).replace_schema_metadata(meta)
    tmp = out_path.with_name(out_path.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, out_path)
    return out_path


def materialize_keyset(root: DatasetRoot, version_tag: str, where: str, out_path: Path) -> Path:
    v = root.read_version(version_tag)
    files = manifest_files(root, v)
    con = duckdb.connect()
    table = run_query(
        con, f"SELECT {_COLS} FROM read_parquet(?) WHERE {where} ORDER BY key", [files]
    )
    return _write(Path(out_path), table, root, v.tag, where)


def keyset_from_keys(root: DatasetRoot, version_tag: str, keys: list[str], out_path: Path) -> Path:
    dups = sorted(k for k, c in Counter(keys).items() if c > 1)[:5]
    if dups:
        raise PlanError(f"duplicate keys in key list, e.g. {dups}")
    v = root.read_version(version_tag)
    con = duckdb.connect()
    con.register("wanted", pa.table({"key": pa.array(keys, pa.string())}))
    files = manifest_files(root, v)
    table = run_query(
        con,
        f"SELECT m.key, {', '.join('m.' + c for c in _COLS.split(', '))} FROM read_parquet(?) m "
        "JOIN wanted w ON m.key = w.key ORDER BY m.key",
        [files],
    )
    found = set(table.column("key").to_pylist())
    unknown = [k for k in keys if k not in found][:5]
    if unknown:
        raise PlanError(f"unknown keys for version {v.tag}, e.g. {unknown}")
    return _write(Path(out_path), table.drop(["key"]), root, v.tag, None)


def load_keyset(path: Path, root: DatasetRoot, version_tag: str) -> pa.Table:
    table = pq.read_table(path)
    md = table.schema.metadata or {}
    expected = root.version_hash(version_tag)
    if md.get(b"dp.version_hash", b"").decode() != expected:
        raise KeysetMismatchError(
            f"key-set {path} was built against version hash "
            f"{md.get(b'dp.version_hash', b'?').decode()}, not {version_tag} ({expected[:12]}…)"
        )
    con = duckdb.connect()
    con.register("ks", table)
    n_dup = con.execute(
        f"SELECT count(*) FROM (SELECT {_COLS} FROM ks GROUP BY {_COLS} HAVING count(*) > 1)"
    ).fetchone()[0]
    if n_dup:
        raise PlanError(f"key-set contains {n_dup} duplicate coordinates")
    return table


def keyset_digest(table: pa.Table) -> str:
    h = hashlib.sha256()
    for col in ("partition_id", "shard_id", "sample_index_in_shard"):
        h.update(col.encode())
        h.update(str(table.column(col).to_pylist()).encode())
    return h.hexdigest()[:16]
