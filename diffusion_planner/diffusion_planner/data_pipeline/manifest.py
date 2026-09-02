"""Parquet manifest fragments: index columns + existing sidecar fields (spec §3)."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from diffusion_planner.data_pipeline.sidecar import SIDECAR_FIELDS

INDEX_FIELDS: list[tuple[str, pa.DataType]] = [
    ("key", pa.string()),
    ("partition_id", pa.string()),
    ("rel_dir", pa.string()),
    ("shard_id", pa.int32()),
    ("sample_index_in_shard", pa.int32()),
    ("offset", pa.int64()),
    ("size", pa.int64()),
    ("source_sha256", pa.binary(32)),
    ("payload_sha256", pa.binary(32)),
]
PLANNING_COLUMNS = ["partition_id", "shard_id", "sample_index_in_shard", "offset", "size"]
_META_PREFIX = b"dp."


def manifest_schema() -> pa.Schema:
    return pa.schema(
        [pa.field(n, t) for n, t in INDEX_FIELDS]
        + [pa.field(n, t, nullable=True) for n, t in SIDECAR_FIELDS]
    )


@dataclass
class ManifestRow:
    key: str
    partition_id: str
    rel_dir: str
    shard_id: int
    sample_index_in_shard: int
    offset: int
    size: int
    source_sha256: bytes
    payload_sha256: bytes
    sidecar: dict


def rows_to_table(rows: list[ManifestRow]) -> pa.Table:
    rows = sorted(rows, key=lambda r: r.key)
    cols: dict[str, list] = {n: [] for n, _ in INDEX_FIELDS}
    for n, _ in SIDECAR_FIELDS:
        cols[n] = []
    for r in rows:
        for n, _ in INDEX_FIELDS:
            cols[n].append(getattr(r, n))
        for n, _ in SIDECAR_FIELDS:
            cols[n].append(r.sidecar.get(n))
    return pa.Table.from_pydict(cols, schema=manifest_schema())


def meta_rev(table: pa.Table) -> str:
    h = hashlib.sha256()
    keys = table.column("key").to_pylist()
    for n, _ in SIDECAR_FIELDS:
        h.update(n.encode())
        for k, v in zip(keys, table.column(n).to_pylist()):
            h.update(f"{k}={v!r}\n".encode())
    return h.hexdigest()[:16]


def write_manifest(path: Path, table: pa.Table, metadata: dict[str, str]) -> None:
    path = Path(path)
    meta = {(_META_PREFIX + k.encode()): v.encode() for k, v in metadata.items()}
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), **meta})
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd", use_dictionary=True)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_manifest(path: Path, columns: list[str] | None = None) -> pa.Table:
    return pq.read_table(path, columns=columns)


def read_metadata(path: Path) -> dict[str, str]:
    md = pq.read_schema(path).metadata or {}
    return {
        k[len(_META_PREFIX) :].decode(): v.decode()
        for k, v in md.items()
        if k.startswith(_META_PREFIX)
    }
