"""ShardReader: query / get / iter over a dataset version (spec §5b)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import duckdb
import numpy as np
import pyarrow as pa

from diffusion_planner.data_pipeline import encoding
from diffusion_planner.data_pipeline import tar_shards as T
from diffusion_planner.data_pipeline.defaults import SEEK_THRESHOLD
from diffusion_planner.data_pipeline.errors import PlanError
from diffusion_planner.data_pipeline.keyset import manifest_files
from diffusion_planner.data_pipeline.versioning import DatasetRoot


class ShardReader:
    def __init__(self, root: Path, version: str = "latest"):
        self.root = DatasetRoot(root)
        self.version = self.root.read_version(version)
        self.version_hash = self.root.version_hash(self.version.tag)
        self._files = manifest_files(self.root, self.version)
        self._con = duckdb.connect()

    def query(self, where: str, columns: list[str] | None = None) -> pa.Table:
        """Query manifest with WHERE clause; reserved column names (offset, size) must be double-quoted in WHERE.

        Args:
            where: WHERE clause expression; `;` is forbidden; reserved names like `offset` must be `"offset"`
            columns: columns to select (auto-quoted if reserved); None for all

        Raises:
            PlanError: if WHERE contains `;` or if DuckDB syntax error occurs (likely reserved column not quoted)
        """
        if ";" in where:
            raise PlanError("WHERE clause must be a single expression (no ';')")
        if columns:
            cols = ", ".join(f'"{c}"' if c in ("offset", "size") else c for c in columns)
        else:
            cols = "*"
        try:
            return (
                self._con.execute(
                    f"SELECT {cols} FROM read_parquet(?) WHERE {where} ORDER BY key", [self._files]
                )
                .arrow()
                .read_all()
            )
        except (duckdb.ParserException, duckdb.BinderException, duckdb.CatalogException) as e:
            raise PlanError(
                f'invalid WHERE clause ({e}); double-quote reserved column names, e.g. "offset"'
            ) from e

    def shard_path(self, partition_id: str, shard_id: int) -> Path:
        e = self.version.partitions[partition_id]
        return self.root.shards_dir_for(e.pid, e.data_rev) / e.shards[shard_id]

    def get(self, key: str) -> dict[str, np.ndarray]:
        rows = self._con.execute(
            'SELECT partition_id, shard_id, "offset", "size" FROM read_parquet(?) WHERE key = ?',
            [self._files, key],
        ).fetchall()
        if not rows:
            raise KeyError(key)
        partition_id, shard_id, offset, size = rows[0]
        with open(self.shard_path(partition_id, shard_id), "rb") as f:
            return encoding.decode_sample(T.read_member(f, offset, size))

    def iter(
        self, where: str, *, training_view: bool = False
    ) -> Iterator[tuple[str, dict[str, np.ndarray]]]:
        sel = self.query(
            where, ["key", "partition_id", "shard_id", "sample_index_in_shard", "offset", "size"]
        ).to_pylist()
        by_shard: dict[tuple[str, int], list[dict]] = {}
        for r in sel:
            by_shard.setdefault((r["partition_id"], r["shard_id"]), []).append(r)
        decode = encoding.decode_for_training if training_view else encoding.decode_sample
        for (pid, sid), rows in sorted(by_shard.items()):
            path = self.shard_path(pid, sid)
            total = len(T.list_members(path))
            if len(rows) / max(total, 1) >= SEEK_THRESHOLD:
                wanted = {r["sample_index_in_shard"]: r["key"] for r in rows}
                for idx, payload in T.iter_members(path):
                    if idx in wanted:
                        yield wanted[idx], decode(payload)
            else:
                with open(path, "rb") as f:
                    for r in sorted(rows, key=lambda r: r["offset"]):
                        yield r["key"], decode(T.read_member(f, r["offset"], r["size"]))
