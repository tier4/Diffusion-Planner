"""pack / remove / scrub (spec §4)."""

from __future__ import annotations

import hashlib
import shutil
import uuid
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from diffusion_planner.data_pipeline import FORMAT_VERSION, PACKER_VERSION, encoding
from diffusion_planner.data_pipeline import partition as P
from diffusion_planner.data_pipeline import tar_shards as T
from diffusion_planner.data_pipeline import versioning as V
from diffusion_planner.data_pipeline.defaults import SHARD_SIZE_BYTES
from diffusion_planner.data_pipeline.errors import (
    EncodingError,
    IntegrityError,
    PlanError,
    RuleMismatchError,
    SourceChangedError,
)
from diffusion_planner.data_pipeline.manifest import (
    ManifestRow,
    meta_rev,
    read_manifest,
    rows_to_table,
    write_manifest,
)
from diffusion_planner.data_pipeline.sidecar import is_rejected, neighbor_ids_of, parse_sidecar


@dataclass
class PackOptions:
    source: Path
    dest: Path
    base: str
    tag: str
    rule: P.PartitionRule
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    path_list: list[str] | None = None
    partitions: list[str] | None = None
    sync: bool = False
    replace_all: bool = False
    shard_size_bytes: int = SHARD_SIZE_BYTES
    seed: int = 42
    workers: int = 1
    drop_skipped: bool = True
    with_neighbor_ids: bool = False
    force: bool = False
    require_marker: str | None = None
    source_namespace: str | None = None


@dataclass
class PartitionBuild:
    entry: V.PartitionEntry
    build_shards_dir: Path | None
    build_manifest: Path | None
    build_relation: Path | None
    reused: bool
    rejected: int = 0
    missing_sidecars: int = 0


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _data_rev(fingerprint: str, shard_size: int, seed: int) -> str:
    text = f"{fingerprint}\n{encoding.recipe_hash()}\nformat={FORMAT_VERSION}\nshard_size={shard_size}\nseed={seed}"
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _resolve_base(root: V.DatasetRoot, base: str) -> V.Version | None:
    return None if base == "none" else root.read_version(base)


def _build_partition(
    opts: PackOptions,
    build_dir: Path,
    partition_id: str,
    samples: list[P.Sample],
    base: V.Version | None,
) -> PartitionBuild:
    stats: dict[Path, P.FileStat] = {}
    kept: list[tuple[P.Sample, bytes, bytes | None, bytes, bytes | None, dict]] = []
    rejected = missing = 0
    for s in samples:
        stats[s.npz_path] = P.stat_of(s.npz_path)
        npz_bytes = s.npz_path.read_bytes()
        sc_bytes = None
        if s.sidecar_path is not None:
            stats[s.sidecar_path] = P.stat_of(s.sidecar_path)
            sc_bytes = s.sidecar_path.read_bytes()
        else:
            missing += 1
        fields = parse_sidecar(sc_bytes)
        if opts.drop_skipped and is_rejected(fields):
            rejected += 1
            continue
        kept.append(
            (
                s,
                npz_bytes,
                sc_bytes,
                hashlib.sha256(npz_bytes).digest(),
                hashlib.sha256(sc_bytes).digest() if sc_bytes is not None else None,
                fields,
            )
        )
    fp = P.fingerprint([(s.key, nsha, ssha) for s, _, _, nsha, ssha, _ in kept])
    data_rev = _data_rev(fp, opts.shard_size_bytes, opts.seed)
    pid = P.pid_of(partition_id)
    base_entry = base.partitions.get(partition_id) if base else None
    if (
        base_entry
        and not opts.force
        and base_entry.source_fingerprint == fp
        and base_entry.data_rev == data_rev
    ):
        return PartitionBuild(base_entry, None, None, None, True, rejected, missing)

    rng = np.random.default_rng(
        np.random.SeedSequence([opts.seed, zlib.crc32(partition_id.encode()), FORMAT_VERSION])
    )
    order = rng.permutation(len(kept))
    shards_dir = build_dir / "shards" / f"{pid}@{data_rev}"
    writer = T.ShardWriter(shards_dir, opts.shard_size_bytes)
    rows: list[ManifestRow] = []
    relation: list[tuple[str, list[str] | None]] = []
    for i in order:
        s, npz_bytes, sc_bytes, nsha, _, fields = kept[i]
        arrays = encoding.load_npz_bytes(npz_bytes)
        payload = encoding.encode_sample(arrays)
        if not encoding.arrays_bitexact(arrays, encoding.decode_sample(payload)):
            raise EncodingError(f"{s.key}: re-encoded arrays differ from source")
        rec = writer.add(payload)
        rows.append(
            ManifestRow(
                s.key,
                partition_id,
                s.rel_dir,
                rec.shard_id,
                rec.sample_index,
                rec.offset,
                rec.size,
                nsha,
                rec.payload_sha256,
                fields,
            )
        )
        if opts.with_neighbor_ids:
            relation.append((s.key, neighbor_ids_of(sc_bytes)))
    shard_names = writer.close()
    # quiescence check (spec §4.1)
    for path, before in stats.items():
        if not path.exists() or P.stat_of(path) != before:
            raise SourceChangedError(f"source changed during pack: {path}")
    for s in samples:
        if (s.sidecar_path is None) != (not s.npz_path.with_suffix(".json").exists()):
            raise SourceChangedError(f"sidecar appeared/disappeared during pack: {s.key}")
    table = rows_to_table(rows)
    mrev = meta_rev(table)
    manifest_path = build_dir / "manifest" / f"{pid}@{data_rev}.{mrev}.parquet"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(
        manifest_path,
        table,
        {
            "format_version": str(FORMAT_VERSION),
            "packer_version": PACKER_VERSION,
            "recipe_hash": encoding.recipe_hash(),
            "source_fingerprint": fp,
            "data_rev": data_rev,
            "meta_rev": mrev,
            "partition_id": partition_id,
            "shards": ",".join(shard_names),
        },
    )
    rel_path = None
    if opts.with_neighbor_ids:
        rel_path = build_dir / "relations" / f"{pid}@{data_rev}.neighbor_ids.parquet"
        rel_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "key": [k for k, _ in relation],
                    "neighbor_ids": pa.array([v for _, v in relation], pa.list_(pa.string())),
                }
            ),
            rel_path,
        )
    _verify_build(shards_dir, shard_names, table, kept)
    entry = V.PartitionEntry(partition_id, pid, data_rev, mrev, shard_names, len(rows), fp)
    return PartitionBuild(entry, shards_dir, manifest_path, rel_path, False, rejected, missing)


def _verify_build(shards_dir: Path, shard_names: list[str], table: pa.Table, kept) -> None:
    by_key = {s.key: npz for s, npz, *_ in kept}
    rows = table.to_pylist()
    per_shard: dict[int, list[dict]] = {}
    for r in rows:
        per_shard.setdefault(r["shard_id"], []).append(r)
    for sid, name in enumerate(shard_names):
        listed = T.list_members(shards_dir / name)
        expect = sorted(per_shard.get(sid, []), key=lambda r: r["sample_index_in_shard"])
        if [i for i, _, _ in listed] != [r["sample_index_in_shard"] for r in expect]:
            raise IntegrityError(f"{name}: member list differs from manifest")
        with open(shards_dir / name, "rb") as f:
            for (idx, off, size), r in zip(listed, expect):
                if (off, size) != (r["offset"], r["size"]):
                    raise IntegrityError(f"{name}[{idx}]: offset/size mismatch")
                payload = T.read_member(f, off, size)
                if hashlib.sha256(payload).digest() != r["payload_sha256"]:
                    raise IntegrityError(f"{name}[{idx}]: payload sha mismatch")
                if idx % 100 == 0 and not encoding.arrays_bitexact(
                    encoding.load_npz_bytes(by_key[r["key"]]), encoding.decode_sample(payload)
                ):
                    raise IntegrityError(f"{name}[{idx}]: spot decode differs from source")


def _check_unique_keys(root: V.DatasetRoot, version: V.Version) -> None:
    files = [
        str(root.manifest_path_for(e.pid, e.data_rev, e.meta_rev))
        for e in version.partitions.values()
    ]
    if not files:
        return
    dup = duckdb.sql(
        f"SELECT key FROM read_parquet({files!r}) GROUP BY key HAVING count(*) > 1 LIMIT 5"
    ).fetchall()
    if dup:
        raise PlanError(f"duplicate keys across partitions, e.g. {[k for (k,) in dup]}")


def _publish(
    root: V.DatasetRoot, journal: V.Journal, version: V.Version, builds: list[PartitionBuild]
) -> None:
    journal.advance("verified")
    for b in builds:
        if b.reused:
            continue
        target = root.shards_dir_for(b.entry.pid, b.entry.data_rev)
        if target.exists():
            if sorted(p.name for p in target.iterdir()) != sorted(b.entry.shards):
                raise IntegrityError(f"existing revision dir {target} differs from build")
            shutil.rmtree(b.build_shards_dir)
        else:
            shutil.move(str(b.build_shards_dir), str(target))
        mpath = root.manifest_path_for(b.entry.pid, b.entry.data_rev, b.entry.meta_rev)
        if not mpath.exists():
            shutil.move(str(b.build_manifest), str(mpath))
        if b.build_relation is not None:
            (root.root / "relations").mkdir(exist_ok=True)
            shutil.move(str(b.build_relation), str(root.root / "relations" / b.build_relation.name))
    journal.advance("moved")
    _check_unique_keys(root, version)
    root.write_version(version)
    journal.advance("version_written")
    root.set_latest(version.tag)
    journal.advance("catalog_updated")


def pack(opts: PackOptions) -> V.Version:
    root = V.DatasetRoot(opts.dest)
    root.ensure_layout()
    if opts.require_marker and not (Path(opts.source) / opts.require_marker).exists():
        raise SourceChangedError(f"completion marker missing: {opts.require_marker}")
    namespace = opts.source_namespace or str(Path(opts.source).resolve())
    with V.writer_lock(root):
        base = _resolve_base(root, opts.base)
        groups = P.discover(opts.source, opts.rule, opts.include, opts.exclude, opts.path_list)
        if opts.partitions is not None:
            groups = {k: v for k, v in groups.items() if k in set(opts.partitions)}
        # only guard against a mismatched rule/namespace when it would actually build something
        # under the new scheme; a partition-scoped pack that resolves to no groups is a no-op.
        if (
            base
            and not opts.replace_all
            and groups
            and (base.rule_hash != opts.rule.rule_hash or base.source_namespace != namespace)
        ):
            raise RuleMismatchError(
                "partition rule or source namespace differs from base; pass --replace-all"
            )
        build_dir = root.builds_dir / uuid.uuid4().hex
        build_dir.mkdir(parents=True)
        journal = V.Journal(build_dir)
        builds = [
            _build_partition(opts, build_dir, pid, samples, base) for pid, samples in groups.items()
        ]
        journal.advance("built")
        partitions = {} if (base is None or opts.replace_all) else dict(base.partitions)
        if opts.sync and base is not None:
            for pid in list(partitions):
                if pid not in groups and P.is_selected(
                    pid + "/x.npz", list(opts.include), list(opts.exclude)
                ):
                    del partitions[pid]
        for b in builds:
            partitions[b.entry.partition_id] = b.entry
        version = V.Version(
            opts.tag,
            partitions,
            opts.rule.rule_hash,
            namespace,
            None if base is None else base.tag,
            _now(),
            PACKER_VERSION,
            encoding.recipe_hash(),
            FORMAT_VERSION,
        )
        _publish(root, journal, version, builds)
        for b in builds:
            print(
                f"{b.entry.partition_id}: {b.entry.sample_count} kept, {b.rejected} rejected, "
                f"{b.missing_sidecars} missing sidecars, {'reused' if b.reused else 'built'}, "
                f"{len(b.entry.shards)} shards, data_rev={b.entry.data_rev} meta_rev={b.entry.meta_rev}"
            )
        return version


def remove(dest: Path, base: str, tag: str, partition_ids: list[str]) -> V.Version:
    root = V.DatasetRoot(dest)
    with V.writer_lock(root):
        b = root.read_version(base)
        missing = [p for p in partition_ids if p not in b.partitions]
        if missing:
            raise PlanError(f"partitions not in {base}: {missing}")
        parts = {k: v for k, v in b.partitions.items() if k not in set(partition_ids)}
        v = V.Version(
            tag,
            parts,
            b.rule_hash,
            b.source_namespace,
            b.tag,
            _now(),
            PACKER_VERSION,
            b.recipe_hash,
            b.format_version,
        )
        journal = V.Journal(root.builds_dir / uuid.uuid4().hex)
        journal.advance("built")
        journal.advance("verified")
        journal.advance("moved")
        root.write_version(v)
        journal.advance("version_written")
        root.set_latest(tag)
        journal.advance("catalog_updated")
        return v


def scrub(dest: Path, tag: str) -> dict[str, int]:
    root = V.DatasetRoot(dest)
    v = root.read_version(tag)
    members = mismatches = shards = 0
    for e in v.partitions.values():
        table = read_manifest(
            root.manifest_path_for(e.pid, e.data_rev, e.meta_rev),
            columns=["shard_id", "sample_index_in_shard", "offset", "size", "payload_sha256"],
        )
        expect = {(r["shard_id"], r["sample_index_in_shard"]): r for r in table.to_pylist()}
        for sid, name in enumerate(e.shards):
            shards += 1
            shard_rows = sum(1 for (s, _) in expect if s == sid)
            for idx, payload in T.iter_members(
                root.shards_dir_for(e.pid, e.data_rev) / name, expected_count=shard_rows
            ):
                members += 1
                r = expect.get((sid, idx))
                if (
                    r is None
                    or hashlib.sha256(payload).digest() != r["payload_sha256"]
                    or len(payload) != r["size"]
                ):
                    mismatches += 1
    report = {"members": members, "mismatches": mismatches, "shards": shards}
    if mismatches:
        raise IntegrityError(f"scrub found {mismatches} mismatching members: {report}")
    return report
