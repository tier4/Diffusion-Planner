"""Dataset root layout, immutable versions, catalog, writer lock, journal, GC (spec §3)."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from diffusion_planner.data_pipeline.errors import VersionExistsError

PHASES = ["built", "verified", "moved", "version_written", "catalog_updated"]


@dataclass
class PartitionEntry:
    partition_id: str
    pid: str
    data_rev: str
    meta_rev: str
    shards: list[str]
    sample_count: int
    source_fingerprint: str


@dataclass
class Version:
    tag: str
    partitions: dict[str, PartitionEntry]
    rule_hash: str
    source_namespace: str
    base_tag: str | None
    created_at: str
    packer_version: str
    recipe_hash: str
    format_version: int

    def to_json(self) -> str:
        d = asdict(self)
        d["partitions"] = {k: asdict(v) for k, v in sorted(self.partitions.items())}
        return json.dumps(d, sort_keys=True, indent=2) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "Version":
        d = json.loads(text)
        d["partitions"] = {k: PartitionEntry(**v) for k, v in d["partitions"].items()}
        return cls(**d)

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()


def _fsync_dir(d: Path) -> None:
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


class DatasetRoot:
    def __init__(self, root: Path):
        self.root = Path(root)

    versions_dir = property(lambda s: s.root / "versions")
    catalog_path = property(lambda s: s.root / "catalog.json")
    shards_dir = property(lambda s: s.root / "shards")
    manifest_dir = property(lambda s: s.root / "manifest")
    builds_dir = property(lambda s: s.root / "builds")
    lock_path = property(lambda s: s.root / ".writer.lock")

    def ensure_layout(self) -> None:
        for d in (self.versions_dir, self.shards_dir, self.manifest_dir, self.builds_dir):
            d.mkdir(parents=True, exist_ok=True)

    def shards_dir_for(self, pid: str, data_rev: str) -> Path:
        return self.shards_dir / f"{pid}@{data_rev}"

    def manifest_path_for(self, pid: str, data_rev: str, meta_rev: str) -> Path:
        return self.manifest_dir / f"{pid}@{data_rev}.{meta_rev}.parquet"

    def list_versions(self) -> list[str]:
        return sorted(p.stem for p in self.versions_dir.glob("*.json"))

    def latest(self) -> str | None:
        if not self.catalog_path.exists():
            return None
        return json.loads(self.catalog_path.read_text()).get("latest")

    def read_version(self, tag: str) -> Version:
        if tag == "latest":
            tag = self.latest()
            if tag is None:
                raise FileNotFoundError("catalog has no latest version")
        return Version.from_json((self.versions_dir / f"{tag}.json").read_text())

    def version_hash(self, tag: str) -> str:
        if tag == "latest":
            tag = self.read_version("latest").tag
        return hashlib.sha256((self.versions_dir / f"{tag}.json").read_bytes()).hexdigest()

    def write_version(self, v: Version) -> None:
        final = self.versions_dir / f"{v.tag}.json"
        data = v.to_json().encode()
        tmp = final.with_name(final.name + f".tmp.{os.getpid()}")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, final)  # create-if-absent: fails if final exists
        except FileExistsError:
            # Compare ignoring created_at for idempotency
            existing_version = Version.from_json(final.read_text())
            new_dict = asdict(v)
            existing_dict = asdict(existing_version)
            new_dict["created_at"] = existing_dict["created_at"]
            if new_dict == existing_dict:
                return  # idempotent rerun
            raise VersionExistsError(
                f"version {v.tag!r} exists with different content; tags are immutable"
            )
        finally:
            tmp.unlink(missing_ok=True)
        _fsync_dir(self.versions_dir)

    def set_latest(self, tag: str) -> None:
        if not (self.versions_dir / f"{tag}.json").exists():
            raise FileNotFoundError(tag)
        _write_atomic(self.catalog_path, (json.dumps({"latest": tag}) + "\n").encode())


@contextmanager
def writer_lock(root: DatasetRoot, timeout_s: float = 600.0):
    root.ensure_layout()
    fd = os.open(root.lock_path, os.O_RDWR | os.O_CREAT, 0o664)
    deadline = time.monotonic() + timeout_s
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"could not acquire writer lock {root.lock_path} within {timeout_s}s"
                    )
                time.sleep(0.05)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


class Journal:
    def __init__(self, build_dir: Path):
        self.build_dir = Path(build_dir)
        self.path = self.build_dir / "journal.json"

    @property
    def phases(self) -> list[str]:
        return json.loads(self.path.read_text())["phases"] if self.path.exists() else []

    @property
    def phase(self) -> str | None:
        p = self.phases
        return p[-1] if p else None

    def advance(self, phase: str) -> None:
        done = self.phases
        if phase not in PHASES or (done and PHASES.index(phase) <= PHASES.index(done[-1])):
            raise ValueError(f"cannot advance journal from {done} to {phase!r}")
        self.build_dir.mkdir(parents=True, exist_ok=True)
        _write_atomic(
            self.path,
            json.dumps({"build_id": self.build_dir.name, "phases": done + [phase]}).encode(),
        )


def gc_roots(root: DatasetRoot) -> list[str]:
    tags = set(root.list_versions())
    if (latest := root.latest()) is not None:
        tags.add(latest)
    return sorted(tags)


def referenced_artifacts(root: DatasetRoot, tags: list[str]) -> set[Path]:
    refs: set[Path] = set()
    for tag in tags:
        v = root.read_version(tag)
        for e in v.partitions.values():
            refs.add(root.shards_dir_for(e.pid, e.data_rev))
            refs.add(root.manifest_path_for(e.pid, e.data_rev, e.meta_rev))
    return refs


def gc(root: DatasetRoot, dry_run: bool) -> list[Path]:
    refs = referenced_artifacts(root, gc_roots(root))
    victims = [p for p in sorted(root.shards_dir.iterdir()) if p not in refs]
    victims += [p for p in sorted(root.manifest_dir.glob("*.parquet")) if p not in refs]
    for b in sorted(root.builds_dir.iterdir()) if root.builds_dir.exists() else []:
        if Journal(b).phase != "catalog_updated":
            victims.append(b)
    if not dry_run:
        for p in victims:
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    return victims


def prune_version(root: DatasetRoot, tag: str) -> None:
    if tag == root.latest():
        raise ValueError("refusing to prune the version catalog.latest points to")
    (root.versions_dir / f"{tag}.json").unlink()
    _fsync_dir(root.versions_dir)
