"""Partition rule, discovery, fingerprints. Paths are opaque (spec §3)."""

from __future__ import annotations

import base64
import fnmatch
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from diffusion_planner.data_pipeline.errors import PlanError
from diffusion_planner.data_pipeline.sidecar import sidecar_path_for

FileStat = tuple[int, int, int]


@dataclass(frozen=True)
class PartitionRule:
    depth: int | None = None
    regex: str | None = None

    def __post_init__(self):
        if (self.depth is None) == (self.regex is None):
            raise ValueError("exactly one of depth / regex must be given")
        if self.depth is not None and self.depth < 1:
            raise ValueError("depth must be >= 1")

    def partition_of(self, key: str) -> str:
        if self.depth is not None:
            parts = PurePosixPath(key).parts
            if len(parts) - 1 < self.depth:  # key's last component is the frame stem
                raise PlanError(f"key {key!r} shallower than partition depth {self.depth}")
            return "/".join(parts[: self.depth])
        m = re.match(self.regex, key)
        if not m:
            raise PlanError(f"key {key!r} does not match partition regex")
        return m.group("partition") if "partition" in m.groupdict() else m.group(1)

    @property
    def rule_hash(self) -> str:
        text = f"depth={self.depth}" if self.depth is not None else f"regex={self.regex}"
        return hashlib.sha256(text.encode()).hexdigest()


def pid_of(partition_id: str) -> str:
    digest = hashlib.sha256(partition_id.encode("utf-8")).digest()
    return base64.b32encode(digest).decode("ascii").lower()[:16]


@dataclass(frozen=True)
class Sample:
    key: str
    rel_dir: str
    npz_path: Path
    sidecar_path: Path | None
    partition_id: str


def stat_of(path: Path) -> FileStat:
    st = os.stat(path)
    return (st.st_ino, st.st_size, st.st_mtime_ns)


def is_selected(rel: str, include, exclude) -> bool:
    if include and not any(fnmatch.fnmatch(rel, g) for g in include):
        return False
    return not any(fnmatch.fnmatch(rel, g) for g in exclude)


def _rel_npz_paths(source: Path, path_list) -> list[str]:
    source = source.resolve()
    if path_list is None:
        return sorted(p.relative_to(source).as_posix() for p in source.rglob("*.npz"))
    out, seen = [], set()
    for raw in path_list:
        p = Path(raw)
        p = (source / p) if not p.is_absolute() else p
        p = p.resolve()
        try:
            rel = p.relative_to(source).as_posix()
        except ValueError:
            raise ValueError(f"path outside source: {raw}")
        if not p.is_file():
            raise FileNotFoundError(raw)
        if rel in seen:
            raise ValueError(f"duplicate path in path list: {raw}")
        seen.add(rel)
        out.append(rel)
    return sorted(out)


def discover(
    source: Path,
    rule: PartitionRule,
    include=(),
    exclude=(),
    path_list=None,
) -> dict[str, list[Sample]]:
    source = Path(source).resolve()
    groups: dict[str, list[Sample]] = {}
    for rel in _rel_npz_paths(source, path_list):
        if not is_selected(rel, list(include), list(exclude)):
            continue
        key = rel[: -len(".npz")]
        npz = source / rel
        sc = sidecar_path_for(npz)
        groups.setdefault(rule.partition_of(key), []).append(
            Sample(
                key=key,
                rel_dir=PurePosixPath(key).parent.as_posix(),
                npz_path=npz,
                sidecar_path=sc if sc.is_file() else None,
                partition_id=rule.partition_of(key),
            )
        )
    return {k: sorted(v, key=lambda s: s.key) for k, v in sorted(groups.items())}


def fingerprint(entries) -> str:
    lines = sorted(f"{k}\t{n.hex()}\t{s.hex() if s is not None else '-'}" for k, n, s in entries)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_HMS = re.compile(r"^\d{2}-\d{2}-\d{2}$")


def _pattern(name: str) -> str:
    if _DATE.match(name):
        return "<DATE>"
    if _HMS.match(name):
        return "<HH-MM-SS>"
    if name.startswith("seed_"):
        return "seed_*"
    if name.startswith("route_"):
        return "route_*"
    return name


@dataclass
class InspectReport:
    n_npz: int = 0
    npz_depth_histogram: dict[int, int] = field(default_factory=dict)
    dir_name_patterns: dict[int, Counter] = field(default_factory=dict)
    sidecar_variants: Counter = field(default_factory=Counter)
    missing_sidecars: int = 0
    non_sidecar_jsons: int = 0
    partitions: dict[str, int] = field(default_factory=dict)
    outside_rule: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"npz files: {self.n_npz}",
            f"npz depth histogram: {dict(sorted(self.npz_depth_histogram.items()))}",
        ]
        for lvl in sorted(self.dir_name_patterns):
            lines.append(f"level {lvl}: {self.dir_name_patterns[lvl].most_common(8)}")
        lines.append(
            f"sidecar variants (top-level key sets): {[(sorted(k), c) for k, c in self.sidecar_variants.most_common()]}"
        )
        lines.append(
            f"missing sidecars: {self.missing_sidecars}; non-sidecar jsons: {self.non_sidecar_jsons}"
        )
        lines.append(f"partitions ({len(self.partitions)}):")
        lines += [f"  {p}: {n}" for p, n in self.partitions.items()]
        if self.outside_rule:
            lines.append(f"OUTSIDE RULE ({len(self.outside_rule)}): {self.outside_rule[:10]}")
        return "\n".join(lines)


def inspect_tree(source: Path, rule: PartitionRule, include, exclude) -> InspectReport:
    source = Path(source).resolve()
    rep = InspectReport()
    npz_rels = [
        r for r in _rel_npz_paths(source, None) if is_selected(r, list(include), list(exclude))
    ]
    npz_stems = {r[:-4] for r in npz_rels}
    for js in source.rglob("*.json"):
        if js.relative_to(source).as_posix()[:-5] not in npz_stems:
            rep.non_sidecar_jsons += 1
    sampled: Counter = Counter()
    for rel in npz_rels:
        rep.n_npz += 1
        parts = PurePosixPath(rel).parts
        rep.npz_depth_histogram[len(parts)] = rep.npz_depth_histogram.get(len(parts), 0) + 1
        for lvl, name in enumerate(parts[:-1], start=1):
            rep.dir_name_patterns.setdefault(lvl, Counter())[_pattern(name)] += 1
        key = rel[:-4]
        try:
            pid = rule.partition_of(key)
        except PlanError:
            rep.outside_rule.append(key)
            continue
        rep.partitions[pid] = rep.partitions.get(pid, 0) + 1
        sc = sidecar_path_for(source / rel)
        if not sc.is_file():
            rep.missing_sidecars += 1
        elif sampled[pid] < 200:
            sampled[pid] += 1
            try:
                rep.sidecar_variants[frozenset(json.loads(sc.read_text()).keys())] += 1
            except Exception:
                rep.sidecar_variants[frozenset({"<malformed>"})] += 1
    rep.partitions = dict(sorted(rep.partitions.items()))
    return rep
