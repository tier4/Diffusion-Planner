import json
import threading
import time

import pytest
from diffusion_planner.data_pipeline import versioning as V
from diffusion_planner.data_pipeline.errors import VersionExistsError


def _version(tag, parts):
    return V.Version(
        tag=tag,
        partitions=parts,
        rule_hash="r",
        source_namespace="src",
        base_tag=None,
        created_at="2026-09-02T00:00:00Z",
        packer_version="0.1.0",
        recipe_hash="h",
        format_version=1,
    )


def _entry(pid_name, data_rev, meta_rev):
    return V.PartitionEntry(
        pid_name, V_pid(pid_name), data_rev, meta_rev, ["shard-0000.tar"], 3, "fp"
    )


def V_pid(x):
    from diffusion_planner.data_pipeline.partition import pid_of

    return pid_of(x)


def test_layout_paths(tmp_path):
    root = V.DatasetRoot(tmp_path)
    root.ensure_layout()
    assert root.shards_dir_for("abc", "d1") == tmp_path / "shards" / "abc@d1"
    assert root.manifest_path_for("abc", "d1", "m1") == tmp_path / "manifest" / "abc@d1.m1.parquet"
    assert (tmp_path / "versions").is_dir() and (tmp_path / "builds").is_dir()


def test_version_roundtrip_and_create_if_absent(tmp_path):
    root = V.DatasetRoot(tmp_path)
    root.ensure_layout()
    v = _version("v1", {"p/a": _entry("p/a", "d1", "m1")})
    root.write_version(v)
    assert V.Version.from_json(root.read_version("v1").to_json()) == v
    root.write_version(v)  # identical content → idempotent
    v2 = _version("v1", {"p/a": _entry("p/a", "d2", "m1")})
    with pytest.raises(VersionExistsError):
        root.write_version(v2)
    assert root.latest() is None
    root.set_latest("v1")
    assert root.latest() == "v1" and root.read_version("latest").tag == "v1"
    assert len(root.version_hash("v1")) == 64


def test_version_idempotent_with_different_created_at(tmp_path):
    """Writing the same version with different created_at should be accepted (idempotent)."""
    root = V.DatasetRoot(tmp_path)
    root.ensure_layout()
    v1 = _version("v1", {"p/a": _entry("p/a", "d1", "m1")})
    root.write_version(v1)

    # Re-run with different timestamp (simulating a rerun of the same pack)
    v1_rerun = V.Version(
        tag="v1",
        partitions={"p/a": _entry("p/a", "d1", "m1")},
        rule_hash="r",
        source_namespace="src",
        base_tag=None,
        created_at="2026-09-02T01:00:00Z",  # Different timestamp
        packer_version="0.1.0",
        recipe_hash="h",
        format_version=1,
    )
    # Should not raise - idempotent rerun is accepted
    root.write_version(v1_rerun)

    # But different data_rev should still raise
    v1_different = V.Version(
        tag="v1",
        partitions={"p/a": _entry("p/a", "d2", "m1")},  # Different data_rev
        rule_hash="r",
        source_namespace="src",
        base_tag=None,
        created_at="2026-09-02T00:00:00Z",
        packer_version="0.1.0",
        recipe_hash="h",
        format_version=1,
    )
    with pytest.raises(VersionExistsError):
        root.write_version(v1_different)


def test_writer_lock_is_exclusive(tmp_path):
    root = V.DatasetRoot(tmp_path)
    root.ensure_layout()
    order = []

    def holder():
        with V.writer_lock(root):
            order.append("a-in")
            time.sleep(0.3)
            order.append("a-out")

    t = threading.Thread(target=holder)
    t.start()
    time.sleep(0.05)
    with V.writer_lock(root):
        order.append("b-in")
    t.join()
    assert order == ["a-in", "a-out", "b-in"]


def test_gc_roots_and_gc(tmp_path):
    root = V.DatasetRoot(tmp_path)
    root.ensure_layout()
    for pid_name, d, m in [("p/a", "d1", "m1"), ("p/a", "d2", "m1"), ("p/b", "d3", "m3")]:
        root.shards_dir_for(V_pid(pid_name), d).mkdir(parents=True)
        (root.shards_dir_for(V_pid(pid_name), d) / "shard-0000.tar").write_bytes(b"x")
        root.manifest_path_for(V_pid(pid_name), d, m).write_bytes(b"y")
    root.write_version(_version("v1", {"p/a": _entry("p/a", "d1", "m1")}))
    root.write_version(
        _version("v2", {"p/a": _entry("p/a", "d2", "m1"), "p/b": _entry("p/b", "d3", "m3")})
    )
    root.set_latest("v2")
    orphan = root.shards_dir_for("zzzz", "d9")
    orphan.mkdir()
    (orphan / "shard-0000.tar").write_bytes(b"o")
    assert set(V.gc_roots(root)) == {"v1", "v2"}
    planned = V.gc(root, dry_run=True)
    assert planned == [orphan] and orphan.exists()
    V.gc(root, dry_run=False)
    assert (
        not orphan.exists() and root.shards_dir_for(V_pid("p/a"), "d1").exists()
    )  # v1 still a root
    with pytest.raises(ValueError):
        V.prune_version(root, "v2")  # latest cannot be pruned
    V.prune_version(root, "v1")
    deleted = V.gc(root, dry_run=False)
    assert (
        root.shards_dir_for(V_pid("p/a"), "d1") in deleted
        and not root.shards_dir_for(V_pid("p/a"), "d1").exists()
    )
    assert root.shards_dir_for(V_pid("p/a"), "d2").exists()


def test_journal_phases(tmp_path):
    j = V.Journal(tmp_path / "builds" / "b1")
    assert j.phase is None
    j.advance("built")
    j.advance("verified")
    assert j.phase == "verified"
    assert json.loads((tmp_path / "builds/b1/journal.json").read_text())["phases"] == [
        "built",
        "verified",
    ]
    with pytest.raises(ValueError):
        j.advance("built")  # phases are monotonic


def test_gc_on_empty_directory(tmp_path):
    """gc() on a bare root should return [] without raising."""
    root = V.DatasetRoot(tmp_path)
    result = V.gc(root, dry_run=True)
    assert result == []
