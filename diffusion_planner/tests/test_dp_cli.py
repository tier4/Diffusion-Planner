import json

import pytest
from diffusion_planner.data_pipeline import pack_shards as CLI
from diffusion_planner.data_pipeline import tar_shards as T
from diffusion_planner.data_pipeline.partition import pid_of
from diffusion_planner.data_pipeline.versioning import DatasetRoot
from tests.dp_fixtures import make_tree

LAYOUT = [
    ("pA/mX/manual/2026-01-01/t1/r", 6, "full"),
    ("pA/mX/auto/2026-01-02/t1/r", 2, "full"),
    ("pB/mY/manual/2026-01-03/t1/r", 3, "psim"),
]


def test_inspect_prints_report_and_requires_rule(tmp_path, capsys):
    make_tree(tmp_path / "src", LAYOUT)
    with pytest.raises(SystemExit) as e:
        CLI.main(["inspect", "--source", str(tmp_path / "src")])
    assert e.value.code == 2  # rule is required, no default
    assert (
        CLI.main(
            [
                "inspect",
                "--source",
                str(tmp_path / "src"),
                "--partition-depth",
                "4",
                "--exclude",
                "*/auto/*",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert (
        "npz files: 9" in out
        and "pA/mX/manual/2026-01-01: 6" in out
        and "auto" not in out.split("partitions")[1]
    )


def test_pack_remove_gc_scrub_export_keyset(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    make_tree(src, LAYOUT)
    common = [
        "--source",
        str(src),
        "--dest",
        str(dst),
        "--partition-depth",
        "4",
        "--exclude",
        "*/auto/*",
    ]
    assert (
        CLI.main(["pack", *common, "--base", "none", "--tag", "v1", "--shard-size-gb", "0.0001"])
        == 0
    )
    root = DatasetRoot(dst)
    assert root.latest() == "v1" and set(root.read_version("v1").partitions) == {
        "pA/mX/manual/2026-01-01",
        "pB/mY/manual/2026-01-03",
    }
    assert (
        CLI.main(["pack", *common, "--base", "none", "--tag", "v1", "--shard-size-gb", "0.0001"])
        == 0
    )  # identical recipe → idempotent
    assert (
        CLI.main(
            [
                "remove",
                "--dest",
                str(dst),
                "--base",
                "v1",
                "--tag",
                "v2",
                "--partition",
                "pB/mY/manual/2026-01-03",
            ]
        )
        == 0
    )
    assert CLI.main(["scrub", "--dest", str(dst), "--tag", "v2"]) == 0
    assert (
        CLI.main(
            [
                "keyset",
                "--dest",
                str(dst),
                "--tag",
                "v2",
                "--where",
                "is_skipped IS NOT TRUE",
                "--out",
                str(tmp_path / "ks.parquet"),
            ]
        )
        == 0
    )
    assert (tmp_path / "ks.parquet").exists()
    assert (
        CLI.main(
            [
                "export",
                "--dest",
                str(dst),
                "--tag",
                "v2",
                "--where",
                "project_id = 'projA'",
                "--out",
                str(tmp_path / "exp"),
            ]
        )
        == 0
    )
    assert json.loads((tmp_path / "exp/export_manifest.json").read_text())["n"] == 6
    assert CLI.main(["gc", "--dest", str(dst), "--dry-run"]) == 0
    assert (
        CLI.main(["prune-version", "--dest", str(dst), "--tag", "v2"]) == 1
    )  # latest cannot be pruned
    assert CLI.main(["prune-version", "--dest", str(dst), "--tag", "v1"]) == 0
    assert CLI.main(["gc", "--dest", str(dst)]) == 0
    assert not root.shards_dir_for(
        root.read_version("v2").partitions["pA/mX/manual/2026-01-01"].pid, "x"
    ).exists()
    # pB's revision was only referenced by v1 → gone after prune+gc
    assert not any(
        p.name.startswith(pid_of("pB/mY/manual/2026-01-03")) for p in root.shards_dir.iterdir()
    )


def test_pack_refuses_existing_tag_with_different_content(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    make_tree(src, LAYOUT[:1])
    common = ["--source", str(src), "--dest", str(dst), "--partition-depth", "4"]
    assert CLI.main(["pack", *common, "--base", "none", "--tag", "v1"]) == 0
    assert (
        CLI.main(["pack", *common, "--base", "none", "--tag", "v1", "--seed", "7"]) == 1
    )  # different shuffle → different content


def test_pack_source_namespace_override_allows_relocated_dataset_root(tmp_path, monkeypatch):
    """A dataset root copied to another machine can no longer be extended incrementally: the
    base revision recorded the original absolute --source path as its namespace, and the
    resolved --source on the new machine will never match it. --source-namespace lets the
    caller supply the base's recorded value explicitly so the mismatch guard can be satisfied
    without --replace-all (which would require every source file to still be present).

    data_rev/meta_rev/shards alone cannot prove the base's partition was *reused* rather than
    deterministically rebuilt to an identical revision (src2 has the same content as src1, so
    a full rebuild would land on the exact same tuple). Capturing (inode, mtime_ns) of the
    published shard/manifest files doesn't prove it either: `_publish` never overwrites an
    already-published shard directory that verifies byte-identical, it just discards the
    redundant rebuild — so a full rebuild that reproduces the base's exact revision leaves the
    same (inode, mtime_ns) behind as a genuine reuse would (verified empirically against this
    file's own `_build_partition` before writing this test: disabling the reuse short-circuit
    entirely still left the published files' stat untouched). The only thing that actually
    distinguishes reuse from a deterministic rebuild is whether pass 2 of `_build_partition`
    (the npz-decode-and-shard-write loop) ran at all — so this test spies on
    `tar_shards.ShardWriter`, the object that loop instantiates once per partition it builds,
    and asserts it is never constructed during the incremental pack for the (single, fully
    reused) partition under test. --workers defaults to 1, which runs `_build_partition`
    serially in this process (see packer._run_builds), so the spy sees every call.

    The spy is installed before the very FIRST pack (not just the reuse pack at the end) so
    that this test has a positive control: that first pack has no base to reuse from, so it
    MUST build the partition, and the assertion right after it proves the spy actually fires.
    Without that, an accidental change to how `packer.py` imports `ShardWriter` (e.g. switching
    to `from .tar_shards import ShardWriter`, which `monkeypatch.setattr(T, ...)` would no
    longer intercept) would make `shard_writer_calls == []` trivially true for the wrong
    reason, and this test would pass even though it had stopped testing anything.
    """
    src1, src2, dst = tmp_path / "src1", tmp_path / "src2", tmp_path / "dst"
    make_tree(src1, LAYOUT[:1])

    shard_writer_calls = []
    real_shard_writer = T.ShardWriter

    def _spy_shard_writer(*args, **kwargs):
        shard_writer_calls.append(args)
        return real_shard_writer(*args, **kwargs)

    monkeypatch.setattr(T, "ShardWriter", _spy_shard_writer)

    common1 = ["--source", str(src1), "--dest", str(dst), "--partition-depth", "4"]
    assert CLI.main(["pack", *common1, "--base", "none", "--tag", "v1"]) == 0
    # Positive control: nothing exists to reuse yet, so this pack must actually build the
    # partition — proving the spy fires at all before the negative assertion below relies on
    # it staying silent.
    assert len(shard_writer_calls) == 1
    shard_writer_calls.clear()

    root = DatasetRoot(dst)
    v1 = root.read_version("v1")
    recorded_namespace = v1.source_namespace
    assert recorded_namespace == str(src1.resolve())

    # Same content, different source directory — stands in for "the dataset root's original
    # source tree is no longer at the path recorded in the base revision".
    make_tree(src2, LAYOUT[:1])
    common2 = ["--source", str(src2), "--dest", str(dst), "--partition-depth", "4"]
    assert CLI.main(["pack", *common2, "--base", "v1", "--tag", "v2"]) == 1
    assert DatasetRoot(dst).latest() == "v1"  # rejected pack must not publish v2
    shard_writer_calls.clear()  # the rejected pack is expected to have built nothing either

    assert (
        CLI.main(
            [
                "pack",
                *common2,
                "--base",
                "v1",
                "--tag",
                "v2",
                "--source-namespace",
                recorded_namespace,
            ]
        )
        == 0
    )
    # Pass 2 (npz decode + tar-shard write) never ran: the base's partition was reused outright,
    # not rebuilt to a matching revision.
    assert shard_writer_calls == []
    v2 = DatasetRoot(dst).read_version("v2")
    assert v2.source_namespace == recorded_namespace
    assert set(v2.partitions) == set(v1.partitions)
    e1 = v1.partitions["pA/mX/manual/2026-01-01"]
    e2 = v2.partitions["pA/mX/manual/2026-01-01"]
    assert (e2.data_rev, e2.meta_rev, tuple(e2.shards)) == (
        e1.data_rev,
        e1.meta_rev,
        tuple(e1.shards),
    )


def test_pack_rejects_workers_below_one(tmp_path, capsys):
    src, dst = tmp_path / "src", tmp_path / "dst"
    make_tree(src, LAYOUT[:1])
    common = ["--source", str(src), "--dest", str(dst), "--partition-depth", "4"]
    assert CLI.main(["pack", *common, "--base", "none", "--tag", "v1", "--workers", "0"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and "--workers" in err
    # rejected before pack() ever ran (no writer_lock, no dest scaffolding)
    assert not dst.exists()


def test_pack_quiet_suppresses_progress(tmp_path, capsys):
    """Minor: progress was on by default with no opt-out; --quiet sets progress=False."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    make_tree(src, LAYOUT[:1])
    common = ["--source", str(src), "--dest", str(dst), "--partition-depth", "4"]
    assert CLI.main(["pack", *common, "--base", "none", "--tag", "v1", "--quiet"]) == 0
    err = capsys.readouterr().err
    assert "pack:" not in err


def test_keyset_empty_where_error(tmp_path, capsys):
    src, dst = tmp_path / "src", tmp_path / "dst"
    make_tree(src, LAYOUT[:1])
    common = ["--source", str(src), "--dest", str(dst), "--partition-depth", "4"]
    CLI.main(["pack", *common, "--base", "none", "--tag", "v1"])
    # `pack` reports progress on stderr by default (PackOptions.progress=True); drain it
    # here so the assertion below is about the `keyset` command's own stderr only.
    capsys.readouterr()
    assert (
        CLI.main(
            [
                "keyset",
                "--dest",
                str(dst),
                "--tag",
                "v1",
                "--where",
                "",
                "--out",
                str(tmp_path / "ks.parquet"),
            ]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert err.startswith("error:")


def test_keyset_reserved_column_where_error(tmp_path, capsys):
    """Finding #7: pack_shards keyset --where 'offset > 0' -> exit 1 with 'error:' on stderr."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    make_tree(src, LAYOUT[:1])
    common = ["--source", str(src), "--dest", str(dst), "--partition-depth", "4"]
    CLI.main(["pack", *common, "--base", "none", "--tag", "v1"])
    # `pack` reports progress on stderr by default (PackOptions.progress=True); drain it
    # here so the assertion below is about the `keyset` command's own stderr only.
    capsys.readouterr()
    assert (
        CLI.main(
            [
                "keyset",
                "--dest",
                str(dst),
                "--tag",
                "v1",
                "--where",
                "offset > 0",
                "--out",
                str(tmp_path / "ks.parquet"),
            ]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert err.startswith("error:")
