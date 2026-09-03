import json

import pytest
from diffusion_planner.data_pipeline import pack_shards as CLI
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


def test_pack_rejects_workers_below_one(tmp_path, capsys):
    src, dst = tmp_path / "src", tmp_path / "dst"
    make_tree(src, LAYOUT[:1])
    common = ["--source", str(src), "--dest", str(dst), "--partition-depth", "4"]
    assert CLI.main(["pack", *common, "--base", "none", "--tag", "v1", "--workers", "0"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and "--workers" in err
    # rejected before pack() ever ran (no writer_lock, no dest scaffolding)
    assert not dst.exists()


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
