import pytest
from diffusion_planner.data_pipeline import partition as P
from diffusion_planner.data_pipeline.errors import PlanError
from tests.dp_fixtures import make_tree

LAYOUT = [
    ("projA/mapX/manual/2026-01-01/t1/route_0", 3, "full"),
    ("projA/mapX/manual/2026-01-02/t1/route_0", 2, "full"),
    ("projA/mapX/auto/2026-01-03/t1/route_0", 2, "full"),
    ("psim/loc_seed_1/manual/seed_1_x/bag_0/route_0", 2, "psim"),
]


def test_rule_requires_exactly_one():
    with pytest.raises(ValueError):
        P.PartitionRule()
    with pytest.raises(ValueError):
        P.PartitionRule(depth=2, regex="x")
    assert P.PartitionRule(depth=4).partition_of("a/b/c/d/e/f_00000001") == "a/b/c/d"
    assert P.PartitionRule(regex=r"^(?P<partition>[^/]+/[^/]+)").partition_of("a/b/c/x") == "a/b"
    with pytest.raises(PlanError):
        P.PartitionRule(regex=r"^zzz(?P<partition>.*)").partition_of("a/b/c")
    assert P.PartitionRule(depth=4).rule_hash != P.PartitionRule(depth=3).rule_hash


def test_pid_is_artifact_safe_and_stable():
    pid = P.pid_of("projA/mapX/manual/2026-01-01")
    assert len(pid) == 16 and pid.isalnum() and pid == pid.lower()
    assert pid == P.pid_of("projA/mapX/manual/2026-01-01") != P.pid_of("other")


def test_discover_groups_pairs_and_filters(tmp_path):
    make_tree(tmp_path, LAYOUT)
    (tmp_path / "projA/mapX/auto/2026-01-03/t1/control_mode_4_intervals.json").write_text("{}")
    parts = P.discover(tmp_path, P.PartitionRule(depth=4), exclude=["*/auto/*"])
    assert set(parts) == {
        "projA/mapX/manual/2026-01-01",
        "projA/mapX/manual/2026-01-02",
        "psim/loc_seed_1/manual/seed_1_x",
    }
    s = parts["projA/mapX/manual/2026-01-01"]
    assert [x.key for x in s] == sorted(x.key for x in s) and len(s) == 3
    assert (
        s[0].sidecar_path is not None and s[0].rel_dir == "projA/mapX/manual/2026-01-01/t1/route_0"
    )
    only = P.discover(tmp_path, P.PartitionRule(depth=4), include=["psim/*"])
    assert set(only) == {"psim/loc_seed_1/manual/seed_1_x"}


def test_discover_path_list_validation(tmp_path):
    keys = make_tree(tmp_path, LAYOUT[:1])
    lst = [str(tmp_path / f"{k}.npz") for k in keys]
    parts = P.discover(tmp_path, P.PartitionRule(depth=4), path_list=lst)
    assert sum(len(v) for v in parts.values()) == 3
    with pytest.raises(ValueError):
        P.discover(tmp_path, P.PartitionRule(depth=4), path_list=lst + [lst[0]])
    with pytest.raises(FileNotFoundError):
        P.discover(tmp_path, P.PartitionRule(depth=4), path_list=[str(tmp_path / "nope.npz")])
    with pytest.raises(ValueError):
        P.discover(tmp_path, P.PartitionRule(depth=4), path_list=["/etc/passwd.npz"])


def test_fingerprint_is_order_independent_and_content_sensitive():
    a = [("k1", b"\x01" * 32, b"\x02" * 32), ("k2", b"\x03" * 32, None)]
    assert P.fingerprint(a) == P.fingerprint(list(reversed(a)))
    assert P.fingerprint(a) != P.fingerprint([("k1", b"\x01" * 32, b"\x09" * 32), a[1]])


def test_inspect_report(tmp_path):
    make_tree(tmp_path, LAYOUT)
    rep = P.inspect_tree(tmp_path, P.PartitionRule(depth=4), [], [])
    assert rep.n_npz == 9 and rep.npz_depth_histogram == {7: 9}
    assert rep.partitions["projA/mapX/manual/2026-01-01"] == 3
    assert "<DATE>" in rep.dir_name_patterns[4]
    assert "npz" in rep.render() and "partitions" in rep.render()
