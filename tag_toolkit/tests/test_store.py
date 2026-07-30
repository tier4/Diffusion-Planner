"""Unit tests for tag_toolkit (no RDMA access)."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pytest

from tag_toolkit import (
    TagStore,
    expand_source,
    format_buckets,
    list_known_tags,
    read_tags,
    route_of,
)
from tag_toolkit.sidecar import write_tags

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "scripts"),
)
from write_site_split_tags import apply_path_tags, parse_site_split


def _frame(dir: Path, name: str, tags: list[str] | None = None) -> Path:
    dir.mkdir(parents=True, exist_ok=True)
    npz = dir / f"{name}.npz"
    npz.write_bytes(b"")
    data = {"timestamp": 1}
    if tags is not None:
        data["tags"] = tags
    (dir / f"{name}.json").write_text(json.dumps(data) + "\n")
    return npz


@pytest.fixture
def tiny(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    bag = root / "prd_jt" / "1423_shinagawa_odaiba" / "manual" / "2025-02-04" / "10-34-24" / "routes"
    bag.mkdir(parents=True)
    _frame(bag, "a_00000000_00000031", ["lateral:turn"])
    _frame(bag, "a_00000000_00000032", ["lateral:turn", "longitudinal:yield"])
    _frame(bag, "a_00000000_00000033", [])
    return root


@pytest.fixture
def two_routes(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    a = root / "prd_jt" / "1423_shinagawa_odaiba" / "manual" / "2025-02-04" / "10-34-24" / "routes"
    b = root / "erga" / "2416_odaiba" / "auto" / "2026-04-15" / "14-00-15" / "routes"
    _frame(a, "a_00000000_00000031", ["lateral:turn", "site:1423_shinagawa_odaiba", "split:manual"])
    _frame(a, "a_00000000_00000032", ["longitudinal:yield", "site:1423_shinagawa_odaiba", "split:manual"])
    _frame(b, "b_00000000_00000031", ["lateral:lane_keeping", "site:2416_odaiba", "split:auto"])
    return root


def test_expand_dir_and_path_list(tiny: Path, tmp_path: Path) -> None:
    paths = expand_source(tiny)
    assert len(paths) == 3
    listing = tmp_path / "list.json"
    listing.write_text(json.dumps([str(p) for p in paths]))
    listed = expand_source(listing)
    assert len(listed) == 3
    ghost = tmp_path / "ghost.json"
    ghost.write_text(json.dumps(["/nonexistent/a.npz", "/nonexistent/b.npz"]))
    assert [str(p) for p in expand_source(ghost)] == [
        "/nonexistent/a.npz",
        "/nonexistent/b.npz",
    ]


def test_path_list_preserves_order_no_resolve(tmp_path: Path) -> None:
    listing = tmp_path / "order.json"
    listing.write_text(json.dumps(["/z/c.npz", "/a/b.npz", "/z/c.npz"]))
    paths = expand_source(listing)
    assert [str(p) for p in paths] == ["/z/c.npz", "/a/b.npz"]


def test_route_of_strips_routes_dir(tiny: Path) -> None:
    npz = expand_source(tiny)[0]
    route = route_of(npz)
    assert route.name == "10-34-24"
    assert (route / "routes").is_dir()


def test_add_remove_query_group_by(tiny: Path) -> None:
    store = TagStore(tiny)
    # default granularity is route: one bag has turn frames
    assert len(store.query("lateral:turn")) == 1
    assert len(store.query("lateral:turn", granularity="frame")) == 2

    store.add_tags(None, ["split:manual", "site:1423_shinagawa_odaiba"])
    route = store.route_paths()[0]
    assert "split:manual" in store.tags_for(route)
    assert len(store.query("lateral:turn", granularity="frame")) == 2

    store.remove_tags(None, ["lateral:turn"])
    assert store.query("lateral:turn") == []
    assert store.query("lateral:turn", granularity="frame") == []

    buckets = store.group_by(["site", "split"])
    assert sum(b.count for b in buckets) == 1
    assert buckets[0].routes == 1
    assert buckets[0].frame_count == 3
    assert buckets[0].members == [route]
    assert buckets[0].values["site"] == "1423_shinagawa_odaiba"


def test_route_union_semantics(two_routes: Path) -> None:
    store = TagStore(two_routes)
    # turn is only on one frame of route A, but the route still carries it
    routes = store.query("lateral:turn")
    assert len(routes) == 1
    assert routes[0].name == "10-34-24"
    # route A union has both turn and yield even though no single frame has both
    assert len(store.query({"all": ["lateral:turn", "longitudinal:yield"]})) == 1
    assert (
        len(
            store.query(
                {"all": ["lateral:turn", "longitudinal:yield"]},
                granularity="frame",
            )
        )
        == 0
    )

    buckets = store.group_by(["site", "split"], members=True)
    assert len(buckets) == 2
    assert sum(b.count for b in buckets) == 2
    assert sum(b.frame_count or 0 for b in buckets) == 3


def test_format_buckets_uses_count_header(two_routes: Path) -> None:
    store = TagStore(two_routes)
    text = format_buckets(store.group_by(["site", "split"]), ["site", "split"])
    assert "count" in text.splitlines()[0]
    assert "frames" not in text.splitlines()[0]


def test_replace_dimension(tiny: Path) -> None:
    store = TagStore(tiny)
    store.replace_tags(None, dimension="lateral", values=["lane_keeping"])
    for path in store.npz_paths():
        assert store.tag_values(path, "lateral") == ["lane_keeping"]


def test_set_replace_mode(tiny: Path) -> None:
    store = TagStore(tiny)
    store.set_tags(None, ["site:x"], mode="replace")
    for path in store.npz_paths():
        assert store.tags_for(path) == ["site:x"]


def test_clause_all_any_not(tiny: Path) -> None:
    store = TagStore(tiny)
    # frame: only one npz has both
    assert len(store.query({"all": ["lateral:turn", "longitudinal:yield"]}, granularity="frame")) == 1
    assert len(store.query({"any": ["lateral:turn", "longitudinal:yield"]}, granularity="frame")) == 2
    assert len(store.query({"not": "lateral:turn"}, granularity="frame")) == 1
    # route: union has both; not turn matches nothing
    assert len(store.query({"all": ["lateral:turn", "longitudinal:yield"]})) == 1
    assert len(store.query({"not": "lateral:turn"})) == 0


def test_parse_site_split_and_apply(tiny: Path) -> None:
    paths = expand_source(tiny)
    site, split = parse_site_split(paths[0])
    assert site == "1423_shinagawa_odaiba"
    assert split == "manual"
    n = apply_path_tags(tiny)
    assert n == 3
    for path in paths:
        tags = read_tags(path)
        assert "site:1423_shinagawa_odaiba" in tags
        assert "split:manual" in tags
    assert "lateral:turn" in read_tags(paths[0])


def test_generate_from_labeled_layout_is_route_standard(tmp_path: Path) -> None:
    route_dir = (
        tmp_path
        / "data"
        / "prd_jt"
        / "1423_shinagawa_odaiba"
        / "manual"
        / "2025-02-04"
        / "10-34-24"
    )
    npz = _frame(route_dir / "routes", "frame")
    assert route_of(npz) == route_dir
    assert route_of(route_dir) == route_dir
    assert parse_site_split(npz) == ("1423_shinagawa_odaiba", "manual")


def test_split_labels_refine_generate_from_labeled_manual(tmp_path: Path) -> None:
    route_dir = (
        tmp_path
        / "data"
        / "prd_jt"
        / "1423_shinagawa_odaiba"
        / "manual"
        / "2025-02-04"
        / "10-34-24"
    )
    npz = _frame(route_dir / "routes", "frame", ["lateral:turn"])
    n = apply_path_tags(
        route_dir,
        split_labels={"prd_jt/1423_shinagawa_odaiba/2025-02-04/10-34-24": "train"},
    )
    assert n == 1
    assert read_tags(npz) == [
        "lateral:turn",
        "site:1423_shinagawa_odaiba",
        "split:train",
    ]


def test_parse_unknown_layout(tmp_path: Path) -> None:
    npz = _frame(tmp_path / "weird", "frame")
    assert parse_site_split(npz) == ("unknown", "unknown")


def test_taxonomy_helper_warns() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tags = list_known_tags()
    assert any("may not match" in str(w.message) for w in caught)
    assert "split:manual" in tags
    assert "site:unknown" in tags


def test_missing_tags_field(tmp_path: Path) -> None:
    npz = _frame(tmp_path, "x")
    assert read_tags(npz) == []
    write_tags(npz, ["split:auto"])
    assert read_tags(npz) == ["split:auto"]


def test_tags_for_route_without_source(tmp_path: Path) -> None:
    route_dir = (
        tmp_path
        / "data"
        / "prd_jt"
        / "1423_shinagawa_odaiba"
        / "manual"
        / "2025-02-04"
        / "10-34-24"
    )
    _frame(route_dir / "routes", "a", ["lateral:turn"])
    _frame(route_dir / "routes", "b", ["longitudinal:yield"])
    store = TagStore()
    tags = store.tags_for(route_dir)
    assert tags == ["lateral:turn", "longitudinal:yield"]


def test_partial_path_list_is_source_relative_union(tmp_path: Path) -> None:
    route_dir = (
        tmp_path
        / "data"
        / "prd_jt"
        / "1423_shinagawa_odaiba"
        / "manual"
        / "2025-02-04"
        / "10-34-24"
    )
    a = _frame(route_dir / "routes", "a", ["lateral:turn"])
    _frame(route_dir / "routes", "b", ["longitudinal:yield"])
    listing = tmp_path / "partial.json"
    listing.write_text(json.dumps([str(a)]))
    store = TagStore(listing)
    assert store.tags_for(route_dir) == ["lateral:turn"]
    assert store.query("longitudinal:yield") == []
    # Without a source, scan the whole route directory.
    assert TagStore().tags_for(route_dir) == ["lateral:turn", "longitudinal:yield"]


def test_group_by_multi_value_total_is_unique(tmp_path: Path) -> None:
    root = tmp_path / "data"
    bag = root / "prd_jt" / "1423_shinagawa_odaiba" / "manual" / "2025-02-04" / "10-34-24" / "routes"
    _frame(bag, "a", ["site:alpha", "site:beta", "lateral:turn"])
    store = TagStore(root)
    buckets = store.group_by(["site"])
    assert len(buckets) == 2
    assert sum(b.count for b in buckets) == 2
    text = format_buckets(buckets, ["site"])
    assert text.splitlines()[-1].split()[-1] == "1"


def test_invalid_sidecar_tag_skipped_with_warning(tmp_path: Path) -> None:
    npz = _frame(tmp_path / "bag" / "routes", "x", ["lateral:turn", "BAD TAG"])
    store = TagStore(tmp_path / "bag")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tags = store.tags_for(npz)
    assert tags == ["lateral:turn"]
    assert any("BAD TAG" in str(w.message) for w in caught)


def test_mutate_preflight_missing_sidecar(tmp_path: Path) -> None:
    bag = tmp_path / "bag" / "routes"
    bag.mkdir(parents=True)
    npz = bag / "orphan.npz"
    npz.write_bytes(b"")
    store = TagStore(tmp_path / "bag")
    with pytest.raises(FileNotFoundError, match="missing sidecar"):
        store.add_tags(None, ["lateral:turn"])


def test_flat_layout_route_of(tmp_path: Path) -> None:
    route_dir = tmp_path / "bag_time"
    npz = _frame(route_dir, "frame", ["split:auto"])
    assert route_of(npz) == route_dir
    assert TagStore().tags_for(route_dir) == ["split:auto"]
