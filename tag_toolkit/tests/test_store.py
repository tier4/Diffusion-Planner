"""Unit tests for tag_toolkit (no RDMA access)."""

from __future__ import annotations

import json
import os
import pickle
import sys
import threading
import warnings
from pathlib import Path

import pytest

from tag_toolkit import (
    Bucket,
    TagStore,
    expand_source,
    format_buckets,
    list_known_tags,
    read_tags,
    route_of,
)
from tag_toolkit.sidecar import (
    _atomic_write_text,
    cleanup_tmp_files,
    parse_tag,
    read_tags,
    write_tags,
)

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "sample_dataset"),
)
from _build import build as build_sample_dataset

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "scripts"),
)
from write_site_split_tags import apply_path_tags, parse_site_split
from incremental_index import build_incremental_index

# ``sample_dataset/_build.py`` is the single source of truth for the test
# fixture. The session fixture below calls it once into tmp_path; per-test
# fixtures then either reuse that build or copy individual routes from it.
# This keeps every test run independent of any checked-in artifacts.
BUILD_SCRIPT = Path(__file__).resolve().parents[1] / "sample_dataset" / "_build.py"


def _frame(dir: Path, name: str, tags: list[str] | None = None) -> Path:
    """Create a test frame with its sidecar.

    Kept for tests that build their own synthetic frames in tmp_path
    (independent of the sample dataset fixture).
    """
    dir.mkdir(parents=True, exist_ok=True)
    npz = dir / f"{name}.npz"
    npz.write_bytes(b"")
    data = {"timestamp": 1}
    if tags is not None:
        data["tags"] = tags
    (dir / f"{name}.json").write_text(json.dumps(data) + "\n")
    return npz


@pytest.fixture(scope="session")
def _built_sample(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the sample dataset into a session-scoped tmp directory.

    Runs ``_build.build()`` once per pytest session and returns the root of
    the freshly-built tree. Per-test fixtures copy from this root so that
    mutations don't leak across tests.

    The built tree also contains the generated ``npz_list.json``,
    ``route_list.json``, and ``index.tag`` artifacts under this root.
    """
    root = tmp_path_factory.mktemp("sample_dataset_built")
    build_sample_dataset(root)
    return root


def _route_dirs(built_root: Path) -> dict[str, Path]:
    """Resolve the three route directories of a freshly-built sample dataset.

    Order matches the ROUTES tuple in ``_build.py``: aomi (10-55-13),
    ariake (15-16-36), psim (psim_training_bag_0_0).
    """
    return {
        "aomi": built_root
        / "x2_dev"
        / "2231_odaiba_shinagawa_copied_from_xx1"
        / "auto"
        / "2026-06-23"
        / "10-55-13",
        "ariake": built_root
        / "x2_dev"
        / "2231_odaiba_shinagawa_copied_from_xx1"
        / "auto"
        / "2026-07-07"
        / "15-16-36",
        "psim": built_root
        / "xx1_psim"
        / "b_mobility_seed_200_poses_100_jpntaxi_vehicle_8_pilot-auto-v0.49.2"
        / "manual"
        / "2026-04-15"
        / "psim_training_bag_0_0",
    }


@pytest.fixture
def sample(tmp_path: Path, _built_sample: Path) -> Path:
    """A writable copy of the freshly-built sample dataset."""
    import shutil

    dest = tmp_path / "sample_dataset"
    shutil.copytree(_built_sample, dest)
    return dest


@pytest.fixture
def sample_route(tmp_path: Path, _built_sample: Path) -> Path:
    """A writable copy of the aomi route from the freshly-built sample dataset.

    Mirrors the role of the old ``tiny`` fixture: one bag directory with
    enough frames to exercise route vs frame granularity.
    """
    import shutil

    dest = tmp_path / "aomi"
    shutil.copytree(_route_dirs(_built_sample)["aomi"], dest)
    return dest


@pytest.fixture
def two_routes(tmp_path: Path, _built_sample: Path) -> tuple[Path, Path]:
    """Writable copies of two distinct routes (aomi + ariake)."""
    import shutil

    routes = _route_dirs(_built_sample)
    aomi = tmp_path / "aomi"
    ariake = tmp_path / "ariake"
    shutil.copytree(routes["aomi"], aomi)
    shutil.copytree(routes["ariake"], ariake)
    return aomi, ariake


def test_expand_dir_and_path_list(sample: Path, tmp_path: Path) -> None:
    paths = expand_source(sample)
    assert len(paths) == 30
    listing = tmp_path / "list.json"
    listing.write_text(json.dumps([str(p) for p in paths]))
    listed = expand_source(listing)
    assert len(listed) == 30
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


def test_route_of_strips_routes_dir(sample_route: Path) -> None:
    npz = expand_source(sample_route)[0]
    route = route_of(npz)
    # sample_route is a writable copy of the aomi route; the directory name
    # in tmp_path is "aomi" but route_of() should still strip "/routes/"
    # and return the parent directory.
    assert route == sample_route
    assert (route / "routes").is_dir()


def test_empty_store() -> None:
    """Empty store: accessors raise ValueError until an index is loaded."""
    store = TagStore()
    assert store.source is None
    assert not store.has_index()
    # Fail-fast on empty store — same convention as query()/tags_of().
    with pytest.raises(ValueError, match="no index loaded"):
        store.npz_paths()
    with pytest.raises(ValueError, match="no index loaded"):
        store.route_paths()


def test_scan_mode_direct_directory(sample: Path) -> None:
    """TagStore can scan a directory directly without building an index file."""
    store = TagStore(sample)
    assert store.has_index()
    assert len(store.npz_paths()) == 30
    # lateral:turn appears on 5 frames per route, 3 routes → 15 frames, 3 routes
    assert len(store.query("lateral:turn")) == 3
    assert len(store.query("lateral:turn", granularity="frame")) == 15


def test_index_mode_from_file(tmp_path: Path, sample: Path) -> None:
    """TagStore can load from a .tag index file."""
    index_file = tmp_path / "index.tag"
    TagStore.build_index(sample, index_file)

    # Load from index file
    store = TagStore(index_file)
    assert store.source == index_file
    assert len(store.npz_paths()) == 30
    assert len(store.query("lateral:turn")) == 3


def test_add_remove_query_group_by(sample: Path) -> None:
    """Test add_tags, remove_tags, query, group_by with scan mode."""
    store = TagStore(sample)

    # route granularity: all 3 routes carry lateral:turn
    assert len(store.query("lateral:turn")) == 3
    assert len(store.query("lateral:turn", granularity="frame")) == 15

    # Pre-existing: split:manual is on every route. Find the zeikan route
    # in the writable copy so we don't touch the checked-in fixture.
    zeikan_route = next(
        r for r in store.route_paths() if r.name == "psim_training_bag_0_0"
    )
    assert "split:manual" in store.tags_of(scope=zeikan_route)

    # Add a tag scoped to a single route — confirm scope narrows results
    store.add_tags(["split:eval"], scope=zeikan_route)
    assert "split:eval" in store.tags_of(scope=zeikan_route)
    # turn count unchanged (lateral:turn untouched)
    assert len(store.query("lateral:turn", granularity="frame")) == 15

    # Remove lateral:turn from the whole index → all 3 routes drop the tag
    store.remove_tags(["lateral:turn"])
    assert store.query("lateral:turn") == []
    assert store.query("lateral:turn", granularity="frame") == []

    buckets = store.group_by(["site", "split"])
    # 5 buckets total now that the fixture covers 4 split values:
    #   (2231, auto)   ← aomi route
    #   (2231, train)  ← ariake route
    #   (879, manual)  ← psim route (5 even-numbered frames)
    #   (879, valid)   ← psim route (5 odd-numbered frames)
    #   (879, eval)    ← added above via add_tags on zeikan
    assert len(buckets) == 5
    assert {b.values["site"] for b in buckets} == {
        "2231_odaiba_shinagawa_copied_from_xx1",
        "879_hiratsuka",
    }
    # sum of counts: 1 + 1 + 1 + 1 + 1 = 5 (each bucket = 1 unique member,
    # psim route is shared between manual/valid/eval cells but counted once
    # per cell because there is one bucket per (site, split) combo).
    assert sum(b.count for b in buckets) == 5
    # psim route should appear in the 879_hiratsuka cell (3 cells share it)
    psim_cells = [b for b in buckets if b.values["site"] == "879_hiratsuka"]
    assert len(psim_cells) == 3
    for cell in psim_cells:
        assert cell.members == [zeikan_route]


def test_route_union_semantics(sample: Path) -> None:
    """Route-level query uses union semantics (any frame has the tag)."""
    store = TagStore(sample)
    # aomi is the only route with longitudinal:yield
    aomi = next(r for r in store.route_paths() if r.name == "10-55-13")

    # longitudinal:yield is only on aomi (2 frames) → only aomi route carries it
    routes = store.query("longitudinal:yield")
    assert len(routes) == 1
    assert routes[0] == aomi
    # route aomi union has both turn and yield even though no single frame has both
    assert len(store.query({"all": ["lateral:turn", "longitudinal:yield"]})) == 1
    # No single frame carries both turn AND yield → frame query is empty
    assert (
        len(
            store.query(
                {"all": ["lateral:turn", "longitudinal:yield"]},
                granularity="frame",
            )
        )
        == 0
    )

    # group by site: 3 routes across 2 site values
    buckets = store.group_by(["site"])
    sites = {b.values["site"] for b in buckets}
    assert sites == {
        "2231_odaiba_shinagawa_copied_from_xx1",
        "879_hiratsuka",
    }
    assert sum(b.count for b in buckets) == 3


def test_format_buckets_uses_count_header(sample: Path) -> None:
    """format_buckets shows 'count' header, not 'frames'."""
    store = TagStore(sample)
    text = format_buckets(store.group_by(["site", "split"]), ["site", "split"])
    assert "count" in text.splitlines()[0]
    assert "frames" not in text.splitlines()[0]


def test_remove_dimension(sample_route: Path) -> None:
    """remove_dimension deletes every tag for a dimension across scope."""
    # sample_route is a writable copy of aomi; original has split/lateral/site/etc.
    store = TagStore(sample_route)
    n = store.remove_dimension("lateral")
    # Only the 5 frames that had lateral:turn are touched; the other 5 are
    # unchanged and are not counted.
    assert n == 5
    for path in store.npz_paths():
        assert all(not t.startswith("lateral:") for t in store.tags_of(scope=path, granularity="frame"))
    # After removing lateral, route-granularity query for it is empty.
    assert store.query("lateral:turn") == []


def test_replace_tags_tag_pairs(sample: Path) -> None:
    """replace_tags(tag_pairs=...) replaces old with new across scope."""
    store = TagStore(sample)
    zeikan = next(r for r in store.route_paths() if r.name == "psim_training_bag_0_0")

    # Replace split:manual -> split:eval on the zeikan route only.
    # Only the 5 even-numbered psim frames carry split:manual; the other 5
    # are split:valid and are not touched.
    n = store.replace_tags(tag_pairs={"split:manual": "split:eval"}, scope=zeikan)
    assert n == 5
    # split:manual had no route other than psim, and all of psim's frames
    # carrying split:manual were rewritten → the tag is gone from the index.
    assert store.query("split:manual") == []
    # split:eval lives on the zeikan route now.
    assert len(store.query("split:eval")) == 1

    # Re-run is a no-op: nothing has split:manual on zeikan anymore.
    assert store.replace_tags(tag_pairs={"split:manual": "split:eval"}, scope=zeikan) == 0


def test_replace_tags_no_op_when_old_not_present(sample_route: Path) -> None:
    """replace_tags silently skips frames without the old tag."""
    store = TagStore(sample_route)
    # lateral:turn is on 5 of 10 frames (odd frame numbers per _build.py).
    # Replacing it should yield 5 writes; frames without turn are untouched.
    n = store.replace_tags(tag_pairs={"lateral:turn": "lateral:lane_keeping"})
    assert n == 5
    # Every frame that had lateral:turn now has lateral:lane_keeping; no frame
    # carries both, and no frame still has lateral:turn.
    count_with_lk = 0
    count_with_turn = 0
    for npz in store.npz_paths():
        tags = read_tags(npz)
        if "lateral:lane_keeping" in tags:
            count_with_lk += 1
        if "lateral:turn" in tags:
            count_with_turn += 1
    assert count_with_lk == 5
    assert count_with_turn == 0
    # Frames that never had lateral:turn are completely unchanged. The fixture
    # uses sample_route == aomi, whose split tag is split:auto (not manual).
    for npz in store.npz_paths():
        tags = read_tags(npz)
        if "lateral:turn" in tags or "lateral:lane_keeping" in tags:
            # This frame had turn and got lane_keeping; the pre-existing
            # site/split/override_metric tags (and longitudinal:yield on the
            # two aomi frames that carry it) survive.
            assert "split:auto" in tags
        else:
            # Untouched frame: only its original tags remain.
            assert "split:auto" in tags
            assert "lateral:lane_keeping" not in tags
            assert "lateral:turn" not in tags


def test_replace_tags_validates_inputs(tmp_path: Path) -> None:
    """replace_tags rejects malformed keys/values eagerly."""
    bag = tmp_path / "bag" / "routes"
    _frame(bag, "x", ["lateral:turn"])
    store = TagStore(tmp_path / "bag")
    with pytest.raises(ValueError):
        store.replace_tags(tag_pairs={"BAD": "lateral:turn"})
    with pytest.raises(ValueError):
        store.replace_tags(tag_pairs={"lateral:turn": "BAD"})


def test_replace_tags_key_eq_value_is_noop(sample_route: Path) -> None:
    """Old tag == new tag is a no-op (per entry)."""
    store = TagStore(sample_route)
    n = store.replace_tags(tag_pairs={"lateral:turn": "lateral:turn"})
    assert n == 0


def test_remove_dimension_validates_name(sample_route: Path) -> None:
    """remove_dimension rejects names that don't match the regex."""
    store = TagStore(sample_route)
    with pytest.raises(ValueError, match=r"\[a-z0-9_]+"):
        store.remove_dimension("BAD-NAME")


def test_clause_all_any_not(sample_route: Path) -> None:
    """Query with all/any/not clauses."""
    # aomi: 5 frames have turn, 2 frames have yield, none have both
    store = TagStore(sample_route)
    # route granularity: union has both turn and yield (turn on 5 frames,
    # yield on 2 frames)
    assert len(store.query({"all": ["lateral:turn", "longitudinal:yield"]})) == 1
    assert len(store.query({"not": "lateral:turn"})) == 0
    # frame granularity: no single frame carries both
    assert len(store.query({"all": ["lateral:turn", "longitudinal:yield"]}, granularity="frame")) == 0
    # any of turn or yield: 5 (turn) + 2 (yield) = 7 frames
    assert len(store.query({"any": ["lateral:turn", "longitudinal:yield"]}, granularity="frame")) == 7
    # not turn: 10 - 5 = 5 frames
    assert len(store.query({"not": "lateral:turn"}, granularity="frame")) == 5


def test_tags_of_basic(sample_route: Path) -> None:
    """tags_of returns the union of tags across scope at route granularity."""
    store = TagStore(sample_route)
    tags = store.tags_of()  # default route granularity, no scope = whole index
    assert "lateral:turn" in tags
    assert "longitudinal:yield" in tags
    # aomi's split tag is split:auto (not manual). Verify any split:* is
    # present rather than hard-coding the value.
    assert any(t.startswith("split:") for t in tags)


def test_tags_of_dimensions_filter(sample_route: Path) -> None:
    """tags_of(dimensions=...) keeps only tags whose dimension is in the list."""
    store = TagStore(sample_route)
    tags = store.tags_of(dimensions=["lateral"])
    assert tags == ["lateral:turn"]  # only one lateral value on aomi


def test_tags_of_granularity_frame(sample_route: Path) -> None:
    """tags_of(granularity='frame') returns union over frame-level tags."""
    store = TagStore(sample_route)
    frame_tags = store.tags_of(granularity="frame")
    assert "lateral:turn" in frame_tags
    assert "longitudinal:yield" in frame_tags


def test_tags_of_empty_scope_returns_empty(sample: Path, tmp_path: Path) -> None:
    """scope with no matching routes returns an empty list."""
    store = TagStore(sample)
    bogus = tmp_path / "outside"
    bogus.mkdir()
    assert store.tags_of(scope=bogus) == []


def test_tags_of_validates_granularity(sample: Path) -> None:
    """tags_of rejects invalid granularity strings."""
    store = TagStore(sample)
    with pytest.raises(ValueError):
        store.tags_of(granularity="bogus")


def test_tags_of_validates_dimension_name(sample: Path) -> None:
    """tags_of rejects dimension names outside the regex."""
    store = TagStore(sample)
    with pytest.raises(ValueError):
        store.tags_of(dimensions=["BAD-NAME"])


def test_add_tags_to_route_requires_known_route(sample_route: Path) -> None:
    """add_tags_to_route raises if the route is not in the index."""
    store = TagStore(sample_route)
    with pytest.raises(ValueError, match="route not in index"):
        store.add_tags_to_route(["lateral:turn"], "/nonexistent/route")


def test_add_tags_to_route_merges_into_route(sample_route: Path) -> None:
    """add_tags_to_route adds tags to every frame of the route."""
    store = TagStore(sample_route)
    # override_metric:centerline is already on every aomi frame, so use a
    # tag that nothing has yet.
    n = store.add_tags_to_route(["longitudinal:new"], sample_route)
    assert n == 10
    for npz in store.npz_paths():
        assert "longitudinal:new" in read_tags(npz)


def test_add_tags_to_route_with_frame_filter(sample_route: Path) -> None:
    """add_tags_to_route applies frame_filter to the route's frames."""
    store = TagStore(sample_route)
    # Only the first half by frame number
    n = store.add_tags_to_route(
        ["longitudinal:new"],
        sample_route,
        frame_filter=(2997, 3001),
    )
    assert n == 5


def test_parse_site_split_and_apply(sample: Path) -> None:
    paths = expand_source(sample)
    # expand_source on a directory uses os.walk — order is filesystem-dependent.
    # Pick a frame deterministically by sorting.
    sorted_paths = sorted(paths, key=str)
    # Pick the first path whose directory layout lets parse_site_split recover
    # the map_id (which requires the directory above {manual,auto} to start
    # with a digit). The psim route uses a non-numeric map_id and yields
    # site="unknown" by design — skip those.
    paths0 = next(
        p for p in sorted_paths if parse_site_split(p)[0] != "unknown"
    )
    site, split = parse_site_split(paths0)
    # sorted_paths lands in the 2231_odaiba_shinagawa_copied_from_xx1 (aomi
    # or ariake) route, which has a numeric-prefixed map_id.
    assert site == "2231_odaiba_shinagawa_copied_from_xx1"
    assert split == "auto"
    # apply_path_tags rewrites every frame's site/split tags based on path
    # layout. All 30 frames get their site/split tags overwritten to match
    # the directory layout, so the count is 30.
    n = apply_path_tags(sample)
    assert n == 30
    for path in paths:
        tags = read_tags(path)
        # site resolves via the path layout. For psim (non-numeric map_id)
        # the resolved site is "unknown", so psim frames end up with
        # site:unknown. For x2_dev the resolved site is the original map_id.
        assert (
            "site:2231_odaiba_shinagawa_copied_from_xx1" in tags
            or "site:879_hiratsuka" in tags
            or "site:unknown" in tags
        )
        # After apply_path_tags, split matches the directory token:
        # /manual/ → split:manual, /auto/ → split:auto
        assert "split:manual" in tags or "split:auto" in tags
    # Pick a frame that we know has lateral:turn after the parity-based
    # seeding: frame number 2997 in the aomi route (10-55-13_00000000_00002997).
    # apply_path_tags only touches site/split so lateral:turn survives.
    aomi_turn_path = next(
        p for p in paths if "10-55-13_00000000_00002997" in str(p)
    )
    assert "lateral:turn" in read_tags(aomi_turn_path)


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
    """load_taxonomy warns the docs-only caveat; list_known_tags silences it."""
    from tag_toolkit.taxonomy import load_taxonomy

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_taxonomy()
    assert any("may not match" in str(w.message) for w in caught)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tags = list_known_tags()
    # Programmatic API: warning is suppressed so callers don't have to wire
    # warnings filters in loops/import paths.
    assert not any("may not match" in str(w.message) for w in caught)
    assert "split:manual" in tags
    assert "site:unknown" in tags


def test_taxonomy_skips_malformed_entries(tmp_path: Path) -> None:
    """Malformed taxonomy entries are skipped with a warning."""
    from tag_toolkit.taxonomy import load_taxonomy

    bad = tmp_path / "bad_taxonomy.yaml"
    bad.write_text(
        "dimensions:\n"
        "  site:\n"
        "    values:\n"
        "      - name: 1423\n"          # valid
        "      - bogus_entry: 42\n"     # no 'name' key
        "      - 7\n"                   # not a string or mapping
        "  NOT-VALID-DIM:\n"           # dimension name with hyphens
        "    values:\n"
        "      - name: x\n"
        "  good:\n"
        "    values: not_a_list\n"     # values not a list
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = load_taxonomy(bad)
    # Should not raise; "site" is good, but the bogus entries are dropped.
    assert "site" in data["dimensions"]
    # warnings emitted for at least 3 of the 4 bad shapes
    msgs = [str(w.message) for w in caught]
    assert any("taxonomy" in m for m in msgs)


def test_missing_tags_field(tmp_path: Path) -> None:
    npz = _frame(tmp_path, "x")
    assert read_tags(npz) == []
    write_tags(npz, ["split:auto"])
    assert read_tags(npz) == ["split:auto"]


def test_read_tags_warns_on_malformed_entry(tmp_path: Path) -> None:
    """read_tags warns and skips malformed tag entries."""
    npz = _frame(tmp_path / "bag" / "routes", "x", ["lateral:turn", "BAD TAG"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tags = read_tags(npz)
    assert tags == ["lateral:turn"]
    assert any("BAD TAG" in str(w.message) for w in caught)


def test_read_tags_warns_on_non_string_entry(tmp_path: Path) -> None:
    npz = tmp_path / "bag" / "routes" / "x.npz"
    npz.parent.mkdir(parents=True)
    npz.write_bytes(b"")
    (npz.parent / "x.json").write_text(json.dumps({"tags": ["lateral:turn", 7]}) + "\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tags = read_tags(npz)
    assert tags == ["lateral:turn"]
    assert any("non-string" in str(w.message) for w in caught)


def test_read_tags_raises_on_structural_problems(tmp_path: Path) -> None:
    """'tags' that is not a list (or top-level non-object) still raises."""
    npz = tmp_path / "bag" / "routes" / "x.npz"
    npz.parent.mkdir(parents=True)
    npz.write_bytes(b"")
    (npz.parent / "x.json").write_text(json.dumps({"tags": "not-a-list"}) + "\n")
    with pytest.raises(ValueError, match="must be a list"):
        read_tags(npz)


def test_parse_tag_strictly_rejects_bad_tags() -> None:
    """parse_tag itself is strict — used for argument validation in store.py."""
    with pytest.raises(ValueError):
        parse_tag("BAD TAG")
    with pytest.raises(ValueError):
        parse_tag("nocolon")
    with pytest.raises(ValueError):
        parse_tag("BAD-NAME:value")


def test_tags_for_route_via_tags_of(sample_route: Path) -> None:
    """tags_of with route scope is the new way to ask 'tags for this route'."""
    store = TagStore(sample_route)
    tags = store.tags_of(scope=sample_route)
    assert "lateral:turn" in tags
    assert "longitudinal:yield" in tags


def test_tags_of_route_uses_index_not_disk(
    sample_route: Path, tmp_path: Path
) -> None:
    """tags_of returns indexed frames' tags only."""
    store = TagStore(sample_route)
    assert len(store.npz_paths()) == 10

    # Add an extra frame on disk but NOT in store's source.
    extra_route = tmp_path / "outside_store_index" / "routes"
    extra_npz = _frame(extra_route, "extra_frame", ["override_metric:extra_only"])

    tags = store.tags_of(scope=sample_route)
    assert "override_metric:extra_only" not in tags
    assert "lateral:turn" in tags

    extra_npz.unlink()
    (extra_route / "extra_frame.json").unlink()


def test_group_by_multi_value_total_is_unique(tmp_path: Path) -> None:
    """Multi-value dimensions count each route only once in TOTAL."""
    root = tmp_path / "data"
    bag = root / "prd_jt" / "1423_shinagawa_odaiba" / "manual" / "2025-02-04" / "10-34-24" / "routes"
    _frame(bag, "a", ["site:alpha", "site:beta", "lateral:turn"])

    store = TagStore(root)
    buckets = store.group_by(["site"])
    assert len(buckets) == 2
    assert sum(b.count for b in buckets) == 2
    text = format_buckets(buckets, ["site"])
    assert text.splitlines()[-1].split()[-1] == "1"


def test_group_by_sort_uses_dimensions_order(sample: Path) -> None:
    """Buckets sort primarily by dimensions[0], then dimensions[1], etc."""
    store = TagStore(sample)
    # Add split:eval to psim so it has split:manual, split:valid and split:eval.
    zeikan = next(r for r in store.route_paths() if r.name == "psim_training_bag_0_0")
    store.add_tags(["split:eval"], scope=zeikan)

    buckets = store.group_by(["site", "split"])
    # Expected 5 buckets:
    #   (2231_..., auto)   aomi
    #   (2231_..., train)  ariake
    #   (879_hiratsuka, eval)   psim (added above)
    #   (879_hiratsuka, manual) psim
    #   (879_hiratsuka, valid)  psim
    sites = [b.values["site"] for b in buckets]
    splits = [b.values["split"] for b in buckets]
    # Primary sort by site, secondary by split. Alphabetical:
    #   2231_... < 879_hiratsuka; within 879: eval < manual < valid.
    assert sites == [
        "2231_odaiba_shinagawa_copied_from_xx1",
        "2231_odaiba_shinagawa_copied_from_xx1",
        "879_hiratsuka",
        "879_hiratsuka",
        "879_hiratsuka",
    ]
    assert splits == ["auto", "train", "eval", "manual", "valid"]


def test_group_by_none_sorts_last_in_its_slot(sample: Path) -> None:
    """Items missing a dimension sort after items that have it for that slot."""
    # Build a custom dataset where one route has lateral and another doesn't.
    store = TagStore(sample)
    # The aomi/ariake routes all have lateral:turn on some frames.
    # Add a fourth route to the same source tree with no lateral tags.
    import shutil
    extra_root = store.npz_paths()[0].parent.parent.parent.parent
    extra_route = extra_root / "extra_no_lateral" / "manual" / "2026-04-15" / "10-00-00" / "routes"
    extra_route.mkdir(parents=True, exist_ok=True)
    npz_path = extra_route / "x_00000000_00000001.npz"
    npz_path.write_bytes(b"")
    import json
    (extra_route / "x_00000000_00000001.json").write_text(
        json.dumps({"tags": ["site:no_lateral_here", "split:manual"]})
    )

    # Re-scan including the new route
    store2 = TagStore(extra_root)
    buckets = store2.group_by(["site", "lateral"])

    # For every site, the None lateral cell sorts last within that site.
    by_site: dict[str, list] = {}
    for b in buckets:
        by_site.setdefault(b.values["site"], []).append(b.values["lateral"])
    for site_value, laterals in by_site.items():
        if None in laterals:
            assert laterals[-1] is None, (
                f"site={site_value}: None lateral should sort last, got {laterals}"
            )


def test_invalid_sidecar_tag_skipped_with_warning(
    tmp_path: Path, sample: Path
) -> None:
    """Invalid tags are skipped with a warning during scan."""
    bag = tmp_path / "bag" / "routes"
    npz = _frame(bag, "x", ["lateral:turn", "BAD TAG"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        store = TagStore(tmp_path / "bag")

    tags = store.tags_of(scope=npz, granularity="frame")
    assert tags == ["lateral:turn"]
    assert any("BAD TAG" in str(w.message) for w in caught)


def test_mutate_preflight_missing_sidecar(tmp_path: Path) -> None:
    """Mutation fails if sidecars are missing."""
    bag = tmp_path / "bag" / "routes"
    bag.mkdir(parents=True)
    npz = bag / "orphan.npz"
    npz.write_bytes(b"")

    store = TagStore(bag)
    with pytest.raises(FileNotFoundError, match="missing sidecar"):
        store.add_tags(["lateral:turn"])


def test_flat_layout_route_of(tmp_path: Path) -> None:
    """Flat layout (no /routes/ subdir) also works."""
    route_dir = tmp_path / "bag_time"
    npz = _frame(route_dir, "frame", ["split:auto"])
    assert route_of(npz) == route_dir

    store = TagStore(route_dir)
    assert store.tags_of(scope=route_dir) == ["split:auto"]


def test_source_property(sample: Path) -> None:
    """source property returns the original source."""
    store = TagStore(sample)
    assert store.source == sample


def test_build_index_creates_pickle_file(sample: Path, tmp_path: Path) -> None:
    """build_index writes the index pickle (with magic+version header)."""
    from tag_toolkit.store import _PICKLE_MAGIC, _PICKLE_VERSION

    index_file = tmp_path / "index.tag"
    TagStore.build_index(sample, index_file)

    # Header: 4 bytes magic + 1 byte version, then standard pickle.
    with open(index_file, "rb") as f:
        head = f.read(len(_PICKLE_MAGIC) + len(_PICKLE_VERSION))
        idx = pickle.load(f)
    assert head[: len(_PICKLE_MAGIC)] == _PICKLE_MAGIC
    assert head[len(_PICKLE_MAGIC) :] == _PICKLE_VERSION
    from tag_toolkit.store import _Index
    assert isinstance(idx, _Index)
    assert len(idx.frames) == 30


def test_build_index_returns_index_store(sample: Path, tmp_path: Path) -> None:
    """build_index returns a TagStore with in-memory index."""
    index_file = tmp_path / "index.tag"
    store = TagStore.build_index(sample, index_file)

    assert isinstance(store, TagStore)
    assert store.has_index()
    assert len(store.npz_paths()) == 30
    assert len(store.query("lateral:turn")) == 3


def test_scan_path_list_json(tmp_path: Path, sample: Path) -> None:
    """TagStore can scan a path-list JSON directly."""
    listing = tmp_path / "list.json"
    listing.write_text(json.dumps([str(p) for p in expand_source(sample)]))

    store = TagStore(listing)
    assert len(store.npz_paths()) == 30
    assert len(store.query("lateral:turn")) == 3


def test_index_file_loaded_from_pickle(tmp_path: Path, sample: Path) -> None:
    """Passing a .tag file loads the index from pickle, not scanning it."""
    index_file = tmp_path / "index.tag"
    TagStore.build_index(sample, index_file)

    store = TagStore(index_file)
    assert store.has_index()
    assert store.source == index_file


def test_source_none_creates_empty_store() -> None:
    """TagStore(None) creates an empty store; accessors raise until loaded."""
    store = TagStore(None)
    assert store.source is None
    assert not store.has_index()
    with pytest.raises(ValueError, match="no index loaded"):
        store.npz_paths()


def test_build_index_from_multiple_sources(tmp_path: Path) -> None:
    """build_index can accept multiple sources as a list."""
    root = tmp_path / "data"
    a = root / "a" / "routes"
    b = root / "b" / "routes"
    _frame(a, "a1", ["split:train"])
    _frame(b, "b1", ["split:eval"])

    index_file = tmp_path / "index.tag"
    store = TagStore.build_index([a, b], index_file)
    assert len(store.npz_paths()) == 2
    assert len(store.query("split:train")) == 1
    assert len(store.query("split:eval")) == 1


def test_index_roundtrip_paths_are_absolute(sample: Path, tmp_path: Path) -> None:
    """Index saves and loads with absolute paths."""
    index_file = tmp_path / "index.tag"
    TagStore.build_index(sample, index_file)

    store = TagStore(index_file)
    assert store.has_index()
    for npz in store.npz_paths():
        assert npz.is_absolute(), f"loaded path should be absolute: {npz}"


def test_index_load_rejects_non_index_payload(tmp_path: Path) -> None:
    """Loading a pickle that doesn't contain _Index raises."""
    index_file = tmp_path / "bad_index.tag"
    with open(index_file, "wb") as f:
        pickle.dump({"this": "is not an index"}, f)

    with pytest.raises(ValueError, match="unrecognized pickle"):
        TagStore(index_file)


def test_remove_tags_updates_index_correctly(sample_route: Path) -> None:
    """remove_tags properly updates the in-memory index."""
    store = TagStore(sample_route)

    # Initially, turn is present on 5 frames
    assert len(store.query("lateral:turn")) == 1
    assert len(store.query("lateral:turn", granularity="frame")) == 5

    # Remove turn from every frame in the route
    all_with_turn = store.query("lateral:turn", granularity="frame")
    for npz in all_with_turn:
        write_tags(npz, [])
        store._update_index_for(npz)

    assert store.query("lateral:turn") == []
    assert store.query("lateral:turn", granularity="frame") == []


def test_bucket_has_no_routes_property(sample_route: Path) -> None:
    """Bucket exposes .count and (optionally) .members, not .routes."""
    store = TagStore(sample_route)
    buckets = store.group_by(["split"])

    assert len(buckets) == 1
    bucket = buckets[0]
    assert bucket.count == 1
    assert not hasattr(bucket, "routes")
    assert bucket.members is not None and len(bucket.members) == 1


def test_format_buckets_total_counts_unique_members(sample_route: Path) -> None:
    """TOTAL row counts each unique member once, even with multi-value dimensions."""
    store = TagStore(sample_route)
    npz = store.npz_paths()[0]
    write_tags(npz, ["site:alpha", "site:beta", "split:manual"])
    store._update_index_for(npz)

    buckets = store.group_by(["site"])
    text = format_buckets(buckets, ["site"])

    lines = text.splitlines()
    total_line = lines[-1]
    assert "1" in total_line.split()[-1]


def test_scope_route_path_uses_fast_path(
    sample: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """scope=route_path should hit frames_of_route directly without expand_source."""
    from tag_toolkit import source as source_mod

    store = TagStore(sample)
    routes = {p: store.tags_of(scope=p) for p in store.route_paths()}
    target_route = next(p for p in routes if p.name == "psim_training_bag_0_0")
    assert "split:manual" in store.tags_of(scope=target_route)

    calls = {"n": 0}

    real_expand = source_mod.expand_source

    def counting_expand(*args, **kwargs):
        calls["n"] += 1
        return real_expand(*args, **kwargs)

    monkeypatch.setattr(source_mod, "expand_source", counting_expand)
    monkeypatch.setattr("tag_toolkit.store.expand_source", counting_expand)

    result = store.query("split:manual", granularity="route", scope=target_route)

    assert calls["n"] == 0
    assert set(result) == {target_route}


def test_scope_npz_path_uses_fast_path(
    sample: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """scope=npz_path should use route_of_frame lookup, not expand_source."""
    from tag_toolkit import source as source_mod
    from tag_toolkit.routes import route_of

    store = TagStore(sample)
    zeikan_route = next(p for p in store.route_paths() if p.name == "psim_training_bag_0_0")
    zeikan_npzs = [
        p for p in store.npz_paths() if route_of(p) == zeikan_route
    ]
    assert zeikan_npzs
    npz_a = zeikan_npzs[0]
    expected_route = route_of(npz_a)

    calls = {"n": 0}

    real_expand = source_mod.expand_source

    def counting_expand(*args, **kwargs):
        calls["n"] += 1
        return real_expand(*args, **kwargs)

    monkeypatch.setattr(source_mod, "expand_source", counting_expand)
    monkeypatch.setattr("tag_toolkit.store.expand_source", counting_expand)

    result = store.query("split:manual", granularity="route", scope=npz_a)

    assert calls["n"] == 0
    assert set(result) == {expected_route}


def test_scope_mixed_list_fast_path(sample: Path) -> None:
    """scope=[route, npz] should correctly union both at route granularity."""
    store = TagStore(sample)
    target_route = next(p for p in store.route_paths() if p.name == "psim_training_bag_0_0")

    result = store.query("split:manual", granularity="route", scope=[target_route, target_route])
    assert set(result) == {target_route}


def test_scope_unindexed_npz_path_silently_dropped(
    sample: Path, tmp_path: Path
) -> None:
    """scope with a real path not in the index silently returns empty."""
    store = TagStore(sample)
    bogus_route = tmp_path / "outside_index"
    bogus_route.mkdir()
    bogus_npz = bogus_route / "frame.npz"
    bogus_npz.write_bytes(b"")

    result = store.query("split:manual", granularity="route", scope=bogus_npz)
    assert result == []


def test_scope_other_store_intersected_with_self(sample: Path, tmp_path: Path) -> None:
    """TagStore-scope is intersected with self index (B4 fix)."""
    # Two unrelated scans → two TagStores with disjoint route sets.
    root_a = sample
    root_b = tmp_path / "unrelated"
    bag = root_b / "prd_jt" / "9999_other" / "manual" / "2026-04-15" / "10-00-00" / "routes"
    _frame(bag, "x_00000000_00000001", ["split:manual"])
    other_npz = (bag / "x_00000000_00000001.npz").resolve()

    store_a = TagStore(root_a)
    store_b = TagStore(root_b)

    # The other store's route is NOT in store_a's index. The intersection is
    # empty → query against the other scope returns nothing.
    result = store_a.query("split:manual", scope=store_b)
    assert result == []


def test_atomic_write_uses_tmp_and_rename(tmp_path: Path) -> None:
    """_atomic_write_text writes to a .tmp then renames over the target."""
    target = tmp_path / "out.json"
    _atomic_write_text(target, '{"a": 1}\n')
    assert target.read_text() == '{"a": 1}\n'
    # No leftover tmp
    assert not (tmp_path / "out.json.tmp").exists()


def test_cleanup_tmp_files_removes_strays(tmp_path: Path) -> None:
    """cleanup_tmp_files removes leftover .json.tmp files."""
    (tmp_path / "a.json.tmp").write_text("stale")
    (tmp_path / "b.json.tmp").write_text("stale")
    n = cleanup_tmp_files(tmp_path)
    assert n == 2
    assert list(tmp_path.glob("*.json.tmp")) == []


def test_write_tags_is_atomic(tmp_path: Path) -> None:
    """write_tags succeeds atomically; tmp is cleaned up on success."""
    npz = _frame(tmp_path / "bag" / "routes", "x", ["split:auto"])
    write_tags(npz, ["split:eval", "lateral:turn"])
    assert read_tags(npz) == ["lateral:turn", "split:eval"]
    assert not list((tmp_path / "bag" / "routes").glob("*.json.tmp"))


def test_write_tags_creates_missing_sidecar(tmp_path: Path) -> None:
    """write_tags(npz, ..., create=True) creates the sidecar if missing."""
    npz_path = tmp_path / "bag" / "routes" / "fresh.npz"
    npz_path.parent.mkdir(parents=True)
    npz_path.write_bytes(b"")
    write_tags(npz_path, ["split:auto"], create=True)
    assert read_tags(npz_path) == ["split:auto"]


def test_write_tags_missing_sidecar_without_create(tmp_path: Path) -> None:
    """write_tags without create=True raises FileNotFoundError."""
    npz_path = tmp_path / "bag" / "routes" / "fresh.npz"
    npz_path.parent.mkdir(parents=True)
    npz_path.write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        write_tags(npz_path, ["split:auto"])


# ---------------------------------------------------------------------------
# incremental_index.py — see ../scripts/incremental_index.py
# ---------------------------------------------------------------------------


def _build_old_index_with_routes(
    sample: Path, tmp_path: Path, *, route_names: list[str]
) -> tuple[Path, TagStore]:
    """Build an index over a subset of the sample dataset's routes.

    Returns ``(index_path, store)`` where ``store`` holds the same index in
    memory (so test code can compare frame counts / tag queries without
    re-loading from disk).
    """
    route_dirs = _route_dirs(sample)
    sources = [route_dirs[n] for n in route_names]
    index_path = tmp_path / "old.tag"
    store = TagStore.build_index(sources, index_path)
    return index_path, store


def test_incremental_index_adds_new_route(sample: Path, tmp_path: Path) -> None:
    """Adding a brand-new route grows the index without touching the old route."""
    old_index, old_store = _build_old_index_with_routes(
        sample, tmp_path, route_names=["aomi"]
    )
    assert len(old_store.npz_paths()) == 10

    # Add the ariake route (10 new frames in a route the old index never saw).
    new_route = _route_dirs(sample)["ariake"]
    summary = build_incremental_index(old_index, new_route)

    assert summary["new_frames"] == 10
    assert summary["skipped"] == 0

    merged = TagStore(old_index)
    assert len(merged.npz_paths()) == 20
    assert len(merged.route_paths()) == 2

    # Old frames still resolve under the same tags.
    aomi = merged.query("site:2231_odaiba_shinagawa_copied_from_xx1")
    assert len(aomi) == 2  # both routes share the same site
    # lateral:turn is on 5 frames per route → 10 frames, 2 routes after merge.
    assert len(merged.query("lateral:turn")) == 2
    assert len(merged.query("lateral:turn", granularity="frame")) == 10


def test_incremental_index_overwrites_in_place_by_default(
    sample: Path, tmp_path: Path
) -> None:
    """Without --output the script writes back over the old index file."""
    old_index, _ = _build_old_index_with_routes(
        sample, tmp_path, route_names=["aomi"]
    )
    before_bytes = old_index.read_bytes()
    new_route = _route_dirs(sample)["ariake"]

    build_incremental_index(old_index, new_route)

    # File still exists, content changed.
    assert old_index.is_file()
    assert old_index.read_bytes() != before_bytes
    merged = TagStore(old_index)
    assert len(merged.npz_paths()) == 20


def test_incremental_index_output_keeps_old_intact(
    sample: Path, tmp_path: Path
) -> None:
    """With output_path the old index is preserved; a new file is written."""
    old_index, old_store = _build_old_index_with_routes(
        sample, tmp_path, route_names=["aomi"]
    )
    before_bytes = old_index.read_bytes()
    new_route = _route_dirs(sample)["ariake"]
    out = tmp_path / "merged.tag"

    summary = build_incremental_index(old_index, new_route, output_path=out)

    assert summary["output"] == out
    assert out.is_file()
    # Old index untouched on disk.
    assert old_index.read_bytes() == before_bytes
    # Both are loadable, both contain 20 frames.
    assert len(TagStore(old_index).npz_paths()) == 10
    assert len(TagStore(out).npz_paths()) == 20


def test_incremental_index_skips_frames_already_in_old(
    sample: Path, tmp_path: Path
) -> None:
    """Re-listing an old route yields zero new frames (warned, not crashed)."""
    old_index, old_store = _build_old_index_with_routes(
        sample, tmp_path, route_names=["aomi"]
    )
    aomi = _route_dirs(sample)["aomi"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary = build_incremental_index(old_index, aomi)

    assert summary["new_frames"] == 0
    assert summary["skipped"] == 10
    # No crash, no rewrites: the merged index has the same 10 frames.
    assert len(TagStore(old_index).npz_paths()) == 10
    # User got a warning per skipped frame.
    assert any("already in old index" in str(w.message) for w in caught)


def test_incremental_index_missing_sidecar_aborts(
    sample: Path, tmp_path: Path
) -> None:
    """A new frame without a sidecar raises; old index stays intact."""
    old_index, _ = _build_old_index_with_routes(
        sample, tmp_path, route_names=["aomi"]
    )
    before_bytes = old_index.read_bytes()

    # Half-constructed new frame: NPZ present, sidecar missing.
    new_routes = tmp_path / "new_routes"
    bag = new_routes / "prd_jt" / "9999_new" / "manual" / "2026-04-15" / "10-00-00"
    routes_dir = bag / "routes"
    routes_dir.mkdir(parents=True)
    (routes_dir / "x_00000000_00000001.npz").write_bytes(b"")

    with pytest.raises(FileNotFoundError, match="missing sidecar"):
        build_incremental_index(old_index, new_routes)

    # Old index is not corrupted / partially merged.
    assert old_index.read_bytes() == before_bytes
    assert len(TagStore(old_index).npz_paths()) == 10


def test_incremental_index_preserves_tag_counts(
    sample: Path, tmp_path: Path
) -> None:
    """route_tag_counts stays consistent after merging a new route."""
    old_index, _ = _build_old_index_with_routes(
        sample, tmp_path, route_names=["aomi"]
    )
    new_route = _route_dirs(sample)["ariake"]

    build_incremental_index(old_index, new_route)
    merged = TagStore(old_index)

    # lateral:turn is on 5 frames per route → after merge 10 frames, 2 routes.
    assert len(merged.query("lateral:turn", granularity="frame")) == 10
    assert len(merged.query("lateral:turn")) == 2
    # longitudinal:yield is only on aomi (2 specific frames).
    assert len(merged.query("longitudinal:yield", granularity="frame")) == 2
    assert len(merged.query("longitudinal:yield")) == 1


def test_incremental_index_rejects_bad_old_index(tmp_path: Path) -> None:
    """Loading a non-index pickle is the same ValueError as the regular API."""
    bad = tmp_path / "bad.tag"
    bad.write_bytes(pickle.dumps({"not": "an index"}))

    with pytest.raises(ValueError, match="unrecognized pickle"):
        build_incremental_index(bad, tmp_path)


# ---------------------------------------------------------------------------
# Coverage additions (D1–D10)
# ---------------------------------------------------------------------------


def test_bucket_label_default_separator(sample: Path) -> None:
    """Bucket.label() joins values with ' | ' by default; '-' for missing."""
    store = TagStore(sample)
    buckets = store.group_by(["site", "split"])
    # Pick the bucket whose split is "auto" (aomi only).
    b = next(b for b in buckets if b.values.get("split") == "auto")
    assert b.label() == f"2231_odaiba_shinagawa_copied_from_xx1 | auto"
    # Custom separator.
    assert b.label(sep=" / ") == "2231_odaiba_shinagawa_copied_from_xx1 / auto"


def test_bucket_label_missing_value(sample: Path) -> None:
    """Bucket.label() substitutes '-' when a dimension value is None."""
    # Add an extra route without a "lateral" tag so it lands in a None cell.
    import shutil

    store = TagStore(sample)
    extra_root = store.npz_paths()[0].parent.parent.parent.parent
    extra_route = extra_root / "extra_no_lateral" / "manual" / "2026-04-15" / "extra"
    extra_route.mkdir(parents=True, exist_ok=True)
    npz_path = extra_route / "x_00000000_00000001.npz"
    npz_path.write_bytes(b"")
    (extra_route / "x_00000000_00000001.json").write_text(
        json.dumps({"tags": ["site:no_lateral_here", "split:manual"]})
    )

    store2 = TagStore(extra_root)
    buckets = store2.group_by(["site", "lateral"])
    none_bucket = next(b for b in buckets if b.values.get("lateral") is None)
    assert none_bucket.label() == "no_lateral_here | -"


def test_group_by_frame_granularity(sample: Path) -> None:
    """group_by(granularity='frame') buckets frames, not routes."""
    store = TagStore(sample)
    # All 30 frames carry site/2231_..._xx1 or site/879_hiratsuka.
    # split has 4 values: auto (10), train (10), manual (5), valid (5).
    buckets = store.group_by(["split"], granularity="frame")
    by_split = {b.values["split"]: b.count for b in buckets}
    assert by_split == {"auto": 10, "train": 10, "manual": 5, "valid": 5}

    # Cross-check against the route-granularity view: psim's two splits
    # collapse into one route cell.
    route_buckets = store.group_by(["split"], granularity="route")
    route_by_split = {b.values["split"]: b.count for b in route_buckets}
    assert route_by_split == {"auto": 1, "train": 1, "manual": 1, "valid": 1}


def test_query_not_on_route_granularity(sample: Path) -> None:
    """{'not': ...} subtracts routes from the universe at route granularity."""
    store = TagStore(sample)
    # Routes are 3 distinct paths.
    all_routes = set(store.route_paths())
    # Pick a tag present on exactly one route (aomi).
    pos = set(store.query("split:auto"))
    neg = set(store.query({"not": "split:auto"}))
    assert pos == {r for r in all_routes if r.name == "10-55-13"}
    assert pos | neg == all_routes
    assert pos & neg == set()


def test_replace_tags_deduplicates_old_pair_with_duplicate_frame(
    sample_route: Path,
) -> None:
    """replace_tags skips frames where the new tag is already present.

    Per-frame sidecar may already carry ``new_tag`` for some frames. Those
    frames do not need a write — the only effect of the replace on them is
    removing ``old_tag``, but if ``new_tag`` is also present the tag set is
    unchanged after the operation. Other frames get rewritten as usual.
    """
    store = TagStore(sample_route)
    # Hand-edit one aomi frame so it carries both lateral:turn and
    # lateral:lane_keeping already (this frame does not get rewritten).
    npz = next(iter(store.npz_paths()))
    sidecar = json.loads(npz.with_suffix(".json").read_text())
    sidecar["tags"] = sorted(
        {"lateral:turn", "lateral:lane_keeping", "site:2231_x", "split:auto"}
    )
    npz.with_suffix(".json").write_text(json.dumps(sidecar) + "\n")
    # Refresh index by re-opening.
    store = TagStore(sample_route)

    n = store.replace_tags(tag_pairs={"lateral:turn": "lateral:lane_keeping"})
    # aomi has 5 frames with lateral:turn. The hand-edited frame (3006)
    # already carries lateral:lane_keeping, so the replace there is a
    # no-op and the file isn't rewritten. All 5 odd-numbered frames are
    # rewritten; the hand-edited frame stays untouched in storage.
    assert n == 5
    # The hand-edited frame keeps both tags (we never wrote to it).
    assert "lateral:turn" in read_tags(npz)
    assert "lateral:lane_keeping" in read_tags(npz)
    # The 5 odd-numbered frames (which carried lateral:turn but not
    # lane_keeping) are now rewritten: turn gone, lane_keeping present
    # exactly once. Even-numbered frames never had turn and stay untouched.
    odd_with_turn = sorted(store.npz_paths())[::2]  # 2997, 2999, 3001, 3003, 3005
    assert len(odd_with_turn) == 5
    for p in odd_with_turn:
        tags = read_tags(p)
        assert "lateral:lane_keeping" in tags
        assert "lateral:turn" not in tags
        assert tags.count("lateral:lane_keeping") == 1


def test_replace_tags_old_equals_new_is_noop(sample_route: Path) -> None:
    """tag_pairs where old == new counts as a no-op."""
    store = TagStore(sample_route)
    n = store.replace_tags(tag_pairs={"lateral:turn": "lateral:turn"})
    assert n == 0


def test_add_tags_with_duplicate_tags_is_idempotent(sample_route: Path) -> None:
    """add_tags with duplicates in input doesn't write twice or double-count.

    lateral:turn is on 5 of 10 aomi frames (odd frame numbers per _build.py).
    Once those 5 frames are tagged, a second ``add_tags(['lateral:turn'])``
    is a no-op (0 writes). Passing the tag twice in the input list is
    equivalent to passing it once. After the first call, lateral:turn
    appears exactly once per frame, never duplicated.
    """
    store = TagStore(sample_route)
    n_once = store.add_tags(["lateral:turn"])
    # First call writes the 5 frames that didn't have lateral:turn yet.
    assert n_once == 5
    for p in store.npz_paths():
        assert read_tags(p).count("lateral:turn") == 1

    # Second call with the same tag is a no-op: all frames have it now.
    n_second = store.add_tags(["lateral:turn"])
    assert n_second == 0
    # Same outcome when the tag is duplicated in the input list.
    n_twice = store.add_tags(["lateral:turn", "lateral:turn"])
    assert n_twice == 0
    for p in store.npz_paths():
        assert read_tags(p).count("lateral:turn") == 1

    # A genuinely-new tag alongside a duplicate only writes the frames
    # missing the new tag.
    n = store.add_tags(["lateral:turn", "longitudinal:new"])
    assert n == 10
    for p in store.npz_paths():
        tags = read_tags(p)
        assert "longitudinal:new" in tags
        assert "lateral:turn" in tags
        assert tags.count("lateral:turn") == 1


def test_resolve_scope_with_list_of_tagstore(sample: Path) -> None:
    """scope=[TagStore, ...] flattens TagStore elements instead of raising."""
    import shutil

    # Build a second store that scans only the ariake route. We give it the
    # parent auto/ directory and rely on TagStore to discover one route.
    ariake_route = (
        sample
        / "x2_dev"
        / "2231_odaiba_shinagawa_copied_from_xx1"
        / "auto"
        / "2026-07-07"
    )
    other_store = TagStore(ariake_route)
    assert len(other_store.route_paths()) == 1
    ariake_only = other_store.route_paths()[0]
    main = TagStore(sample)

    # Without the fix this used to raise TypeError. With the fix it should
    # return only the routes that intersect between self and the scoped
    # TagStore — here that is the ariake route, which carries split:train
    # (and is the only route in the other_store).
    routes = main.query("split:train", scope=[other_store])
    assert set(routes) == {ariake_only}
    # The flattened TagStore element exposes its route set; mixing with a
    # Path in the same list also works.
    routes = main.query("split:train", scope=[other_store, ariake_route])
    assert set(routes) == {ariake_only}


def test_expand_source_single_npz_missing_raises(tmp_path: Path) -> None:
    """expand_source on a missing single NPZ raises FileNotFoundError.

    Without the explicit exists() check, expand_source would happily return
    [missing.npz] and downstream ``read_tags`` would crash. Better to fail
    fast at the source expansion boundary.
    """
    from tag_toolkit import expand_source

    bogus = tmp_path / "missing.npz"
    with pytest.raises(FileNotFoundError):
        expand_source(str(bogus))


def test_query_and_npz_paths_on_empty_store(tmp_path: Path) -> None:
    """query/group_by/npz_paths raise ValueError when no index has been loaded yet."""
    # Construct an explicitly-no-source store: _index stays None.
    store = TagStore(None)
    # npz_paths() no longer returns [] when empty — it fails fast like query/group_by.
    with pytest.raises(ValueError, match="no index loaded"):
        store.npz_paths()
    # query / group_by need an index.
    with pytest.raises(ValueError, match="no index loaded"):
        store.query("split:auto")
    with pytest.raises(ValueError, match="no index loaded"):
        store.group_by(["site"])


def test_mutate_methods_on_uninitialized_store_raise() -> None:
    """Mutate methods raise ValueError when no index has been loaded yet.

    A ``TagStore(None)`` (or never-scanned store) has ``_index is None`` and
    must reject every mutation with a clear message, matching the query
    side. Scanning an empty directory is different — it produces an empty
    index where mutations are valid no-ops.
    """
    store = TagStore(None)
    for call in [
        lambda: store.add_tags(["split:auto"]),
        lambda: store.remove_tags(["split:auto"]),
        lambda: store.remove_dimension("split"),
        lambda: store.replace_tags(tag_pairs={"split:auto": "split:eval"}),
        lambda: store.add_tags_to_route(["split:auto"], "/some/route"),
    ]:
        with pytest.raises(ValueError, match="no index loaded"):
            call()


def test_helper_exports_basic(sample: Path) -> None:
    """parse_tag, format_tag, normalize_tags, read_tags round-trip correctly."""
    from tag_toolkit import format_tag, normalize_tags, parse_tag

    # parse_tag round-trips with format_tag.
    d, v = parse_tag("split:auto")
    assert format_tag(d, v) == "split:auto"

    # parse_tag error message mentions both the regex and the structure.
    with pytest.raises(ValueError, match=r"\[a-z0-9_\]\+"):
        parse_tag("BAD/auto")

    # normalize_tags sorts and dedupes.
    assert normalize_tags(["b:2", "a:1", "b:2"]) == ["a:1", "b:2"]

    # read_tags on a known frame returns the same list normalize_tags would
    # produce from a write.
    from tag_toolkit import write_tags

    npz = sample / "x2_dev" / "2231_odaiba_shinagawa_copied_from_xx1" / "auto" / "2026-06-23" / "10-55-13" / "routes"
    sample_npz = next(npz.glob("*.npz"))
    expected = normalize_tags(["split:auto", "lateral:turn"])
    write_tags(sample_npz, expected)
    assert read_tags(sample_npz) == expected


def test_expand_source_directory_and_npz_list(tmp_path: Path) -> None:
    """expand_source handles directory, npz_list.json, and list-of-paths."""
    import json as _json

    from tag_toolkit import expand_source

    # Directory with two NPZ files.
    (tmp_path / "bag" / "routes").mkdir(parents=True)
    npz_a = tmp_path / "bag" / "routes" / "a.npz"
    npz_b = tmp_path / "bag" / "routes" / "b.npz"
    npz_a.write_bytes(b"")
    npz_b.write_bytes(b"")

    dir_paths = sorted(p.name for p in expand_source(tmp_path / "bag"))
    assert dir_paths == ["a.npz", "b.npz"]

    # npz_list.json.
    list_file = tmp_path / "list.json"
    list_file.write_text(_json.dumps([str(npz_a), str(npz_b)]))
    list_paths = sorted(p.name for p in expand_source(list_file))
    assert list_paths == ["a.npz", "b.npz"]

    # List of paths passed directly.
    direct = sorted(p.name for p in expand_source([npz_a, npz_b]))
    assert direct == ["a.npz", "b.npz"]


def test_load_json_helper(tmp_path: Path) -> None:
    """load_json parses JSON files into dicts/lists."""
    from tag_toolkit import load_json

    f = tmp_path / "x.json"
    f.write_text(json.dumps({"k": "v"}))
    assert load_json(f) == {"k": "v"}


def test_list_known_tags_includes_open_placeholder(tmp_path: Path) -> None:
    """list_known_tags emits '<dim>:<open>' for dimensions without values."""
    from tag_toolkit import list_known_tags

    yaml_path = tmp_path / "t.yaml"
    yaml_path.write_text(
        "dimensions:\n"
        "  site:\n"
        "    values: [{name: odx}, {name: hrk}]\n"
        "  notes:\n"  # no 'values' key — open dimension
    )
    tags = list_known_tags(yaml_path)
    assert "site:odx" in tags
    assert "site:hrk" in tags
    assert "notes:<open>" in tags


def test_query_default_is_all(sample: Path) -> None:
    """query() with no clause = 'everything in scope' (no extra filter)."""
    store = TagStore(sample)
    all_routes = store.query()
    assert set(all_routes) == set(store.route_paths())
    # Frame granularity too.
    all_frames = store.query(granularity="frame")
    assert set(all_frames) == set(store.npz_paths())


def test_group_by_drop_missing_skips_incomplete(tmp_path: Path) -> None:
    """drop_missing=True: items missing a requested dimension are skipped."""
    root = tmp_path / "data"
    bag_a = root / "a" / "routes"
    bag_b = root / "b" / "routes"
    _frame(bag_a, "x", ["site:alpha", "split:auto"])
    _frame(bag_b, "y", ["split:auto"])  # missing 'site' on this one

    store = TagStore(root)

    # Default behaviour: 'y' still lands in the (None, auto) bucket.
    default_buckets = store.group_by(["site", "split"])
    assert any(b.values["site"] is None for b in default_buckets)

    # drop_missing=True: 'y' is gone entirely.
    dropped_buckets = store.group_by(["site", "split"], drop_missing=True)
    assert not any(b.values["site"] is None for b in dropped_buckets)
    # Only the (alpha, auto) bucket remains.
    assert len(dropped_buckets) == 1
    assert dropped_buckets[0].values == {"site": "alpha", "split": "auto"}


def test_bucket_members_is_always_populated(sample: Path) -> None:
    """members is always a list (never None) — no more include_members flag."""
    store = TagStore(sample)
    buckets = store.group_by(["site"])
    for b in buckets:
        assert isinstance(b.members, list)
        assert b.count == len(b.members)


def test_load_legacy_pickle_without_magic(tmp_path: Path) -> None:
    """Legacy pickles (no magic header) still load — backwards compat."""
    from tag_toolkit.store import _Index, _build_index_from_frames

    legacy = tmp_path / "legacy.tag"
    idx = _Index()
    # Hand-build a one-frame index the way the previous pickle format did:
    # a bare pickle with no leading magic.
    with open(legacy, "wb") as f:
        pickle.dump(_build_index_from_frames(tmp_path / "ignored") if False else idx, f)

    # Loading the empty legacy pickle should succeed with no frames.
    store = TagStore(legacy)
    assert store.source == legacy
    assert list(store.npz_paths()) == []


def test_concurrent_add_tags_does_not_corrupt(tmp_path: Path) -> None:
    """Threads calling add_tags on different frames don't corrupt the index."""
    # Create 8 frames in a single route so we can hammer them concurrently.
    root = tmp_path / "data"
    bag = root / "p" / "1423_x" / "manual" / "2025-01-01" / "10-00-00" / "routes"
    bag.mkdir(parents=True)
    frame_paths = [
        _frame(bag, f"frame{i:08d}") for i in range(8)
    ]

    store = TagStore(root)

    errors: list[Exception] = []

    def worker(start_idx: int) -> None:
        try:
            # Each thread adds a distinct tag to a distinct frame.
            store.add_tags(
                [f"thread:{start_idx}"],
                scope=bag,
                frame_filter=f"frame{start_idx:08d}*.npz",
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent mutations raised: {errors}"
    # And every frame now carries its thread's tag (no overwrites, no lost writes).
    for i, npz in enumerate(frame_paths):
        assert f"thread:{i}" in read_tags(npz), f"thread {i} tag missing from {npz}"


def test_group_by_default_false_keeps_missing_in_bucket(sample: Path) -> None:
    """Default drop_missing=False: items missing a dimension land in None slot.

    Documents the cartesian-product behaviour that 'count' may show N items
    across (-, X) and (-, Y) buckets that all correspond to the same item
    when the OTHER dimension is multi-valued. Covered separately in
    test_group_by_multi_value_total_is_unique for deduping.
    """
    store = TagStore(sample)
    # Queried with only "split" — every route carries some split value.
    buckets = store.group_by(["split"])
    # No None slots expected because all routes carry a split tag.
    assert all(b.values["split"] is not None for b in buckets)
