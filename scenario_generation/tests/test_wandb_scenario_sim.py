import sys
from unittest.mock import MagicMock

# Mock wandb module if not installed in local environment
mock_wandb = MagicMock()
sys.modules.setdefault("wandb", mock_wandb)

from scenario_generation.scenario_sim_viewer_export import FAILURE_CATEGORIES
from scenario_generation.wandb_scenario_sim import (
    build_scenario_sim_scalars,
    build_scenario_sim_series_panels,
    build_scenario_sim_wandb_payload,
    load_case_rows,
    pick_diverse_worst_cases,
    series_point,
)


def _case(
    route: str = "c",
    kind: str = "Failure",
    clearance: float | None = 1.0,
    collisions: int = 0,
    borders: int = 0,
    brakes: int = 0,
    progress: float | None = None,
    **extra,
):
    """A case row carrying only what these tests exercise; absent blocks default to zero."""
    row = {
        "route": route,
        "result_kind": kind,
        "object": {"collision_count": collisions, "clearance_min_m": clearance},
    }
    if borders:
        row["road_border"] = {"collision_count": borders}
    if brakes:
        row["strong_brake"] = {"count": brakes}
    if progress is not None:
        row["progress_m"] = progress
    row.update(extra)
    return row


def test_bucket_selection():
    """Distinct failure modes get their own bucket; a count-defined bucket ranks by that count;
    an unmeasured clearance is not evidence of danger; braking hard is not a failure."""
    rows = [
        _case("err", terminated="worker_failed"),  # never ran
        _case("frozen", max_speed_mps=0.1, progress=3.0),  # ran and stopped dead
        _case("collision", collisions=1, clearance=0.0, progress=20.0),
        _case("unmeasured", collisions=1, clearance=None, progress=1.0),
        _case("rb_few", borders=1, clearance=0.5),
        _case("rb_many", borders=4, clearance=3.0),
        _case("brake_few", brakes=1, clearance=0.5),
        _case("brake_many", brakes=6, clearance=3.0),
        _case("pass_tight", kind="Pass", clearance=1.0),
        _case("pass_braked", kind="Pass", clearance=4.0, brakes=9),
    ]

    selected = pick_diverse_worst_cases(rows)
    assert selected["worst_error"]["route"] == "err"
    assert selected["worst_frozen_standstill"]["route"] == "frozen"
    assert selected["worst_collision"]["route"] == "collision"
    assert selected["worst_road_departure"]["route"] == "rb_many"
    assert selected["worst_strong_brake"]["route"] == "brake_many"
    assert selected["best_pass"]["route"] == "pass_braked"


def test_build_scenario_sim_scalars():
    rows = [
        _case(kind="Pass", clearance=2.0, progress=100.0),
        _case(collisions=1, clearance=0.0, progress=50.0),
        _case(borders=2, clearance=None, progress=60.0),
    ]

    scalars = build_scenario_sim_scalars(rows, prefix="scenario_sim")
    assert scalars["scenario_sim/total_cases"] == 3
    assert scalars["scenario_sim/passed_cases"] == 1
    assert scalars["scenario_sim/pass_rate"] == 33.33
    assert scalars["scenario_sim/collision_cases"] == 1
    assert scalars["scenario_sim/road_border_cases"] == 1
    assert scalars["scenario_sim/mean_case_min_clearance_m"] == 1.0
    assert scalars["scenario_sim/mean_progress_m"] == 70.0

    # The categories partition the run, which is what lets W&B stack them, and every one is
    # emitted: a key missing from some steps would read as a gap rather than the zero it is.
    counts = {k: v for k, v in scalars.items() if "/category/" in k}
    assert sum(counts.values()) == len(rows)
    assert set(counts) == {f"scenario_sim/category/{c.lower()}" for c in FAILURE_CATEGORIES}


def test_build_scenario_sim_wandb_payload(tmp_path):
    media_dir = tmp_path / "media" / "uuid123"
    media_dir.mkdir(parents=True)
    video_file = media_dir / "uuid123_route0.mp4"
    video_file.write_text("dummy mp4")

    rows = [
        _case(
            "ego_speed2p7778",
            collisions=1,
            clearance=0.0,
            progress=10.0,
            case_key="uuid123_route0",
            scenario="uuid123",
        )
    ]

    payload = build_scenario_sim_wandb_payload(rows, media_root=tmp_path, prefix="test_sim")
    assert payload["test_sim/total_cases"] == 1
    assert payload["test_sim/collision_cases"] == 1
    assert "test_sim/cases_table" in payload
    assert "test_sim/videos/worst_collision" in payload


def test_load_case_rows(tmp_path):
    """A run in flight can leave a half-written last line, which must not fail the read.

    A closed-loop segments.jsonl is not a source: its rows carry no verdict."""
    (tmp_path / "cases.jsonl").write_text('{"a": 1}\n\n{"a": 2}\n{"a":')
    assert load_case_rows(tmp_path) == [{"a": 1}, {"a": 2}]

    segments = tmp_path / "sub"
    segments.mkdir()
    (segments / "segments.jsonl").write_text('{"route": "r"}\n')
    assert load_case_rows(segments) == []
    assert load_case_rows(tmp_path / "absent") == []


def test_series_panels_cover_every_point():
    points = [
        series_point(f"epoch{e:04d}", e, build_scenario_sim_scalars([_case(kind="Pass")]))
        for e in (10, 20)
    ]
    # build_scenario_sim_wandb_payload shares this mock, so the assertions below must read
    # this call rather than whichever test ran last.
    mock_wandb.Table.reset_mock()
    panels = build_scenario_sim_series_panels(points)

    assert set(panels) == {"scenario_sim/series/summary"}
    table = mock_wandb.Table.call_args.kwargs
    assert [r[0] for r in table["data"]] == ["epoch0010", "epoch0020"]
    assert [r[1] for r in table["data"]] == [10, 20]
    # A count stays a number so W&B can sort and plot it; only the ratios are formatted.
    assert table["data"][0][2] == 1 and table["data"][0][4] == "100.00%"
    assert build_scenario_sim_series_panels([]) == {}


def test_failure_categories_covers_every_classifier_return():
    """The tuple is hand-maintained next to a function returning bare literals, so tie them.

    Asserting the emitted keys against the tuple is circular; this reads the classifier itself.
    """
    import inspect
    import re

    from scenario_generation import scenario_sim_viewer_export

    body = inspect.getsource(scenario_sim_viewer_export.classify_failure)
    returned = set(re.findall(r'return "([A-Z_]+)"', body))
    assert returned, "no literal returns found -- this guard would silently pass"
    assert returned <= set(FAILURE_CATEGORIES), returned - set(FAILURE_CATEGORIES)
