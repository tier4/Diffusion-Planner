"""Contract tests for the scenario_sim viewer export.

These assert the shape a consumer outside this repo depends on.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from scenario_generation.scenario_sim_viewer_export import (
    _NAMES_SIDECAR,
    ViewerTree,
    export,
    key_aliases,
    load_submitted,
)

# Shaped like a suite's map path, because the map id is read out of it. Nothing on disk.
_MAP = "/map/example_project/1/1-0001/lanelet2_map.osm"
_SIDECAR = {"categories": {"Z": "label"}, "scenarios": {}}
_DESCRIPTION = "what this scenario is"
_XOSC = (
    '<OpenSCENARIO><FileHeader description="{d}"/>'
    '<ParameterDeclarations><ParameterDeclaration name="speed" value="8.3"/>'
    "</ParameterDeclarations><Storyboard/></OpenSCENARIO>"
)
_SC1 = str(uuid.uuid4())
_SC2 = str(uuid.uuid4())


def _rel(scenario_id: str, case: int = 5) -> str:
    return f"proj/{scenario_id}/{case}/scenario_0.xosc"


def _row() -> dict:
    """A segment row with the block layout ``aggregate`` rolls up."""
    clearance = {
        "miss_thresh_m": 1.0,
        "collision_steps": 0,
        "collision_count": 0,
        "miss_steps": 0,
        "miss_count": 0,
        "clearance_min_m": 2.0,
        "clearance_mean_m": 3.0,
        "clearance_p5_m": 2.1,
        "clearance_finite_steps": 3,
    }
    return {
        "n_steps_run": 3,
        "terminated": "max_steps",
        "progress_m": 10.0,
        "object": dict(clearance),
        "road_border": dict(clearance),
        "red_light_violation": {"steps": 0, "count": 0, "measured": False},
        # inf is the in-band "no braking event" value the rollout writes.
        "strong_brake": {
            "thresh_mps2": -2.5,
            "strongest_mps2": float("inf"),
            "steps": 0,
            "count": 0,
        },
        "reproducer": {"expand_count": 0, "snap_count": 0, "repeat_steps": 0, "normal_steps": 3},
        "map_path": _MAP,
    }


def _make_run(root: Path, rels: list[str], *, submitted: list[str] | None = None) -> Path:
    """A run directory in the shape the suite driver leaves behind."""
    root.mkdir(parents=True, exist_ok=True)
    suite = root.parent / "suite"
    (suite / "scenarios").mkdir(parents=True, exist_ok=True)
    (root / "run_context.txt").write_text(
        f"jobs=4 max_steps=1700 draw_every=4 scenario_root={suite / 'scenarios'}\n"
    )
    # The suite ships its names beside itself, and each scenario carries its own description.
    (suite / _NAMES_SIDECAR).write_text(json.dumps(_SIDECAR))
    for rel in rels:
        osc = suite / "scenarios" / rel
        osc.parent.mkdir(parents=True, exist_ok=True)
        osc.write_text(_XOSC.format(d=_DESCRIPTION))
    (root / "work.json").write_text(
        json.dumps(
            [
                [str(root / key_aliases(r)[0]), str(suite / "scenarios" / r)]
                for r in (submitted or rels)
            ]
        )
    )
    trace = [
        json.dumps(
            {"k": k, "ego": [float(k), 0.0], "speed": 1.0, "clearance_m": 2.0, "collision": False}
        )
        for k in range(3)
    ]
    for rel in rels:
        case = root / key_aliases(rel)[0]
        case.mkdir(parents=True, exist_ok=True)
        (case / "row.json").write_text(json.dumps(_row()))
        (case / "rollout.jsonl").write_text("\n".join(trace) + "\n")
        (case / f"{case.name}.mp4").write_bytes(b"\x00")
    return root


def test_the_listing_is_three_files_at_the_run_root(tmp_path):
    """The listing must not cost a read per scenario."""
    export(_make_run(tmp_path / "run", [_rel(_SC1), _rel(_SC1, case=6)]), tmp_path / "out")
    out = tmp_path / "out"

    for name in ("run.json", "scenarios.json", "cases.jsonl"):
        assert (out / name).is_file()
    assert not list(out.glob("*/summary.json")), "per-scenario summary files are back"

    cases = [json.loads(ln) for ln in (out / "cases.jsonl").read_text().splitlines()]
    assert len(cases) == 2
    tree = ViewerTree(out)
    for case in cases:
        assert case["scenario"] == _SC1
        stem = case["route"]
        assert tree.video(_SC1, stem).is_file()
        assert tree.rollout(_SC1, stem).is_file()
        assert tree.colormap(_SC1, stem, "clearance").is_file()


def test_exported_json_carries_no_non_json_constants(tmp_path):
    """A consumer outside Python cannot read a file containing Infinity or NaN."""
    export(_make_run(tmp_path / "run", [_rel(_SC1)]), tmp_path / "out")

    def reject(token):
        raise AssertionError(f"non-JSON constant in output: {token}")

    for path in (tmp_path / "out").rglob("*.json"):
        json.loads(path.read_text(), parse_constant=reject)
    for path in (tmp_path / "out").rglob("*.jsonl"):
        for line in path.read_text().splitlines():
            json.loads(line, parse_constant=reject)


def test_unmeasured_families_are_not_reported_as_zero(tmp_path):
    """A family nobody observed must stay distinguishable from one measured at zero."""
    export(_make_run(tmp_path / "run", [_rel(_SC1)]), tmp_path / "out")
    entry = json.loads((tmp_path / "out" / "scenarios.json").read_text())[_SC1]

    assert "mean_route_completion" not in entry["summary"]
    assert entry["summary"]["red_light_violation"]["measured"] is False
    assert {"mean_route_completion", "reproducer"} <= set(entry["unmeasured_keys"])
    case = json.loads((tmp_path / "out" / "cases.jsonl").read_text().splitlines()[0])
    assert "route_completion" not in case


def test_missing_cases_are_stated_against_the_submitted_list(tmp_path):
    """Failures are absent rows; the artifacts on disk cannot reveal them."""
    run = _make_run(tmp_path / "run", [_rel(_SC1)], submitted=[_rel(_SC1), _rel(_SC1, case=6)])
    export(run, tmp_path / "out")

    entry = json.loads((tmp_path / "out" / "scenarios.json").read_text())[_SC1]
    assert "produced no row" in entry["error"]
    assert json.loads((tmp_path / "out" / "run.json").read_text())["submitted_cases"] == 2


def test_a_suite_with_no_scenario_id_fails(tmp_path):
    """A suite whose paths carry no scenario id is unreadable, not an empty run."""
    with pytest.raises(SystemExit, match="resolved nothing"):
        export(_make_run(tmp_path / "run", ["cat/a.xosc"]), tmp_path / "out")


def test_work_list_entries_resolve_against_the_manifest(tmp_path):
    """Entries resolve against the manifest's own directory; absolute ones pass through."""
    run = tmp_path / "run"
    run.mkdir()
    absolute = str(tmp_path / "suite" / _rel(_SC2))
    (run / "work.json").write_text(
        json.dumps([[str(run / "case_a"), _rel(_SC1)], [str(run / "case_b"), absolute]])
    )
    got = dict(load_submitted(run))
    assert got["case_a"] == str(run / _rel(_SC1))
    assert got["case_b"] == absolute


_JUNIT_FAILURE = """<?xml version="1.0"?>
<testsuites name="s" failures="1" errors="0" tests="1">
  <testsuite name="5" failures="1" errors="0" tests="1">
    <testcase name="scenario_0">
      <failure type="SimulationFailure" message="CustomCommandAction typed &quot;exitFailure&quot; \
was triggered by the anonymous Condition (OpenSCENARIO.Storyboard.Story[0]): Is [ego] \
colliding with Npc1?&#10;Unmet success conditions:&#10;  - &quot;goal_position&quot;" />
    </testcase>
  </testsuite>
</testsuites>
"""


def test_a_case_that_never_decided_is_not_a_failure(tmp_path):
    """Reaching a verdict and being cut off before one are different states.

    The row's ``result_kind`` cannot tell them apart: the interpreter presets it to a timeout
    failure, so a case that never decided still reads ``Failure``.
    """
    run = _make_run(tmp_path / "run", [_rel(_SC1), _rel(_SC1, case=6)])
    osp_out = run / key_aliases(_rel(_SC1))[0] / "osp_out"
    osp_out.mkdir(parents=True)
    (osp_out / "result.junit.xml").write_text(_JUNIT_FAILURE)
    export(run, tmp_path / "out")

    verdicts = [
        json.loads(line)["verdict"]
        for line in (tmp_path / "out" / "cases.jsonl").read_text().splitlines()
    ]
    decided = [v for v in verdicts if v["decided"]]
    assert [v for v in verdicts if not v["decided"]] == [{"decided": False}]
    assert decided[0]["kind"] == "Failure"
    assert decided[0]["unmet"] == ["goal_position"]

    entry = json.loads((tmp_path / "out" / "scenarios.json").read_text())[_SC1]
    assert entry["verdicts"] == {"pass": 0, "failure": 1, "error": 0, "undecided": 1}
    assert sum(entry["verdicts"].values()) == entry["n_cases"]


def test_the_suite_names_travel_as_the_suite_wrote_them(tmp_path):
    """The export copies the sidecar rather than reading it, so what a name means stays the
    reader's business."""
    export(_make_run(tmp_path / "run", [_rel(_SC1)]), tmp_path / "out")

    assert json.loads(ViewerTree(tmp_path / "out").names.read_text()) == _SIDECAR


def test_a_case_carries_what_its_expansion_set(tmp_path):
    """Cases are told apart by their parameters; naming them from those is the reader's job."""
    export(_make_run(tmp_path / "run", [_rel(_SC1)]), tmp_path / "out")

    out = tmp_path / "out"
    cases = [json.loads(ln) for ln in (out / "cases.jsonl").read_text().splitlines()]
    assert cases[0]["parameters"] == {"speed": "8.3"}
    entry = json.loads((out / "scenarios.json").read_text())[_SC1]
    assert entry["description"] == _DESCRIPTION
