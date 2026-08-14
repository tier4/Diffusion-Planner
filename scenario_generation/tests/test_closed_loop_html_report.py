"""Unit tests for the closed-loop HTML report's vehicle_type field/filter."""

from __future__ import annotations

import json
from pathlib import Path

from scenario_generation.closed_loop_html_report import build_html_report, collect_site_data

_MINIMAL_SUMMARY = {"n_segments": 1, "mean_route_completion": 0.9}
_MINIMAL_SEGMENT = {
    "segment": [0, 1],
    "route": "route_0",
    "n_steps_run": 1,
    "terminated": "completed",
    "route_completion": 0.9,
    "progress_m": 1.0,
}


def _make_site(out_root: Path, site_name: str) -> None:
    d = out_root / site_name
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps(_MINIMAL_SUMMARY))
    (d / "segments.jsonl").write_text(json.dumps(_MINIMAL_SEGMENT) + "\n")


def test_collect_site_data_adds_vehicle_type(tmp_path: Path):
    _make_site(tmp_path, "site_a")
    items, summaries = collect_site_data(
        tmp_path, ["site_a"], site_vehicle_types={"site_a": "vehicle_x"}
    )
    assert summaries[0]["vehicle_type"] == "vehicle_x"
    assert items[0]["vehicle_type"] == "vehicle_x"


def test_collect_site_data_without_map_is_empty_string(tmp_path: Path):
    _make_site(tmp_path, "site_a")
    items, summaries = collect_site_data(tmp_path, ["site_a"])
    assert summaries[0]["vehicle_type"] == ""
    assert items[0]["vehicle_type"] == ""


def test_noobj_label_resolves_base_site_vehicle_type(tmp_path: Path):
    _make_site(tmp_path, "site_a__noobj")
    items, summaries = collect_site_data(
        tmp_path, ["site_a__noobj"], site_vehicle_types={"site_a": "vehicle_x"}
    )
    assert summaries[0]["vehicle_type"] == "vehicle_x"
    assert items[0]["vehicle_type"] == "vehicle_x"


def test_build_html_report_embeds_vehicle_filter(tmp_path: Path):
    _make_site(tmp_path, "site_a")
    _make_site(tmp_path, "site_b")
    report_path = build_html_report(
        tmp_path,
        ["site_a", "site_b"],
        site_vehicle_types={"site_a": "vehicle_x", "site_b": "vehicle_y"},
    )
    html = report_path.read_text(encoding="utf-8")
    assert 'id="vehicleFilter"' in html
    assert '"vehicle_type": "vehicle_x"' in html
    assert '"vehicle_type": "vehicle_y"' in html
