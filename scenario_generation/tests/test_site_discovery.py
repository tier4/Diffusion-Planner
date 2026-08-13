"""Unit tests for site_discovery's site/vehicle-type resolution."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scenario_generation.site_discovery import (
    discover_sites_from_json,
    discover_sites_with_vehicles_from_json,
)

# Layout: <root>/<project>/<area_map>/<split>/<date>/<time>
ROOT = "/data"

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_list(tmp_path: Path, entries: list[str]) -> Path:
    p = tmp_path / "path_list.json"
    p.write_text(json.dumps(entries))
    return p


def test_groups_multiple_roots_under_one_site(tmp_path: Path):
    entries = [
        f"{ROOT}/proj_a/site_1/manual/2026-01-01/10-00-00",
        f"{ROOT}/proj_a/site_1/manual/2026-01-02/11-00-00",
    ]
    path = _write_list(tmp_path, entries)
    sites = discover_sites_with_vehicles_from_json(path)
    assert set(sites) == {"site_1"}
    assert len(sites["site_1"]["npz_roots"]) == 2
    assert sites["site_1"]["project"] == "proj_a"
    assert sites["site_1"]["vehicle_type"] == "proj_a"  # no map -> falls back to project name


def test_resolves_vehicle_type_from_map(tmp_path: Path):
    entries = [
        f"{ROOT}/proj_a/site_1/manual/2026-01-01/10-00-00",
        f"{ROOT}/proj_b/site_2/manual/2026-01-01/10-00-00",
    ]
    path = _write_list(tmp_path, entries)
    vehicle_map = {"proj_a": "vehicle_x", "proj_b": "vehicle_x"}
    sites = discover_sites_with_vehicles_from_json(path, vehicle_map)
    assert sites["site_1"]["vehicle_type"] == "vehicle_x"
    assert sites["site_2"]["vehicle_type"] == "vehicle_x"


def test_falls_back_to_project_name_when_missing_from_map(tmp_path: Path, capsys):
    entries = [f"{ROOT}/proj_unknown/site_1/manual/2026-01-01/10-00-00"]
    path = _write_list(tmp_path, entries)
    sites = discover_sites_with_vehicles_from_json(path, {"proj_a": "vehicle_x"})
    assert sites["site_1"]["vehicle_type"] == "proj_unknown"
    assert "proj_unknown" in capsys.readouterr().err


def test_splits_site_name_collision_across_vehicle_types(tmp_path: Path, capsys):
    entries = [
        f"{ROOT}/proj_a/site_1/manual/2026-01-01/10-00-00",
        f"{ROOT}/proj_b/site_1/manual/2026-01-01/10-00-00",
    ]
    path = _write_list(tmp_path, entries)
    vehicle_map = {"proj_a": "vehicle_x", "proj_b": "vehicle_y"}
    sites = discover_sites_with_vehicles_from_json(path, vehicle_map)
    assert set(sites) == {"vehicle_x__site_1", "vehicle_y__site_1"}
    assert sites["vehicle_x__site_1"]["project"] == "proj_a"
    assert sites["vehicle_y__site_1"]["project"] == "proj_b"
    assert "site_1" in capsys.readouterr().err


def test_splits_three_way_vehicle_type_collision(tmp_path: Path, capsys):
    entries = [
        f"{ROOT}/proj_a/site_1/manual/2026-01-01/10-00-00",
        f"{ROOT}/proj_b/site_1/manual/2026-01-01/10-00-00",
        f"{ROOT}/proj_c/site_1/manual/2026-01-01/10-00-00",
    ]
    path = _write_list(tmp_path, entries)
    vehicle_map = {"proj_a": "vehicle_x", "proj_b": "vehicle_y", "proj_c": "vehicle_z"}
    sites = discover_sites_with_vehicles_from_json(path, vehicle_map)
    assert set(sites) == {"vehicle_x__site_1", "vehicle_y__site_1", "vehicle_z__site_1"}
    assert sites["vehicle_z__site_1"]["project"] == "proj_c"
    assert "site_1" not in sites  # the bare site name must not silently reappear


def test_legacy_wrapper_returns_only_npz_roots(tmp_path: Path):
    entries = [
        f"{ROOT}/proj_a/site_1/manual/2026-01-01/10-00-00",
        f"{ROOT}/proj_b/site_2/manual/2026-01-01/10-00-00",
    ]
    path = _write_list(tmp_path, entries)
    legacy = discover_sites_from_json(path)
    full = discover_sites_with_vehicles_from_json(path)
    assert set(legacy) == set(full)
    assert all(legacy[name] == full[name]["npz_roots"] for name in legacy)


@pytest.mark.parametrize(
    "filename",
    [
        "project_vehicle_map.json",
        "closed_loop_project_vehicle_map.json",
    ],
)
def test_project_vehicle_map_file_is_gitignored(filename: str):
    """The ``project_vehicle_map`` JSON is supplied at runtime, so it must stay untracked.

    Covers both ways the ``*project_vehicle_map*.json`` glob is reached: the bare name and a
    prefixed one.
    """
    try:
        result = subprocess.run(
            ["git", "check-ignore", "-q", filename],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:  # pragma: no cover - git is expected in dev/CI
        pytest.skip("git not available")
    assert result.returncode == 0, f"{filename} is not ignored (rc={result.returncode})"
