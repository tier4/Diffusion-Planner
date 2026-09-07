"""Verify closed_loop_* fields in TrainConfig are properly defined."""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_closed_loop_npz_root_is_list():
    """closed_loop_npz_root must be list (nargs='+') in CLI, not str."""
    from diffusion_planner.config import GRPOConfig, build_parser

    gp = build_parser(GRPOConfig, "test")
    for action in gp._actions:
        if action.dest == "closed_loop_npz_root":
            assert action.nargs == "+", (
                f"closed_loop_npz_root nargs={action.nargs!r}, should be '+' (list) — "
                "this is kosuke55 bug #2"
            )
            return
    pytest.fail("closed_loop_npz_root not found in GRPOConfig parser")


def test_closed_loop_config_fields_cli_marked():
    """Verify closed_loop_* fields are cli-marked in TrainConfig."""
    from diffusion_planner.config import TrainConfig

    cli_field_names = {
        f.name for f in TrainConfig.__dataclass_fields__.values() if f.metadata.get("cli")
    }

    expected_cli_fields = [
        "closed_loop_npz_root",
    ]

    for field_name in expected_cli_fields:
        assert field_name in cli_field_names, (
            f"{field_name} should be marked with cli() in TrainConfig"
        )


def test_train_predictor_uses_config_build_parser():
    """Verify train_predictor.py uses config.build_parser."""
    train_predictor = _REPO_ROOT / "diffusion_planner" / "train_predictor.py"
    source = train_predictor.read_text(encoding="utf-8")
    assert "from diffusion_planner.config import" in source
    assert "build_parser" in source


def test_resolve_closed_loop_defaults_modes_to_objects(tmp_path):
    """Omitting --closed_loop_object_modes ⇒ every input gets mode 'objects'.

    Smoke test for the position-based pairing contract: callers don't have to
    spell out ``objects`` N times when they only want one mode.
    """
    import importlib.util
    import json
    import sys

    rcl_path = _REPO_ROOT / "diffusion_planner" / "run_all_groups_closed_loop.py"
    spec = importlib.util.spec_from_file_location("run_all_groups_closed_loop", rcl_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    j1 = tmp_path / "a.json"
    j1.write_text(json.dumps({"x": ["/x"]}))
    j2 = tmp_path / "b.json"
    j2.write_text(json.dumps({"y": ["/y"]}))

    entries = mod.resolve_closed_loop_inputs([str(j1), str(j2)])
    assert [e["mode"] for e in entries] == ["objects", "objects"]


def test_resolve_closed_loop_duplicate_path_keeps_each_mode(tmp_path):
    """Same JSON listed twice ⇒ two entries, each with its own mode.

    This is the supported use case: run one dataset under multiple modes.
    Without this contract, ``sites.json objects sites.json noobj`` would
    silently dedupe to a single ``objects`` run.
    """
    import importlib.util
    import json
    import sys

    rcl_path = _REPO_ROOT / "diffusion_planner" / "run_all_groups_closed_loop.py"
    spec = importlib.util.spec_from_file_location("run_all_groups_closed_loop", rcl_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    j = tmp_path / "sites.json"
    j.write_text(json.dumps({"all": ["/data/s"]}))

    entries = mod.resolve_closed_loop_inputs([str(j), str(j)], modes=["objects", "noobj"])

    assert len(entries) == 2
    assert [e["mode"] for e in entries] == ["objects", "noobj"]
    assert entries[0]["name"] == entries[1]["name"] == "sites"
