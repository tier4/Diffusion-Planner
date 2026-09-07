from pathlib import Path

import pytest
from diffusion_planner.config import (
    TrainConfig,
    build_config,
    build_parser,
    resolve_paths,
    to_command_line,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NORMALIZATION_JSON = _REPO_ROOT / "diffusion_planner" / "normalization.json"


def _load_run_all_groups():
    """Lazy-load run_all_groups_closed_loop for the cross-module pairing tests."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "run_all_groups_closed_loop",
        _REPO_ROOT / "diffusion_planner" / "run_all_groups_closed_loop.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_build_parser_required_and_defaults(tmp_path: Path):
    parser = build_parser(TrainConfig, "test parser")
    train_list = str(tmp_path / "train.json")
    valid_list = str(tmp_path / "valid.json")

    args = parser.parse_args(
        [
            "--exp_name",
            "test_run",
            "--train_set_list",
            train_list,
            "--valid_set_list",
            valid_list,
        ]
    )
    assert args.exp_name == "test_run"
    assert args.train_set_list == train_list
    assert args.valid_set_list == valid_list
    assert args.use_wandb is True
    assert args.closed_loop_draw_workers == 4
    assert args.scenario_sim_driver == ""
    assert args.scenario_based_open_loop_list == ""
    assert args.scenario_based_open_loop_only is False
    assert args.batch_size == 512
    assert args.train_epochs == 80
    assert args.save_utd == 10


def test_resolve_paths(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.json").write_text("[]")
    (tmp_path / "valid.json").write_text("[]")

    parser = build_parser(TrainConfig, "test parser")
    args = parser.parse_args(
        [
            "--exp_name",
            "test_run",
            "--train_set_list",
            "train.json",
            "--valid_set_list",
            "valid.json",
        ]
    )
    resolve_paths(args, TrainConfig)
    assert Path(args.train_set_list).is_absolute()
    assert Path(args.valid_set_list).is_absolute()
    assert args.train_set_list == str((tmp_path / "train.json").resolve())


def test_to_command_line(tmp_path: Path):
    parser = build_parser(TrainConfig, "test parser")
    train_list = str(tmp_path / "train.json")
    valid_list = str(tmp_path / "valid.json")

    args = parser.parse_args(
        [
            "--exp_name",
            "test_run",
            "--train_set_list",
            train_list,
            "--valid_set_list",
            valid_list,
            "--closed_loop_draw_workers",
            "8",
            "--scenario_sim_driver",
            "/tmp/driver.sh",
        ]
    )
    cmd = to_command_line(args, cls=TrainConfig, exclude=("output_root",))
    assert "--exp_name" in cmd
    assert "test_run" in cmd
    assert "--train_set_list" in cmd
    assert "--closed_loop_draw_workers" in cmd
    assert "8" in cmd
    assert "--scenario_sim_driver" in cmd
    assert "/tmp/driver.sh" in cmd
    assert "--use_wandb" not in cmd


def test_scenario_sim_validate_hook(tmp_path: Path, monkeypatch):
    """Test scenario_sim_validate contract: out-of-process invocation with CKPT/OUT env."""
    from types import SimpleNamespace

    from diffusion_planner.train import scenario_sim_validate

    # Case 1: Disabled when scenario_sim_driver is empty
    called_cmds = []

    def fake_run(cmd, env=None, **kwargs):
        called_cmds.append((cmd, env))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)

    args_disabled = SimpleNamespace(scenario_sim_driver="")
    scenario_sim_validate(args_disabled, epoch=0, ckpt_path="/path/ckpt.pth", out_dir="/path/out")
    assert len(called_cmds) == 0

    # Case 2: Invokes bash driver with CKPT and OUT in env
    args_enabled = SimpleNamespace(scenario_sim_driver="/opt/run_suite.sh")
    scenario_sim_validate(args_enabled, epoch=4, ckpt_path="/path/ckpt.pth", out_dir="/path/out")
    assert len(called_cmds) == 1
    cmd, env = called_cmds[0]
    assert cmd == ["bash", "/opt/run_suite.sh"]
    assert env["CKPT"] == "/path/ckpt.pth"
    assert env["OUT"] == "/path/out"


def test_build_config(tmp_path: Path):
    parser = build_parser(TrainConfig, "test parser")
    train_list = str(tmp_path / "train.json")
    valid_list = str(tmp_path / "valid.json")

    args = parser.parse_args(
        [
            "--exp_name",
            "test_run",
            "--train_set_list",
            train_list,
            "--valid_set_list",
            valid_list,
        ]
    )
    config = build_config(
        TrainConfig,
        args,
        num_workers=2,
        normalization_file_path=str(_NORMALIZATION_JSON),
    )
    assert isinstance(config, TrainConfig)
    assert config.exp_name == "test_run"
    assert config.train_set_list == train_list
    assert config.num_workers == 2
    assert config.save_dir != ""
    assert config.state_normalizer is None
    assert config.observation_normalizer is None


def test_resolve_closed_loop_duplicate_path_keeps_each_mode(tmp_path: Path, monkeypatch):
    """Same JSON listed twice with [objects, noobj] ⇒ 2 entries, each mode preserved.

    End-to-end check that duplicate JSON paths (used to run the same dataset under
    multiple modes) propagate through ``resolve_closed_loop_inputs`` without
    dedupe. Without this, ``--closed_loop_object_modes sites.json objects sites.json noobj``
    would silently collapse to a single ``objects`` run.
    """
    import json
    from types import SimpleNamespace

    mod = _load_run_all_groups()

    json_path = tmp_path / "sites.json"
    json_path.write_text(json.dumps({"alpha": ["/data/a"]}))

    captured = []

    def fake_run_one_group(model, model_args, npz_paths, out_dir, cfg, **kwargs):
        captured.append({"mode": kwargs.get("mode"), "out_dir": str(out_dir)})

    monkeypatch.setattr(mod, "run_one_group", fake_run_one_group)
    monkeypatch.setattr(mod, "log_closed_loop_to_wandb", lambda *a, **k: None)

    cfg = SimpleNamespace(
        closed_loop_npz_root=[str(json_path), str(json_path)],
        closed_loop_object_modes=["objects", "noobj"],
        closed_loop_near_miss_thresh=0.5,
        closed_loop_search_radius=1.5,
        closed_loop_warmup_steps=0,
        closed_loop_unstick_after=300,
        closed_loop_unstick_advance_m=5.0,
        closed_loop_unstick_radius_mult=10.0,
        closed_loop_unstick_teleport_after=300,
        closed_loop_draw_every=4,
        closed_loop_replan_interval=1,
        closed_loop_abort_deviation_m=50.0,
        closed_loop_abort_after=30,
        closed_loop_abort_max_snaps=0,
        closed_loop_fps=10,
        closed_loop_seg_len=100000,
        closed_loop_draw_workers=2,
        closed_loop_wandb_video_pick="worst",
        closed_loop_colormap_metrics=[],
        render_media=False,
        device="cpu",
        ddp=False,
        wandb_project_name="",
    )

    ok = mod.run_closed_loop_main(
        model=None,
        model_args=None,
        cfg=cfg,
        out_root=tmp_path,
        wandb_run=None,
        only_json=None,
        render_media=False,
    )
    assert ok is True
    by_mode = {c["mode"]: c["out_dir"] for c in captured}
    assert by_mode == {
        "objects": str(tmp_path / "sites" / "alpha"),
        "noobj": str(tmp_path / "sites__noobj" / "alpha"),
    }
