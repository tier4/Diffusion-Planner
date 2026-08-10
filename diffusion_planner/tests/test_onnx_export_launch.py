"""Subprocess-based ONNX export: command building, artifact copying and the
detached launch, so the training loop never stalls behind the CPU trace and a
save_utd + best checkpoint is only traced once."""

from pathlib import Path

from diffusion_planner.utils import onnx_export as oe


def test_build_export_command_round_trips_through_parser():
    cmd = oe.build_export_command(
        config_json_path="/tmp/args.json",
        ckpt_path="/tmp/best_model.pth",
        output_dirs=[Path("/tmp/epoch0010"), Path("/tmp/best_model")],
        output_prefix="diffusion_planner",
        use_ema=False,
        use_simplify=False,
        opset_version=20,
        external_data=False,
    )
    # sys.executable -m diffusion_planner.utils.onnx_export <args...>
    assert cmd[1:3] == ["-m", "diffusion_planner.utils.onnx_export"]

    args = oe._build_arg_parser().parse_args(cmd[3:])
    assert args.config_json == "/tmp/args.json"
    assert args.ckpt == "/tmp/best_model.pth"
    assert args.output_dirs == ["/tmp/epoch0010", "/tmp/best_model"]
    assert args.output_prefix == "diffusion_planner"
    assert args.opset_version == 20
    assert args.use_ema is False
    assert args.use_simplify is False
    assert args.external_data is False


def test_copy_exported_artifacts_copies_all_graphs(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    names = [
        "diffusion_planner.onnx",
        "diffusion_planner_encoder.onnx",
        "diffusion_planner_decoder.onnx",
        "diffusion_planner_turn_indicator.onnx",
        "diffusion_planner.onnx.data",  # external-data sidecar
    ]
    for name in names:
        (src / name).write_bytes(b"onnx")
    (src / "unrelated.txt").write_text("keep out")

    dst1 = tmp_path / "dst1"
    dst2 = tmp_path / "dst2"  # not pre-created on purpose
    oe.copy_exported_artifacts(src, [dst1, dst2], "diffusion_planner")

    for dst in [dst1, dst2]:
        assert sorted(p.name for p in dst.iterdir()) == sorted(names)


def test_main_exports_once_then_copies(tmp_path, monkeypatch):
    calls = []

    def fake_export(**kwargs):
        calls.append(kwargs)
        (Path(kwargs["output_dir"]) / "diffusion_planner.onnx").write_bytes(b"onnx")

    monkeypatch.setattr(oe, "export_checkpoint_onnx", lambda **kw: fake_export(**kw))

    primary = tmp_path / "epoch0010"
    secondary = tmp_path / "best_model"
    primary.mkdir()
    oe.main(
        [
            "--config-json",
            str(tmp_path / "args.json"),
            "--ckpt",
            str(tmp_path / "best_model.pth"),
            "--output-dir",
            str(primary),
            "--output-dir",
            str(secondary),
        ]
    )

    # exported exactly once, into the first dir
    assert len(calls) == 1
    assert calls[0]["output_dir"] == primary
    # copied to the second dir
    assert (secondary / "diffusion_planner.onnx").read_bytes() == b"onnx"


def test_launch_is_detached_and_cpu_only(tmp_path, monkeypatch):
    captured = {}

    class FakeProc:
        def poll(self):
            return None

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr(oe.subprocess, "Popen", fake_popen)

    out_dir = tmp_path / "epoch0010"
    out_dir.mkdir()
    proc = oe.launch_checkpoint_onnx_export(
        config_json_path=str(out_dir / "args.json"),
        ckpt_path=str(out_dir / "best_model.pth"),
        output_dirs=[out_dir],
        output_prefix="diffusion_planner",
        use_ema=False,
        use_simplify=False,
        opset_version=20,
        external_data=False,
    )

    assert isinstance(proc, FakeProc)
    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert (out_dir / "onnx_export.log").exists()


def test_launch_failure_returns_none(tmp_path, monkeypatch):
    def fake_popen(cmd, **kwargs):
        raise OSError("spawn failed")

    monkeypatch.setattr(oe.subprocess, "Popen", fake_popen)

    out_dir = tmp_path / "epoch0010"
    out_dir.mkdir()
    proc = oe.launch_checkpoint_onnx_export(
        config_json_path="c",
        ckpt_path="k",
        output_dirs=[out_dir],
        output_prefix="diffusion_planner",
        use_ema=False,
        use_simplify=False,
        opset_version=20,
        external_data=False,
    )
    assert proc is None
