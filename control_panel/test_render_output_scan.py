import json

from control_panel.app import _scan_outputs


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")


def test_scan_outputs_keeps_distinct_model_lora_runs(tmp_path):
    run_a = tmp_path / "scene_a__j6_base__run_a-lora_epoch_010"
    run_b = tmp_path / "scene_a__j6_base__run_b-lora_epoch_010"
    for run, lora_label in (
        (run_a, "run_a/lora_epoch_010"),
        (run_b, "run_b/lora_epoch_010"),
    ):
        run.mkdir(parents=True)
        (run / "render_meta.json").write_text(
            json.dumps(
                {
                    "model_label": "j6_base/best_model.pth",
                    "lora_label": lora_label,
                }
            )
        )
        _touch(run / "clip.webm")
        _touch(run / "00000.png")
        _touch(run / "00001.png")

    webms, stills, msg = _scan_outputs(str(tmp_path), one_still_per_scene=True)

    assert len(webms) == 2
    assert len(stills) == 2
    assert "2 webm, 2 scene still" in msg


def test_scan_outputs_filters_with_metadata_and_path_fallback(tmp_path):
    tagged = tmp_path / "scene_a__new"
    tagged.mkdir()
    (tagged / "render_meta.json").write_text(
        json.dumps(
            {
                "model_label": "j6_base/best_model.pth",
                "lora_label": "20260626-144944_lowlr-codex/lora_epoch_010",
            }
        )
    )
    _touch(tagged / "clip.webm")
    _touch(tagged / "00000.png")

    legacy = tmp_path / "legacy__j6_base__lora_epoch_009"
    _touch(legacy / "clip.webm")
    _touch(legacy / "00000.png")

    webms, stills, msg = _scan_outputs(
        str(tmp_path),
        one_still_per_scene=True,
        model_filter="j6_base",
        lora_filter="lowlr-codex",
    )

    assert webms == [str(tagged / "clip.webm")]
    assert stills == [str(tagged / "00000.png")]
    assert "model contains 'j6_base'" in msg
    assert "lora contains 'lowlr-codex'" in msg

    webms, stills, _ = _scan_outputs(
        str(tmp_path),
        one_still_per_scene=True,
        model_filter="j6_base",
        lora_filter="lora_epoch_009",
    )

    assert webms == [str(legacy / "clip.webm")]
    assert stills == [str(legacy / "00000.png")]
