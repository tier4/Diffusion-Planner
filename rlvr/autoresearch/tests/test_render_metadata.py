from pathlib import Path

from rlvr.autoresearch.tools import render_metadata as M


def test_render_tag_includes_model_lora_and_unique_stamp(monkeypatch):
    monkeypatch.setattr(M, "run_stamp", lambda: "STAMP")

    tag = M.render_tag(
        Path("/workspace/models/j6_base/best_model.pth"),
        Path("/workspace/runs/train_ranked_sft/20260626-144944_lowlr-codex/lora_epoch_010"),
    )

    assert tag == "j6_base-best_model.pth__20260626-144944_lowlr-codex-lora_epoch_010__STAMP"


def test_write_render_meta_creates_parent_and_json(tmp_path):
    out = tmp_path / "render" / "scene"

    M.write_render_meta(out, model_label="j6_base/best_model.pth", lora_label="run/lora_010")

    meta_path = out / "render_meta.json"
    assert meta_path.exists()
    assert '"model_label": "j6_base/best_model.pth"' in meta_path.read_text()
