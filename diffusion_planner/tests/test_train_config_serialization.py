"""Tests for the training configuration JSON contract."""

import json

from diffusion_planner.config import TrainConfig


def test_train_config_runtime_fields_are_not_saved_as_json(tmp_path):
    """The closed-loop pass-condition cache must not break args.json writing."""
    cfg = TrainConfig(
        exp_name="serialization_test",
        save_dir=str(tmp_path),
    )

    args_dict = {
        key: value
        for key, value in vars(cfg).items()
        if not key.startswith("_")
    }
    args_dict["major_version"] = 5

    output = tmp_path / "args.json"
    output.write_text(json.dumps(args_dict, indent=4), encoding="utf-8")

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert "_closed_loop_pass_conditions_loaded" not in saved
    assert saved["major_version"] == 5
