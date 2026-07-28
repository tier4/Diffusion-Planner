import sys
from argparse import Namespace
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from train_run import build_optional_args, parse_args


def _required_args() -> list[str]:
    return [
        "--exp_name",
        "test",
        "--train_set_list",
        "train.json",
        "--valid_set_list",
        "valid.json",
    ]


def test_repeat_augmentation_flags_default_to_not_forwarded():
    args = parse_args(_required_args())

    assert args.force_aug_on_repeat is None
    assert args.repeat_aug_pool is None
    assert "--force_aug_on_repeat" not in build_optional_args(args)
    assert "--repeat_aug_pool" not in build_optional_args(args)


def test_repeat_augmentation_flags_are_parsed_and_forwarded(tmp_path):
    cluster_json = tmp_path / "clusters.json"
    cluster_json.write_text("{}")
    args = parse_args(
        [
            *_required_args(),
            "--cluster_json",
            str(cluster_json),
            "--force_aug_on_repeat",
            "True",
            "--repeat_aug_pool",
            "flip,state_perturbation",
        ]
    )

    optional = build_optional_args(args)

    assert args.force_aug_on_repeat is True
    assert optional[-4:] == [
        "--force_aug_on_repeat",
        "True",
        "--repeat_aug_pool",
        "flip,state_perturbation",
    ]


def test_explicit_false_is_forwarded():
    args = Namespace(
        resume_model_path=None,
        wandb_run_id=None,
        wandb_project_name=None,
        cluster_json=None,
        cluster_weight_alpha=None,
        force_aug_on_repeat=False,
        repeat_aug_pool=None,
    )

    assert build_optional_args(args) == ["--force_aug_on_repeat", "False"]


def test_multi_augmentation_enable_flags_are_forwarded():
    args = parse_args(
        [
            *_required_args(),
            "--use_flip_augment",
            "True",
            "--use_neighbor_dropout",
            "True",
            "--use_neighbor_noise",
            "True",
            "--use_turn_indicator_dropout",
            "True",
        ]
    )

    assert build_optional_args(args)[-8:] == [
        "--use_flip_augment",
        "True",
        "--use_neighbor_dropout",
        "True",
        "--use_neighbor_noise",
        "True",
        "--use_turn_indicator_dropout",
        "True",
    ]
