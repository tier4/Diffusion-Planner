"""Every entrypoint that trains with frenet must honour the --frenet_* flags.

The regression this pins: the GRPO entrypoint built the augmenter with only
``augment_prob`` and ``device``, so it accepted ``--frenet_dy_max 0.25`` and then trained at
the 2.0 default. A run's recorded configuration must be the configuration it used.
"""

import ast
import pathlib
from types import SimpleNamespace

from diffusion_planner.utils.data_augmentation_frenet import frenet_augmenter_from_args

REPO = pathlib.Path(__file__).resolve().parents[2]
ENTRYPOINTS = (
    REPO / "diffusion_planner" / "diffusion_planner" / "train.py",
    REPO / "diffusion_planner" / "train_grpo_predictor.py",
)


def _args(**over):
    base = dict(
        augment_prob=0.5,
        device="cpu",
        frenet_n_draws=16,
        frenet_dy_max=2.0,
        frenet_dth_max=0.17,
        frenet_merge_times=[2.0, 3.0, 4.0, 5.0],
        frenet_anchors=[2.0, 3.0],
        frenet_acc0_fracs=[0.0, -0.5, 0.5, -1.0, 1.0],
        frenet_seed=0,
        frenet_ranked_temp_s=1.0,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_defaults_round_trip():
    aug = frenet_augmenter_from_args(_args())
    assert aug.dy_max == 2.0
    assert aug.knobs.merge_times == (2.0, 3.0, 4.0, 5.0)


def test_custom_values_reach_the_augmenter():
    aug = frenet_augmenter_from_args(
        _args(frenet_dy_max=0.25, frenet_n_draws=4, frenet_seed=7, frenet_merge_times=[2.0])
    )
    assert aug.dy_max == 0.25
    assert aug.n_draws == 4
    assert aug.knobs.merge_times == (2.0,)


def test_string_lists_from_argparse_are_coerced():
    """argparse hands list fields back as strings; the factory owns the coercion."""
    aug = frenet_augmenter_from_args(
        _args(frenet_merge_times=["2", "3"], frenet_dy_max="1.5", frenet_n_draws="8")
    )
    assert aug.knobs.merge_times == (2.0, 3.0)
    assert isinstance(aug.dy_max, float) and aug.dy_max == 1.5
    assert aug.n_draws == 8


def test_no_entrypoint_constructs_the_augmenter_directly():
    """Both entrypoints must go through the factory — a direct constructor call is how the
    GRPO path silently dropped the flags in the first place."""
    for path in ENTRYPOINTS:
        tree = ast.parse(path.read_text())
        direct = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "FrenetStatePerturbationTensor"
        ]
        assert not direct, f"{path.name} constructs the augmenter directly; use the factory"
