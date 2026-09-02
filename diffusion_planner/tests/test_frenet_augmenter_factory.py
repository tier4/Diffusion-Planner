"""Every entrypoint that trains with frenet must honour the --frenet_* flags.

The regression this pins: the GRPO entrypoint built the augmenter with only
``augment_prob`` and ``device``, so it accepted ``--frenet_dy_max 0.25`` and then trained at
the 2.0 default. A run's recorded configuration must be the configuration it used.
"""

import ast
import pathlib
from types import SimpleNamespace

import pytest
from diffusion_planner.utils.data_augmentation_frenet import (
    FrenetStatePerturbationTensor,
    frenet_augmenter_from_args,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
# constructing any of these in an entrypoint is how a knob gets silently dropped
DIRECT_CONSTRUCTORS = {
    "FrenetStatePerturbationTensor",
    "BridgeStatePerturbation",
    "StatePerturbation",
}
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
        frenet_recovery_rounds=0,
        frenet_toward_parked_prob=0.0,
        frenet_min_clearance=0.0,
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
            and n.func.id in DIRECT_CONSTRUCTORS
        ]
        assert not direct, f"{path.name} constructs the augmenter directly; use the factory"


def test_recovery_rounds_defaults_to_off():
    """A vetoed scene falls back to plain GT unless recovery is explicitly asked for."""
    assert frenet_augmenter_from_args(_args()).recovery_rounds == 0


def test_recovery_rounds_reaches_the_augmenter():
    aug = frenet_augmenter_from_args(_args(frenet_recovery_rounds=2))
    assert aug.recovery_rounds == 2


def test_recovery_rounds_from_argparse_string_is_coerced():
    assert frenet_augmenter_from_args(_args(frenet_recovery_rounds="3")).recovery_rounds == 3


def test_negative_recovery_rounds_is_refused():
    with pytest.raises(ValueError, match="recovery_rounds"):
        FrenetStatePerturbationTensor(augment_prob=1.0, device="cpu", recovery_rounds=-1)


def test_factory_dispatches_every_augment_type():
    from diffusion_planner.utils.augmenter_factory import AUGMENT_TYPES, augmenter_from_args
    from diffusion_planner.utils.data_augmentation import StatePerturbation
    from diffusion_planner.utils.data_augmentation_bridge import (
        StatePerturbation as BridgeStatePerturbation,
    )

    def full(**over):
        return _args(
            use_data_augment=True,
            num_refine=20,
            ego_past_noise_std=0.1,
            use_smoothing_future_trajectory=False,
            **over,
        )

    assert set(AUGMENT_TYPES) == {"quintic", "bridge", "frenet"}
    got = {t: augmenter_from_args(full(augment_type=t)) for t in AUGMENT_TYPES}
    assert isinstance(got["frenet"], FrenetStatePerturbationTensor)
    assert isinstance(got["bridge"], BridgeStatePerturbation)
    assert type(got["quintic"]) is StatePerturbation


def test_factory_returns_none_when_augmentation_is_off():
    from diffusion_planner.utils.augmenter_factory import augmenter_from_args

    assert augmenter_from_args(_args(use_data_augment=False, augment_type="frenet")) is None


def test_factory_refuses_an_unknown_augment_type():
    """The ladder this replaced fell through to quintic; a silent substitution is worse
    than an error for a caller that asked for something else."""
    from diffusion_planner.utils.augmenter_factory import augmenter_from_args

    with pytest.raises(ValueError, match="unknown augment_type"):
        augmenter_from_args(_args(use_data_augment=True, augment_type="nope"))


def test_toward_parked_and_min_clearance_default_off():
    aug = frenet_augmenter_from_args(_args())
    assert aug.toward_parked_prob == 0.0 and aug.min_clearance == 0.0


def test_toward_parked_and_min_clearance_reach_the_augmenter():
    aug = frenet_augmenter_from_args(
        _args(frenet_toward_parked_prob="0.3", frenet_min_clearance="0.2")
    )
    assert aug.toward_parked_prob == 0.3 and aug.min_clearance == 0.2


def test_toward_parked_prob_outside_unit_interval_is_refused():
    with pytest.raises(ValueError, match="toward_parked_prob"):
        FrenetStatePerturbationTensor(augment_prob=1.0, device="cpu", toward_parked_prob=1.5)


def test_negative_min_clearance_is_refused():
    with pytest.raises(ValueError, match="min_clearance"):
        FrenetStatePerturbationTensor(augment_prob=1.0, device="cpu", min_clearance=-0.1)
