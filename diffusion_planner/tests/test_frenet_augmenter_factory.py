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
        ego_past_noise_std=0.0,
        frenet_hist_jitter_lat=0.0,
        frenet_hist_jitter_lon=0.0,
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


def test_frenet_reads_the_shared_ego_past_noise_std():
    """Frenet honours the same flag as quintic -- there is no separate frenet knob.

    The base class is still handed 0.0: it would scale the RECORDED history, which
    frenet discards. The value reaches frenet's own scaling of the rewritten history.
    """
    assert frenet_augmenter_from_args(_args(ego_past_noise_std="0.1")).past_noise_std == 0.1
    aug = frenet_augmenter_from_args(_args(ego_past_noise_std="0.0"))
    assert aug.past_noise_std == 0.0
    assert aug._ego_past_noise_std == 0.0, "the recorded history must never be scaled"


def test_hist_jitter_defaults_off_and_reaches_the_augmenter():
    plain = frenet_augmenter_from_args(_args())
    assert plain.hist_jitter_lat == 0.0 and plain.hist_jitter_lon == 0.0
    aug = frenet_augmenter_from_args(
        _args(frenet_hist_jitter_lat="0.25", frenet_hist_jitter_lon="0.1")
    )
    assert aug.hist_jitter_lat == 0.25 and aug.hist_jitter_lon == 0.1


def test_negative_past_noise_std_is_refused():
    with pytest.raises(ValueError, match="past_noise_std"):
        FrenetStatePerturbationTensor(augment_prob=1.0, device="cpu", ego_past_noise_std=-0.1)


def test_seed_and_past_noise_std_are_real_command_line_flags():
    """Both were plain dataclass fields, so the parser rejected them outright.

    The A/B they are needed for cannot run without them: ``--ego_past_noise_std 0`` is
    the no-history-noise control, and ``--seed`` is the only way to measure the
    run-to-run noise floor of an otherwise deterministic training.
    """
    from diffusion_planner.config import TrainConfig, build_config, build_parser

    parser = build_parser(TrainConfig, description="t")
    base = build_config(TrainConfig, parser.parse_args([]))
    assert (base.seed, base.ego_past_noise_std) == (3407, 0.1), "a default moved"
    assert (base.lr_schedule, base.augment_type) == ("constant", "quintic"), "a default moved"

    over = build_config(
        TrainConfig, parser.parse_args(["--seed", "1234", "--ego_past_noise_std", "0.0"])
    )
    assert over.seed == 1234
    assert over.ego_past_noise_std == 0.0


def test_every_arg_the_factory_reads_exists_on_the_real_config():
    """Every ``args.<name>`` the augmenter factories read must be a real TrainConfig field.

    The other tests here build a SimpleNamespace and hand-supply the attributes, so a
    field deleted from TrainConfig still "passes" while every real training crashes at
    startup with AttributeError. That happened: `frenet_recovery_rounds` was removed by
    an edit to a neighbouring field and nothing caught it until a training was launched.
    This reads the attribute names straight out of the source and checks them against
    the dataclass, so the two cannot drift again.
    """
    import ast
    import re
    from dataclasses import fields
    from pathlib import Path

    from diffusion_planner.config.train_config import TrainConfig

    declared = {f.name for f in fields(TrainConfig)}
    root = Path(__file__).resolve().parents[1] / "diffusion_planner" / "utils"
    read: set[str] = set()
    for src in ("data_augmentation_frenet.py", "augmenter_factory.py"):
        tree = ast.parse((root / src).read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "args"
            ):
                read.add(node.attr)
    assert read, "found no args.* reads — the AST walk is broken, not the config"
    missing = sorted(read - declared)
    assert not missing, f"read from args but not declared on TrainConfig: {missing}"
