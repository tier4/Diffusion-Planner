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
        ego_past_noise_mode="scale",
        use_data_augment=True,
        # the builder resolves the effective value itself, which needs these two
        augment_type="frenet",
    )
    base.update(over)
    # Deliberately NOT pre-populating ego_past_noise_std_effective. Doing so hid a
    # regression where the builder read that field unresolved and raised TypeError on
    # any freshly parsed config; the builder resolves for itself, so the stand-in must
    # arrive unresolved exactly as a real parsed config does.
    base.setdefault("use_data_augment", True)
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
        # ego_past_noise_std unset: bridge rejects it outright and the other two resolve
        # it per augmenter, so None is the only value valid for all three.
        return _args(
            use_data_augment=True,
            num_refine=20,
            ego_past_noise_std=None,
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


def test_the_noise_mode_defaults_to_scale_and_reaches_the_augmenter():
    """One magnitude, one mode. The two mechanisms are mutually exclusive."""
    plain = frenet_augmenter_from_args(_args())
    assert plain.past_noise_mode == "scale"
    jit = frenet_augmenter_from_args(_args(ego_past_noise_mode="jitter", ego_past_noise_std="0.3"))
    assert jit.past_noise_mode == "jitter" and jit.past_noise_std == 0.3


def test_an_unknown_noise_mode_is_refused():
    with pytest.raises(ValueError, match="ego_past_noise_mode"):
        frenet_augmenter_from_args(_args(ego_past_noise_mode="wobble"))


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
    assert (base.seed, base.ego_past_noise_std) == (3407, None), "a default moved"
    assert (base.lr_schedule, base.augment_type) == ("constant", "quintic"), "a default moved"

    # The sentinel is only half the contract; what it RESOLVES to is the thing that
    # reaches a training, and that is what silently moved once already.
    from diffusion_planner.utils.augment_defaults import past_noise_std_for

    for augment_type, expected in (("quintic", 0.1), ("frenet", 0.0)):
        resolved = past_noise_std_for(
            build_config(TrainConfig, parser.parse_args(["--augment_type", augment_type]))
        )
        assert resolved == expected, f"{augment_type} history noise moved to {resolved}"

    # bridge has no entry on purpose: resolve_history_noise returns before resolving for
    # it, so reaching this lookup means a caller bypassed the resolver, and that caller
    # should get a KeyError rather than a silent 0.0 for a knob bridge cannot honour.
    with pytest.raises(KeyError):
        past_noise_std_for(
            build_config(TrainConfig, parser.parse_args(["--augment_type", "bridge"]))
        )

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


# ─────────── the CLI default is the one that ships; test THAT, not the class ───────────
#
# Every behavioural test builds the augmenter directly, where the class default for the
# history noise is 0.0. The value a training actually gets comes from the parser, and the
# two diverged once already: the flag was threaded to frenet and picked up quintic's 0.1,
# silently changing the frenet recipe and leaving existing frenet checkpoints
# unreproducible at their own seed. These go through the real parser for that reason.


def _from_cli(*argv):
    """Build an augmenter the way a training does: real parser, real config, real factory.

    ``device`` is forced to CPU afterwards. ``TrainConfig.device`` defaults to "cuda", so
    without this the augmenter's constructor raises ``RuntimeError: No CUDA GPUs are
    available`` on a CPU-only host and these tests never reach their assertions. The
    device is not what is under test here -- the flag resolution is -- and every other
    test in this file constructs on CPU for the same reason.
    """
    from diffusion_planner.config.config_cli import build_config, build_parser
    from diffusion_planner.config.train_config import TrainConfig
    from diffusion_planner.utils.augmenter_factory import augmenter_from_args

    args = build_config(TrainConfig, build_parser(TrainConfig).parse_args(list(argv)))
    args.device = "cpu"
    return augmenter_from_args(args)


def _past_noise(aug):
    return getattr(aug, "past_noise_std", getattr(aug, "_ego_past_noise_std", None))


def test_frenet_does_not_inherit_quintics_history_noise():
    """Frenet is 0.0, as on tier4-main, so a stock frenet run is unchanged by this branch.

    The two defaults diverge deliberately: quintic scales a RECORDED history, frenet one
    it rewrote kinematically from the perturbed polyline. This is the assertion that
    keeps a shared flag name from quietly merging the two recipes -- it did exactly that
    once, which left existing frenet checkpoints unreproducible at their own seed.
    Measured either way: 0.1 sat inside the seed spread on an 8-arm A/B, so nothing is
    given up by defaulting it off.
    """
    assert _past_noise(_from_cli("--augment_type", "frenet")) == 0.0


def test_frenet_history_noise_is_still_reachable():
    """Off by default is not gone: the knob still applies to frenet when asked for."""
    assert _past_noise(_from_cli("--augment_type", "frenet", "--ego_past_noise_std", "0.1")) == 0.1


def test_quintic_keeps_the_history_noise_it_has_always_had():
    assert _past_noise(_from_cli("--augment_type", "quintic")) == 0.1


@pytest.mark.parametrize("augment_type,value", [("frenet", 0.3), ("quintic", 0.0)])
def test_an_explicit_flag_wins_for_either_augmenter(augment_type, value):
    """The per-augmenter default must not make the flag un-sweepable."""
    aug = _from_cli("--augment_type", augment_type, "--ego_past_noise_std", str(value))
    assert _past_noise(aug) == value


def test_bridge_refuses_a_flag_it_cannot_honour():
    """Accepting it would record a value in args.json that the run never applied --
    the exact recorded-config-is-not-used-config defect the factory exists to stop."""
    with pytest.raises(ValueError, match="not supported by augment_type=bridge"):
        _from_cli("--augment_type", "bridge", "--ego_past_noise_std", "0.4")


def test_bridge_is_fine_without_the_flag():
    assert _from_cli("--augment_type", "bridge") is not None


# ───────── the SAVED configuration must be the configuration APPLIED ─────────
#
# Both trainers write args.json before they build the augmenter, so a sentinel default
# that only resolves inside the factory is recorded as null while the run trains with a
# number. That makes the saved experiment configuration inaccurate and its meaning
# dependent on a future code default rather than on the file.


def _resolved_config(*argv):
    from diffusion_planner.config.config_cli import build_config, build_parser
    from diffusion_planner.config.train_config import TrainConfig
    from diffusion_planner.utils.augment_defaults import resolve_history_noise

    args = build_config(TrainConfig, build_parser(TrainConfig).parse_args(list(argv)))
    args.device = "cpu"
    resolve_history_noise(args)  # what the trainers call before serializing
    return args


@pytest.mark.parametrize("augment_type", ["frenet", "quintic"])
def test_saved_config_equals_applied_config(augment_type):
    """Serialize the way the trainers do, then build the augmenter, and compare."""
    import json

    from diffusion_planner.utils.augmenter_factory import augmenter_from_args

    args = _resolved_config("--augment_type", augment_type)
    saved = json.loads(
        json.dumps({"ego_past_noise_std_effective": args.ego_past_noise_std_effective})
    )
    applied = _past_noise(augmenter_from_args(args))

    assert saved["ego_past_noise_std_effective"] is not None, "args.json would record null"
    assert saved["ego_past_noise_std_effective"] == applied, (
        f"{augment_type}: args.json records {saved['ego_past_noise_std_effective']} "
        f"but the augmenter trains with {applied}"
    )


def test_the_resolved_value_survives_a_json_round_trip():
    """A reader of args.json must get the effective number, not a sentinel."""
    import json

    args = _resolved_config("--augment_type", "quintic")
    assert json.loads(json.dumps(args.ego_past_noise_std_effective)) == 0.1


def test_an_explicit_zero_is_recorded_as_zero_not_as_unset():
    """A deliberate 0 must be distinguishable in the saved config from an unset flag."""
    args = _resolved_config("--augment_type", "quintic", "--ego_past_noise_std", "0")
    assert args.ego_past_noise_std_effective == 0.0


@pytest.mark.parametrize(
    "augment_type,expected", [("quintic", 0.1), ("frenet", 0.0), ("bridge", None)]
)
def test_resolving_twice_changes_nothing(augment_type, expected):
    """The factory calls the resolver defensively, so it runs twice in a real run.

    Covers BRIDGE, which is the case that broke: resolving an omitted bridge value to
    0.0 made the second call mistake it for an explicitly supplied unsupported flag and
    raise, so every ordinary bridge run failed. An unset bridge value stays None.
    """
    from diffusion_planner.utils.augment_defaults import resolve_history_noise

    args = _resolved_config("--augment_type", augment_type)
    resolve_history_noise(args)
    resolve_history_noise(args)
    assert args.ego_past_noise_std_effective == expected


@pytest.mark.parametrize("augment_type", ["quintic", "frenet", "bridge"])
def test_the_exact_production_ordering_builds_an_augmenter(augment_type):
    """resolve -> serialize -> factory (which resolves again) -> build, for all three.

    The ordering a real trainer uses. Asserting only "it builds" is the point: this is
    the path that raised for bridge.
    """
    from diffusion_planner.utils.augmenter_factory import augmenter_from_args

    args = _resolved_config("--augment_type", augment_type)
    assert augmenter_from_args(args) is not None


def test_bridge_still_rejects_the_flag_at_resolve_time():
    """The rejection moved to the resolver; it must not have been lost in the move."""
    with pytest.raises(ValueError, match="not supported by augment_type=bridge"):
        _resolved_config("--augment_type", "bridge", "--ego_past_noise_std", "0.4")
    # an unset bridge value stays None: the knob does not apply to bridge, and writing a
    # number here is what made the resolver's second call reject every bridge run
    assert _resolved_config("--augment_type", "bridge").ego_past_noise_std_effective is None


@pytest.mark.parametrize(
    "argv",
    [
        ["--augment_type", "frenet"],
        ["--augment_type", "frenet", "--ego_past_noise_std", "0.3"],
        [
            "--augment_type",
            "frenet",
            "--ego_past_noise_std",
            "0.3",
            "--ego_past_noise_mode",
            "jitter",
        ],
    ],
)
def test_the_direct_builder_works_on_a_freshly_parsed_config(argv):
    """frenet_augmenter_from_args is a public entrypoint, not only reached via the factory.

    It briefly read `ego_past_noise_std_effective` -- which is None until something
    resolves it -- and so raised TypeError on any parsed config, even one carrying an
    explicit --ego_past_noise_std. Nothing caught it because the namespace fixture in
    this file pre-populated that field; it no longer does.
    """
    from diffusion_planner.config.config_cli import build_config, build_parser
    from diffusion_planner.config.train_config import TrainConfig
    from diffusion_planner.utils.data_augmentation_frenet import frenet_augmenter_from_args

    args = build_config(TrainConfig, build_parser(TrainConfig).parse_args(argv))
    args.device = "cpu"
    aug = frenet_augmenter_from_args(args)  # no separate resolve step
    assert aug.past_noise_std is not None
    if "--ego_past_noise_std" in argv:
        assert aug.past_noise_std == float(argv[argv.index("--ego_past_noise_std") + 1])
