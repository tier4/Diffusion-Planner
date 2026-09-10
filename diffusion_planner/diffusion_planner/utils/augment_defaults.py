"""Per-augmenter defaults for knobs that are shared by name but not by history.

Lives in its own module because both :mod:`augmenter_factory` and
:mod:`data_augmentation_frenet` need it and the factory imports the frenet builder,
so a resolver in either of those would be a cycle.
"""

# `--ego_past_noise_std` unset means "the default for this augmenter", and the two
# supported augmenters disagree on purpose. Quintic has perturbed the RECORDED ego
# history at 0.1 since it was written. Frenet stays at 0.0, matching `tier4-main`,
# which hard-passes 0.0 with the reason in-line: frenet already rewrites the past
# kinematically from the perturbed polyline, so a multiplicative scale on top would
# perturb a history this augmenter synthesised rather than one it observed.
#
# The knob is still available to frenet -- `--ego_past_noise_std 0.1`, or the lateral
# `--ego_past_noise_mode jitter` -- it is simply off unless asked for.
#
# Measured before choosing it, so the 0.0 is a decision rather than an omission: on a
# 2-seed, 8-arm A/B (~1,113 perturbed closed-loop rollouts per arm, both trackers,
# replan intervals 1 and 3), 0.1 vs 0.0 moved recovery 28.3% -> 30.2% at replan 1 and
# 52.7% -> 52.2% at replan 3, with lost% 20.3 -> 19.8 and 7.7 -> 8.0. Every one of
# those is inside the seed spread (17.2 points on recovered% at replan 1; 1.0 on lost%
# at replan 3), so the evidence neither justifies nor forbids the perturbation. With
# nothing measured to gain, the default is the one that leaves a stock
# `--augment_type frenet` run byte-identical to main.
# No bridge entry on purpose: resolve_history_noise returns before resolving for
# bridge, so a lookup here can only come from a caller that bypassed it -- and that
# caller should get the KeyError this dict advertises, not a silent 0.0.
DEFAULT_PAST_NOISE_STD = {"quintic": 0.1, "frenet": 0.0}


def past_noise_std_for(args) -> float:
    """The history-noise std an augmenter should use, honouring an explicit flag.

    ``None`` (the unset default) resolves per augmenter; any number the caller passed
    wins for every augmenter, so a sweep over the flag still sweeps all of them.

    Idempotent, so it is safe to call after :func:`resolve_history_noise` has already
    written the resolved number back onto ``args``.

    Raises:
        KeyError: on an ``augment_type`` with no recorded default, which means a new
            augmenter was added without deciding what this knob means for it.
    """
    if args.ego_past_noise_std is not None:
        return float(args.ego_past_noise_std)
    try:
        return DEFAULT_PAST_NOISE_STD[args.augment_type]
    except KeyError:
        raise KeyError(
            f"no history-noise default recorded for augment_type={args.augment_type!r}. "
            f"Known: {sorted(DEFAULT_PAST_NOISE_STD)}. bridge is absent on purpose -- it "
            "does not perturb the ego history, and resolve_history_noise returns before "
            "reaching here, so this means the resolver was bypassed. A new augmenter "
            "needs an entry (or an early return) rather than a default it never chose."
        ) from None


def resolve_history_noise(args) -> None:
    """Record the EFFECTIVE history-noise value on ``args``, without destroying the flag.

    Call once at startup, before the config is serialized. Both trainers write
    ``args.json`` before building the augmenter, so a sentinel that only resolved
    inside the factory was recorded as null while the run trained with a number.

    The resolved value goes to ``ego_past_noise_std_effective`` and the raw flag is left
    exactly as the user gave it. Overwriting the flag in place looked simpler but broke
    round-tripping: ``run_lifelong_r2lpl_rounds`` reads a base run's ``args.json`` and
    re-feeds the recorded fields as explicit flags, so a quintic run recording 0.1 and a
    later round overriding ``augment_type`` to bridge died at startup on a flag the user
    never passed.

    ``None`` in the effective field means "no history perturbation is applied" -- either
    augmentation is off entirely, or the augmenter does not support the knob.

    Also the only place the mode and bridge rejections can live: both need to tell "the
    user passed a value" from "unset", which is what resolution would destroy.

    Idempotent.

    Raises:
        ValueError: on ``jitter`` with a non-frenet augmenter, or on an explicit
            ``--ego_past_noise_std`` with ``augment_type=bridge``.
    """
    if args.ego_past_noise_mode == "jitter" and args.augment_type != "frenet":
        raise ValueError(
            f"--ego_past_noise_mode jitter is only implemented for augment_type=frenet, "
            f"got {args.augment_type}; drop the flag or pick augment_type=frenet"
        )
    if args.augment_type == "bridge" and args.ego_past_noise_std is not None:
        raise ValueError(
            "--ego_past_noise_std is not supported by augment_type=bridge "
            "(the bridge augmenter does not perturb the ego history); "
            "drop the flag or pick another augment_type"
        )
    # Nothing is applied when augmentation is off or the augmenter has no such knob, so
    # the effective value stays None rather than recording a number no run ever used.
    if not args.use_data_augment or args.augment_type == "bridge":
        args.ego_past_noise_std_effective = None
        return
    args.ego_past_noise_std_effective = past_noise_std_for(args)
