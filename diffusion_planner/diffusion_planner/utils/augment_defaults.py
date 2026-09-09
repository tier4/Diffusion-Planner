"""Per-augmenter defaults for knobs that are shared by name but not by history.

Lives in its own module because both :mod:`augmenter_factory` and
:mod:`data_augmentation_frenet` need it and the factory imports the frenet builder,
so a resolver in either of those would be a cycle.
"""

# `--ego_past_noise_std` unset means "the default for this augmenter". Quintic has
# perturbed the recorded history at 0.1 since it was written. Frenet now matches it.
# Bridge has no history perturbation at all, so it has no default to speak of and
# rejects the flag rather than accepting and ignoring it.
#
# The frenet 0.1 is a DELIBERATE DEFAULT CHANGE, not an inherited value. `tier4-main`
# hard-passed 0.0 to the frenet augmenter, so a stock `--augment_type frenet` run on
# this branch trains a different distribution than the same command on main, and
# reproducing an existing frenet checkpoint needs `--ego_past_noise_std 0` explicitly.
#
# Measured before choosing it: on a 2-seed, 8-arm A/B (~1,113 perturbed closed-loop
# rollouts per arm, both trackers, replan intervals 1 and 3), 0.1 vs 0.0 moved recovery
# 28.3% -> 30.2% at replan 1 and 52.7% -> 52.2% at replan 3, with lost% 20.3 -> 19.8 and
# 7.7 -> 8.0. Every one of those is inside the seed spread (17.2 points on recovered% at
# replan 1; 1.0 on lost% at replan 3), so the evidence says the perturbation neither
# helps nor hurts. It is on by default because the flag is uniform across augmenters
# that support it, NOT because it was shown to improve anything.
DEFAULT_PAST_NOISE_STD = {"quintic": 0.1, "bridge": 0.0, "frenet": 0.1}


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
    return DEFAULT_PAST_NOISE_STD[args.augment_type]


def resolve_history_noise(args) -> None:
    """Write the EFFECTIVE history-noise value onto ``args``, in place.

    Call this once at startup, **before the config is serialized**. Both trainers write
    ``args.json`` before they build the augmenter, so without this a stock frenet run
    records ``ego_past_noise_std: null`` while training with 0.1 -- a saved experiment
    configuration that is not the configuration used, and one whose meaning depends on
    a future code default rather than on the file. That is the defect the augmenter
    factory exists to prevent, one level up.

    Also the only place the bridge rejection can live. The check needs to distinguish
    "the user passed a value" from "unset", which is exactly the information this
    function destroys, so it has to run first.

    Idempotent: a second call is a no-op, since the value is then already a float.

    Raises:
        ValueError: if ``--ego_past_noise_std`` was passed with an ``augment_type``
            that cannot honour it.
    """
    if args.ego_past_noise_mode == "jitter" and args.augment_type != "frenet":
        raise ValueError(
            f"--ego_past_noise_mode jitter is only implemented for augment_type=frenet, "
            f"got {args.augment_type}; drop the flag or pick augment_type=frenet"
        )
    if args.augment_type == "bridge":
        if args.ego_past_noise_std is not None:
            raise ValueError(
                "--ego_past_noise_std is not supported by augment_type=bridge "
                "(the bridge augmenter does not perturb the ego history); "
                "drop the flag or pick another augment_type"
            )
        # Left as None deliberately, NOT resolved to a number. Bridge applies no history
        # perturbation, so None is the honest record: the knob does not apply, rather
        # than applying at 0. Writing 0.0 here also breaks the second call -- the factory
        # calls this defensively, and a resolved 0.0 is indistinguishable from a user
        # passing 0.0, so the rejection above would fire on every ordinary bridge run.
        return
    args.ego_past_noise_std = past_noise_std_for(args)
