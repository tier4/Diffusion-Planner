"""Per-augmenter defaults for knobs that are shared by name but not by history.

Lives in its own module because both :mod:`augmenter_factory` and
:mod:`data_augmentation_frenet` need it and the factory imports the frenet builder,
so a resolver in either of those would be a cycle.
"""

# `--ego_past_noise_std` unset means "whatever this augmenter has always used", which is
# NOT the same number for each. Quintic has perturbed the recorded history at 0.1 since
# it was written. Frenet rewrites the history kinematically from the perturbed polyline
# and hard-coded 0.0 before the flag was threaded to it, so inheriting quintic's 0.1
# would silently change every frenet run and leave existing frenet checkpoints
# unreproducible at their own seed. Bridge has no history perturbation at all.
#
# There is no measured reason to change the frenet default either: on a 2-seed, 8-arm
# A/B (~1,110 perturbed closed-loop rollouts per arm), turning it on moved recovery
# 28.3% -> 30.2%, well inside the 17.2-point spread between the control's own two seeds.
DEFAULT_PAST_NOISE_STD = {"quintic": 0.1, "bridge": 0.0, "frenet": 0.0}


def past_noise_std_for(args) -> float:
    """The history-noise std an augmenter should use, honouring an explicit flag.

    ``None`` (the unset default) resolves per augmenter; any number the caller passed
    wins for every augmenter, so a sweep over the flag still sweeps all of them.

    Raises:
        KeyError: on an ``augment_type`` with no recorded default, which means a new
            augmenter was added without deciding what this knob means for it.
    """
    if args.ego_past_noise_std is not None:
        return float(args.ego_past_noise_std)
    return DEFAULT_PAST_NOISE_STD[args.augment_type]
