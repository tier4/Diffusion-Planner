"""One place where a parsed config becomes an augmenter.

Both training entrypoints used to carry the same ``if augment_type == ...`` ladder,
and a knob added to one of them was silently dropped by the other: the GRPO path
accepted every ``--frenet_*`` flag and then trained at the defaults, so a run's
recorded configuration was not the configuration it used. ``frenet_augmenter_from_args``
fixed that for one augmenter; this fixes it for all of them, so there is a single
place a flag has to be threaded through.

Unlike the ladder it replaces, an unrecognised ``augment_type`` raises instead of
falling through to quintic — the parser's ``Literal`` already constrains the flag,
and a programmatic caller deserves the error rather than a silent substitution.
"""

from diffusion_planner.utils.augment_defaults import (
    past_noise_std_for,
    resolve_history_noise,
)
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.data_augmentation_bridge import (
    StatePerturbation as BridgeStatePerturbation,
)
from diffusion_planner.utils.data_augmentation_frenet import frenet_augmenter_from_args

AUGMENT_TYPES = ("quintic", "bridge", "frenet")


def augmenter_from_args(args):
    """Build the augmenter a config asks for, or None when augmentation is off.

    Args:
        args: parsed config carrying ``use_data_augment``, ``augment_type`` and the
            per-augmenter knobs.

    Returns:
        The augmenter, or ``None`` if ``use_data_augment`` is false.

    Raises:
        ValueError: on an ``augment_type`` outside :data:`AUGMENT_TYPES`.
    """
    if not args.use_data_augment:
        return None
    if args.augment_type == "frenet":
        return frenet_augmenter_from_args(args)
    if args.augment_type == "bridge":
        # Bridge takes neither the history noise nor the quintic refinement knobs. The
        # rejection of an explicitly-passed --ego_past_noise_std lives in
        # resolve_history_noise, which the trainers call before serializing the config:
        # it needs to tell "passed" from "unset", and by the time the factory runs the
        # value has been resolved to a number either way. Called here defensively for
        # any caller that skipped the resolver.
        resolve_history_noise(args)
        return BridgeStatePerturbation(augment_prob=args.augment_prob, device=args.device)
    if args.augment_type == "quintic":
        return StatePerturbation(
            augment_prob=args.augment_prob,
            num_refine=args.num_refine,
            device=args.device,
            ego_past_noise_std=past_noise_std_for(args),
            use_smoothing_future_trajectory=args.use_smoothing_future_trajectory,
        )
    raise ValueError(f"unknown augment_type {args.augment_type!r}; expected one of {AUGMENT_TYPES}")
