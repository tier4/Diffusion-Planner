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

from diffusion_planner.utils.augment_defaults import resolve_history_noise
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
    # Above the type dispatch, so the mode contract is enforced on EVERY path. It used to
    # be called only in the bridge branch, which meant a caller that skipped the
    # trainers' own call got `augment_type=quintic, ego_past_noise_mode=jitter` silently
    # honoured as scale -- the record-it-and-ignore-it mismatch this module exists to
    # stop. Idempotent, so the trainers calling it first costs nothing.
    resolve_history_noise(args)
    if args.augment_type == "frenet":
        return frenet_augmenter_from_args(args)
    if args.augment_type == "bridge":
        return BridgeStatePerturbation(augment_prob=args.augment_prob, device=args.device)
    if args.augment_type == "quintic":
        return StatePerturbation(
            augment_prob=args.augment_prob,
            num_refine=args.num_refine,
            device=args.device,
            ego_past_noise_std=args.ego_past_noise_std_effective,
            use_smoothing_future_trajectory=args.use_smoothing_future_trajectory,
        )
    raise ValueError(f"unknown augment_type {args.augment_type!r}; expected one of {AUGMENT_TYPES}")
