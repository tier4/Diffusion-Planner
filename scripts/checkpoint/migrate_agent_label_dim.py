"""Widen a checkpoint's agent-label projection for the unknown class.

`AGENT_LABEL_DIM` gained a fourth column (index 3 = unknown), which widens exactly
one tensor in the model: the neighbor agent `OneHotEncoder` projection weight,
`(hidden_dim, num_classes)`. Both checkpoint loaders call `load_state_dict`
strictly, so checkpoints written before that change fail to load. This script
right-pads that one weight with a zero column and writes a new checkpoint file,
leaving everything else in the checkpoint dict untouched.

The padding is zero so the widening is a no-op for what the checkpoint already
learned: the three known one-hot inputs produce bit-identical embeddings, an
unknown one-hot produces a zero vector, and an all-zero row (an empty neighbor
slot) keeps producing a zero vector.

The migrated checkpoint must be used with `training.warm_start=true`, because a
full-state resume also restores the optimizer state, whose momentum buffer for
this tensor still has the old width.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import argcomplete
import torch

from diffusion_planner.data.dimensions import AGENT_LABEL_DIM

LABEL_PROJECTION_SUFFIX = "label_encoder.projection.weight"


def find_label_projection_key(state_dict: dict[str, Any]) -> str:
    """Locate the single agent-label projection weight in a model state dict."""
    keys = [key for key in state_dict if key.endswith(LABEL_PROJECTION_SUFFIX)]
    if not keys:
        raise RuntimeError(
            f"no tensor ending in {LABEL_PROJECTION_SUFFIX!r} in the checkpoint; "
            "this does not look like a diffusion planner checkpoint"
        )
    if len(keys) > 1:
        raise RuntimeError(
            f"expected exactly one tensor ending in {LABEL_PROJECTION_SUFFIX!r}, "
            f"found {len(keys)}: {keys}"
        )
    return keys[0]


def pad_label_projection(weight: torch.Tensor, *, target_classes: int) -> torch.Tensor:
    """Right-pad a `(hidden_dim, num_classes)` weight with zero columns."""
    if weight.ndim != 2:
        raise RuntimeError(
            f"agent-label projection weight must be 2D, got shape {tuple(weight.shape)}"
        )
    num_classes = weight.shape[1]
    if num_classes == target_classes:
        raise RuntimeError(
            f"checkpoint already has {num_classes} agent-label columns, which matches "
            f"AGENT_LABEL_DIM={target_classes}; it needs no migration and padding it "
            "again would make it too wide"
        )
    if num_classes > target_classes:
        raise RuntimeError(
            f"checkpoint has {num_classes} agent-label columns, more than "
            f"AGENT_LABEL_DIM={target_classes}; this script only widens"
        )
    padding = weight.new_zeros(weight.shape[0], target_classes - num_classes)
    return torch.cat([weight, padding], dim=1)


def migrate_checkpoint(
    source: Path, destination: Path, *, target_classes: int = AGENT_LABEL_DIM
) -> None:
    """Write a copy of `source` whose agent-label projection has `target_classes` columns."""
    checkpoint: dict[str, Any] = torch.load(
        source, map_location="cpu", weights_only=False
    )
    if "model" not in checkpoint:
        raise RuntimeError(f"{source} has no 'model' entry: keys={list(checkpoint)}")

    state_dict = checkpoint["model"]
    key = find_label_projection_key(state_dict)
    old_weight = state_dict[key]
    new_weight = pad_label_projection(old_weight, target_classes=target_classes)
    state_dict[key] = new_weight

    print(f"source: {source}")
    print(f"  {key}")
    print(f"    {tuple(old_weight.shape)} -> {tuple(new_weight.shape)}")
    print(f"  tensors unchanged: {len(state_dict) - 1} of {len(state_dict)}")
    print(
        f"  epoch={checkpoint.get('epoch')} global_step={checkpoint.get('global_step')}"
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(destination)
    print(f"wrote: {destination}")
    print("load it with training.resume_from=<path> training.warm_start=true")


def main() -> None:
    """Migrate one checkpoint to the current `AGENT_LABEL_DIM`."""
    parser = argparse.ArgumentParser(
        description=(
            "Right-pad a checkpoint's agent-label projection weight with zero columns "
            f"so it loads against AGENT_LABEL_DIM={AGENT_LABEL_DIM}"
        )
    )
    parser.add_argument("source", type=Path, help="checkpoint to read (never modified)")
    parser.add_argument(
        "destination",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "checkpoint to write (default: the source with a "
            f"'_label{AGENT_LABEL_DIM}' suffix)"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite the destination if it already exists",
    )
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    source: Path = args.source.expanduser()
    if not source.is_file():
        parser.error(f"source checkpoint does not exist: {source}")

    destination: Path = (
        args.destination.expanduser()
        if args.destination is not None
        else source.with_name(f"{source.stem}_label{AGENT_LABEL_DIM}{source.suffix}")
    )
    if destination.resolve() == source.resolve():
        parser.error(
            "destination must differ from source; this script never writes in place"
        )
    if destination.exists() and not args.force:
        parser.error(
            f"destination already exists (pass --force to overwrite): {destination}"
        )

    migrate_checkpoint(source, destination)


if __name__ == "__main__":
    main()
