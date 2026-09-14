"""Single-file training checkpoints for Accelerate-based training."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

import torch
from accelerate import Accelerator
from timm.scheduler.scheduler import Scheduler
from torch import nn
from torch.optim import Optimizer

ModelT = TypeVar("ModelT", bound=nn.Module)


def load_model(
    path: str | Path,
    model_factory: Callable[..., ModelT],
) -> ModelT:
    """Construct a model from its saved config and load its weights on CPU."""
    checkpoint_path = Path(path).expanduser()
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model_config = dict(checkpoint["model_config"])
    model_config.pop("_target_", None)
    model = model_factory(**model_config)
    model.load_state_dict(checkpoint["model"])
    return model


def save_checkpoint(
    accelerator: Accelerator,
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Scheduler,
    *,
    model_config: Mapping[str, Any],
    epoch: int,
    step_in_epoch: int,
    global_step: int,
    steps_per_epoch: int,
    provenance: Mapping[str, Any] | None = None,
) -> None:
    """Atomically save model and training state from the main process.

    ``provenance`` carries the resolved run config, the git state and the dataset
    fingerprint (see ``utils.provenance``). It is optional so that older callers and tests
    keep working, and it is omitted from the file entirely when not supplied.
    """
    checkpoint_path = Path(path)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    model_state = model.state_dict()
    state = {
        "model": model_state,
        "model_config": dict(model_config),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "global_step": global_step,
        "steps_per_epoch": steps_per_epoch,
        "world_size": accelerator.num_processes,
    }
    if provenance is not None:
        state["provenance"] = dict(provenance)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, temporary_path)
    temporary_path.replace(checkpoint_path)
    print(f"saved checkpoint: {checkpoint_path}")


def load_checkpoint(
    accelerator: Accelerator,
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Scheduler,
    *,
    steps_per_epoch: int,
    warm_start: bool = False,
) -> tuple[int, int]:
    """Restore a full training state or warm-start from model weights only."""
    checkpoint_path = Path(path).expanduser()
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )

    model.load_state_dict(checkpoint["model"])
    if not warm_start:
        saved_steps_per_epoch = int(checkpoint["steps_per_epoch"])
        if saved_steps_per_epoch != steps_per_epoch:
            raise RuntimeError(
                "steps_per_epoch changed since the checkpoint was created: "
                f"saved={saved_steps_per_epoch}, current={steps_per_epoch}"
            )
        saved_world_size = int(checkpoint["world_size"])
        if saved_world_size != accelerator.num_processes:
            raise RuntimeError(
                "world size changed since the checkpoint was created: "
                f"saved={saved_world_size}, current={accelerator.num_processes}"
            )
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
    accelerator.wait_for_everyone()

    epoch = 0 if warm_start else int(checkpoint["epoch"])
    global_step = 0 if warm_start else int(checkpoint["global_step"])
    if accelerator.is_main_process:
        if warm_start:
            print(f"warm-started model weights: {checkpoint_path}")
        else:
            print(
                f"loaded checkpoint: {checkpoint_path} "
                f"(restart_epoch={epoch + 1}, global_step={global_step})"
            )
    return epoch, global_step
