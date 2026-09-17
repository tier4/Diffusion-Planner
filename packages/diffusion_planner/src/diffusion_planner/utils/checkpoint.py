"""Single-file training checkpoints for Accelerate-based training."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

import torch
from accelerate import Accelerator
from timm.scheduler.scheduler import Scheduler
from torch import nn
from torch.optim import Optimizer

ModelT = TypeVar("ModelT", bound=nn.Module)


def resolve_target(target: str) -> Callable[..., Any]:
    """Import the class or function named by a Hydra ``_target_`` string."""
    module_name, _, attribute = target.rpartition(".")
    if not module_name or not attribute:
        raise ValueError(f"Invalid _target_: {target!r}")
    return getattr(importlib.import_module(module_name), attribute)


def load_model(
    path: str | Path,
    model_factory: Callable[..., ModelT] | None = None,
) -> ModelT:
    """Construct a model from its saved config and load its weights on CPU.

    Without ``model_factory`` the class is resolved from the checkpoint's
    ``_target_``, so flow-matching and PLUTO checkpoints load the same way.
    """
    checkpoint_path = Path(path).expanduser()
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model_config = dict(checkpoint["model_config"])
    target = model_config.pop("_target_", None)
    if model_factory is None:
        if target is None:
            raise ValueError(
                f"{checkpoint_path} has no model _target_; pass model_factory"
            )
        model_factory = resolve_target(str(target))
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
) -> None:
    """Atomically save model and training state from the main process."""
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
