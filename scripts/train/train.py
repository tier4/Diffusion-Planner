"""Accelerate-based planner training entry point."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

import wandb
from diffusion_planner.utils.checkpoint import load_checkpoint, save_checkpoint
from diffusion_planner.utils.lr_scheduler import (
    build_lr_scheduler,
    describe_lr_scheduler,
)

# Loss-dict entries logged under train/loss/; other scalars go under train/.
LOSS_KEYS = (
    "total",
    "trajectory",
    "turn_indicator",
    "regression",
    "classification",
    "neighbor",
    "yaw_regularization",
)


def _is_count(name: str) -> bool:
    """Count-like entries are summed across processes instead of averaged."""
    return name.endswith(("_count", "_counts", "_correct"))


def _output_layers(planner: torch.nn.Module) -> tuple[torch.nn.Module, ...]:
    """Head layers kept in AdamW; models may declare them, the flow model does not."""
    if hasattr(planner, "output_layers"):
        return tuple(planner.output_layers())
    return (
        planner.trajectory_decoder.output_projection.fc2,
        planner.turn_indicator_decoder.classifier,
    )


@hydra.main(version_base=None, config_path="../../configs", config_name="train/train")
def main(config: DictConfig) -> None:
    """Train on one or more devices managed by Accelerate."""
    compile_enabled = bool(config.accelerator.compile)
    accelerator: Accelerator = hydra.utils.instantiate(config.accelerator)
    set_seed(int(config.seed), device_specific=True)

    run_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{config.experiment_name}"
    checkpoint_dir = Path(str(config.training.checkpoint_dir)) / run_name

    if accelerator.is_main_process:
        wandb_config = OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True
        )
        wandb.init(
            name=run_name,
            config=wandb_config,  # type: ignore
        )

    loader = hydra.utils.instantiate(config.dataloader)
    if len(loader) == 0:
        raise RuntimeError(
            "dataloader has no batches; reduce batch_size or disable drop_last"
        )

    planner: torch.nn.Module = hydra.utils.instantiate(config.model)
    loss_function = hydra.utils.instantiate(config.loss)
    checkpoint_model = planner
    model_config = OmegaConf.to_container(
        config.model, resolve=True, throw_on_missing=True
    )
    if accelerator.is_main_process:
        parameter_count = sum(parameter.numel() for parameter in planner.parameters())
        trainable_parameter_count = sum(
            parameter.numel()
            for parameter in planner.parameters()
            if parameter.requires_grad
        )
        print(
            f"parameters={parameter_count / 1e6:.2f}M "
            f"trainable={trainable_parameter_count / 1e6:.2f}M"
        )
    optimizer = hydra.utils.instantiate(
        config.optimizer,
        model=planner,
        output_layers=_output_layers(planner),
        verbose=accelerator.is_main_process,
    )
    planner, optimizer, loader = accelerator.prepare(planner, optimizer, loader)
    total_epochs = int(config.training.total_epochs)
    steps_per_epoch = len(loader)
    total_steps = total_epochs * steps_per_epoch
    scheduler = build_lr_scheduler(optimizer, config.scheduler, total_steps)
    scheduler.step_update(0)

    start_epoch = 0
    global_step = 0
    if config.training.resume_from is not None:
        start_epoch, global_step = load_checkpoint(
            accelerator,
            str(config.training.resume_from),
            checkpoint_model,
            optimizer,
            scheduler,
            steps_per_epoch=steps_per_epoch,
            warm_start=bool(config.training.warm_start),
        )

    @torch.compile(fullgraph=False, disable=not compile_enabled)
    def optimizer_step() -> None:
        optimizer.step()

    if accelerator.is_main_process:
        print(
            f"training {len(loader.dataset)} frames on "
            f"{accelerator.num_processes} process(es), configured batch size "
            f"{config.dataloader.batch_size}"
        )
        print(
            f"epochs={total_epochs} steps_per_epoch={steps_per_epoch} "
            f"total_steps={total_steps}"
        )
        print(
            f"torch_compile={compile_enabled} "
            f"backend={config.accelerator.compile_backend} "
            f"mode={config.accelerator.compile_mode} "
            f"dynamic={bool(config.accelerator.compile_dynamic)} "
            f"optimizer_compiled={compile_enabled}"
        )
        print(describe_lr_scheduler(config.scheduler, total_steps))
        print(f"run_name={run_name} checkpoint_dir={checkpoint_dir}")
        print(f"loss={config.loss._target_}")

    planner.train()
    optimizer.zero_grad(set_to_none=True)
    log_interval = int(config.training.log_interval)
    checkpoint_interval = int(config.training.checkpoint_interval)
    for epoch in range(start_epoch, total_epochs):
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        progress = tqdm(
            loader,
            desc=f"epoch {epoch + 1}/{total_epochs}",
            disable=not accelerator.is_main_process,
            dynamic_ncols=True,
        )
        for step_in_epoch, batch in enumerate(progress, start=1):
            losses = loss_function(planner, batch)
            loss = losses["total"]
            accelerator.backward(loss)
            gradient_norm = accelerator.clip_grad_norm_(
                planner.parameters(), float(config.training.max_grad_norm)
            )
            optimizer_step()
            optimizer.zero_grad(set_to_none=True)
            if accelerator.optimizer_step_was_skipped:
                continue

            global_step += 1
            scheduler.step_update(global_step)
            if global_step % log_interval == 0:
                gradient_norm_value = (
                    gradient_norm.detach().float()
                    if gradient_norm is not None
                    else torch.full((), torch.nan, device=loss.device)
                )
                scalar_names = [
                    name
                    for name, value in losses.items()
                    if torch.is_tensor(value)
                    and value.ndim == 0
                    and not _is_count(name)
                ]
                scalars = accelerator.reduce(
                    torch.stack(
                        [losses[name].detach().float() for name in scalar_names]
                        + [gradient_norm_value]
                    ),
                    reduction="mean",
                )
                counts = {
                    name: accelerator.reduce(
                        value.detach().to(torch.float32), reduction="sum"
                    )
                    for name, value in losses.items()
                    if torch.is_tensor(value) and _is_count(name)
                }
                metric_values: dict[str, float] = {}
                for name, value in zip(scalar_names, scalars[:-1], strict=True):
                    prefix = "train/loss/" if name in LOSS_KEYS else "train/"
                    metric_values[f"{prefix}{name}"] = value.item()
                metric_values["train/grad_norm"] = scalars[-1].item()
                metric_values["train/learning_rate"] = optimizer.param_groups[0]["lr"]
                if {
                    "turn_indicator_correct",
                    "turn_indicator_valid_count",
                } <= counts.keys():
                    metric_values["train/turn_indicator_accuracy"] = (
                        counts["turn_indicator_correct"]
                        / counts["turn_indicator_valid_count"].clamp_min(1)
                    ).item()
                if "mode_target_counts" in counts:
                    mode_counts = counts["mode_target_counts"]
                    fractions = mode_counts / mode_counts.sum().clamp_min(1)
                    for index, fraction in enumerate(fractions.tolist()):
                        metric_values[f"train/mode_target_fraction/{index:02d}"] = (
                            fraction
                        )
                if accelerator.is_main_process:
                    progress.set_postfix(
                        loss=f"{metric_values['train/loss/total']:.5f}",
                        lr=f"{metric_values['train/learning_rate']:.2e}",
                    )
                    wandb.log(metric_values, step=global_step)
                    print(
                        f"step={global_step}/{total_steps} "
                        f"loss={metric_values['train/loss/total']:.5f} "
                        f"grad_norm={metric_values['train/grad_norm']:.3f} "
                        f"lr={metric_values['train/learning_rate']:.2e}"
                    )
            if accelerator.is_main_process and global_step % checkpoint_interval == 0:
                save_checkpoint(
                    accelerator,
                    checkpoint_dir / "latest.pth",
                    checkpoint_model,
                    optimizer,
                    scheduler,
                    model_config=model_config,  # pyright: ignore[reportArgumentType]
                    epoch=epoch,
                    step_in_epoch=step_in_epoch,
                    global_step=global_step,
                    steps_per_epoch=steps_per_epoch,
                )
        if accelerator.is_main_process:
            save_checkpoint(
                accelerator,
                checkpoint_dir / f"epoch_{epoch + 1:04d}.pth",
                checkpoint_model,
                optimizer,
                scheduler,
                model_config=model_config,  # pyright: ignore[reportArgumentType]
                epoch=epoch + 1,
                step_in_epoch=0,
                global_step=global_step,
                steps_per_epoch=steps_per_epoch,
            )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
