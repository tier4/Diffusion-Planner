"""Training-loop integration for scenario-specific open-loop validation."""

import json
from pathlib import Path

import wandb

from diffusion_planner.scenario_based_open_loop.open_loop import (
    run_scenario_based_open_loop_validation,
)
from diffusion_planner.utils import ddp


def scenario_based_open_loop_validate(
    model, args, epoch: int, output_dir: str | Path | None = None
) -> None:
    """Run configured open-loop metrics and publish summaries to W&B.

    This is called by rank 0 only. The list JSON selects NPZ files per metric;
    the training configuration provides metric parameters, and the resulting
    scalar summaries are logged under the open-loop W&B namespace.
    """
    if not args.scenario_based_open_loop_list:
        return

    output_root = Path(output_dir) if output_dir is not None else None
    summary = run_scenario_based_open_loop_validation(
        ddp.get_model(model, args.ddp),
        args,
        visualization_dir=output_root / "visualization" if output_root else None,
        details_dir=output_root / "details" if output_root else None,
    )
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    log = {
        f"scenario_based_open_loop/{metric_name}/{key}": value
        for metric_name, values in summary.items()
        for key, value in values.items()
    }
    if not log:
        return
    wandb.log(log, step=epoch + 1)
    print(
        "scenario-based-open-loop @epoch {}: {}".format(
            epoch + 1,
            ", ".join(summary),
        )
    )
