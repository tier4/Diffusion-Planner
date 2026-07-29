"""Training-loop integration for scenario-specific open-loop validation."""

import wandb

from diffusion_planner.scenario_based_open_loop.open_loop import (
    run_scenario_based_open_loop_validation,
)
from diffusion_planner.utils import ddp


def scenario_based_open_loop_validate(model, args, epoch: int) -> None:
    """Run configured open-loop metrics and publish summaries to W&B.

    This is called by rank 0 only. The list JSON selects NPZ files per metric;
    the training configuration provides metric parameters, and the resulting
    scalar summaries are logged under the open-loop W&B namespace.
    """
    if not args.scenario_based_open_loop_list:
        return

    summary = run_scenario_based_open_loop_validation(ddp.get_model(model, args.ddp), args)
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
            ", ".join(
                f"{metric_name}={int(values['sample_count'])} samples"
                for metric_name, values in summary.items()
            ),
        )
    )
