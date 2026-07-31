"""Visualization helpers for scenario-specific open-loop predictions."""

from pathlib import Path


def visualize_scenario_prediction(
    inputs: dict,
    prediction,
    save_path: str | Path,
    title: str,
    show_neighbors: bool = False,
    view_range: float = 60.0,
) -> None:
    """Render one input scene with the predicted ego trajectory overlaid.

    The input NPZ convention stores heading as ``(x, y, heading)`` for the ego
    history and goal pose. It is converted to the cosine/sine representation
    expected by the shared input visualizer before rendering.
    """
    import matplotlib.pyplot as plt
    import torch

    from diffusion_planner.train_epoch import heading_to_cos_sin
    from diffusion_planner.utils.visualize_input import visualize_inputs

    sample_inputs = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            sample_inputs[key] = value[:1]
        else:
            sample_inputs[key] = value

    # The scenario NPZs store headings as (x, y, heading), while the shared
    # visualization helpers expect (x, y, cos(heading), sin(heading)).
    for key in ("ego_agent_past", "goal_pose"):
        if key in sample_inputs:
            sample_inputs[key] = heading_to_cos_sin(sample_inputs[key])

    fig, ax = plt.subplots(figsize=(8, 8))
    visualize_inputs(
        sample_inputs,
        ax=ax,
        view_ranges=[view_range],
        show_neighbors=show_neighbors,
        show_ego_future=False,
        route_color="#00A6D6",
        route_label="Route centerline",
    )

    prediction_xy = prediction.detach().float().cpu().numpy()
    ax.plot(
        prediction_xy[:, 0],
        prediction_xy[:, 1],
        color="orange",
        linewidth=2,
        label="scenario-based prediction",
    )
    ax.scatter(
        prediction_xy[-1, 0],
        prediction_xy[-1, 1],
        color="black",
        marker="x",
        label="final point",
    )
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
