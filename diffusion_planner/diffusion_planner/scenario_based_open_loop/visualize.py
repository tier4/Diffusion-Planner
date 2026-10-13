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
    import torch

    from diffusion_planner.train_epoch import heading_to_cos_sin
    from diffusion_planner.utils.visualize_input import visualize_inputs
    from scenario_generation.canvas import Figure

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

    fig = Figure((8, 8), dpi=120)
    ax = fig.panel((0.1, 0.07, 0.87, 0.87))
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
    ax.lines(
        [prediction_xy[:, :2]],
        "#ffa500",
        width=2,
        z=2,
        label="Diffusion Planner output",
    )
    ax.markers(
        prediction_xy[-1, :2],
        "x",
        size=6,
        color="#000000",
        z=1,
        label="final point",
    )
    ax.title(title)
    ax.legend("upper left")

    save_path = Path(save_path)
    fig.save(save_path)
