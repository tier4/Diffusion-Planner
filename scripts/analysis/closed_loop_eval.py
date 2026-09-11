"""Compare two planners over many closed-loop rollouts.

Open-loop attention says what a model looks at when handed a recorded scene.
This asks the harder question: when the model drives the scene it is looking at,
does the difference show up in behaviour, and does its attention drift as the
rollout leaves the recorded distribution?

Both models roll out from the same start scenes with the same seeds, so every
statistic is paired. The absolute stall rate depends on this rollout's own
fidelity — it re-plans on a reconstructed history — but both models are subject
to identically the same limitation, so the comparison between them is fair.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import argcomplete
import numpy as np
import plotly.graph_objects as go
import torch
from tqdm import tqdm

from diffusion_planner.analysis import (
    SceneTokenLayout,
    capture_fusion_attention,
    ego_query_attention,
    rollout,
)
from diffusion_planner.data import PlannerDataset
from diffusion_planner.data.transforms import (
    PlannerDataNormalizer,
    PlannerUnknownLabelAugmentation,
)
from diffusion_planner.models.diffusion_planner import DiffusionPlanner
from diffusion_planner.utils.checkpoint import load_model
from diffusion_planner.visualizer.schema import EgoIndex

TRACKED_BLOCKS = ("lanes", "neighbors", "road_borders", "route_lanes")


def _numpy_frame(frame: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value))
        for key, value in frame.items()
    }


def _block_shares(
    model: DiffusionPlanner,
    frame: dict[str, Any],
    normalizer: PlannerDataNormalizer,
    layout: SceneTokenLayout,
    device: torch.device,
) -> dict[str, float]:
    """Share of ego-query attention per tracked block, for one frame."""
    normalized = normalizer({key: np.asarray(v) for key, v in frame.items()})
    data = {
        key: torch.as_tensor(value, device=device).unsqueeze(0)
        for key, value in normalized.items()
    }
    with capture_fusion_attention(model) as capture, torch.no_grad():
        model.scene_encoder(data)
    row = ego_query_attention(capture, layout)[0].detach().float().cpu().numpy()
    total = float(row.sum()) or 1.0
    shares = {}
    for name in TRACKED_BLOCKS:
        block = layout.block(name)
        shares[name] = float(row[block.start : block.stop].sum()) / total
    return shares


def _evaluate(
    models: dict[str, DiffusionPlanner],
    dataset: PlannerDataset,
    indices: list[int],
    device: torch.device,
    *,
    steps: int,
    replan_every: int,
    sampling_steps: int,
    min_speed: float,
) -> dict[str, Any]:
    widen = PlannerUnknownLabelAugmentation()
    normalizer = PlannerDataNormalizer()
    layout = SceneTokenLayout.from_dimensions()
    per_scene: dict[str, dict[str, list[float]]] = {
        label: defaultdict(list) for label in models
    }
    cumulative: dict[str, list[list[float]]] = {label: [] for label in models}
    used = 0

    for index in tqdm(indices, desc="rollouts", unit="scene"):
        start = dict(widen(_numpy_frame(dataset[index])))
        speed = float(start["ego_agent_past"][-1, EgoIndex.VELOCITY])
        if speed < min_speed:
            continue
        ideal_step = speed * 0.1
        results: dict[str, dict[str, Any]] = {}
        complete = True
        for label, model in models.items():
            lengths: list[float] = []
            travelled: list[float] = []
            previous = 0.0
            final_frame = start
            for state in rollout(
                model,
                start,
                steps=steps,
                device=str(device),
                num_sampling_steps=sampling_steps,
                replan_every=replan_every,
                stop_beyond_map=False,
            ):
                lengths.append(state.travelled_m - previous)
                previous = state.travelled_m
                travelled.append(state.travelled_m)
                final_frame = state.frame
            if len(lengths) < steps:
                complete = False
                break
            results[label] = {
                "lengths": lengths,
                "travelled": travelled,
                "start_shares": _block_shares(model, start, normalizer, layout, device),
                "final_shares": _block_shares(
                    model, final_frame, normalizer, layout, device
                ),
            }
        if not complete:
            continue
        used += 1
        for label, data in results.items():
            lengths = data["lengths"]
            retention = float(np.mean(lengths)) / ideal_step
            final_retention = float(np.mean(lengths[-5:])) / ideal_step
            per_scene[label]["retention"].append(retention)
            per_scene[label]["final_retention"].append(final_retention)
            per_scene[label]["travelled_m"].append(data["travelled"][-1])
            per_scene[label]["stalled"].append(1.0 if final_retention < 0.25 else 0.0)
            per_scene[label]["speed_cv"].append(
                float(np.std(lengths) / max(np.mean(lengths), 1e-9))
            )
            for name in TRACKED_BLOCKS:
                per_scene[label][f"start/{name}"].append(data["start_shares"][name])
                per_scene[label][f"final/{name}"].append(data["final_shares"][name])
                per_scene[label][f"drift/{name}"].append(
                    data["final_shares"][name] - data["start_shares"][name]
                )
            cumulative[label].append(data["travelled"])
    return {"per_scene": per_scene, "cumulative": cumulative, "scenes": used}


def _paired(a: dict[str, list[float]], b: dict[str, list[float]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted(set(a) & set(b)):
        left = np.asarray(a[key], dtype=float)
        right = np.asarray(b[key], dtype=float)
        if left.size < 2 or left.size != right.size:
            continue
        difference = right - left
        stderr = float(difference.std(ddof=1) / math.sqrt(difference.size))
        mean = float(difference.mean())
        out[key] = {
            "a": float(left.mean()),
            "b": float(right.mean()),
            "difference": mean,
            "stderr": stderr,
            "sigmas": abs(mean) / stderr if stderr > 0 else math.nan,
            "scenes": int(difference.size),
        }
    return out


def _save(figure: go.Figure, path: Path, width: int = 1000, height: int = 600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(figure.to_image(format="png", width=width, height=height, scale=2))


def _plot_retention_ecdf(
    per_scene: dict[str, dict[str, list[float]]], labels: tuple[str, str], path: Path
) -> None:
    """Where the two models differ is the low-retention tail, so show the whole curve."""
    figure = go.Figure()
    for colour, label in zip(("#8aa8bd", "#2a7fb8"), labels, strict=False):
        values = np.sort(np.asarray(per_scene[label]["final_retention"], dtype=float))
        figure.add_trace(
            go.Scatter(
                x=values,
                y=np.arange(1, values.size + 1) / values.size,
                name=label,
                line={"color": colour, "width": 2, "shape": "hv"},
            )
        )
    figure.add_vline(
        x=0.25,
        line={"color": "#c4472f", "dash": "dash"},
        annotation_text="stall threshold",
        annotation_position="top right",
    )
    figure.update_layout(
        title="Closed loop: speed retention over the final five steps, per scene",
        xaxis_title="mean step length ÷ the ego's initial speed × dt",
        yaxis_title="fraction of scenes at or below",
        template="plotly_white",
        margin={"l": 70, "r": 40, "t": 60, "b": 50},
        xaxis={"range": [0.0, 1.6]},
    )
    _save(figure, path)


def _plot_cumulative(
    cumulative: dict[str, list[list[float]]], labels: tuple[str, str], path: Path
) -> None:
    figure = go.Figure()
    for colour, label in zip(("#8aa8bd", "#2a7fb8"), labels, strict=False):
        rows = cumulative[label]
        if not rows:
            continue
        matrix = np.asarray(rows, dtype=float)
        mean = matrix.mean(axis=0)
        stderr = matrix.std(axis=0, ddof=1) / math.sqrt(matrix.shape[0])
        steps = list(range(1, matrix.shape[1] + 1))
        figure.add_trace(
            go.Scatter(
                x=steps + steps[::-1],
                y=list(mean + stderr) + list((mean - stderr)[::-1]),
                fill="toself",
                fillcolor=colour,
                opacity=0.3,
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(x=steps, y=mean, name=label, line={"color": colour, "width": 2})
        )
    figure.update_layout(
        title="Closed loop: cumulative distance travelled, mean ± standard error",
        xaxis_title="rollout step",
        yaxis_title="metres",
        template="plotly_white",
        margin={"l": 70, "r": 40, "t": 60, "b": 50},
    )
    _save(figure, path)


def _plot_drift(paired: dict[str, Any], labels: tuple[str, str], path: Path) -> None:
    """Does attention move as the model drives the scene it is looking at?"""
    names = [n for n in TRACKED_BLOCKS if f"drift/{n}" in paired]
    figure = go.Figure(
        [
            go.Bar(
                name=f"{labels[0]}",
                x=names,
                y=[paired[f"drift/{n}"]["a"] for n in names],
                marker_color="#8aa8bd",
            ),
            go.Bar(
                name=f"{labels[1]}",
                x=names,
                y=[paired[f"drift/{n}"]["b"] for n in names],
                marker_color="#2a7fb8",
                error_y={
                    "type": "data",
                    "array": [paired[f"drift/{n}"]["stderr"] for n in names],
                },
            ),
        ]
    )
    figure.update_layout(
        barmode="group",
        title="Attention drift over a rollout: final share minus starting share",
        yaxis_title="change in share of ego-query attention",
        template="plotly_white",
        margin={"l": 70, "r": 40, "t": 60, "b": 50},
    )
    _save(figure, path)


def _plot_stall(
    per_scene: dict[str, dict[str, list[float]]], labels: tuple[str, str], path: Path
) -> None:
    rates, errors = [], []
    for label in labels:
        values = np.asarray(per_scene[label]["stalled"], dtype=float)
        rate = float(values.mean())
        rates.append(rate * 100.0)
        errors.append(100.0 * math.sqrt(max(rate * (1 - rate), 0.0) / values.size))
    figure = go.Figure(
        go.Bar(
            x=list(labels),
            y=rates,
            error_y={"type": "data", "array": errors},
            marker_color=["#8aa8bd", "#2a7fb8"],
            text=[f"{r:.1f}%" for r in rates],
            textposition="outside",
        )
    )
    figure.update_layout(
        title="Closed loop: scenes where the ego stalled, with binomial standard error",
        yaxis_title="percent of scenes",
        template="plotly_white",
        margin={"l": 70, "r": 40, "t": 60, "b": 50},
    )
    _save(figure, path, width=700)


def main() -> None:
    """Run paired rollouts and write plots plus a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=512)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--replan-every", type=int, default=10)
    parser.add_argument("--sampling-steps", type=int, default=10)
    parser.add_argument("--min-speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = PlannerDataset(str(args.parquet), file_capacity=8)
    generator = np.random.default_rng(args.seed)
    indices = sorted(
        generator.choice(
            len(dataset), size=min(args.scenes, len(dataset)), replace=False
        ).tolist()
    )
    labels = (args.label_a, args.label_b)
    models = {
        args.label_a: load_model(args.model_a, DiffusionPlanner).to(device).eval(),
        args.label_b: load_model(args.model_b, DiffusionPlanner).to(device).eval(),
    }
    print(f"{len(indices)} candidate scenes, {args.steps} steps each, on {device}")

    result = _evaluate(
        models,
        dataset,
        indices,
        device,
        steps=args.steps,
        replan_every=args.replan_every,
        sampling_steps=args.sampling_steps,
        min_speed=args.min_speed,
    )
    paired = _paired(result["per_scene"][labels[0]], result["per_scene"][labels[1]])

    plots = args.out_dir / "plots"
    _plot_retention_ecdf(result["per_scene"], labels, plots / "cl_retention_ecdf.png")
    _plot_cumulative(result["cumulative"], labels, plots / "cl_cumulative.png")
    _plot_drift(paired, labels, plots / "cl_attention_drift.png")
    _plot_stall(result["per_scene"], labels, plots / "cl_stall_rate.png")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "closed_loop.json").write_text(
        json.dumps(
            {
                "model_a": {"label": labels[0], "path": str(args.model_a)},
                "model_b": {"label": labels[1], "path": str(args.model_b)},
                "scenes": result["scenes"],
                "steps": args.steps,
                "replan_every": args.replan_every,
                "paired": paired,
            },
            indent=2,
        )
    )

    print(f"\nusable scenes: {result['scenes']}")
    print(f"  {'metric':<22}{labels[0]:>11}{labels[1]:>11}{'diff':>11}{'sigmas':>8}")
    for key, value in sorted(paired.items(), key=lambda i: -i[1]["sigmas"]):
        print(
            f"  {key:<22}{value['a']:>11.4f}{value['b']:>11.4f}"
            f"{value['difference']:>+11.4f}{value['sigmas']:>8.1f}"
        )
    print(f"\nwrote {args.out_dir}/closed_loop.json and {plots}/*.png")


if __name__ == "__main__":
    main()
