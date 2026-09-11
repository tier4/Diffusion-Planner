"""Ten closed-loop example figures comparing two planners on the same rollouts.

Open-loop attention says what a model looks at when handed a recorded scene.
These figures ask what happens once the model drives the scene: where the ego
actually goes, whether it holds speed, and whether its attention stays put as
the rollout leaves the recorded distribution.

Scene drawing is the port of ``rlvr.autoresearch.visualize_scenes.draw_scene``
in ``trend_examples``, so the visual language matches the open-loop set.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
from typing import Any

import argcomplete
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from diffusion_planner.analysis import (  # noqa: E402
    SceneTokenLayout,
    annotate_records,
    capture_fusion_attention,
    ego_query_attention,
    rollout,
    token_records,
)
from diffusion_planner.data import PlannerDataset  # noqa: E402
from diffusion_planner.data.transforms import (  # noqa: E402
    PlannerDataNormalizer,
    PlannerUnknownLabelAugmentation,
)
from diffusion_planner.models.diffusion_planner import DiffusionPlanner  # noqa: E402
from diffusion_planner.utils.checkpoint import load_model  # noqa: E402
from diffusion_planner.visualizer.schema import EgoIndex  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "trend_examples", Path(__file__).with_name("trend_examples.py")
)
assert _spec is not None and _spec.loader is not None
_te = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_te)

MODEL_COLORS = _te.MODEL_COLORS
TRACKED = ("lanes", "neighbors", "road_borders", "route_lanes")
STALL = 0.25


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
) -> tuple[dict[str, float], list[dict[str, Any]], float]:
    """Per-block shares, the annotated records, and the lead-agent share."""
    normalized = normalizer({key: np.asarray(v) for key, v in frame.items()})
    data = {
        key: torch.as_tensor(value, device=device).unsqueeze(0)
        for key, value in normalized.items()
    }
    with capture_fusion_attention(model) as capture, torch.no_grad():
        model.scene_encoder(data)
    row = ego_query_attention(capture, layout)[0].detach().float().cpu().numpy()
    records = annotate_records(token_records(frame, row, layout), frame)
    total = float(row.sum()) or 1.0
    shares = {}
    for name in TRACKED:
        block = layout.block(name)
        shares[name] = float(row[block.start : block.stop].sum()) / total
    ahead = [
        r
        for r in records
        if r["block"] == "neighbors"
        and r.get("bearing") == "ahead"
        and r["distance_m"] is not None
    ]
    lead = (
        min(ahead, key=lambda r: r["distance_m"])["attention"] / total if ahead else 0.0
    )
    return shares, records, lead


def _run(
    models: dict[str, DiffusionPlanner],
    start: dict[str, Any],
    device: torch.device,
    normalizer: PlannerDataNormalizer,
    layout: SceneTokenLayout,
    *,
    steps: int,
    replan_every: int,
    track_attention: bool,
) -> dict[str, dict[str, Any]]:
    """Roll both models from one start, collecting frames, path and attention."""
    out: dict[str, dict[str, Any]] = {}
    for label, model in models.items():
        frames: list[dict[str, Any]] = []
        lengths: list[float] = []
        path: list[tuple[float, float]] = []
        shares: list[dict[str, float]] = []
        leads: list[float] = []
        previous = 0.0
        for state in rollout(
            model,
            start,
            steps=steps,
            device=str(device),
            num_sampling_steps=10,
            replan_every=replan_every,
            stop_beyond_map=False,
        ):
            frames.append(state.frame)
            lengths.append(state.travelled_m - previous)
            previous = state.travelled_m
            path.append((state.world_x_m, state.world_y_m))
            if track_attention:
                block, _records, lead = _block_shares(
                    model, state.frame, normalizer, layout, device
                )
                shares.append(block)
                leads.append(lead)
        out[label] = {
            "frames": frames,
            "lengths": lengths,
            "path": path,
            "shares": shares,
            "leads": leads,
        }
    return out


def _retentions(
    result: dict[str, dict[str, Any]], labels: tuple[str, str], ideal: float
) -> dict[str, float]:
    return {
        label: float(np.mean(result[label]["lengths"][-5:])) / ideal for label in labels
    }


def _save(figure: Any, path: Path, dpi: int = 110) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _filmstrip(
    result: dict[str, dict[str, Any]],
    labels: tuple[str, str],
    steps: list[int],
    ideal: float,
    title: str,
    path: Path,
) -> None:
    columns = len(steps)
    figure = plt.figure(figsize=(7.5 * columns, 6.0 * 2 + 4.5))
    grid = figure.add_gridspec(3, columns, height_ratios=[4, 4, 1.6], hspace=0.18)
    for row, label in enumerate(labels):
        for column, step in enumerate(steps):
            ax = figure.add_subplot(grid[row, column])
            _te._draw_scene(ax, result[label]["frames"][step], show_gt=False)
            ax.set_xlim(-25.0, 65.0)
            ax.set_ylim(-30.0, 30.0)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.15)
            if row == 0 and column == 0:
                ax.legend(
                    handles=_te._legend_handles(False),
                    fontsize=7,
                    loc="upper left",
                    framealpha=0.85,
                )
            ax.set_title(f"{label} · rollout step {step}", fontsize=9)
    ax = figure.add_subplot(grid[2, :])
    for colour, label in zip(MODEL_COLORS, labels, strict=True):
        ax.plot(
            range(1, len(result[label]["lengths"]) + 1),
            result[label]["lengths"],
            color=colour,
            lw=2,
            alpha=0.85,
            label=label,
        )
    ax.axhline(ideal, color="red", ls="--", lw=1.5, alpha=0.7, label="recorded speed")
    ax.axhline(ideal * STALL, color="#888", ls=":", lw=1.5, label="stall threshold")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("metres per step")
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=8, loc="upper right")
    figure.suptitle(title, fontsize=14)
    _save(figure, path, dpi=95)


def _paths(
    results: list[tuple[int, dict[str, dict[str, Any]]]],
    labels: tuple[str, str],
    path: Path,
) -> None:
    """Where the ego actually goes, in the starting frame, across several scenes."""
    count = min(len(results), 6)
    figure, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, (index, result) in zip(axes.ravel(), results[:count], strict=False):
        for colour, label in zip(MODEL_COLORS, labels, strict=True):
            xy = np.asarray(result[label]["path"], dtype=float)
            ax.plot(xy[:, 0], xy[:, 1], color=colour, lw=2, alpha=0.85, label=label)
            ax.plot(xy[-1, 0], xy[-1, 1], "o", color=colour, ms=7, mew=0)
        ax.plot(0.0, 0.0, "k*", ms=12, label="start")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15)
        ax.set_title(f"scene {index}", fontsize=9)
        ax.set_xlabel("x (m)", fontsize=8)
        ax.set_ylabel("y (m)", fontsize=8)
    axes.ravel()[0].legend(fontsize=8, loc="best")
    for ax in axes.ravel()[count:]:
        ax.axis("off")
    figure.suptitle(
        "Closed loop: ego path in the starting frame, same scenes and seeds",
        fontsize=14,
    )
    _save(figure, path)


def _drift_trace(
    result: dict[str, dict[str, Any]], labels: tuple[str, str], index: int, path: Path
) -> None:
    """Block shares across a rollout, so drift is visible as it happens."""
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for ax, block in zip(axes.ravel(), TRACKED, strict=True):
        for colour, label in zip(MODEL_COLORS, labels, strict=True):
            series = [s[block] for s in result[label]["shares"]]
            ax.plot(
                range(1, len(series) + 1),
                series,
                color=colour,
                lw=2,
                alpha=0.85,
                label=label,
            )
        ax.set_title(block, fontsize=10)
        ax.grid(True, alpha=0.15)
        ax.set_ylabel("share of attention", fontsize=8)
    axes[1][0].set_xlabel("rollout step")
    axes[1][1].set_xlabel("rollout step")
    axes[0][0].legend(fontsize=8, loc="best")
    figure.suptitle(f"Attention drift during one rollout — scene {index}", fontsize=14)
    _save(figure, path)


def _lead_agent(
    results: list[tuple[int, dict[str, dict[str, Any]]]],
    labels: tuple[str, str],
    path: Path,
) -> None:
    """Attention on the nearest agent ahead, averaged across scenes."""
    figure, ax = plt.subplots(figsize=(11, 6))
    for colour, label in zip(MODEL_COLORS, labels, strict=True):
        traces = [r[label]["leads"] for _index, r in results if r[label]["leads"]]
        if not traces:
            continue
        length = min(len(t) for t in traces)
        matrix = np.asarray([t[:length] for t in traces], dtype=float) * 100.0
        mean = matrix.mean(axis=0)
        stderr = matrix.std(axis=0, ddof=1) / math.sqrt(matrix.shape[0])
        steps = np.arange(1, length + 1)
        ax.fill_between(steps, mean - stderr, mean + stderr, color=colour, alpha=0.25)
        ax.plot(steps, mean, color=colour, lw=2, label=f"{label} (n={matrix.shape[0]})")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("attention on the nearest agent ahead (%)")
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=9)
    figure.suptitle(
        "Does the model keep watching the vehicle it is following?", fontsize=14
    )
    _save(figure, path)


def _speed_grid(
    results: list[tuple[int, dict[str, dict[str, Any]]]],
    labels: tuple[str, str],
    ideals: dict[int, float],
    path: Path,
) -> None:
    count = min(len(results), 8)
    figure, axes = plt.subplots(2, 4, figsize=(22, 9), sharex=True)
    for ax, (index, result) in zip(axes.ravel(), results[:count], strict=False):
        ideal = ideals[index]
        for colour, label in zip(MODEL_COLORS, labels, strict=True):
            series = result[label]["lengths"]
            ax.plot(
                range(1, len(series) + 1),
                series,
                color=colour,
                lw=1.6,
                alpha=0.9,
                label=label,
            )
        ax.axhline(ideal, color="red", ls="--", lw=1.2, alpha=0.7)
        ax.axhline(ideal * STALL, color="#888", ls=":", lw=1.2)
        ax.set_title(f"scene {index}", fontsize=9)
        ax.grid(True, alpha=0.15)
    axes[0][0].legend(fontsize=8, loc="best")
    for ax in axes.ravel()[count:]:
        ax.axis("off")
    figure.suptitle(
        "Per-step speed across eight scenes; dashed is the recorded speed, "
        "dotted the stall threshold",
        fontsize=14,
    )
    _save(figure, path)


def _scatter(
    per_scene: dict[str, list[float]], labels: tuple[str, str], path: Path
) -> None:
    """The pairing itself: one point per scene."""
    a = np.asarray(per_scene[labels[0]], dtype=float)
    b = np.asarray(per_scene[labels[1]], dtype=float)
    figure, ax = plt.subplots(figsize=(8.5, 8))
    limit = float(max(a.max(), b.max())) * 1.05
    ax.plot([0, limit], [0, limit], color="#888", ls="--", lw=1)
    ax.axvline(STALL, color="#c4472f", ls=":", lw=1.2)
    ax.axhline(STALL, color="#c4472f", ls=":", lw=1.2)
    both = (a >= STALL) & (b >= STALL)
    ax.scatter(
        a[both], b[both], s=22, color="#8aa8bd", alpha=0.6, label="neither stalls"
    )
    only_a = (a < STALL) & (b >= STALL)
    ax.scatter(
        a[only_a],
        b[only_a],
        s=42,
        color="#2a7fb8",
        label=f"only {labels[0]} stalls (n={int(only_a.sum())})",
    )
    only_b = (a >= STALL) & (b < STALL)
    ax.scatter(
        a[only_b],
        b[only_b],
        s=42,
        color="#c4472f",
        label=f"only {labels[1]} stalls (n={int(only_b.sum())})",
    )
    ax.set_xlabel(f"{labels[0]} final speed retention")
    ax.set_ylabel(f"{labels[1]} final speed retention")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=9, loc="lower right")
    figure.suptitle(
        "Paired retention, one point per scene; above the diagonal favours "
        f"{labels[1]}",
        fontsize=13,
    )
    _save(figure, path)


def _divergence(
    results: list[tuple[int, dict[str, dict[str, Any]]]],
    labels: tuple[str, str],
    path: Path,
) -> None:
    """How far apart the two models' egos drift, given the same start."""
    figure, ax = plt.subplots(figsize=(11, 6))
    traces = []
    for _index, result in results:
        a = np.asarray(result[labels[0]]["path"], dtype=float)
        b = np.asarray(result[labels[1]]["path"], dtype=float)
        length = min(len(a), len(b))
        traces.append(np.linalg.norm(a[:length] - b[:length], axis=1))
    if not traces:
        return
    length = min(len(t) for t in traces)
    matrix = np.asarray([t[:length] for t in traces], dtype=float)
    steps = np.arange(1, length + 1)
    mean = matrix.mean(axis=0)
    stderr = matrix.std(axis=0, ddof=1) / math.sqrt(matrix.shape[0])
    ax.fill_between(steps, mean - stderr, mean + stderr, color="#2a7fb8", alpha=0.25)
    ax.plot(
        steps, mean, color="#2a7fb8", lw=2, label=f"mean over {matrix.shape[0]} scenes"
    )
    ax.plot(
        steps,
        np.quantile(matrix, 0.9, axis=0),
        color="#c4472f",
        lw=1.5,
        ls="--",
        label="90th percentile",
    )
    ax.set_xlabel("rollout step")
    ax.set_ylabel("distance between the two egos (m)")
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=9)
    figure.suptitle(
        "Same start, same seed: how far the two models' rollouts separate", fontsize=14
    )
    _save(figure, path)


def main() -> None:
    """Render ten closed-loop example figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=220)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--replan-every", type=int, default=10)
    parser.add_argument("--min-speed", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = PlannerDataset(str(args.parquet), file_capacity=8)
    generator = np.random.default_rng(args.seed)
    labels = (args.label_a, args.label_b)
    models = {
        args.label_a: load_model(args.model_a, DiffusionPlanner).to(device).eval(),
        args.label_b: load_model(args.model_b, DiffusionPlanner).to(device).eval(),
    }
    widen = PlannerUnknownLabelAugmentation()
    normalizer = PlannerDataNormalizer()
    layout = SceneTokenLayout.from_dimensions()
    plots = args.out_dir / "plots"

    indices = sorted(
        generator.choice(
            len(dataset), size=min(args.scenes, len(dataset)), replace=False
        ).tolist()
    )
    collected: list[tuple[int, dict[str, dict[str, Any]]]] = []
    retention: dict[str, list[float]] = {label: [] for label in labels}
    ideals: dict[int, float] = {}
    a_stalls: list[tuple[float, int, dict[str, dict[str, Any]]]] = []
    b_stalls: list[tuple[float, int, dict[str, dict[str, Any]]]] = []
    healthy: list[tuple[float, int, dict[str, dict[str, Any]]]] = []

    for index in tqdm(indices, desc="rollouts", unit="scene"):
        start = dict(widen(_numpy_frame(dataset[index])))
        speed = float(start["ego_agent_past"][-1, EgoIndex.VELOCITY])
        if speed < args.min_speed:
            continue
        ideal = speed * 0.1
        track = len(collected) < 24  # attention tracking is the expensive part
        result = _run(
            models,
            start,
            device,
            normalizer,
            layout,
            steps=args.steps,
            replan_every=args.replan_every,
            track_attention=track,
        )
        if any(len(result[label]["lengths"]) < args.steps for label in labels):
            continue
        finals = _retentions(result, labels, ideal)
        ideals[index] = ideal
        for label in labels:
            retention[label].append(finals[label])
        collected.append((index, result))
        gap = finals[labels[1]] - finals[labels[0]]
        if finals[labels[0]] < STALL <= finals[labels[1]]:
            a_stalls.append((gap, index, result))
        elif finals[labels[1]] < STALL <= finals[labels[0]]:
            b_stalls.append((-gap, index, result))
        elif min(finals.values()) > 0.75:
            healthy.append((abs(gap), index, result))

    print(f"usable scenes: {len(collected)}")
    snapshots = [0, args.steps // 2 - 1, args.steps - 1]
    tracked = [item for item in collected if item[1][labels[0]]["shares"]]

    a_stalls.sort(key=lambda item: -item[0])
    for rank, (gap, index, result) in enumerate(a_stalls[:2], start=1):
        _filmstrip(
            result,
            labels,
            snapshots,
            ideals[index],
            f"{labels[0]} stalls, {labels[1]} does not — scene {index} "
            f"(retention gap {gap:+.2f})",
            plots / f"example_cl_stall_{rank}.png",
        )
        print(f"stall example {rank}: scene {index}, gap {gap:+.3f}")

    if b_stalls:
        b_stalls.sort(key=lambda item: -item[0])
        gap, index, result = b_stalls[0]
        _filmstrip(
            result,
            labels,
            snapshots,
            ideals[index],
            f"Counter-example: {labels[1]} stalls and {labels[0]} does not — "
            f"scene {index}",
            plots / "example_cl_counter.png",
        )
        print(f"counter-example: scene {index}, gap {-gap:+.3f}")
    else:
        print("counter-example: none found")

    if healthy:
        healthy.sort(key=lambda item: item[0])
        _gap, index, result = healthy[0]
        _filmstrip(
            result,
            labels,
            snapshots,
            ideals[index],
            f"Control: both models track the recorded speed — scene {index}",
            plots / "example_cl_healthy.png",
        )
        print(f"healthy control: scene {index}")

    if tracked:
        index, result = tracked[0]
        _drift_trace(result, labels, index, plots / "example_cl_drift_trace.png")
        _lead_agent(tracked, labels, plots / "example_cl_lead_agent.png")
        first = tracked[0][1]
        starts = {label: first[label]["frames"][0] for label in labels}
        finals = {label: first[label]["frames"][-1] for label in labels}
        records = {}
        titles = []
        for label in labels:
            row = []
            for when, frame in (("start", starts[label]), ("final", finals[label])):
                _shares, entries, _lead = _block_shares(
                    models[label], frame, normalizer, layout, device
                )
                records[(label, when)] = entries
                total = sum(r["attention"] for r in entries) or 1.0
                agents = (
                    sum(r["attention"] for r in entries if r["block"] == "neighbors")
                    / total
                )
                row.append(f"{label} · {when} · agent share {agents:.1%}")
            titles.append(row)
        _te._grid_2x2(
            [
                [starts[labels[0]], finals[labels[0]]],
                [starts[labels[1]], finals[labels[1]]],
            ],
            [
                [records[(labels[0], "start")], records[(labels[0], "final")]],
                [records[(labels[1], "start")], records[(labels[1], "final")]],
            ],
            titles,
            f"Attention at the start and end of one rollout — scene {tracked[0][0]}",
            plots / "example_cl_drift_scene.png",
            _te._extent(starts[labels[0]], list(records.values()), None),
        )

    _paths(collected, labels, plots / "example_cl_paths.png")
    _speed_grid(collected, labels, ideals, plots / "example_cl_speed_grid.png")
    _scatter(retention, labels, plots / "example_cl_retention_scatter.png")
    _divergence(collected, labels, plots / "example_cl_divergence.png")
    print(f"wrote closed-loop examples to {plots}")


if __name__ == "__main__":
    main()
