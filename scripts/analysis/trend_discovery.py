"""Search two planners for attention differences, open loop and closed loop.

Rather than testing a fixed hypothesis, this scans a wide metric set over paired
frames and ranks every metric by how consistently the two models differ, so
trends nobody thought to look for can surface. It then runs many closed-loop
rollouts, which is the only way to see whether an attention difference shows up
in behaviour once the model drives the scene it is looking at.

Everything is paired: both models see the same frames and the same rollout
starts in the same order, and differences are averaged per frame. Sigma is the
mean paired difference over its standard error, so it measures consistency
across frames, not causation.
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
    AGENT_CLASS_NAMES,
    BEARING_BUCKETS,
    BOUNDARY_BUCKETS,
    SceneTokenLayout,
    annotate_records,
    capture_fusion_attention,
    ego_query_attention,
    fusion_dimensions,
    nearest_route_rank,
    rollout,
    token_records,
)
from diffusion_planner.data import PlannerDataset
from diffusion_planner.data.transforms import (
    PlannerDataNormalizer,
    PlannerUnknownLabelAugmentation,
)
from diffusion_planner.models.diffusion_planner import DiffusionPlanner
from diffusion_planner.utils.checkpoint import load_model
from diffusion_planner.visualizer.schema import EgoIndex

DISTANCE_BANDS = ((0.0, 20.0, "0-20m"), (20.0, 50.0, "20-50m"), (50.0, 1e9, "50m+"))
KEY_BLOCKS = ("lanes", "neighbors", "road_borders", "route_lanes", "ego_history")


def _numpy_frame(frame: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value))
        for key, value in frame.items()
    }


def _capture_all_layers(
    model: DiffusionPlanner,
    frame: dict[str, Any],
    normalizer: PlannerDataNormalizer,
    layout: SceneTokenLayout,
    device: torch.device,
    layer_count: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Ego-query attention averaged over layers, plus one row per layer."""
    normalized = normalizer({key: np.asarray(v) for key, v in frame.items()})
    data = {
        key: torch.as_tensor(value, device=device).unsqueeze(0)
        for key, value in normalized.items()
    }
    with capture_fusion_attention(model) as capture, torch.no_grad():
        model.scene_encoder(data)
    mean = ego_query_attention(capture, layout)[0].detach().float().cpu().numpy()
    per_layer = [
        ego_query_attention(capture, layout, layer=index)[0]
        .detach()
        .float()
        .cpu()
        .numpy()
        for index in range(layer_count)
    ]
    return mean, per_layer


def _entropy(values: np.ndarray) -> float:
    """Attention entropy normalized to [0, 1], 1 being perfectly uniform."""
    positive = values[values > 0.0]
    if positive.size <= 1:
        return 0.0
    probabilities = positive / positive.sum()
    return float(
        -(probabilities * np.log(probabilities)).sum() / math.log(positive.size)
    )


def _frame_metrics(
    records: list[dict[str, Any]],
    attention: np.ndarray,
    per_layer: list[np.ndarray],
    layout: SceneTokenLayout,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    total = sum(record["attention"] for record in records) or 1.0

    by_block: dict[str, float] = defaultdict(float)
    tokens: dict[str, int] = defaultdict(int)
    for record in records:
        by_block[record["block"]] += record["attention"]
        tokens[record["block"]] += 1
    for name, value in by_block.items():
        metrics[f"block/{name}"] = value / total
        share_tokens = tokens[name] / max(len(records), 1)
        metrics[f"selectivity/{name}"] = (
            (value / total) / share_tokens if share_tokens > 0 else math.nan
        )

    neighbor_total = by_block.get("neighbors", 0.0) or 1.0
    per_class: dict[str, float] = defaultdict(float)
    per_bearing: dict[str, float] = defaultdict(float)
    for record in records:
        if record["block"] != "neighbors":
            continue
        per_class[record.get("agent_class", "?")] += record["attention"]
        if "bearing" in record:
            per_bearing[record["bearing"]] += record["attention"]
    for name in (*AGENT_CLASS_NAMES, "unlabeled"):
        metrics[f"class/{name}"] = per_class.get(name, 0.0) / neighbor_total
    for name in BEARING_BUCKETS:
        metrics[f"agent_bearing/{name}"] = per_bearing.get(name, 0.0) / neighbor_total

    lane_total = by_block.get("lanes", 0.0) or 1.0
    for bucket in BOUNDARY_BUCKETS:
        flag = f"lane_{bucket}"
        metrics[f"lane/{bucket}"] = (
            sum(
                r["attention"] for r in records if r["block"] == "lanes" and r.get(flag)
            )
            / lane_total
        )

    # Where attention lands in range, and how sharply it is focused.
    spatial = [r for r in records if r["distance_m"] is not None]
    spatial_total = sum(r["attention"] for r in spatial) or 1.0
    for low, high, name in DISTANCE_BANDS:
        metrics[f"range/{name}"] = (
            sum(r["attention"] for r in spatial if low <= r["distance_m"] < high)
            / spatial_total
        )
    if spatial:
        metrics["range/mean_attended_m"] = (
            sum(r["attention"] * r["distance_m"] for r in spatial) / spatial_total
        )

    ordered = sorted((r["attention"] for r in records), reverse=True)
    metrics["focus/top1"] = ordered[0] / total if ordered else math.nan
    metrics["focus/top10"] = sum(ordered[:10]) / total
    metrics["focus/entropy"] = _entropy(attention[attention > 0.0])
    metrics["focus/valid_tokens"] = float(len(records))

    top_block = max(by_block.items(), key=lambda item: item[1])[0] if by_block else ""
    for name in layout.names:
        metrics[f"top_block/{name}"] = 1.0 if name == top_block else 0.0

    rank, share = nearest_route_rank(records)
    if rank is not None:
        metrics["route/nearest_rank"] = float(rank)
        metrics["route/nearest_is_first"] = 1.0 if rank == 1 else 0.0

    # Per layer, so a trend that only exists deep in the stack is visible.
    for index, row in enumerate(per_layer):
        layer_total = float(row.sum()) or 1.0
        for name in KEY_BLOCKS:
            block = layout.block(name)
            metrics[f"layer{index}/{name}"] = (
                float(row[block.start : block.stop].sum()) / layer_total
            )
        metrics[f"layer{index}/entropy"] = _entropy(row[row > 0.0])
    return metrics


def _paired(a: dict[str, list[float]], b: dict[str, list[float]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted(set(a) | set(b)):
        left = np.asarray(a.get(key, []), dtype=float)
        right = np.asarray(b.get(key, []), dtype=float)
        if left.size == 0 or left.size != right.size:
            continue
        finite = np.isfinite(left) & np.isfinite(right)
        if finite.sum() < 2:
            continue
        left, right = left[finite], right[finite]
        difference = right - left
        stderr = float(difference.std(ddof=1) / math.sqrt(difference.size))
        mean = float(difference.mean())
        scale = float(max(abs(left.mean()), 1e-12))
        out[key] = {
            "a": float(left.mean()),
            "b": float(right.mean()),
            "difference": mean,
            "relative": mean / scale,
            "stderr": stderr,
            "sigmas": abs(mean) / stderr if stderr > 0 else math.nan,
            "frames": int(difference.size),
        }
    return out


def _open_loop(
    models: dict[str, DiffusionPlanner],
    dataset: PlannerDataset,
    indices: list[int],
    device: torch.device,
) -> dict[str, Any]:
    widen = PlannerUnknownLabelAugmentation()
    normalizer = PlannerDataNormalizer()
    layout = SceneTokenLayout.from_dimensions()
    layer_count = fusion_dimensions(next(iter(models.values())))[0]
    collected: dict[str, dict[str, list[float]]] = {
        k: defaultdict(list) for k in models
    }

    for index in tqdm(indices, desc="open loop", unit="frame"):
        frame = dict(widen(_numpy_frame(dataset[index])))
        for label, model in models.items():
            attention, per_layer = _capture_all_layers(
                model, frame, normalizer, layout, device, layer_count
            )
            records = annotate_records(token_records(frame, attention, layout), frame)
            for key, value in _frame_metrics(
                records, attention, per_layer, layout
            ).items():
                collected[label][key].append(value)
    labels = list(models)
    return {
        "paired": _paired(collected[labels[0]], collected[labels[1]]),
        "layer_count": layer_count,
    }


def _closed_loop(
    models: dict[str, DiffusionPlanner],
    dataset: PlannerDataset,
    indices: list[int],
    device: torch.device,
    *,
    steps: int,
    replan_every: int,
    sampling_steps: int,
) -> dict[str, Any]:
    """Roll each start scene forward under both models and compare behaviour."""
    widen = PlannerUnknownLabelAugmentation()
    traces: dict[str, list[list[float]]] = {label: [] for label in models}
    summary: dict[str, dict[str, list[float]]] = {
        label: defaultdict(list) for label in models
    }

    for index in tqdm(indices, desc="closed loop", unit="scene"):
        frame = dict(widen(_numpy_frame(dataset[index])))
        speed0 = float(frame["ego_agent_past"][-1, EgoIndex.VELOCITY])
        if speed0 < 1.0:
            continue  # A parked ego has no speed to retain.
        for label, model in models.items():
            previous, lengths = 0.0, []
            for state in rollout(
                model,
                frame,
                steps=steps,
                device=str(device),
                num_sampling_steps=sampling_steps,
                replan_every=replan_every,
                stop_beyond_map=False,
            ):
                lengths.append(state.travelled_m - previous)
                previous = state.travelled_m
            if not lengths:
                continue
            traces[label].append(lengths)
            ideal = speed0 * 0.1
            summary[label]["travelled_m"].append(previous)
            summary[label]["retention"].append(float(np.mean(lengths)) / ideal)
            summary[label]["final_retention"].append(
                float(np.mean(lengths[-5:])) / ideal
            )
            summary[label]["stalled"].append(
                1.0 if float(np.mean(lengths[-5:])) < 0.25 * ideal else 0.0
            )
    labels = list(models)
    return {
        "paired": _paired(summary[labels[0]], summary[labels[1]]),
        "traces": {label: rows for label, rows in traces.items()},
        "scenes": len(traces[labels[0]]),
    }


def _save(figure: go.Figure, path: Path, width: int = 1000, height: int = 600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(figure.to_image(format="png", width=width, height=height, scale=2))


def _plot_top_differences(
    paired: dict[str, Any], labels: tuple[str, str], path: Path, top: int = 18
) -> None:
    rows = [
        (key, value)
        for key, value in paired.items()
        if np.isfinite(value["sigmas"]) and not key.startswith(("layer", "top_block"))
    ]
    rows.sort(key=lambda item: -item[1]["sigmas"])
    rows = rows[:top][::-1]
    # Plot relative change, not the raw difference: the metrics mix units —
    # shares, selectivity ratios and metres — so one absolute axis would let the
    # metre-valued rows swamp everything else.
    values = [100.0 * value["relative"] for _, value in rows]
    errors = [
        100.0 * value["stderr"] / max(abs(value["a"]), 1e-12) for _, value in rows
    ]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=[key for key, _ in rows],
            orientation="h",
            error_x={"type": "data", "array": errors},
            marker_color=["#2a7fb8" if v > 0 else "#c4472f" for v in values],
            hovertemplate="%{y}: %{x:+.1f}%<extra></extra>",
        )
    )
    figure.update_layout(
        title=(
            f"Largest paired differences ({labels[1]} − {labels[0]}), "
            "ranked by consistency"
        ),
        xaxis_title=f"change relative to {labels[0]} (%)",
        margin={"l": 220, "r": 40, "t": 60, "b": 50},
        template="plotly_white",
    )
    _save(figure, path, height=750)


def _plot_layers(
    paired: dict[str, Any], labels: tuple[str, str], layer_count: int, path: Path
) -> None:
    figure = go.Figure()
    palette = ["#2a7fb8", "#c4472f", "#4a9e5c", "#9a6fb0", "#d8973c"]
    for colour, block in zip(palette, KEY_BLOCKS, strict=False):
        layers = list(range(layer_count))
        a = [paired.get(f"layer{i}/{block}", {}).get("a", math.nan) for i in layers]
        b = [paired.get(f"layer{i}/{block}", {}).get("b", math.nan) for i in layers]
        figure.add_trace(
            go.Scatter(
                x=layers,
                y=a,
                name=f"{block} · {labels[0]}",
                line={"color": colour, "dash": "dot"},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=layers, y=b, name=f"{block} · {labels[1]}", line={"color": colour}
            )
        )
    figure.update_layout(
        title="Share of ego-query attention by fusion layer",
        xaxis_title="fusion layer",
        yaxis_title="share of attention",
        template="plotly_white",
        margin={"l": 70, "r": 40, "t": 60, "b": 50},
    )
    _save(figure, path)


def _plot_grouped(
    paired: dict[str, Any],
    prefix: str,
    labels: tuple[str, str],
    title: str,
    path: Path,
) -> None:
    keys = [k for k in paired if k.startswith(prefix)]
    keys.sort(key=lambda k: -paired[k]["a"])
    if not keys:
        return
    names = [k.split("/", 1)[1] for k in keys]
    figure = go.Figure(
        [
            go.Bar(
                name=labels[0],
                x=names,
                y=[paired[k]["a"] for k in keys],
                marker_color="#8aa8bd",
            ),
            go.Bar(
                name=labels[1],
                x=names,
                y=[paired[k]["b"] for k in keys],
                marker_color="#2a7fb8",
                error_y={"type": "data", "array": [paired[k]["stderr"] for k in keys]},
            ),
        ]
    )
    figure.update_layout(
        barmode="group",
        title=title,
        template="plotly_white",
        margin={"l": 70, "r": 40, "t": 60, "b": 60},
    )
    _save(figure, path)


def _plot_closed_loop(
    traces: dict[str, list[list[float]]], labels: tuple[str, str], path: Path
) -> None:
    figure = go.Figure()
    for colour, label in zip(("#8aa8bd", "#2a7fb8"), labels, strict=False):
        rows = traces.get(label) or []
        if not rows:
            continue
        length = min(len(row) for row in rows)
        matrix = np.asarray([row[:length] for row in rows], dtype=float)
        mean = matrix.mean(axis=0)
        stderr = matrix.std(axis=0, ddof=1) / math.sqrt(matrix.shape[0])
        steps = list(range(1, length + 1))
        figure.add_trace(
            go.Scatter(
                x=steps + steps[::-1],
                y=list(mean + stderr) + list((mean - stderr)[::-1]),
                fill="toself",
                fillcolor=colour,
                opacity=0.25,
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(x=steps, y=mean, name=label, line={"color": colour, "width": 2})
        )
    figure.update_layout(
        title="Closed loop: distance advanced per step, mean ± standard error",
        xaxis_title="rollout step",
        yaxis_title="metres per step",
        template="plotly_white",
        margin={"l": 70, "r": 40, "t": 60, "b": 50},
    )
    _save(figure, path)


def main() -> None:
    """Scan for differences and write plots plus a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--scenes", type=int, default=192)
    parser.add_argument("--rollout-steps", type=int, default=40)
    parser.add_argument("--replan-every", type=int, default=10)
    parser.add_argument("--sampling-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--replot",
        action="store_true",
        help="redraw plots from an existing trends.json instead of recomputing",
    )
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.replot:
        saved = json.loads((args.out_dir / "trends.json").read_text())
        labels = (saved["model_a"]["label"], saved["model_b"]["label"])
        paired = saved["open_loop"]
        layers = 1 + max(
            (int(k[5]) for k in paired if k.startswith("layer") and k[5].isdigit()),
            default=0,
        )
        plots = args.out_dir / "plots"
        _plot_top_differences(paired, labels, plots / "top_differences.png")
        _plot_layers(paired, labels, layers, plots / "by_layer.png")
        _plot_grouped(
            paired,
            "block/",
            labels,
            "Share of ego-query attention by token block",
            plots / "blocks.png",
        )
        _plot_grouped(
            paired,
            "agent_bearing/",
            labels,
            "Share of neighbor attention by bearing",
            plots / "bearing.png",
        )
        _plot_grouped(
            paired,
            "range/",
            labels,
            "Where attention lands, by range",
            plots / "range.png",
        )
        _plot_grouped(
            paired,
            "selectivity/",
            labels,
            "Selectivity by token block",
            plots / "selectivity.png",
        )
        print(f"redrew plots in {plots}")
        return

    device = torch.device(args.device)
    dataset = PlannerDataset(str(args.parquet), file_capacity=8)
    generator = np.random.default_rng(args.seed)
    total = len(dataset)
    frames = sorted(
        generator.choice(total, size=min(args.frames, total), replace=False).tolist()
    )
    scenes = sorted(
        generator.choice(total, size=min(args.scenes, total), replace=False).tolist()
    )
    labels = (args.label_a, args.label_b)
    models = {
        args.label_a: load_model(args.model_a, DiffusionPlanner).to(device).eval(),
        args.label_b: load_model(args.model_b, DiffusionPlanner).to(device).eval(),
    }
    print(
        f"{len(frames)} open-loop frames, {len(scenes)} closed-loop scenes on {device}"
    )

    open_loop = _open_loop(models, dataset, frames, device)
    closed = _closed_loop(
        models,
        dataset,
        scenes,
        device,
        steps=args.rollout_steps,
        replan_every=args.replan_every,
        sampling_steps=args.sampling_steps,
    )

    plots = args.out_dir / "plots"
    _plot_top_differences(open_loop["paired"], labels, plots / "top_differences.png")
    _plot_layers(
        open_loop["paired"], labels, open_loop["layer_count"], plots / "by_layer.png"
    )
    _plot_grouped(
        open_loop["paired"],
        "block/",
        labels,
        "Share of ego-query attention by token block",
        plots / "blocks.png",
    )
    _plot_grouped(
        open_loop["paired"],
        "agent_bearing/",
        labels,
        "Share of neighbor attention by bearing",
        plots / "bearing.png",
    )
    _plot_grouped(
        open_loop["paired"],
        "range/",
        labels,
        "Where attention lands, by range",
        plots / "range.png",
    )
    _plot_grouped(
        open_loop["paired"],
        "selectivity/",
        labels,
        "Selectivity by token block",
        plots / "selectivity.png",
    )
    _plot_closed_loop(closed["traces"], labels, plots / "closed_loop.png")

    report = {
        "model_a": {"label": args.label_a, "path": str(args.model_a)},
        "model_b": {"label": args.label_b, "path": str(args.model_b)},
        "open_loop_frames": len(frames),
        "closed_loop_scenes": closed["scenes"],
        "rollout_steps": args.rollout_steps,
        "replan_every": args.replan_every,
        "open_loop": open_loop["paired"],
        "closed_loop": closed["paired"],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "trends.json").write_text(json.dumps(report, indent=2))

    ranked = sorted(
        (
            (k, v)
            for k, v in open_loop["paired"].items()
            if np.isfinite(v["sigmas"]) and not k.startswith("top_block")
        ),
        key=lambda item: -item[1]["sigmas"],
    )
    print(f"\ntop 20 most consistent differences ({args.label_b} − {args.label_a})")
    print(f"  {'metric':<30}{'A':>10}{'B':>10}{'diff':>11}{'rel%':>9}{'sigmas':>8}")
    for key, value in ranked[:20]:
        print(
            f"  {key:<30}{value['a']:>10.4f}{value['b']:>10.4f}"
            f"{value['difference']:>+11.4f}{value['relative'] * 100:>+9.1f}"
            f"{value['sigmas']:>8.1f}"
        )
    print(f"\nclosed loop over {closed['scenes']} scenes")
    for key, value in sorted(closed["paired"].items()):
        print(
            f"  {key:<22}{value['a']:>10.4f}{value['b']:>10.4f}"
            f"{value['difference']:>+11.4f}{value['sigmas']:>8.1f}"
        )
    print(f"\nwrote {args.out_dir / 'trends.json'} and {plots}/*.png")


if __name__ == "__main__":
    main()
