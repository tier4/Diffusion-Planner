"""Compare two planners' attention over the same frames, paired frame by frame.

Two independently trained models differ for reasons beyond whatever changed
between them, and scene composition is by far the largest of those reasons. So
every statistic here is computed per frame for both models and then averaged as
a *paired difference*, which cancels the scene term. Differences are reported
with a standard error over frames; anything under about two standard errors
should be treated as unresolved rather than as a trend.

Example:
  uv run python scripts/analysis/compare_attention.py \\
    --model-a /path/to/three_class_label4.pth --label-a 3-class \\
    --model-b /path/to/four_class.pth --label-b 4-class \\
    --parquet $HOME/datasets/hdf5_small/indexes/train.parquet \\
    --frames 512 --json-out comparison.json
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
    nearest_route_rank,
    token_records,
)
from diffusion_planner.analysis.scene_edit import PlacedAgent, insert_agent
from diffusion_planner.data import PlannerDataset
from diffusion_planner.data.transforms import (
    PlannerDataNormalizer,
    PlannerUnknownLabelAugmentation,
)
from diffusion_planner.models.diffusion_planner import DiffusionPlanner
from diffusion_planner.utils.checkpoint import load_model

LANE_FLAGS = tuple(f"lane_{name}" for name in BOUNDARY_BUCKETS)


def _numpy_frame(frame: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value))
        for key, value in frame.items()
    }


def _attention(
    model: DiffusionPlanner,
    frame: dict[str, Any],
    normalizer: PlannerDataNormalizer,
    layout: SceneTokenLayout,
    device: torch.device,
) -> np.ndarray:
    normalized = normalizer({key: np.asarray(v) for key, v in frame.items()})
    data = {
        key: torch.as_tensor(value, device=device).unsqueeze(0)
        for key, value in normalized.items()
    }
    with capture_fusion_attention(model) as capture, torch.no_grad():
        model.scene_encoder(data)
    return ego_query_attention(capture, layout)[0].detach().float().cpu().numpy()


def _frame_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    """Per-frame shares, so they can be paired across models."""
    metrics: dict[str, float] = {}
    total = sum(record["attention"] for record in records) or 1.0

    by_block: dict[str, float] = defaultdict(float)
    block_tokens: dict[str, int] = defaultdict(int)
    for record in records:
        by_block[record["block"]] += record["attention"]
        block_tokens[record["block"]] += 1
    for name, value in by_block.items():
        metrics[f"block/{name}"] = value / total
        share_of_tokens = block_tokens[name] / max(len(records), 1)
        metrics[f"selectivity/{name}"] = (
            (value / total) / share_of_tokens if share_of_tokens > 0 else math.nan
        )

    neighbor_total = by_block.get("neighbors", 0.0) or 1.0
    by_class: dict[str, float] = defaultdict(float)
    by_bearing: dict[str, float] = defaultdict(float)
    for record in records:
        if record["block"] != "neighbors":
            continue
        by_class[record.get("agent_class", "?")] += record["attention"]
        if "bearing" in record:
            by_bearing[record["bearing"]] += record["attention"]
    for name in (*AGENT_CLASS_NAMES, "unlabeled"):
        metrics[f"class/{name}"] = by_class.get(name, 0.0) / neighbor_total
    for name in BEARING_BUCKETS:
        metrics[f"agent_bearing/{name}"] = by_bearing.get(name, 0.0) / neighbor_total

    lane_total = by_block.get("lanes", 0.0) or 1.0
    for flag in LANE_FLAGS:
        metrics[f"lane/{flag.removeprefix('lane_')}"] = (
            sum(
                r["attention"] for r in records if r["block"] == "lanes" and r.get(flag)
            )
            / lane_total
        )

    rank, share = nearest_route_rank(records)
    if rank is not None:
        metrics["route/nearest_rank"] = float(rank)
        metrics["route/nearest_is_first"] = 1.0 if rank == 1 else 0.0
        metrics["route/nearest_share"] = float(share or 0.0) / 100.0
    return metrics


def _paired(
    values_a: dict[str, list[float]], values_b: dict[str, list[float]]
) -> dict[str, dict[str, float]]:
    """Mean of each model, plus the paired difference and its standard error."""
    result: dict[str, dict[str, float]] = {}
    for key in sorted(set(values_a) | set(values_b)):
        a = np.asarray(values_a.get(key, []), dtype=float)
        b = np.asarray(values_b.get(key, []), dtype=float)
        if a.size == 0 or b.size == 0 or a.size != b.size:
            continue
        finite = np.isfinite(a) & np.isfinite(b)
        if not finite.any():
            continue
        a, b = a[finite], b[finite]
        difference = b - a
        stderr = (
            float(difference.std(ddof=1) / math.sqrt(difference.size))
            if difference.size > 1
            else math.nan
        )
        mean_difference = float(difference.mean())
        result[key] = {
            "a": float(a.mean()),
            "b": float(b.mean()),
            "difference": mean_difference,
            "stderr": stderr,
            "sigmas": abs(mean_difference) / stderr
            if stderr and stderr > 0
            else math.nan,
            "frames": int(difference.size),
        }
    return result


def _collect(
    model: DiffusionPlanner,
    dataset: PlannerDataset,
    indices: list[int],
    *,
    device: torch.device,
    widen: PlannerUnknownLabelAugmentation,
    normalizer: PlannerDataNormalizer,
    layout: SceneTokenLayout,
    inject: PlacedAgent | None,
    label: str,
) -> tuple[dict[str, list[float]], list[float]]:
    """Per-frame metrics for one model over a fixed frame list."""
    collected: dict[str, list[float]] = defaultdict(list)
    injected: list[float] = []
    for index in tqdm(indices, desc=label, unit="frame"):
        frame = dict(widen(_numpy_frame(dataset[index])))
        slot: int | None = None
        if inject is not None:
            try:
                frame, slot = insert_agent(frame, inject)
            except ValueError:
                continue
        attention = _attention(model, frame, normalizer, layout, device)
        records = annotate_records(token_records(frame, attention, layout), frame)
        for key, value in _frame_metrics(records).items():
            collected[key].append(value)
        if slot is not None:
            injected.extend(
                record["attention_pct"]
                for record in records
                if record["block"] == "neighbors" and record["block_index"] == slot
            )
    return collected, injected


def _print_table(title: str, rows: dict[str, dict[str, float]], prefix: str) -> None:
    selected = {k: v for k, v in rows.items() if k.startswith(prefix)}
    if not selected:
        return
    print(f"\n{title}")
    print(f"  {'metric':<26}{'A':>10}{'B':>10}{'B-A':>11}{'±stderr':>10}{'sigmas':>8}")
    for key, values in sorted(selected.items(), key=lambda i: -abs(i[1]["difference"])):
        name = key.split("/", 1)[1]
        print(
            f"  {name:<26}{values['a']:>10.4f}{values['b']:>10.4f}"
            f"{values['difference']:>+11.4f}{values['stderr']:>10.4f}"
            f"{values['sigmas']:>8.1f}"
        )


def main() -> None:
    """Run both models over one frame list and report paired differences."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--inject-sweep",
        action="store_true",
        help="also sweep an injected agent's label through both models",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = PlannerDataset(str(args.parquet), file_capacity=8)
    generator = np.random.default_rng(args.seed)
    total = len(dataset)
    count = min(args.frames, total)
    indices = sorted(generator.choice(total, size=count, replace=False).tolist())

    widen = PlannerUnknownLabelAugmentation()
    normalizer = PlannerDataNormalizer()
    layout = SceneTokenLayout.from_dimensions()
    models = {
        args.label_a: load_model(args.model_a, DiffusionPlanner).to(device).eval(),
        args.label_b: load_model(args.model_b, DiffusionPlanner).to(device).eval(),
    }
    print(f"{count} frames from {args.parquet} on {device}")
    print(f"  A = {args.label_a}: {args.model_a}")
    print(f"  B = {args.label_b}: {args.model_b}")

    collected = {
        label: _collect(
            model,
            dataset,
            indices,
            device=device,
            widen=widen,
            normalizer=normalizer,
            layout=layout,
            inject=None,
            label=label,
        )[0]
        for label, model in models.items()
    }
    paired = _paired(collected[args.label_a], collected[args.label_b])

    _print_table("share of total attention, by token block", paired, "block/")
    _print_table("selectivity, by token block", paired, "selectivity/")
    _print_table("share of neighbor attention, by agent class", paired, "class/")
    _print_table("share of neighbor attention, by bearing", paired, "agent_bearing/")
    _print_table("share of lane attention, by boundary bucket", paired, "lane/")
    _print_table("nearest route token", paired, "route/")

    report: dict[str, Any] = {
        "model_a": {"label": args.label_a, "path": str(args.model_a)},
        "model_b": {"label": args.label_b, "path": str(args.model_b)},
        "frames": count,
        "paired": paired,
    }

    if args.inject_sweep:
        sweep: dict[str, dict[str, float]] = {}
        print("\ninjected agent, same geometry, label varied (mean share %)")
        for agent_class in (*AGENT_CLASS_NAMES, "unlabeled"):
            placement = PlacedAgent(
                x_m=12.0,
                y_m=2.0,
                speed_mps=1.4,
                width_m=0.8,
                length_m=0.8,
                agent_class=agent_class,
            )
            row: dict[str, float] = {}
            for label, model in models.items():
                _, shares = _collect(
                    model,
                    dataset,
                    indices,
                    device=device,
                    widen=widen,
                    normalizer=normalizer,
                    layout=layout,
                    inject=placement,
                    label=f"{label}/{agent_class}",
                )
                row[label] = float(np.mean(shares)) if shares else math.nan
            sweep[agent_class] = row
            print(
                f"  {agent_class:<12}{args.label_a}={row[args.label_a]:.4f}%  "
                f"{args.label_b}={row[args.label_b]:.4f}%"
            )
        report["inject_sweep"] = sweep

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
