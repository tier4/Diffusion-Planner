"""Visualize per-scene Fusion attention on neighbor-agent tokens.

This is intentionally separate from ``attention_analysis.py``, which produces
dataset-level aggregate statistics.  Here, one scene is rendered with the
ego-token query attention overlaid on each neighbor position.

Examples:
  # Render a specific dataset item.
  uv run python scripts/visualize_neighbor_attention.py \
    --run_dir best_models/20260730/best_model \
    --valid_set_list /path/to/path_list_valid.json \
    --sample_index 123 --device cuda --out_png neighbor_attention.png

  # Search turning scenes and select the one with the most-attended pedestrian.
  uv run python scripts/visualize_neighbor_attention.py \
    --run_dir best_models/20260730/best_model \
    --valid_set_list /path/to/path_list_valid.json \
    --select_class pedestrian --turn_only --candidate_count 128 \
    --device cuda --out_png pedestrian_attention.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusion_planner.dimensions import MAX_NUM_AGENTS, MAX_NUM_NEIGHBORS
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils.visualize_input import visualize_inputs
from matplotlib.lines import Line2D
from token_analysis_common import find_fusion, load_model, patch_fusion, prepare_inputs
from torch.utils.data import DataLoader, Subset

matplotlib.use("Agg", force=True)


NEIGHBOR_OFFSET = 1  # ego occupies token 0
CLASS_NAMES = ("vehicle", "pedestrian", "bicycle")
CLASS_MARKERS = {"vehicle": "o", "pedestrian": "P", "bicycle": "^"}


def load_encoder(run_dir: Path, device: str):
    model, cfg, _ = load_model(run_dir, device)
    return model.encoder, cfg


def neighbor_valid(neighbors: torch.Tensor) -> torch.Tensor:
    """Match NeighborEncoder's validity test on its retained six history rows."""
    return (neighbors[:, :, -6:, :8] != 0).any(dim=(2, 3))


def neighbor_classes(neighbors: torch.Tensor) -> torch.Tensor:
    """Return 0/1/2 for vehicle/pedestrian/bicycle at the current step."""
    return neighbors[:, :, -1, 8:11].argmax(dim=-1)


def layer_indices(layer: str, layer_count: int) -> list[int]:
    if layer == "mean":
        return list(range(layer_count))
    if layer == "last":
        return [layer_count - 1]
    index = int(layer)
    if not 0 <= index < layer_count:
        raise ValueError(f"--layer must be mean, last, or 0..{layer_count - 1}")
    return [index]


def ego_neighbor_attention(store, layer: str) -> torch.Tensor:
    """Return [B, 320] attention from the ego query to neighbor keys."""
    selected = layer_indices(layer, len(store))
    per_layer = [
        store[index]["weights"][:, 0, NEIGHBOR_OFFSET : NEIGHBOR_OFFSET + MAX_NUM_NEIGHBORS]
        for index in selected
    ]
    return torch.stack(per_layer).mean(dim=0)


def movement_and_turn(raw_sample: dict) -> tuple[float, float]:
    future = np.asarray(raw_sample["ego_agent_future"])
    end_x, end_y = future[-1, :2]
    movement = float(np.hypot(end_x, end_y))
    turn_angle = float(abs(np.degrees(np.arctan2(end_y, end_x))))
    return movement, turn_angle


def candidate_indices(
    dataset,
    candidate_count: int,
    move_min_m: float,
    turn_only: bool,
    turn_deg: float,
) -> list[int]:
    target_pool = max(candidate_count * 4, candidate_count)
    stride = max(1, len(dataset) // target_pool)
    selected = []
    for index in range(0, len(dataset), stride):
        movement, turn_angle = movement_and_turn(dataset[index])
        if movement < move_min_m:
            continue
        if turn_only and turn_angle < turn_deg:
            continue
        selected.append(index)
        if len(selected) >= candidate_count:
            break
    if not selected:
        condition = "turning and moving" if turn_only else "moving"
        raise RuntimeError(f"no {condition} candidates found")
    return selected


def select_scene(
    encoder,
    cfg,
    fusion_store,
    dataset,
    indices: list[int],
    select_class: str,
    layer: str,
    batch_size: int,
    device: str,
) -> tuple[int, float]:
    """Choose the sample with the largest single-token attention in a class."""
    class_index = CLASS_NAMES.index(select_class) if select_class != "any" else None
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)
    best_index = None
    best_score = -1.0
    offset = 0
    with torch.no_grad():
        for raw in loader:
            fusion_store.clear()
            encoder(prepare_inputs(dict(raw), cfg, device))
            scores = ego_neighbor_attention(fusion_store, layer)
            neighbors = raw["neighbor_agents_past"]
            valid = neighbor_valid(neighbors).to(scores.device)
            eligible = valid
            if class_index is not None:
                eligible = eligible & (neighbor_classes(neighbors).to(scores.device) == class_index)
            masked_scores = scores.masked_fill(~eligible, -1.0)
            batch_scores = masked_scores.max(dim=1).values
            local = int(batch_scores.argmax())
            score = float(batch_scores[local])
            if score > best_score:
                best_score = score
                best_index = indices[offset + local]
            offset += raw["neighbor_agents_past"].shape[0]
    if best_index is None or best_score < 0:
        raise RuntimeError(f"no valid {select_class} token found in candidate scenes")
    return best_index, best_score


def attention_for_sample(encoder, cfg, fusion_store, sample, layer: str, device: str):
    batch = sample_to_batch(sample)
    with torch.no_grad():
        fusion_store.clear()
        encoder(prepare_inputs(dict(batch), cfg, device))
    scores = ego_neighbor_attention(fusion_store, layer)[0].cpu().numpy()
    layer_scores = np.stack(
        [
            record["weights"][0, 0, NEIGHBOR_OFFSET : NEIGHBOR_OFFSET + MAX_NUM_NEIGHBORS]
            .cpu()
            .numpy()
            for record in fusion_store
        ]
    )
    return batch, scores, layer_scores


def sample_to_batch(sample: dict) -> dict:
    """Convert one raw dataset sample to the batched tensors used for drawing."""
    return {
        key: torch.as_tensor(value).unsqueeze(0)
        for key, value in sample.items()
        if isinstance(value, np.ndarray)
    }


def token_records(sample, scores: np.ndarray, layer_scores: np.ndarray) -> list[dict]:
    neighbors = np.asarray(sample["neighbor_agents_past"])
    valid = (neighbors[:, -6:, :8] != 0).any(axis=(1, 2))
    neighbor_total = float(scores[valid].sum())
    records = []
    for slot in np.flatnonzero(valid):
        current = neighbors[slot, -1]
        class_index = int(np.argmax(current[8:11]))
        score = float(scores[slot])
        records.append(
            {
                "slot": int(slot),
                "class": CLASS_NAMES[class_index],
                "x_m": float(current[0]),
                "y_m": float(current[1]),
                "distance_m": float(np.hypot(current[0], current[1])),
                "attention": score,
                "attention_pct_all_tokens": score * 100.0,
                "attention_pct_within_neighbors": (
                    score / neighbor_total * 100.0 if neighbor_total > 0 else 0.0
                ),
                "attention_per_layer": [float(x) for x in layer_scores[:, slot]],
            }
        )
    return sorted(records, key=lambda item: item["attention"], reverse=True)


def draw_report(
    batch: dict,
    records: list[dict],
    sample_index: int,
    sample_path: str,
    movement_m: float,
    turn_angle_deg: float,
    layer: str,
    top_k: int,
    view_range: float,
    colormap: str,
    marker_size_min: float,
    marker_size_max: float,
    out_png: Path,
    attention_vmax: float | None = None,
    tight_bbox: bool = True,
    attention_label: str = "Attention from ego query",
    title_prefix: str = "Neighbor-token Attention Overlay",
    attention_layer_label: str | None = None,
    prediction=None,
    show_prediction: bool = False,
):
    fig, (scene_ax, rank_ax) = plt.subplots(
        1, 2, figsize=(19, 10), gridspec_kw={"width_ratios": [4.2, 1.15]}
    )
    visual_inputs = {key: value.clone() for key, value in batch.items()}
    visual_inputs["ego_agent_past"] = heading_to_cos_sin(visual_inputs["ego_agent_past"])
    visual_inputs["goal_pose"] = heading_to_cos_sin(visual_inputs["goal_pose"])
    visualize_inputs(visual_inputs, ax=scene_ax, view_ranges=[view_range])
    # Reserve the map corner for token labels rather than the generic ego-state block.
    for text_artist in list(scene_ax.texts):
        text_artist.remove()
    if show_prediction and prediction is not None:
        from visualize_prediction_overlay import draw_prediction_paths

        draw_prediction_paths(scene_ax, batch, prediction, ego=True, neighbors=True)

    values = np.array([record["attention_pct_within_neighbors"] for record in records])
    frame_vmax = float(values.max()) if values.size else 0.0
    vmax = max(attention_vmax if attention_vmax is not None else frame_vmax, 1e-9)
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=vmax)
    cmap = matplotlib.colormaps[colormap]

    for class_name in CLASS_NAMES:
        class_records = [record for record in records if record["class"] == class_name]
        if not class_records:
            continue
        class_values = np.array(
            [record["attention_pct_within_neighbors"] for record in class_records]
        )
        scene_ax.scatter(
            [record["x_m"] for record in class_records],
            [record["y_m"] for record in class_records],
            s=marker_size_min + (marker_size_max - marker_size_min) * class_values / vmax,
            c=class_values,
            cmap=cmap,
            norm=norm,
            marker=CLASS_MARKERS[class_name],
            alpha=0.78,
            edgecolors="black",
            linewidths=0.6,
            zorder=20,
        )

    displayed = records[: min(top_k, len(records))]
    visible_records = [
        record
        for record in records
        if abs(record["x_m"]) <= view_range and abs(record["y_m"]) <= view_range
    ]
    annotated = visible_records[: min(top_k, len(visible_records))]
    record_rank = {record["slot"]: rank for rank, record in enumerate(records, start=1)}
    for record in annotated:
        rank = record_rank[record["slot"]]
        place_left = record["x_m"] > 0
        scene_ax.annotate(
            f"{rank}: {record['class'][0].upper()} {record['attention_pct_within_neighbors']:.1f}%",
            (record["x_m"], record["y_m"]),
            xytext=((-6 if place_left else 6), 7 + 10 * ((rank - 1) % 3)),
            textcoords="offset points",
            ha="right" if place_left else "left",
            fontsize=8,
            weight="bold",
            color="black",
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "black", "alpha": 0.8},
            zorder=30,
        )

    colorbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=scene_ax, fraction=0.035, pad=0.02
    )
    colorbar.set_label(f"Share within neighbor attention (%) — color and marker area")
    legend = [
        Line2D(
            [0],
            [0],
            marker=CLASS_MARKERS[name],
            color="none",
            markerfacecolor="lightgray",
            markeredgecolor="black",
            markersize=9,
            label=name,
        )
        for name in CLASS_NAMES
    ]
    scene_ax.legend(handles=legend, loc="lower left", title="Neighbor class")
    scene_ax.set_title(
        f"{title_prefix} — dataset index {sample_index}\n"
        f"ego-query, {attention_layer_label or f'Fusion layer={layer}'}, "
        f"turn proxy={turn_angle_deg:.1f}°"
    )

    chart_records = list(reversed(displayed))
    labels = [
        f"#{record['slot']} {record['class']}  {record['distance_m']:.1f}m"
        for record in chart_records
    ]
    chart_values = [record["attention_pct_all_tokens"] for record in chart_records]
    colors = [cmap(norm(record["attention_pct_within_neighbors"])) for record in chart_records]
    rank_ax.barh(range(len(chart_records)), chart_values, color=colors, edgecolor="black")
    rank_ax.set_yticks(range(len(chart_records)), labels)
    rank_ax.set_xlabel(f"{attention_label} (% of all valid tokens)")
    rank_ax.set_title(f"Top {len(chart_records)} neighbor tokens")
    rank_ax.grid(axis="x", alpha=0.25)
    rank_ax.tick_params(axis="both", labelsize=8)
    for row, value in enumerate(chart_values):
        rank_ax.text(value, row, f" {value:.3f}%", va="center", fontsize=8)
    rank_ax.text(
        0.0,
        -0.10,
        f"Source NPZ: {Path(sample_path).name}",
        transform=rank_ax.transAxes,
        fontsize=7,
        va="top",
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight" if tight_bbox else None)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overlay ego-query Fusion attention on neighbor agents in one scene."
    )
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--valid_set_list", required=True)
    parser.add_argument("--sample_index", type=int, default=None)
    parser.add_argument(
        "--select_class",
        choices=("any", *CLASS_NAMES),
        default="pedestrian",
        help="class used only when automatically selecting a sample",
    )
    parser.add_argument("--candidate_count", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--move_min_m", type=float, default=5.0)
    parser.add_argument("--turn_only", action="store_true")
    parser.add_argument("--turn_deg", type=float, default=15.0)
    parser.add_argument(
        "--layer", default="mean", help="'mean', 'last', or a zero-based Fusion layer index"
    )
    parser.add_argument("--top_k", type=int, default=12)
    parser.add_argument("--view_range", type=float, default=80.0)
    parser.add_argument(
        "--colormap",
        default="viridis",
        help="Matplotlib sequential colormap used for Attention weight",
    )
    parser.add_argument("--marker_size_min", type=float, default=45.0)
    parser.add_argument("--marker_size_max", type=float, default=950.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_png", default="neighbor_attention.png")
    parser.add_argument(
        "--out_json", default="", help="metadata path; defaults to the PNG path with .json suffix"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.colormap not in matplotlib.colormaps:
        raise ValueError(f"unknown Matplotlib colormap: {args.colormap}")
    if args.marker_size_min <= 0 or args.marker_size_max < args.marker_size_min:
        raise ValueError("--marker_size_max must be >= --marker_size_min > 0")
    run_dir = Path(args.run_dir).resolve()
    dataset = DiffusionPlannerData(args.valid_set_list)
    encoder, cfg = load_encoder(run_dir, args.device)
    fusion = find_fusion(encoder)
    fusion_store = []
    patch_fusion(fusion, fusion_store)

    if args.sample_index is None:
        indices = candidate_indices(
            dataset,
            args.candidate_count,
            args.move_min_m,
            args.turn_only,
            args.turn_deg,
        )
        sample_index, selection_score = select_scene(
            encoder,
            cfg,
            fusion_store,
            dataset,
            indices,
            args.select_class,
            args.layer,
            args.batch_size,
            args.device,
        )
        print(
            f"selected dataset index {sample_index} from {len(indices)} candidates "
            f"(max {args.select_class} attention={selection_score:.6f})",
            flush=True,
        )
    else:
        if not 0 <= args.sample_index < len(dataset):
            raise IndexError(f"--sample_index must be in 0..{len(dataset) - 1}")
        sample_index = args.sample_index

    sample = dataset[sample_index]
    batch, scores, per_layer = attention_for_sample(
        encoder, cfg, fusion_store, sample, args.layer, args.device
    )
    records = token_records(sample, scores, per_layer)
    movement_m, turn_angle_deg = movement_and_turn(sample)
    sample_path = dataset.data_list[sample_index]
    out_png = Path(args.out_png).resolve()
    draw_report(
        batch,
        records,
        sample_index,
        sample_path,
        movement_m,
        turn_angle_deg,
        args.layer,
        args.top_k,
        args.view_range,
        args.colormap,
        args.marker_size_min,
        args.marker_size_max,
        out_png,
    )

    out_json = Path(args.out_json).resolve() if args.out_json else out_png.with_suffix(".json")
    report = {
        "sample_index": sample_index,
        "sample_path": sample_path,
        "movement_m": movement_m,
        "turn_angle_deg": turn_angle_deg,
        "layer": args.layer,
        "attention_definition": (
            "head-averaged Fusion self-attention from the ego-token query; "
            "selected layers are averaged"
        ),
        "visualization": {
            "colormap": args.colormap,
            "marker_size_min": args.marker_size_min,
            "marker_size_max": args.marker_size_max,
            "marker_area_definition": "linear in attention share within valid neighbor tokens",
            "top_k": args.top_k,
            "view_range_m": args.view_range,
        },
        "valid_neighbor_count": len(records),
        "neighbor_attention_pct_all_tokens": float(sum(x["attention"] for x in records) * 100),
        "tokens": records,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as file:
        json.dump(report, file, indent=2)
    print(f"wrote {out_png}", flush=True)
    print(f"wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
