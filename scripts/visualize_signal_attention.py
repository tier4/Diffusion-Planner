"""Visualize attention on lane/route tokens carrying traffic-light attributes."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils.visualize_input import visualize_inputs
from matplotlib.lines import Line2D
from token_analysis_common import load_model, patch_fusion, prepare_inputs
from visualize_all_token_attention import all_token_attention, token_records
from visualize_attention_rollout import attention_rollout, patch_decoder
from visualize_neighbor_attention import find_fusion, movement_and_turn, sample_to_batch

SIGNAL_NAMES = ("green", "yellow", "red", "white", "no_signal")
SIGNAL_COLORS = {
    "green": "#16a34a",
    "yellow": "#eab308",
    "red": "#dc2626",
    "white": "#a855f7",
    "no_signal": "#6b7280",
}


def signal_records(sample, records):
    """Select lane/route tokens with an explicit traffic-light state."""
    result = []
    for record in records:
        if record["class"] not in ("lanes", "route"):
            continue
        key = "lanes" if record["class"] == "lanes" else "route_lanes"
        attributes = np.asarray(sample[key])[record["class_index"], 0, 8:13]
        state_index = int(np.argmax(attributes))
        if attributes[state_index] <= 0 or state_index == 4:
            continue
        item = dict(record)
        item["signal"] = SIGNAL_NAMES[state_index]
        item["signal_color"] = SIGNAL_COLORS[item["signal"]]
        item["attention_pct_signal_tokens"] = 0.0
        result.append(item)
    total = sum(item["attention"] for item in result)
    for item in result:
        item["attention_pct_signal_tokens"] = item["attention"] / max(total, 1e-12) * 100.0
    return sorted(result, key=lambda item: item["attention"], reverse=True)


def draw_signal_report(
    batch,
    sample,
    records,
    sample_index,
    sample_path,
    movement_m,
    turn_angle_deg,
    layer,
    top_k,
    view_range,
    out_png,
    attention_label,
    attention_layer_label,
    attention_vmax=None,
    prediction=None,
    show_prediction: bool = False,
):
    signals = signal_records(sample, records)
    displayed_signals = signals[:top_k]
    fig, (scene_ax, rank_ax) = plt.subplots(
        1, 2, figsize=(19, 10), gridspec_kw={"width_ratios": [4.2, 1.15]}
    )
    visual_inputs = {key: value.clone() for key, value in batch.items()}
    visual_inputs["ego_agent_past"] = heading_to_cos_sin(visual_inputs["ego_agent_past"])
    visual_inputs["goal_pose"] = heading_to_cos_sin(visual_inputs["goal_pose"])
    visualize_inputs(visual_inputs, ax=scene_ax, view_ranges=[view_range])
    for text_artist in list(scene_ax.texts):
        text_artist.remove()
    if show_prediction and prediction is not None:
        from visualize_prediction_overlay import draw_prediction_paths

        draw_prediction_paths(scene_ax, batch, prediction, ego=True, neighbors=True)

    max_value = max(
        attention_vmax or 0.0,
        max((item["attention_pct_all_tokens"] for item in signals), default=0.0),
    )
    for item in displayed_signals:
        key = "lanes" if item["class"] == "lanes" else "route_lanes"
        points = np.asarray(sample[key])[item["class_index"], :, :2]
        valid = np.any(np.abs(points) > 1e-8, axis=1)
        points = points[valid]
        if len(points) < 2:
            continue
        width = 1.5 + 7.0 * item["attention_pct_all_tokens"] / max(max_value, 1e-9)
        scene_ax.plot(
            points[:, 0],
            points[:, 1],
            color=item["signal_color"],
            linewidth=width,
            alpha=0.35 + 0.65 * item["attention_pct_all_tokens"] / max(max_value, 1e-9),
            linestyle="-" if item["class"] == "lanes" else "--",
            zorder=25,
        )
        scene_ax.scatter(
            [item["x_m"]],
            [item["y_m"]],
            s=20 + 180 * item["attention_pct_all_tokens"] / max(max_value, 1e-9),
            color=item["signal_color"],
            edgecolors="black",
            zorder=26,
        )

    visible = [
        item
        for item in displayed_signals
        if abs(item["x_m"]) <= view_range and abs(item["y_m"]) <= view_range
    ]
    for rank, item in enumerate(visible[:top_k], 1):
        scene_ax.annotate(
            f"{rank}: {item['signal']} {item['class']}[{item['class_index']}] "
            f"{item['attention_pct_all_tokens']:.2f}%",
            (item["x_m"], item["y_m"]),
            xytext=(6, 7 + 9 * ((rank - 1) % 3)),
            textcoords="offset points",
            fontsize=7,
            weight="bold",
            color="black",
            bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "black", "alpha": 0.8},
            zorder=30,
        )
    scene_ax.legend(
        handles=[
            Line2D([0], [0], color=color, linewidth=5, label=name)
            for name, color in SIGNAL_COLORS.items()
            if name != "no_signal"
        ],
        loc="lower left",
        title="Traffic-light state (line: lane, dashed: route)",
        fontsize=8,
    )
    scene_ax.set_title(
        f"Signal-aware Attention — dataset index {sample_index}\n"
        f"{attention_layer_label}, movement={movement_m:.1f} m, turn proxy={turn_angle_deg:.1f}°"
    )

    displayed = list(reversed(displayed_signals))
    labels = [
        f"#{x['token_index']} {x['signal']} {x['class']}[{x['class_index']}]" for x in displayed
    ]
    values = [x["attention_pct_all_tokens"] for x in displayed]
    rank_ax.barh(
        range(len(displayed)),
        values,
        color=[x["signal_color"] for x in displayed],
        edgecolor="black",
    )
    rank_ax.set_yticks(range(len(displayed)), labels)
    rank_ax.set_xlabel(f"{attention_label} (% of all valid tokens)")
    rank_ax.set_title(f"Top {len(displayed)} signal-bearing tokens")
    rank_ax.grid(axis="x", alpha=0.25)
    for row, value in enumerate(values):
        rank_ax.text(value, row, f" {value:.3f}%", va="center", fontsize=7)
    total = sum(x["attention_pct_all_tokens"] for x in signals)
    rank_ax.text(
        0,
        -0.12,
        f"Signal-bearing lane/route total: {total:.3f}%\n"
        f"Within-signal ranking uses {len(signals)} tokens.",
        transform=rank_ax.transAxes,
        fontsize=8,
        va="top",
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return signals


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--valid_set_list", required=True)
    parser.add_argument("--sample_index", type=int, required=True)
    parser.add_argument("--layer", default="mean")
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--view_range", type=float, default=80.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_fusion_png", required=True)
    parser.add_argument("--out_rollout_png", required=True)
    parser.add_argument("--out_json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = DiffusionPlannerData(args.valid_set_list)
    sample = dataset[args.sample_index]
    batch = sample_to_batch(sample)
    model, cfg, _ = load_model(Path(args.run_dir).resolve(), args.device)
    fusion_store, decoder_store = [], []
    patch_fusion(find_fusion(model.encoder), fusion_store)
    patch_decoder(model.decoder, decoder_store)
    with torch.no_grad():
        inputs = prepare_inputs(dict(batch), cfg, args.device)
        encoding = model.encoder(inputs)
        model.decoder(encoding, inputs)
    fusion_scores = all_token_attention(fusion_store, args.layer)[0]
    rollout_scores, valid = attention_rollout(fusion_store, decoder_store)
    fusion_records = token_records(sample, fusion_scores.cpu().numpy(), valid.cpu().numpy())
    rollout_records = token_records(sample, rollout_scores.cpu().numpy(), valid.cpu().numpy())
    movement, turn = movement_and_turn(sample)
    common = dict(
        batch=batch,
        sample=sample,
        sample_index=args.sample_index,
        sample_path=dataset.data_list[args.sample_index],
        movement_m=movement,
        turn_angle_deg=turn,
        layer=args.layer,
        top_k=args.top_k,
        view_range=args.view_range,
    )
    fusion_signals = draw_signal_report(
        out_png=Path(args.out_fusion_png),
        attention_label="Fusion attention",
        attention_layer_label="Fusion ego-query attention",
        records=fusion_records,
        **common,
    )
    rollout_signals = draw_signal_report(
        out_png=Path(args.out_rollout_png),
        attention_label="Decoder-to-input rollout",
        attention_layer_label="Decoder cross-attention rolled through Fusion",
        records=rollout_records,
        **common,
    )
    with Path(args.out_json).open("w") as file:
        json.dump(
            {
                "sample_index": args.sample_index,
                "sample_path": dataset.data_list[args.sample_index],
                "fusion_signal_tokens": fusion_signals,
                "rollout_signal_tokens": rollout_signals,
            },
            file,
            indent=2,
        )
    print(f"wrote {args.out_fusion_png}")
    print(f"wrote {args.out_rollout_png}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
