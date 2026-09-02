"""Visualize Fusion attention and Decoder-to-input attention rollout."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import torch
from diffusion_planner.utils.dataset import DiffusionPlannerData
from token_analysis_common import load_model, patch_fusion, prepare_inputs
from visualize_all_token_attention import (
    all_token_attention,
    class_summary,
    draw_report,
    token_records,
)
from visualize_neighbor_attention import draw_report as draw_neighbor_report
from visualize_neighbor_attention import find_fusion, movement_and_turn, sample_to_batch


def patch_decoder(decoder, store):
    """Capture head-averaged Decoder cross-attention for every diffusion call."""
    for layer_index, block in enumerate(decoder.dit.blocks):

        def make_forward(layer, index):
            def forward(x, cross_c, y, attn_mask, cross_attn_mask):
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    layer.adaLN_modulation(y).chunk(6, dim=2)
                )
                modulated_x = layer.norm1(x) * (1 + scale_msa) + shift_msa
                x = (
                    x
                    + gate_msa
                    * layer.attn(
                        modulated_x,
                        modulated_x,
                        modulated_x,
                        key_padding_mask=attn_mask,
                        need_weights=False,
                    )[0]
                )
                modulated_x = layer.norm2(x) * (1 + scale_mlp) + shift_mlp
                x = x + gate_mlp * layer.mlp1(modulated_x)

                query = layer.norm3(x)
                cross_output, weights = layer.cross_attn(
                    query,
                    cross_c,
                    cross_c,
                    key_padding_mask=cross_attn_mask,
                    need_weights=True,
                    average_attn_weights=True,
                )
                store.append(
                    {
                        "layer": index,
                        "weights": weights.detach(),
                        "mask": cross_attn_mask.detach(),
                    }
                )
                x = x + cross_output
                return x + layer.mlp2(layer.norm4(x))

            return forward

        block.forward = make_forward(block, layer_index)


def attention_rollout(fusion_store, decoder_store):
    """Return decoder ego-query relevance propagated through Fusion tokens."""
    if not fusion_store or not decoder_store:
        raise RuntimeError("attention stores are empty")
    n_tokens = fusion_store[0]["weights"].shape[-1]
    rollout = torch.eye(n_tokens, device=fusion_store[0]["weights"].device)
    for record in fusion_store:
        weights = record["weights"][0].float()
        mask = record["mask"][0]
        weights = weights.masked_fill(mask[None, :], 0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        identity = torch.eye(n_tokens, device=weights.device)
        rollout = ((weights + identity) / 2.0) @ rollout

    decoder_rows = []
    for record in decoder_store:
        weights = record["weights"][0, 0].float()  # ego decoder query
        mask = record["mask"][0]
        weights = weights.masked_fill(mask, 0.0)
        decoder_rows.append(weights / weights.sum().clamp_min(1e-12))
    decoder_attention = torch.stack(decoder_rows).mean(dim=0)
    scores = decoder_attention @ rollout
    valid = ~fusion_store[0]["mask"][0]
    scores = scores.masked_fill(~valid, 0.0)
    return scores / scores.sum().clamp_min(1e-12), valid


def neighbor_records(records: list[dict]) -> list[dict]:
    # Copy records because the all-token report may be rendered from the same
    # list.  Re-labeling the copies must not alter the original token classes.
    selected = [dict(record) for record in records if record["class"] == "neighbors"]
    total = sum(record["attention"] for record in selected)
    for slot, record in enumerate(selected):
        record["slot"] = record["class_index"]
        record["class"] = record.get("neighbor_class", "vehicle")
        record["attention_pct_within_neighbors"] = record["attention"] / max(total, 1e-12) * 100.0
    return sorted(selected, key=lambda item: item["attention"], reverse=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--valid_set_list", required=True)
    parser.add_argument("--sample_index", type=int, required=True)
    parser.add_argument("--layer", default="mean")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--view_range", type=float, default=80.0)
    parser.add_argument("--colormap", default="plasma")
    parser.add_argument("--marker_size_min", type=float, default=25.0)
    parser.add_argument("--marker_size_max", type=float, default=700.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_fusion_png", required=True)
    parser.add_argument("--out_rollout_png", required=True)
    parser.add_argument("--out_neighbor_rollout_png", required=True)
    parser.add_argument("--out_json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = DiffusionPlannerData(args.valid_set_list)
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(f"--sample_index must be in 0..{len(dataset) - 1}")

    model, cfg, _ = load_model(Path(args.run_dir).resolve(), args.device)
    fusion_store = []
    decoder_store = []
    patch_fusion(find_fusion(model.encoder), fusion_store)
    patch_decoder(model.decoder, decoder_store)

    sample = dataset[args.sample_index]
    batch = sample_to_batch(sample)
    with torch.no_grad():
        inputs = prepare_inputs(dict(batch), cfg, args.device)
        encoding = model.encoder(inputs)
        model.decoder(encoding, inputs)

    fusion_scores = all_token_attention(fusion_store, args.layer)[0]
    rollout_scores, valid = attention_rollout(fusion_store, decoder_store)
    fusion_records = token_records(sample, fusion_scores.cpu().numpy(), valid.cpu().numpy())
    rollout_records = token_records(sample, rollout_scores.cpu().numpy(), valid.cpu().numpy())
    movement, turn_angle = movement_and_turn(sample)
    sample_path = dataset.data_list[args.sample_index]
    common = dict(
        batch=batch,
        sample_index=args.sample_index,
        sample_path=sample_path,
        movement_m=movement,
        turn_angle_deg=turn_angle,
        layer=args.layer,
        top_k=args.top_k,
        view_range=args.view_range,
        colormap=args.colormap,
        marker_size_min=args.marker_size_min,
        marker_size_max=args.marker_size_max,
    )
    draw_report(
        records=fusion_records,
        out_png=Path(args.out_fusion_png),
        attention_label="Fusion attention from ego query",
        title_prefix="Fusion Attention",
        **common,
    )
    rollout_neighbors = neighbor_records(rollout_records)
    draw_neighbor_report(
        batch=batch,
        records=rollout_neighbors,
        sample_index=args.sample_index,
        sample_path=sample_path,
        movement_m=movement,
        turn_angle_deg=turn_angle,
        layer=args.layer,
        top_k=args.top_k,
        view_range=args.view_range,
        colormap=args.colormap,
        marker_size_min=args.marker_size_min,
        marker_size_max=args.marker_size_max,
        out_png=Path(args.out_neighbor_rollout_png),
        attention_label="Decoder-to-input rollout",
        title_prefix="Neighbor Decoder-to-Input Rollout",
        attention_layer_label="Decoder cross-attention averaged over layers/steps",
    )
    draw_report(
        records=rollout_records,
        out_png=Path(args.out_rollout_png),
        attention_label="Decoder-to-input rollout",
        title_prefix="Decoder-to-Input Attention Rollout",
        attention_layer_label="Decoder cross-attention averaged over layers/steps",
        **common,
    )
    report = {
        "sample_index": args.sample_index,
        "sample_path": sample_path,
        "fusion_attention": {
            "class_summary": class_summary(fusion_records),
            "tokens": fusion_records,
        },
        "rollout_attention": {
            "class_summary": class_summary(rollout_records),
            "tokens": rollout_records,
            "definition": "Decoder ego-query cross-attention rolled through residual Fusion self-attention matrices",
            "decoder_cross_attention_count": len(decoder_store),
        },
    }
    with Path(args.out_json).open("w") as file:
        json.dump(report, file, indent=2)
    print(f"wrote {args.out_fusion_png}")
    print(f"wrote {args.out_rollout_png}")
    print(f"wrote {args.out_neighbor_rollout_png}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
