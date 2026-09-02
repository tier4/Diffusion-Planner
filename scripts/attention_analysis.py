"""Fusion attention analysis by token class (companion to scripts/token_importance.py).

Patches each fusion SelfAttentionBlock to expose attention weights, then
measures, per layer:
  - attention share received by each token class (ego-token query and
    all-valid-query average), vs the class's share of valid tokens
    -> selectivity ratio (=1 means indistinguishable from count-proportional
    dilution; >1 means the model actively prefers the class)
  - value-norm weighted share (alpha * ||W_v x||, approximation ignoring
    per-head structure / out-proj) — attention weight alone can overstate
    contribution of low-value tokens
  - distance-binned attention within lanes / neighbors (ego query)
  - route share in turning vs straight scenes (functional check: does
    attention shift with scenario?)

Usage:
  uv run python scripts/attention_analysis.py \
    --run_dir /path/to/run_dir \
    --valid_set_list /path/to/path_list_valid.json \
    --n_samples 128 --batch_size 8
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from diffusion_planner.utils.dataset import DiffusionPlannerData
from token_analysis_common import (
    find_fusion,
    init_distributed,
    neighbor_dist,
    patch_fusion,
    polyline_dist,
    prepare_inputs,
)
from token_analysis_common import (
    load_model as _load_model,
)
from torch.utils.data import DataLoader, Subset


def load_model(run_dir: Path, device: str):
    model, cfg, _ = _load_model(run_dir, device)
    return model, cfg


def lane_dist(lanes):
    return polyline_dist(lanes, geom_dims=8)


# token layout (dimensions.py defaults, matches V_tail args)
CLASSES = [
    ("ego", 1),
    ("neighbors", 320),
    ("static", 5),
    ("lanes", 140),
    ("route", 25),
    ("polygons", 10),
    ("line_strings", 60),
    ("goal_pose", 1),
    ("ego_shape", 1),
    ("turn_indicator", 1),
]
OFFSETS = {}
_o = 0
for _name, _n in CLASSES:
    OFFSETS[_name] = (_o, _o + _n)
    _o += _n
TOKEN_NUM = _o  # 564

DIST_BINS = [(0, 25), (25, 50), (50, 100), (100, float("inf"))]


def value_norms(block, kv):
    """||W_v x + b_v|| per token (heads concatenated; out_proj ignored)."""
    E = kv.shape[-1]
    W = block.attn.in_proj_weight[2 * E : 3 * E]
    b = block.attn.in_proj_bias[2 * E : 3 * E]
    return (kv @ W.T + b).norm(dim=-1)  # [B, N]


def occupancy_stats(values, slots):
    """Summarize valid-token counts for one token class."""
    a = np.asarray(values)
    p50, p95, p99 = (float(np.percentile(a, q)) for q in (50, 95, 99))
    return {
        "slots": slots,
        "mean": float(a.mean()),
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "max": int(a.max()),
        "mean_utilization_pct": float(a.mean() / slots * 100),
        "p95_capacity": int(np.ceil(p95)),
        "p99_capacity": int(np.ceil(p99)),
    }


def select_moving_indices(dataset, n_samples, move_min_m, turn_deg):
    """Select moving scenes across the full list; run on rank 0 only."""
    stride = max(1, len(dataset) // (n_samples * 2))
    # Check one evenly spaced pass first, then the remaining offsets as needed.
    # This preserves broad dataset coverage without silently stopping after only
    # about 2 * n_samples candidates when moving scenes are sparse.
    cand = [index for offset in range(stride) for index in range(offset, len(dataset), stride)]
    probe = DataLoader(Subset(dataset, cand), batch_size=1, shuffle=False)
    idxs, turning = [], []
    for index, sample in zip(cand, probe):
        future = sample["ego_agent_future"][0]
        if float(np.linalg.norm(future[-1, :2])) < move_min_m:
            continue
        bearing = abs(np.degrees(np.arctan2(float(future[-1, 1]), float(future[-1, 0]))))
        idxs.append(index)
        turning.append(bool(bearing >= turn_deg))
        if len(idxs) >= n_samples:
            break
    return idxs, turning, stride


def broadcast_selection(indices, turning, world_size):
    if world_size == 1:
        return indices, turning
    payload = [indices, turning]
    dist.broadcast_object_list(payload, src=0)
    return payload


def analyze_local(
    model,
    cfg,
    fusion,
    store,
    dataset,
    indices,
    turning,
    batch_size,
    num_workers,
    device,
):
    """Analyze one rank's sample shard and return mergeable Python accumulators."""
    n_layers = len(fusion.blocks)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )
    ego_share = [{c: [] for c, _ in CLASSES} for _ in range(n_layers)]
    all_share = [{c: [] for c, _ in CLASSES} for _ in range(n_layers)]
    vw_share = [{c: [] for c, _ in CLASSES} for _ in range(n_layers)]
    count_share = {c: [] for c, _ in CLASSES}
    valid_count = {c: [] for c, _ in CLASSES}
    total_valid_count = []
    lane_bin_share = [[] for _ in DIST_BINS]
    nbr_bin_share = [[] for _ in DIST_BINS]
    route_share_per_sample = []

    with torch.no_grad():
        for raw in loader:
            l_dist = lane_dist(raw["lanes"]).to(device)
            n_dist = neighbor_dist(raw["neighbor_agents_past"]).to(device)
            inputs_n = prepare_inputs(dict(raw), cfg, device)
            store.clear()
            model.encoder(inputs_n)
            assert len(store) == n_layers
            valid = ~store[0]["mask"]
            vf = valid.float()
            n_valid = vf.sum(dim=1)
            total_valid_count += n_valid.tolist()

            for class_name, _ in CLASSES:
                start, end = OFFSETS[class_name]
                class_count = vf[:, start:end].sum(dim=1)
                valid_count[class_name] += class_count.tolist()
                count_share[class_name] += (class_count / n_valid).tolist()

            per_layer_route = []
            per_layer_lane_bins = [[] for _ in DIST_BINS]
            per_layer_nbr_bins = [[] for _ in DIST_BINS]
            for layer_index, record in enumerate(store):
                weights = record["w"]
                ego_weights = weights[:, 0]
                all_query_weights = (weights * vf.unsqueeze(-1)).sum(dim=1) / n_valid.unsqueeze(-1)
                norms = value_norms(fusion.blocks[layer_index], record["kv"])
                value_weighted = ego_weights * norms
                value_weighted /= value_weighted.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                for class_name, _ in CLASSES:
                    start, end = OFFSETS[class_name]
                    ego_share[layer_index][class_name] += (
                        ego_weights[:, start:end].sum(dim=1).tolist()
                    )
                    all_share[layer_index][class_name] += (
                        all_query_weights[:, start:end].sum(dim=1).tolist()
                    )
                    vw_share[layer_index][class_name] += (
                        value_weighted[:, start:end].sum(dim=1).tolist()
                    )

                lane_start, lane_end = OFFSETS["lanes"]
                neighbor_start, neighbor_end = OFFSETS["neighbors"]
                lane_weights = ego_weights[:, lane_start:lane_end]
                neighbor_weights = ego_weights[:, neighbor_start:neighbor_end]
                for bin_index, (lower, upper) in enumerate(DIST_BINS):
                    lane_mask = (l_dist >= lower) & (l_dist < upper)
                    neighbor_mask = (n_dist >= lower) & (n_dist < upper)
                    lane_total = lane_weights.sum(dim=1)
                    neighbor_total = neighbor_weights.sum(dim=1)
                    per_layer_lane_bins[bin_index].append(
                        torch.where(
                            lane_total > 0,
                            (lane_weights * lane_mask).sum(dim=1) / lane_total.clamp(min=1e-9),
                            torch.nan,
                        )
                    )
                    per_layer_nbr_bins[bin_index].append(
                        torch.where(
                            neighbor_total > 0,
                            (neighbor_weights * neighbor_mask).sum(dim=1)
                            / neighbor_total.clamp(min=1e-9),
                            torch.nan,
                        )
                    )
                route_start, route_end = OFFSETS["route"]
                per_layer_route.append(ego_weights[:, route_start:route_end].sum(dim=1))

            route_share_per_sample += torch.stack(per_layer_route).mean(dim=0).tolist()
            for bin_index in range(len(DIST_BINS)):
                lane_bin_share[bin_index] += (
                    torch.stack(per_layer_lane_bins[bin_index]).mean(dim=0).tolist()
                )
                nbr_bin_share[bin_index] += (
                    torch.stack(per_layer_nbr_bins[bin_index]).mean(dim=0).tolist()
                )

    return {
        "turning": list(turning),
        "ego_share": ego_share,
        "all_share": all_share,
        "vw_share": vw_share,
        "count_share": count_share,
        "valid_count": valid_count,
        "total_valid_count": total_valid_count,
        "lane_bin_share": lane_bin_share,
        "nbr_bin_share": nbr_bin_share,
        "route_share_per_sample": route_share_per_sample,
    }


def merge_payloads(payloads, n_layers):
    """Concatenate per-rank accumulators without changing sample weighting."""
    merged = {
        "turning": [],
        "ego_share": [{c: [] for c, _ in CLASSES} for _ in range(n_layers)],
        "all_share": [{c: [] for c, _ in CLASSES} for _ in range(n_layers)],
        "vw_share": [{c: [] for c, _ in CLASSES} for _ in range(n_layers)],
        "count_share": {c: [] for c, _ in CLASSES},
        "valid_count": {c: [] for c, _ in CLASSES},
        "total_valid_count": [],
        "lane_bin_share": [[] for _ in DIST_BINS],
        "nbr_bin_share": [[] for _ in DIST_BINS],
        "route_share_per_sample": [],
    }
    for payload in payloads:
        merged["turning"].extend(payload["turning"])
        merged["total_valid_count"].extend(payload["total_valid_count"])
        merged["route_share_per_sample"].extend(payload["route_share_per_sample"])
        for class_name, _ in CLASSES:
            merged["count_share"][class_name].extend(payload["count_share"][class_name])
            merged["valid_count"][class_name].extend(payload["valid_count"][class_name])
            for layer_index in range(n_layers):
                for key in ("ego_share", "all_share", "vw_share"):
                    merged[key][layer_index][class_name].extend(
                        payload[key][layer_index][class_name]
                    )
        for bin_index in range(len(DIST_BINS)):
            merged["lane_bin_share"][bin_index].extend(payload["lane_bin_share"][bin_index])
            merged["nbr_bin_share"][bin_index].extend(payload["nbr_bin_share"][bin_index])
    return merged


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--valid_set_list", required=True)
    p.add_argument("--n_samples", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--move_min_m", type=float, default=5.0)
    p.add_argument("--turn_deg", type=float, default=15.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--out_json", default="", help="write aggregated results to this JSON file")
    args = p.parse_args()

    args.device, rank, local_rank, world_size = init_distributed(args.device)
    is_main = rank == 0
    model, cfg = load_model(Path(args.run_dir), args.device)
    fusion = find_fusion(model)
    store = []
    patch_fusion(fusion, store)
    n_layers = len(fusion.blocks)
    if is_main:
        print(f"patched {n_layers} fusion blocks, token_num={TOKEN_NUM}", flush=True)
        print(
            f"data parallel: world_size={world_size}, per-rank batch={args.batch_size}, "
            f"workers/rank={args.num_workers}",
            flush=True,
        )

    dataset = DiffusionPlannerData(args.valid_set_list)
    if is_main:
        idxs, turning, stride = select_moving_indices(
            dataset, args.n_samples, args.move_min_m, args.turn_deg
        )
    else:
        idxs, turning, stride = None, None, None
    idxs, turning = broadcast_selection(idxs, turning, world_size)
    if not idxs:
        if world_size > 1:
            dist.destroy_process_group()
        raise RuntimeError(
            f"no samples moved at least {args.move_min_m} m in the evaluation dataset"
        )
    local_idxs = idxs[rank::world_size]
    local_turning = turning[rank::world_size]
    if is_main:
        print(
            f"selected {len(idxs)} moving samples "
            f"({int(np.sum(turning))} turning, stride={stride}, dataset={len(dataset)})",
            flush=True,
        )
    print(
        f"[rank {rank}/{world_size}, local_rank={local_rank}] analyzing {len(local_idxs)} samples",
        flush=True,
    )
    local_payload = analyze_local(
        model,
        cfg,
        fusion,
        store,
        dataset,
        local_idxs,
        local_turning,
        args.batch_size,
        args.num_workers,
        args.device,
    )
    if world_size > 1:
        payloads = [None] * world_size if is_main else None
        dist.gather_object(local_payload, payloads, dst=0)
    else:
        payloads = [local_payload]
    if not is_main:
        dist.destroy_process_group()
        return
    merged = merge_payloads(payloads, n_layers)
    turning = np.asarray(merged["turning"], dtype=bool)
    ego_share = merged["ego_share"]
    all_share = merged["all_share"]
    vw_share = merged["vw_share"]
    count_share = merged["count_share"]
    valid_count = merged["valid_count"]
    total_valid_count = merged["total_valid_count"]
    lane_bin_share = merged["lane_bin_share"]
    nbr_bin_share = merged["nbr_bin_share"]
    route_share_per_sample = merged["route_share_per_sample"]

    # ---------------- aggregate ----------------
    r_route = np.array(route_share_per_sample)
    results = {
        "n_samples": len(idxs),
        "n_turning": int(turning.sum()),
        "n_layers": n_layers,
        "world_size": world_size,
        "classes": [c for c, _ in CLASSES],
        "token_occupancy": {
            "description": "Valid token counts over the selected moving evaluation samples",
            "total": occupancy_stats(total_valid_count, TOKEN_NUM),
            "by_class": {c: occupancy_stats(valid_count[c], slots) for c, slots in CLASSES},
        },
        "count_share": {c: float(np.mean(count_share[c])) for c, _ in CLASSES},
        "ego_share_per_layer": {
            c: [float(np.mean(ego_share[li][c])) for li in range(n_layers)] for c, _ in CLASSES
        },
        "all_share_avg": {
            c: float(np.mean([np.mean(all_share[li][c]) for li in range(n_layers)]))
            for c, _ in CLASSES
        },
        "vw_share_avg": {
            c: float(np.mean([np.mean(vw_share[li][c]) for li in range(n_layers)]))
            for c, _ in CLASSES
        },
        "dist_bins": ["0-25m", "25-50m", "50-100m", "100m+"],
        "lane_bin_share": [float(np.nanmean(x)) for x in lane_bin_share],
        "nbr_bin_share": [float(np.nanmean(x)) for x in nbr_bin_share],
        "route_share_turning": float(r_route[turning].mean()) if turning.any() else None,
        "route_share_straight": float(r_route[~turning].mean()) if (~turning).any() else None,
    }
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=1)
        print(f"wrote {args.out_json}", flush=True)

    # ---------------- report ----------------
    def pct(x):
        return f"{100 * np.nanmean(x):5.1f}%"

    print("\n=== class attention share (ego-token query), per fusion layer ===")
    hdr = (
        f"{'class':<14}{'count':>7}"
        + "".join(f"  L{li}" + " " * 4 for li in range(n_layers))
        + "   vw(L-avg)  select"
    )
    print(hdr)
    for c, _ in CLASSES:
        cs = np.mean(count_share[c])
        layer_avg = np.mean([np.mean(ego_share[li][c]) for li in range(n_layers)])
        vw_avg = np.mean([np.mean(vw_share[li][c]) for li in range(n_layers)])
        sel = layer_avg / cs if cs > 0 else float("nan")
        row = f"{c:<14}{100 * cs:6.1f}%" + "".join(
            f" {100 * np.mean(ego_share[li][c]):5.1f}%" for li in range(n_layers)
        )
        print(row + f"   {100 * vw_avg:5.1f}%   {sel:5.2f}x")

    print("\n=== class attention share (all-valid-query mean), layer-averaged ===")
    for c, _ in CLASSES:
        cs = np.mean(count_share[c])
        avg = np.mean([np.mean(all_share[li][c]) for li in range(n_layers)])
        sel = avg / cs if cs > 0 else float("nan")
        print(f"{c:<14} count={100 * cs:5.1f}%  attn={100 * avg:5.1f}%  select={sel:5.2f}x")

    print("\n=== ego-query attention within lanes / neighbors, by distance (layer-avg) ===")
    print(f"{'bin':<12}{'lanes':>8}{'neighbors':>11}")
    for bi, (lo, hi) in enumerate(DIST_BINS):
        label = f"{lo:.0f}-{hi:.0f}m" if np.isfinite(hi) else f"{lo:.0f}m+"
        print(f"{label:<12}{pct(lane_bin_share[bi]):>8}{pct(nbr_bin_share[bi]):>11}")

    print("\n=== route share (ego query, layer-avg): turning vs straight ===")
    r = np.array(route_share_per_sample)
    print(f"turning  (n={int(turning.sum())}): {100 * r[turning].mean():5.2f}%")
    print(f"straight (n={int((~turning).sum())}): {100 * r[~turning].mean():5.2f}%")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
