#!/usr/bin/env python3
"""Supervised regression training for the exploration policy (Stage 1).

Sample-efficient alternative to on-policy REINFORCE for low scene counts:
per-scene best guidance params come from an offline grid sweep
(rlvr.autoresearch.tools.sweep_guidance_params), and the policy regresses its
deterministic action (Beta mean mapped to [-1, 1]) toward them:

  - sweep status "solved"        -> target = best combo etas
  - sweep status "already_clean" -> target = 0 for every head (inertness)
  - sweep status "unsolved"      -> EXCLUDED (no valid teaching signal)
  - --normal_scenes              -> target = 0 for every head (inertness)

The planner stays frozen; scene encodings and reference trajectories are
precomputed once and cached, so policy training itself is seconds/epoch.

Usage:
    python -m rlvr.train_explorer_regression \
        --model_path <base.pth> --labels <sweep_labels.json> \
        --normal_scenes <normal.json> --output_dir <dir> \
        --heads lateral,collision [--epochs 200] [--lr 1e-3] \
        [--avoid_weight 5.0] [--val_frac 0.15]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import optim

from exploration_policy.model import ExplorationPolicy, ExplorationPolicyConfig
from exploration_policy.utils import generate_reference_trajectory, run_frozen_encoder
from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.eval_det_avoidance import load_model

HEAD_TO_LABEL_KEY = {"lateral": "eta_lat", "collision": "eta_col", "stretch": "stretch"}

LAMBDA_SPD = 0.2  # stretch = 1 + LAMBDA_SPD * eta — must match the
                  # inference envelope's lambda_spd (build_head_composer)


def label_target(head: str, best: dict) -> float:
    """Map a sweep best-combo entry to the head's eta target in [-1, 1]."""
    key = HEAD_TO_LABEL_KEY[head]
    v = float(best[key])
    if head == "stretch":
        # The sweep records the stretch FACTOR; the policy's eta is symmetric
        # around 1.0 via stretch = 1 + lambda_spd * eta. With lambda_spd 0.2
        # the grid value 0.8 maps to eta -1.0 (saturation); clamp anything
        # beyond to the Beta support.
        eta = (v - 1.0) / LAMBDA_SPD
        return float(max(-1.0, min(1.0, eta)))
    return v


def build_entries(labels_paths: list[str], normal_paths: list[str],
                  heads: list[str],
                  counterfactual_paths: list[str] | None = None):
    """Assemble (scene_path, target_dict, weight_class) tuples.

    counterfactual_paths: zero-target scenes that form PAIRS with solved
    scenes (e.g. neighbor-stripped twins). They get their own weight class
    so the pair's discrimination signal can be weighted symmetrically
    instead of drowning at generic-normal weight.
    """
    entries = []
    n_solved = n_clean = n_unsolved = 0
    seen = set()
    for lp in labels_paths:
        with open(lp) as f:
            data = json.load(f)
        for r in data["scenes"]:
            sp = r["scene_path"]
            if sp in seen:
                continue
            seen.add(sp)
            if r["status"] == "solved":
                targets = {h: label_target(h, r["best"]) for h in heads}
                entries.append((sp, targets, "avoid"))
                n_solved += 1
            elif r["status"] == "already_clean":
                entries.append((sp, {h: 0.0 for h in heads}, "clean"))
                n_clean += 1
            else:
                n_unsolved += 1
    n_normal = 0
    for sp in normal_paths:
        if sp not in seen:
            seen.add(sp)
            entries.append((sp, {h: 0.0 for h in heads}, "normal"))
            n_normal += 1
    n_cf = 0
    for sp in (counterfactual_paths or []):
        if sp not in seen:
            seen.add(sp)
            entries.append((sp, {h: 0.0 for h in heads}, "counterfactual"))
            n_cf += 1
    print(f"[entries] solved={n_solved} clean={n_clean} "
          f"normal-added={n_normal} counterfactual={n_cf} "
          f"unsolved-excluded={n_unsolved} total={len(entries)}")
    return entries


@torch.no_grad()
def precompute_features(model, model_args, entries, device, cache_path: Path,
                        model_path: str = ""):
    """Cache scene_encoding + x_ref per scene (one det pass each)."""
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        cached_model = cache.get("__model_path__", "")
        if model_path and cached_model and cached_model != model_path:
            raise ValueError(
                f"feature cache {cache_path} was built with model "
                f"{cached_model}, not {model_path} — delete the cache or use "
                "a fresh --output_dir (stale encodings would be silently "
                "served otherwise)")
        if model_path and not cached_model:
            print(f"[cache] WARNING: {cache_path} predates the model-path "
                  "stamp — cannot verify it was built with "
                  f"{model_path}. Delete it if in doubt.")
        have = set(cache.keys()) - {"__model_path__"}
        need = {e[0] for e in entries}
        if have >= need:
            print(f"[cache] loaded {len(cache)} features from {cache_path}")
            return cache
        # Partial hit: top up only the missing scenes instead of a full
        # rebuild, then re-save.
        missing = [e for e in entries if e[0] not in have]
        print(f"[cache] topping up {len(missing)} missing of {len(need)} "
              f"entries in {cache_path}")
        cache.setdefault("__model_path__", model_path)
        entries = missing
    else:
        cache = {"__model_path__": model_path}
    for i, (sp, _, _) in enumerate(entries):
        data = load_npz_data(sp, device)
        norm_data = {
            k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()
        }
        norm_data = model_args.observation_normalizer(norm_data)
        x_ref_np = generate_reference_trajectory(model, model_args, norm_data, device)
        x_ref = torch.from_numpy(x_ref_np).unsqueeze(0).to(device)
        norm_data["reference_trajectory"] = x_ref
        enc = run_frozen_encoder(model, norm_data)
        cache[sp] = {"scene_encoding": enc[0].cpu(), "x_ref": x_ref[0].cpu()}
        if (i + 1) % 50 == 0:
            print(f"  [features] {i + 1}/{len(entries)}")
    torch.save(cache, cache_path)
    print(f"[cache] wrote {len(cache)} features to {cache_path}")
    return cache


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--labels", required=True, nargs="+",
                        help="sweep_labels.json file(s)")
    parser.add_argument("--normal_scenes", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--heads", default="lateral,collision")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--avoid_weight", type=float, default=5.0,
                        help="loss weight for solved avoidance scenes vs zero-target scenes")
    parser.add_argument("--counterfactual_scenes", default=None,
                        help="JSON list of zero-target counterfactual twins "
                             "of solved scenes (e.g. neighbor-stripped); "
                             "weighted by --counterfactual_weight instead "
                             "of 1.0")
    parser.add_argument("--counterfactual_weight", type=float, default=1.0)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--head_raw_scale", type=float, default=10.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--init_from", default=None,
                        help="warm-start: load this exploration_policy.pth "
                             "before training (heads/arch must match)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    heads = args.heads.split(",")
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    normal_paths = []
    if args.normal_scenes:
        with open(args.normal_scenes) as f:
            normal_paths = json.load(f)

    cf_paths = []
    if args.counterfactual_scenes:
        with open(args.counterfactual_scenes) as f:
            cf_paths = json.load(f)

    entries = build_entries(args.labels, normal_paths, heads,
                            counterfactual_paths=cf_paths)
    if not entries:
        raise SystemExit("no training entries")

    model, model_args = load_model(args.model_path, device)
    cache = precompute_features(
        model, model_args, entries, device, run_dir / "feature_cache.pt",
        model_path=args.model_path,
    )
    del model
    torch.cuda.empty_cache()

    # --- Tensorize dataset ---
    enc = torch.stack([cache[sp]["scene_encoding"] for sp, _, _ in entries]).to(device)
    xref = torch.stack([cache[sp]["x_ref"] for sp, _, _ in entries]).to(device)
    targets = torch.tensor(
        [[t[h] for h in heads] for _, t, _ in entries], device=device,
    )  # [M, H]
    weights = torch.tensor(
        [args.avoid_weight if cls == "avoid"
         else args.counterfactual_weight if cls == "counterfactual"
         else 1.0 for _, _, cls in entries],
        device=device,
    )
    is_avoid = torch.tensor([cls == "avoid" for _, _, cls in entries], device=device)

    # --- Stratified train/val split (keep avoidance scenes in both) ---
    g = torch.Generator().manual_seed(args.seed)
    val_mask = torch.zeros(len(entries), dtype=torch.bool)
    for cls_mask in (is_avoid.cpu(), ~is_avoid.cpu()):
        idx = torch.nonzero(cls_mask).squeeze(-1)
        perm = idx[torch.randperm(len(idx), generator=g)]
        n_val = max(1, int(len(idx) * args.val_frac))
        val_mask[perm[:n_val]] = True
    train_idx = torch.nonzero(~val_mask).squeeze(-1).to(device)
    val_idx = torch.nonzero(val_mask).squeeze(-1).to(device)
    print(f"[split] train={len(train_idx)} val={len(val_idx)} "
          f"(val avoid={int(is_avoid[val_idx].sum())})")

    ep_config = ExplorationPolicyConfig(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        encoder_hidden_dim=model_args.hidden_dim,
        head_raw_scale=args.head_raw_scale,
        heads=heads,
    )
    policy = ExplorationPolicy(ep_config, ref_seq_len=model_args.future_len).to(device)
    if args.init_from:
        state = torch.load(args.init_from, map_location=device, weights_only=False)
        try:
            policy.load_state_dict(state, strict=True)
        except RuntimeError:
            # Head-spec change (e.g. 2-head -> 3-head: the shared guidance
            # head's Linear changes shape). Transfer everything that fits,
            # report what restarts fresh — loud, not silent.
            compat = {k: v for k, v in state.items()
                      if k in policy.state_dict()
                      and policy.state_dict()[k].shape == v.shape}
            missing = [k for k in policy.state_dict() if k not in compat]
            policy.load_state_dict(compat, strict=False)
            print(f"[warm-start] PARTIAL: {len(compat)}/{len(policy.state_dict())} "
                  f"tensors transferred; fresh: {missing}")
        print(f"[warm-start] loaded {args.init_from}")
    optimizer = optim.AdamW(policy.parameters(), lr=args.lr)

    def pred_etas(idx_batch):
        out = policy(enc[idx_batch], xref[idx_batch], deterministic=True)
        return torch.stack([2.0 * out.dists[h].mean - 1.0 for h in heads], dim=-1)

    def eval_split(idx_split):
        policy.eval()
        with torch.no_grad():
            pred = pred_etas(idx_split)
            err = (pred - targets[idx_split]) ** 2
            mse = (err.mean(dim=-1) * weights[idx_split]).mean()
            av = is_avoid[idx_split]
            metrics = {"weighted_mse": float(mse)}
            if av.any():
                # sign accuracy of the dominant (largest-|target|) head per scene
                t_a, p_a = targets[idx_split][av], pred[av]
                dom = t_a.abs().argmax(dim=-1)
                rows = torch.arange(len(t_a), device=device)
                metrics["avoid_mse"] = float(((t_a - p_a) ** 2).mean())
                metrics["avoid_sign_acc"] = float(
                    (torch.sign(p_a[rows, dom]) == torch.sign(t_a[rows, dom])).float().mean()
                )
                metrics["avoid_pred_absmean"] = float(p_a.abs().mean())
            if (~av).any():
                metrics["inert_pred_absmean"] = float(pred[~av].abs().mean())
        return metrics

    best_val = float("inf")
    best_epoch = 0
    log_rows = []
    for epoch in range(1, args.epochs + 1):
        policy.train()
        perm = train_idx[torch.randperm(len(train_idx), device=device)]
        ep_loss, n_b = 0.0, 0
        for s in range(0, len(perm), args.batch_size):
            b = perm[s : s + args.batch_size]
            pred = pred_etas(b)
            loss = (((pred - targets[b]) ** 2).mean(dim=-1) * weights[b]).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            ep_loss += float(loss)
            n_b += 1

        val_m = eval_split(val_idx)
        train_m = eval_split(train_idx)
        row = {"epoch": epoch, "train_loss": ep_loss / max(n_b, 1),
               **{f"val_{k}": v for k, v in val_m.items()},
               **{f"train_{k}": v for k, v in train_m.items()}}
        log_rows.append(row)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  ep{epoch:3d} train={row['train_loss']:.4f} "
                  f"val_mse={val_m['weighted_mse']:.4f} "
                  f"avoid_sign={val_m.get('avoid_sign_acc', -1):.2f} "
                  f"avoid_|p|={val_m.get('avoid_pred_absmean', -1):.3f} "
                  f"inert_|p|={val_m.get('inert_pred_absmean', -1):.4f}")

        if val_m["weighted_mse"] < best_val - 1e-5:
            best_val = val_m["weighted_mse"]
            best_epoch = epoch
            torch.save(policy.state_dict(), run_dir / "exploration_policy.pth")
        if epoch - best_epoch >= args.patience:
            print(f"  early stop at ep{epoch} (best ep{best_epoch})")
            break

    ep_config.to_json(run_dir / "exploration_policy_config.json")
    with open(run_dir / "train_log.json", "w") as f:
        json.dump(log_rows, f, indent=1)
    with open(run_dir / "train_args.json", "w") as f:
        json.dump(vars(args), f, indent=1)
    print(f"[done] best val_mse={best_val:.4f} @ ep{best_epoch}; "
          f"saved {run_dir / 'exploration_policy.pth'}")


if __name__ == "__main__":
    main()
