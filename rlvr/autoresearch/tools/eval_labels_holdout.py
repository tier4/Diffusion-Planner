#!/usr/bin/env python3
"""Evaluate a trained explorer policy on HELD-OUT sweep-label rows.

Generalization check on base-scene groups the policy never saw (made by
split_labels_holdout): reports weighted eta-MSE, dominant-head sign accuracy
on solved rows, and mean |eta| on already-clean rows (inertness), mirroring
the trainer's own metrics so train-vs-holdout gaps read directly as overfit.

Usage:
    python -m rlvr.autoresearch.tools.eval_labels_holdout \
        --model_path <base.pth> --policy_dir <dir> --holdout <holdout.json>
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from exploration_policy.utils import generate_reference_trajectory, run_frozen_encoder
from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.eval_det_avoidance import load_model
from rlvr.autoresearch.tools.eval_policy_avoidance import load_policy
from rlvr.train_explorer_regression import label_target


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--policy_dir", required=True)
    parser.add_argument("--holdout", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, margs = load_model(args.model_path, device)
    policy, heads = load_policy(args.policy_dir, margs, device)

    with open(args.holdout) as f:
        rows = json.load(f)["scenes"]

    preds, targs, kinds = [], [], []
    for r in rows:
        if r["status"] == "solved":
            t = [label_target(h, r["best"]) for h in heads]
        elif r["status"] == "already_clean":
            t = [0.0] * len(heads)
        else:
            continue
        data = load_npz_data(r["scene_path"], device)
        norm = {k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in data.items()}
        norm = margs.observation_normalizer(norm)
        x_ref = torch.from_numpy(
            generate_reference_trajectory(model, margs, norm, device)
        ).unsqueeze(0).to(device)
        norm["reference_trajectory"] = x_ref
        enc = run_frozen_encoder(model, norm)
        out = policy(enc, x_ref, deterministic=True)
        preds.append([float(2.0 * out.dists[h].mean - 1.0) for h in heads])
        targs.append(t)
        kinds.append(r["status"])

    p = np.array(preds)
    t = np.array(targs)
    solved = np.array([k == "solved" for k in kinds])
    mse = float(((p - t) ** 2).mean())
    res = {"n": len(kinds), "n_solved": int(solved.sum()), "mse": round(mse, 4)}
    if solved.any():
        ps, ts = p[solved], t[solved]
        dom = np.abs(ts).argmax(axis=1)
        idx = np.arange(len(ts))
        res["solved_mse"] = round(float(((ps - ts) ** 2).mean()), 4)
        res["sign_acc"] = round(float(
            (np.sign(ps[idx, dom]) == np.sign(ts[idx, dom])).mean()), 3)
    if (~solved).any():
        res["clean_pred_absmean"] = round(float(np.abs(p[~solved]).mean()), 4)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
