#!/usr/bin/env python3
"""Score evaluation scenes against a latent OOD embedding bank.

Usage:
    python scripts/score_latent_ood.py \
        --model_path /path/to/best_model.pth \
        --args_path /path/to/args.json \
        --eval_list /path/to/eval.json \
        --bank_dir /path/to/latent_ood_bank \
        --output /path/to/latent_ood_scores.jsonl \
        [--method knn|mahalanobis|kmeans|gmm] \
        [--top_k_neighbors 5] \
        [--batch_size 64] \
        [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils.encoder_inference import EncoderInference
from diffusion_planner.utils.latent_ood import LatentOODScorer
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model_path", type=Path, required=True, help="Path to model (.pth checkpoint or .onnx)"
    )
    p.add_argument("--args_path", type=Path, required=True)
    p.add_argument("--eval_list", type=Path, required=True, help="JSON list of eval .npz paths")
    p.add_argument("--bank_dir", type=Path, required=True, help="Pre-built bank directory")
    p.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    p.add_argument(
        "--method",
        type=str,
        default="knn",
        choices=["knn", "mahalanobis", "kmeans", "gmm"],
        help="Scoring method to use",
    )
    p.add_argument(
        "--top_k_neighbors", type=int, default=5, help="Number of nearest neighbors to include"
    )
    p.add_argument("--knn_k", type=int, default=10, help="k for kNN scoring")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading bank from {args.bank_dir}")
    bank_dir = Path(args.bank_dir)

    # Method-specific scorer and data loading
    scorer = None
    maha = None
    skmeans = None
    embeddings_l2 = None
    gmm_scorer = None
    embeddings_raw = None

    if args.method == "knn":
        scorer = LatentOODScorer.load(bank_dir, device=args.device)
        print(f"  Bank: {scorer.num_embeddings} embeddings, dim={scorer.embedding_dim}")
    elif args.method == "mahalanobis":
        from diffusion_planner.utils.scoring_methods import MahalanobisScorer

        maha = MahalanobisScorer.load(bank_dir / "mahalanobis.npz")
    elif args.method == "kmeans":
        from diffusion_planner.utils.scoring_methods import SphericalKMeansScorer

        skmeans = SphericalKMeansScorer.load(bank_dir / "kmeans.npz")
        embeddings_l2 = np.load(bank_dir / "embeddings_l2.npy")
    elif args.method == "gmm":
        from diffusion_planner.utils.scoring_methods import GMMScorer

        gmm_scorer = GMMScorer.load(bank_dir / "gmm.npz", bank_dir / "zscore_params.npz")
        embeddings_raw = np.load(bank_dir / "embeddings.npy")

    print(f"Loading model from {args.model_path}")
    encoder = EncoderInference(args.args_path, args.model_path, args.device)

    with open(args.eval_list) as f:
        npz_paths = json.load(f)

    dataset = DiffusionPlannerData(str(args.eval_list))
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    idx = 0
    total_scored = 0
    level_counts = {0: 0, 1: 0, 2: 0}
    t0 = time.time()

    with open(args.output, "w") as fout, torch.no_grad():
        for batch in tqdm(loader, desc="Scoring"):
            z_raw = encoder.encode_batch(batch, normalize=False)

            if args.method == "knn":
                z = F.normalize(z_raw.float(), dim=-1)
                result = scorer.score(z, k=args.knn_k)
                neighbors_batch = scorer.nearest(z, k=args.top_k_neighbors)

                for i in range(z.shape[0]):
                    path = npz_paths[idx] if idx < len(npz_paths) else f"unknown_{idx}"
                    entry = {
                        "npz_path": path,
                        "frame_idx": idx,
                        "knn_mean": round(result["knn_mean"][i].item(), 6),
                        "knn_min": round(result["knn_min"][i].item(), 6),
                        "knn_kth": round(result["knn_kth"][i].item(), 6),
                    }
                    if "percentile" in result:
                        entry["percentile"] = round(result["percentile"][i].item(), 1)
                        entry["level"] = ["normal", "warning", "high"][result["level"][i].item()]
                        level_counts[result["level"][i].item()] += 1
                    entry["nearest"] = [
                        {
                            "npz_path": n.get("npz_path", ""),
                            "distance": round(n["distance"], 6),
                        }
                        for n in neighbors_batch[i]
                    ]
                    fout.write(json.dumps(entry) + "\n")
                    idx += 1
                    total_scored += 1

            elif args.method == "mahalanobis":
                result = maha.score(z_raw)
                neighbors_batch = maha.nearest(z_raw, k=args.top_k_neighbors)

                for i in range(z_raw.shape[0]):
                    path = npz_paths[idx] if idx < len(npz_paths) else f"unknown_{idx}"
                    entry = {
                        "npz_path": path,
                        "frame_idx": idx,
                        "mahalanobis": round(result["mahalanobis"][i].item(), 6),
                        "nearest": [
                            {
                                "index": n["index"],
                                "distance": round(n["distance"], 6),
                            }
                            for n in neighbors_batch[i]
                        ],
                    }
                    fout.write(json.dumps(entry) + "\n")
                    idx += 1
                    total_scored += 1

            elif args.method == "kmeans":
                result = skmeans.score(z_raw)
                neighbors_batch = skmeans.nearest(z_raw, embeddings_l2, k=args.top_k_neighbors)

                for i in range(z_raw.shape[0]):
                    path = npz_paths[idx] if idx < len(npz_paths) else f"unknown_{idx}"
                    entry = {
                        "npz_path": path,
                        "frame_idx": idx,
                        "kmeans_dist": round(result["kmeans_dist"][i].item(), 6),
                        "cluster_id": int(result["cluster_id"][i].item()),
                        "margin": round(result["margin"][i].item(), 6),
                        "nearest": [
                            {
                                "index": n["index"],
                                "distance": round(n["distance"], 6),
                            }
                            for n in neighbors_batch[i]
                        ],
                    }
                    fout.write(json.dumps(entry) + "\n")
                    idx += 1
                    total_scored += 1

            elif args.method == "gmm":
                result = gmm_scorer.score(z_raw)
                neighbors_batch = gmm_scorer.nearest(z_raw, embeddings_raw, k=args.top_k_neighbors)

                for i in range(z_raw.shape[0]):
                    path = npz_paths[idx] if idx < len(npz_paths) else f"unknown_{idx}"
                    entry = {
                        "npz_path": path,
                        "frame_idx": idx,
                        "gmm_nll": round(result["gmm_nll"][i].item(), 6),
                        "component_id": int(result["component_id"][i].item()),
                        "component_prob": round(result["component_prob"][i].item(), 6),
                        "nearest": [
                            {
                                "index": n["index"],
                                "distance": round(n["distance"], 6),
                            }
                            for n in neighbors_batch[i]
                        ],
                    }
                    fout.write(json.dumps(entry) + "\n")
                    idx += 1
                    total_scored += 1

    elapsed = time.time() - t0
    print(
        f"\nScored {total_scored} frames in {elapsed:.1f}s ({total_scored / elapsed:.0f} frames/s)"
    )
    if args.method == "knn" and any(level_counts.values()):
        print(f"  normal={level_counts[0]}, warning={level_counts[1]}, high={level_counts[2]}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
