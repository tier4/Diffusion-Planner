#!/usr/bin/env python3
"""Build a latent OOD embedding bank from Diffusion Planner training data.

Usage:
    python scripts/build_latent_ood_bank.py \
        --model_path /path/to/best_model.pth \
        --args_path /path/to/args.json \
        --train_list /path/to/train.json \
        --output_dir /path/to/latent_ood_bank \
        [--val_list /path/to/val.json] \
        [--batch_size 64] \
        [--num_workers 4] \
        [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
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
    p.add_argument("--args_path", type=Path, required=True, help="Path to training args.json")
    p.add_argument(
        "--train_list", type=Path, required=True, help="JSON list of training .npz paths"
    )
    p.add_argument("--output_dir", type=Path, required=True, help="Output bank directory")
    p.add_argument(
        "--val_list",
        type=Path,
        default=None,
        help="Optional JSON list of validation .npz for calibration",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--kmeans_k", type=int, default=32, help="Number of clusters for spherical K-means"
    )
    p.add_argument("--gmm_k", type=int, default=16, help="Number of components for GMM")
    p.add_argument("--gmm_covariance_type", type=str, default="diag", help="GMM covariance type")
    p.add_argument(
        "--mahalanobis_eps",
        type=float,
        default=1e-5,
        help="Regularization epsilon for Mahalanobis",
    )
    return p.parse_args()


def extract_embeddings(
    encoder: EncoderInference, data_list: Path, batch_size: int, num_workers: int
):
    dataset = DiffusionPlannerData(str(data_list))
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    with open(data_list) as f:
        npz_paths = json.load(f)

    embeddings = []
    idx = 0
    records = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting embeddings"):
            z = encoder.encode_batch(batch, normalize=False)
            embeddings.append(z.cpu().numpy())

            for i in range(z.shape[0]):
                path = npz_paths[idx] if idx < len(npz_paths) else f"unknown_{idx}"
                records.append({"row": idx, "npz_path": path})
                idx += 1

    return np.concatenate(embeddings, axis=0).astype(np.float32), records


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    args = parse_args()

    print(f"Loading model from {args.model_path}")
    encoder = EncoderInference(args.args_path, args.model_path, args.device)
    hidden_dim = encoder.config.hidden_dim
    print(f"  hidden_dim={hidden_dim}")

    print(f"Extracting training embeddings from {args.train_list}")
    t0 = time.time()
    embeddings, records = extract_embeddings(
        encoder, args.train_list, args.batch_size, args.num_workers
    )
    print(f"  Extracted {embeddings.shape[0]} embeddings in {time.time() - t0:.1f}s")

    # L2 normalize for kNN/Mahalanobis/K-means
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    embeddings_l2 = (embeddings / norms).astype(np.float32)

    metadata = {
        "model_path": str(args.model_path),
        "args_path": str(args.args_path),
        "embedding_source": "encoder_outputs.mean(dim=1)",
        "embedding_dim": int(embeddings.shape[1]),
        "pooling": "mean_all_tokens_after_encoder_mask_zeroing",
        "num_embeddings": int(embeddings.shape[0]),
        "l2_normalized": False,
        "bank_format_version": 2,
        "git_sha": git_sha(),
        "train_list": str(args.train_list),
    }

    # Save raw + L2 embeddings
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "embeddings.npy", embeddings)
    np.save(args.output_dir / "embeddings_l2.npy", embeddings_l2)

    # kNN scorer (uses L2-normalized)
    scorer = LatentOODScorer.build(embeddings_l2, records, metadata, device=args.device)

    if args.val_list:
        print(f"Calibrating on validation set {args.val_list}")
        val_emb, _ = extract_embeddings(encoder, args.val_list, args.batch_size, args.num_workers)
        val_z = torch.from_numpy(val_emb).to(args.device)
        val_scores = []
        for i in tqdm(range(0, val_z.shape[0], args.batch_size), desc="Scoring validation"):
            chunk = val_z[i : i + args.batch_size]
            result = scorer.score(chunk, k=10)
            val_scores.append(result["knn_mean"].cpu().numpy())
        val_scores = np.concatenate(val_scores)
        scorer.calibrate(val_scores)
        print(f"  warning={scorer._calibration['thresholds']['warning']:.4f}")
        print(f"  high={scorer._calibration['thresholds']['high']:.4f}")

    print(f"Saving bank to {args.output_dir}")
    scorer.save(args.output_dir)

    # Fit additional scoring methods
    from diffusion_planner.utils.scoring_methods import (
        GMMScorer,
        MahalanobisScorer,
        SphericalKMeansScorer,
    )

    print("Fitting Mahalanobis...")
    maha = MahalanobisScorer.fit(embeddings_l2, eps=args.mahalanobis_eps)
    maha.save(args.output_dir / "mahalanobis.npz")

    print(f"Fitting Spherical K-means (k={args.kmeans_k})...")
    skmeans = SphericalKMeansScorer.fit(embeddings_l2, k=args.kmeans_k)
    skmeans.save(args.output_dir / "kmeans.npz")

    print(f"Fitting GMM (k={args.gmm_k}, cov={args.gmm_covariance_type})...")
    gmm = GMMScorer.fit(embeddings, k=args.gmm_k, covariance_type=args.gmm_covariance_type)
    gmm.save(args.output_dir / "gmm.npz", args.output_dir / "zscore_params.npz")

    print("Done.")


if __name__ == "__main__":
    main()
