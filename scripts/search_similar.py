#!/usr/bin/env python3
"""Find the most similar training scenes for a query npz file.

Uses the Diffusion Planner encoder to embed the query scene, then searches
against a pre-built embedding bank using kNN (L2 distance on L2-normalized
256-dim embeddings).

Usage:
    uv run python scripts/search_similar.py \
      --query data/or_scene_npz/odaiba/停止_車両/.../frame.npz \
      --bank_dir data/latent_ood_bank_5k \
      --model_path /opt/autoware/mlmodels/diffusion_planner_for_x2/diffusion_planner.onnx \
      --args_path /opt/autoware/mlmodels/diffusion_planner_for_x2/args.json \
      --top_k 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusion_planner.utils.encoder_inference import EncoderInference
from diffusion_planner.utils.latent_ood import LatentOODScorer


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--query", type=Path, required=True, help="Path to query .npz file")
    p.add_argument(
        "--bank_dir", type=Path, required=True, help="Pre-built embedding bank directory"
    )
    p.add_argument("--model_path", type=Path, required=True, help="ONNX model path")
    p.add_argument("--args_path", type=Path, required=True, help="Model args.json path")
    p.add_argument("--top_k", type=int, default=10, help="Number of nearest neighbors")
    p.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    p.add_argument("--output", type=Path, default=None, help="Output JSON path (default: stdout)")
    return p.parse_args()


def load_single_npz(npz_path: Path) -> dict[str, torch.Tensor]:
    """Load a single npz file and convert arrays to tensors with a batch dim."""
    data = dict(np.load(str(npz_path), allow_pickle=True))
    batch = {}
    for key, arr in data.items():
        t = torch.from_numpy(np.array(arr))
        if t.ndim == 0:
            # Scalar (e.g. version) -- skip, not needed by the encoder.
            continue
        batch[key] = t.unsqueeze(0)  # Add batch dimension.
    return batch


def main():
    args = parse_args()

    print(f"Query: {args.query}")
    print(f"Bank:  {args.bank_dir}")

    # EncoderInference signature: (args_path, model_path, device)
    encoder = EncoderInference(args.args_path, args.model_path, args.device)
    scorer = LatentOODScorer.load(args.bank_dir, device=args.device)
    print(f"  Bank: {scorer.num_embeddings} embeddings, dim={scorer.embedding_dim}")

    batch = load_single_npz(args.query)

    with torch.no_grad():
        embedding = encoder.encode_batch(batch)

    neighbors = scorer.nearest(embedding, k=args.top_k)

    results = []
    for neighbor in neighbors[0]:
        results.append(
            {
                "rank": len(results) + 1,
                "distance": round(neighbor["distance"], 6),
                "path": neighbor.get("npz_path", f"index_{neighbor['index']}"),
            }
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(results)} results to {args.output}")
    else:
        print(f"\nTop-{args.top_k} similar training scenes:")
        for r in results:
            print(f"  #{r['rank']}: distance={r['distance']:.4f}  {r['path']}")


if __name__ == "__main__":
    main()
