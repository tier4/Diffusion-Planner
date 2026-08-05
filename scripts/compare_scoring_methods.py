#!/usr/bin/env python3
"""Compare 4 scoring methods on the same eval set and generate an HTML report.

Usage:
    python scripts/compare_scoring_methods.py \
        --model_path /path/to/model.onnx \
        --args_path /path/to/args.json \
        --bank_dir /path/to/latent_ood_bank \
        --eval_list /path/to/eval.json \
        --output_dir /path/to/comparison_output \
        [--query_npz path1.npz path2.npz] \
        [--seed 42] \
        [--viz_module_path ../clip-review-tool/src]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils.encoder_inference import EncoderInference
from diffusion_planner.utils.latent_ood import LatentOODScorer
from diffusion_planner.utils.scoring_methods import (
    GMMScorer,
    MahalanobisScorer,
    SphericalKMeansScorer,
)
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model_path", type=Path, required=True)
    p.add_argument("--args_path", type=Path, required=True)
    p.add_argument("--bank_dir", type=Path, required=True)
    p.add_argument("--eval_list", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--query_npz", type=str, nargs=2, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--viz_module_path", type=Path, default=None)
    p.add_argument(
        "--skip_viz",
        action="store_true",
        help="Skip BEV retrieval visualization (use when clip-review-tool is unavailable)",
    )
    return p.parse_args()


def score_all_methods(encoder, eval_list, bank_dir, batch_size, num_workers, device, output_dir):
    with open(eval_list) as f:
        npz_paths = json.load(f)

    dataset = DiffusionPlannerData(str(eval_list))
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    knn_scorer = LatentOODScorer.load(bank_dir, device=device)
    maha_scorer = MahalanobisScorer.load(bank_dir / "mahalanobis.npz")
    kmeans_scorer = SphericalKMeansScorer.load(bank_dir / "kmeans.npz")
    gmm_scorer = GMMScorer.load(bank_dir / "gmm.npz", bank_dir / "zscore_params.npz")
    embeddings_l2 = np.load(bank_dir / "embeddings_l2.npy")
    embeddings_raw = np.load(bank_dir / "embeddings.npy")

    records = []
    idx = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Scoring all methods"):
            z_raw = encoder.encode_batch(batch, normalize=False)
            z_l2 = F.normalize(z_raw.float(), dim=-1)

            knn_result = knn_scorer.score(z_l2, k=10)
            maha_result = maha_scorer.score(z_raw)
            kmeans_result = kmeans_scorer.score(z_raw)
            gmm_result = gmm_scorer.score(z_raw)

            for i in range(z_raw.shape[0]):
                records.append(
                    {
                        "npz_path": npz_paths[idx],
                        "knn_score": round(knn_result["knn_mean"][i].item(), 6),
                        "mahalanobis_score": round(maha_result["mahalanobis"][i].item(), 6),
                        "kmeans_score": round(kmeans_result["kmeans_dist"][i].item(), 6),
                        "gmm_score": round(gmm_result["gmm_nll"][i].item(), 6),
                    }
                )
                idx += 1

    scores_path = output_dir / "scores.jsonl"
    with open(scores_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Scores written to {scores_path}")

    return (
        records,
        knn_scorer,
        maha_scorer,
        kmeans_scorer,
        gmm_scorer,
        embeddings_l2,
        embeddings_raw,
    )


def compute_rank_correlation(records, output_dir):
    methods = [
        "knn_score",
        "mahalanobis_score",
        "kmeans_score",
        "gmm_score",
    ]
    scores = {m: [r[m] for r in records] for m in methods}

    corr_matrix = np.zeros((4, 4))
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            corr_matrix[i, j] = spearmanr(scores[m1], scores[m2]).statistic

    labels = ["kNN", "Mahalanobis", "K-means", "GMM"]
    csv_path = output_dir / "rank_correlation.csv"
    with open(csv_path, "w") as f:
        f.write("," + ",".join(labels) + "\n")
        for i, label in enumerate(labels):
            f.write(label + "," + ",".join(f"{corr_matrix[i, j]:.4f}" for j in range(4)) + "\n")
    print(f"Rank correlation written to {csv_path}")
    return corr_matrix, labels


def compute_outlier_agreement(records, output_dir, top_n=50):
    methods = [
        "knn_score",
        "mahalanobis_score",
        "kmeans_score",
        "gmm_score",
    ]
    method_labels = ["kNN", "Mahalanobis", "K-means", "GMM"]

    top_per_method = {}
    for m, label in zip(methods, method_labels):
        sorted_records = sorted(records, key=lambda r: r[m], reverse=True)[:top_n]
        top_per_method[label] = {r["npz_path"] for r in sorted_records}

    all_paths = set()
    for paths in top_per_method.values():
        all_paths |= paths

    rows = []
    for path in sorted(all_paths):
        flagged_by = [label for label in method_labels if path in top_per_method[label]]
        rows.append(
            {
                "npz_path": path,
                "flagged_by": ";".join(flagged_by),
                "count": len(flagged_by),
            }
        )

    rows.sort(key=lambda r: r["count"], reverse=True)
    csv_path = output_dir / "outlier_agreement.csv"
    with open(csv_path, "w") as f:
        f.write("npz_path,flagged_by,count\n")
        for r in rows:
            f.write(f"{r['npz_path']},{r['flagged_by']},{r['count']}\n")
    print(f"Outlier agreement written to {csv_path}")


def render_bev_to_base64(npz_path, viz_module_path=None):
    if viz_module_path:
        viz_str = str(viz_module_path)
        if viz_str not in sys.path:
            sys.path.insert(0, viz_str)
    from visualization import PAST_FRAMES, precompute_static, render_frame

    data = {k: v for k, v in np.load(npz_path).items()}
    static = precompute_static(data)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
    render_frame(fig, ax, data, static, PAST_FRAMES, filename=Path(npz_path).stem)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#1A1A1A")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def retrieve_neighbors(
    query_npz_paths,
    encoder,
    scorers,
    embeddings_l2,
    embeddings_raw,
    bank_records,
    device,
    output_dir,
    viz_module_path,
):
    method_names = ["kNN", "Mahalanobis", "K-means", "GMM"]
    knn_scorer, maha_scorer, kmeans_scorer, gmm_scorer = scorers

    retrieval_results = {m: [] for m in method_names}

    for qpath in query_npz_paths:
        query_list_path = output_dir / "_query_tmp.json"
        with open(query_list_path, "w") as f:
            json.dump([qpath], f)
        ds = DiffusionPlannerData(str(query_list_path))
        sample = ds[0]
        batch = {}
        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            if isinstance(v, torch.Tensor):
                v = v.unsqueeze(0)
            batch[k] = v

        with torch.no_grad():
            z_raw = encoder.encode_batch(batch, normalize=False)
            z_l2 = F.normalize(z_raw.float(), dim=-1)

        knn_neighbors = knn_scorer.nearest(z_l2, k=3)
        maha_neighbors = maha_scorer.nearest(z_raw, k=3)
        kmeans_neighbors = kmeans_scorer.nearest(z_raw, embeddings_l2, k=3)
        gmm_neighbors = gmm_scorer.nearest(z_raw, embeddings_raw, k=3)

        for method, neighbors in zip(
            method_names,
            [knn_neighbors, maha_neighbors, kmeans_neighbors, gmm_neighbors],
        ):
            retrieval_results[method].append(
                {
                    "query_path": qpath,
                    "neighbors": neighbors[0],
                }
            )

    return retrieval_results


def generate_html_report(
    records,
    corr_matrix,
    corr_labels,
    retrieval_results,
    bank_records,
    output_dir,
    viz_module_path,
):
    # Render score distribution plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    methods = [
        "knn_score",
        "mahalanobis_score",
        "kmeans_score",
        "gmm_score",
    ]
    labels = ["kNN", "Mahalanobis", "K-means", "GMM"]
    for ax, m, label in zip(axes, methods, labels):
        scores = [r[m] for r in records]
        ax.hist(scores, bins=50, alpha=0.7, color="#2196F3")
        ax.set_title(label)
        ax.set_xlabel("Score")
    fig.tight_layout()
    dist_buf = io.BytesIO()
    fig.savefig(dist_buf, format="png", bbox_inches="tight")
    plt.close(fig)
    dist_b64 = base64.b64encode(dist_buf.getvalue()).decode("ascii")

    # Render correlation heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(corr_labels, rotation=45)
    ax.set_yticks(range(4))
    ax.set_yticklabels(corr_labels)
    for i in range(4):
        for j in range(4):
            ax.text(
                j,
                i,
                f"{corr_matrix[i, j]:.3f}",
                ha="center",
                va="center",
            )
    fig.colorbar(im)
    fig.tight_layout()
    corr_buf = io.BytesIO()
    fig.savefig(corr_buf, format="png", bbox_inches="tight")
    plt.close(fig)
    corr_b64 = base64.b64encode(corr_buf.getvalue()).decode("ascii")

    # Render BEV images for retrieval
    bev_sections = []
    for method_name, queries in retrieval_results.items():
        method_html = f"<h2>{method_name}</h2>"
        for q in queries:
            query_b64 = render_bev_to_base64(q["query_path"], viz_module_path)
            row_html = '<div style="display:flex;align-items:center;gap:10px;margin:10px 0;">'
            row_html += (
                f'<div><p style="color:#aaa;font-size:12px;">Query</p>'
                f'<img src="data:image/png;base64,{query_b64}" '
                f'width="250"></div>'
            )
            row_html += '<span style="color:#aaa;font-size:24px;">→</span>'
            for n in q["neighbors"]:
                # kNN nearest() includes npz_path directly from bank records;
                # other scorers return "index" — resolve via bank_records.
                if "npz_path" in n:
                    n_path = n["npz_path"]
                else:
                    n_path = bank_records[n["index"]].get("npz_path", "")
                n_b64 = render_bev_to_base64(n_path, viz_module_path)
                row_html += (
                    f'<div><p style="color:#aaa;font-size:12px;">'
                    f"d={n['distance']:.4f}</p>"
                    f'<img src="data:image/png;base64,{n_b64}" '
                    f'width="250"></div>'
                )
            row_html += "</div>"
            method_html += row_html
        bev_sections.append(method_html)

    html = (
        "<!DOCTYPE html>\n"
        '<html><head><meta charset="utf-8">'
        "<title>Scoring Method Comparison</title>\n"
        "<style>"
        "body{background:#111;color:#eee;font-family:monospace;padding:20px;}"
        "h1,h2{color:#2196F3;}"
        "img{border:1px solid #333;border-radius:4px;}"
        "</style></head>\n"
        "<body>"
        "<h1>Latent Scoring Method Comparison</h1>\n"
        f"<p>{len(records)} frames scored</p>\n"
        "<h2>Score Distributions</h2>\n"
        f'<img src="data:image/png;base64,{dist_b64}" width="100%">\n'
        "<h2>Rank Correlation (Spearman)</h2>\n"
        f'<img src="data:image/png;base64,{corr_b64}" width="500">\n'
        "<h2>Retrieval Comparison</h2>\n" + "".join(bev_sections) + "</body></html>"
    )

    html_path = output_dir / "comparison_report.html"
    with open(html_path, "w") as f:
        f.write(html)
    print(f"HTML report written to {html_path}")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    encoder = EncoderInference(args.args_path, args.model_path, args.device)

    records, knn_s, maha_s, kmeans_s, gmm_s, emb_l2, emb_raw = score_all_methods(
        encoder,
        args.eval_list,
        args.bank_dir,
        args.batch_size,
        args.num_workers,
        args.device,
        args.output_dir,
    )

    corr_matrix, corr_labels = compute_rank_correlation(records, args.output_dir)
    compute_outlier_agreement(records, args.output_dir)

    if not args.skip_viz:
        # Select query scenes
        with open(args.eval_list) as f:
            npz_paths = json.load(f)
        if args.query_npz:
            query_paths = args.query_npz
        else:
            rng = np.random.RandomState(args.seed)
            idxs = rng.choice(len(npz_paths), size=2, replace=False)
            query_paths = [npz_paths[i] for i in idxs]

        viz_path = args.viz_module_path or (
            Path(__file__).resolve().parent.parent.parent / "clip-review-tool" / "src"
        )

        # Load bank records for npz_path resolution
        bank_records = []
        with open(args.bank_dir / "paths.jsonl") as f:
            for line in f:
                if line.strip():
                    bank_records.append(json.loads(line.strip()))

        retrieval = retrieve_neighbors(
            query_paths,
            encoder,
            (knn_s, maha_s, kmeans_s, gmm_s),
            emb_l2,
            emb_raw,
            bank_records,
            args.device,
            args.output_dir,
            viz_path,
        )

        generate_html_report(
            records,
            corr_matrix,
            corr_labels,
            retrieval,
            bank_records,
            args.output_dir,
            viz_path,
        )
    else:
        print("Skipping BEV visualization (--skip_viz)")


if __name__ == "__main__":
    main()
