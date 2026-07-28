# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate an HTML diagnostic report for cluster-weighted sampling.

Usage:
    python visualize_cluster_report.py \\
        --cluster_json /path/to/cluster_result.json \\
        --output_dir   /path/to/report_output/ \\
        [--max_videos 3] [--workers 1] [--seed 42] [--standalone] \\
        [--cluster_weight_alpha 1.0]

Reads the cluster assignment JSON from cluster.py, computes sampling
statistics, renders BEV video examples per cluster via render-video-txt
(clip-review-tool), and assembles an HTML report.

With --standalone, MP4 videos are converted to GIFs (240px, 3fps) and
embedded as base64 in the HTML, producing a single shareable file.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import random
import shutil
import subprocess
import tempfile
import warnings
import zlib
from datetime import datetime
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_cluster_json(path: str) -> dict[str, list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cluster_index(cluster_id: str) -> tuple[int, int, str]:
    """Sort key that orders ``cluster_id<N>`` numerically (id10 after id2).

    Falls back to lexicographic order for keys that do not match that shape. The
    report reads a JSON that may have been hand-edited or produced elsewhere, so a
    key like ``noise`` must not crash it with a bare ``invalid literal for int()``.
    Mirrors the tolerant ordering train.py uses for its startup log.
    """
    suffix = (
        cluster_id.replace("cluster_id", "", 1)
        if cluster_id.startswith("cluster_id")
        else cluster_id
    )
    return (0, int(suffix), "") if suffix.isdigit() else (1, 0, cluster_id)


def compute_cluster_stats(clusters: dict[str, list[str]], alpha: float = 1.0) -> list[dict]:
    """Compute per-cluster stats. ``alpha`` must match training's --cluster_weight_alpha."""
    # Reject non-finite values (NaN, +/-inf) and anything outside [0, 1]. ``inf``
    # would silently render an all-nan Weight column that still looks like output,
    # and alpha > 1 inverts the weighting. Matches the guard in
    # ClusterWeightedDistributedSampler so the two cannot disagree on what is valid.
    if not (math.isfinite(alpha) and 0 <= alpha <= 1):
        raise ValueError(
            f"alpha={alpha} must be a finite number in [0, 1] "
            f"(1.0 = full inverse weighting, 0.0 = uniform)"
        )

    # Drop zero-count clusters. ``freq`` would be 0, so ``(1/(0 + 1e-8)) ** alpha``
    # renders a weight of ~1e8 in the Weight column -- a number training never
    # applies, because the sampler builds cluster_counts/cluster_multipliers from
    # live matches and omits empty clusters entirely. This report is read to pick
    # alpha, so a phantom 1e8 row is worse than no row.
    sorted_ids = [
        cid for cid in sorted(clusters.keys(), key=cluster_index) if len(clusters[cid]) > 0
    ]
    total = sum(len(v) for v in clusters.values())
    # ClusterWeightedDistributedSampler raises on an all-empty cluster JSON. Silently
    # emitting a report with "Total samples: 0" would have this tool -- the one built
    # to catch a broken cluster JSON before a training launch -- call it fine.
    if total == 0:
        raise ValueError(
            f"Cluster JSON contains no paths ({len(clusters)} clusters, all empty). "
            f"Check cluster JSON."
        )

    raw_weights = {}
    for cid in sorted_ids:
        freq = len(clusters[cid]) / total
        raw_weights[cid] = (1.0 / (freq + 1e-8)) ** alpha

    mean_weight = sum(raw_weights[cid] * len(clusters[cid]) for cid in sorted_ids) / total
    weights = {cid: w / mean_weight for cid, w in raw_weights.items()}

    total_weight = sum(weights[cid] * len(clusters[cid]) for cid in sorted_ids)

    stats = []
    for cid in sorted_ids:
        count = len(clusters[cid])
        w = weights[cid]
        sampling_rate = (w * count) / total_weight
        draws = sampling_rate * total
        stats.append(
            {
                "cluster_id": cid,
                "count": count,
                "pct": 100.0 * count / total,
                "weight": w,
                "sampling_rate": sampling_rate,
                "draws_per_epoch": draws,
                "repeats_per_sample": draws / count if count > 0 else 0.0,
            }
        )
    return stats


def render_bar_chart(stats: list[dict]) -> str:
    stats_sorted = sorted(stats, key=lambda s: s["count"], reverse=True)
    ids = [s["cluster_id"].replace("cluster_id", "") for s in stats_sorted]
    counts = [s["count"] for s in stats_sorted]

    fig, ax = plt.subplots(figsize=(max(6, len(ids) * 0.6), 4))
    ax.bar(ids, counts, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Sample Count")
    ax.set_title("Cluster Size Distribution (sorted by count)")
    ax.tick_params(axis="x", rotation=45 if len(ids) > 15 else 0)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def subsample_cluster_paths(
    clusters: dict[str, list[str]], max_videos: int, seed: int
) -> dict[str, list[str]]:
    result = {}
    for cid, paths in clusters.items():
        # Per-cluster seed offset. Tolerant of non-``cluster_id<N>`` keys so a
        # hand-edited JSON picks videos reproducibly instead of dying on int().
        # crc32, not hash(): str hashing is salted per process, which would make
        # --seed non-reproducible across runs.
        kind, index, _ = cluster_index(cid)
        offset = index if kind == 0 else zlib.crc32(cid.encode()) % 100_000
        rng = random.Random(seed + offset)
        if len(paths) <= max_videos:
            result[cid] = list(paths)
        else:
            result[cid] = rng.sample(paths, max_videos)
    return result


def render_cluster_videos(
    subsampled: dict[str, list[str]], output_dir: str, workers: int
) -> tuple[dict[str, list[str]], list[dict]]:
    if not shutil.which("render-video-txt"):
        raise RuntimeError(
            "render-video-txt not found on PATH. "
            "Install clip-review-tool: pip install -e /path/to/clip-review-tool"
        )

    videos_dir = Path(output_dir) / "videos"
    rendered: dict[str, list[str]] = {}
    errors: list[dict] = []

    for cluster_id, paths in subsampled.items():
        if not paths:
            rendered[cluster_id] = []
            continue

        cluster_dir = videos_dir / cluster_id
        cluster_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for p in paths:
                f.write(p + "\n")
            txt_path = f.name

        try:
            result = subprocess.run(
                ["render-video-txt", txt_path, str(cluster_dir), "--workers", str(workers)],
                check=False,
            )
            if result.returncode != 0:
                warnings.warn(
                    f"render-video-txt exited with code {result.returncode} for {cluster_id}"
                )
        finally:
            Path(txt_path).unlink(missing_ok=True)

        mp4s = sorted(str(p) for p in cluster_dir.glob("*.mp4"))
        rendered[cluster_id] = mp4s

        log_path = cluster_dir / "render_log.jsonl"
        if log_path.exists():
            for line in log_path.read_text().splitlines():
                # render-video-txt is an external tool; a blank or truncated line
                # must not throw away a completed render pass.
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    warnings.warn(f"Skipping malformed render_log.jsonl line in {cluster_id}")
                    continue
                if entry.get("status") == "error":
                    errors.append(
                        {
                            "cluster_id": cluster_id,
                            "file": entry.get("file", ""),
                            "reason": entry.get("reason", "unknown"),
                        }
                    )

    return rendered, errors


def convert_mp4_to_gif(mp4_path: str) -> str:
    gif_path = str(Path(mp4_path).with_suffix(".gif"))
    if Path(gif_path).exists():
        return gif_path
    result = subprocess.run(
        [
            "ffmpeg",
            "-i",
            mp4_path,
            "-vf",
            "fps=3,scale=240:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            "-y",
            gif_path,
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        warnings.warn(f"ffmpeg failed for {mp4_path}: {result.stderr.decode()[-200:]}")
    return gif_path


def _render_errors_section(errors: list[dict]) -> str:
    if not errors:
        return ""
    rows = ""
    for e in errors:
        rows += (
            f"<tr><td>{escape(e['cluster_id'])}</td>"
            f"<td>{escape(e['file'])}</td>"
            f"<td>{escape(e['reason'])}</td></tr>\n"
        )
    return f"""
    <h2>Render Errors</h2>
    <p>{len(errors)} file(s) failed to render:</p>
    <table>
    <tr><th>Cluster</th><th>File</th><th>Reason</th></tr>
    {rows}
    </table>
    """


def generate_html_report(
    stats: list[dict],
    chart_uri: str,
    rendered: dict[str, list[str]],
    errors: list[dict],
    cluster_json_path: str,
    output_dir: str,
    standalone: bool = False,
    alpha: float = 1.0,
) -> str:
    total = sum(s["count"] for s in stats)
    n_clusters = len(stats)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_path = Path(output_dir) / "report.html"

    # --- sort by sample count (largest first) for display ---
    stats_sorted = sorted(stats, key=lambda s: s["count"], reverse=True)

    # --- stats table rows ---
    table_rows = ""
    for s in stats_sorted:
        table_rows += (
            f"<tr>"
            f"<td>{escape(s['cluster_id'])}</td>"
            f"<td>{s['count']:,}</td>"
            f"<td>{s['pct']:.1f}%</td>"
            f"<td>{s['weight']:.3f}</td>"
            f"<td>{s['sampling_rate']:.4f}</td>"
            f"<td>{s['draws_per_epoch']:.1f}</td>"
            f"<td>{s['repeats_per_sample']:.2f}</td>"
            f"</tr>\n"
        )

    # --- per-cluster video/gif galleries ---
    galleries = ""
    for s in stats_sorted:
        cid = s["cluster_id"]
        mp4s = rendered.get(cid, [])
        media_tags = ""
        if standalone:
            for mp4_path in mp4s:
                gif_path = convert_mp4_to_gif(mp4_path)
                if Path(gif_path).exists():
                    with open(gif_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("ascii")
                    media_tags += (
                        f'<img src="data:image/gif;base64,{b64}" '
                        f'style="border-radius:4px;background:#1a1a1a"> '
                    )
        else:
            for mp4_path in mp4s:
                rel = str(Path(mp4_path).relative_to(output_dir))
                media_tags += (
                    f'<video controls preload="none" width="360">'
                    f'<source src="{escape(rel)}" type="video/mp4">'
                    f"</video>\n"
                )
        # Key the placeholder on whether any tag was actually produced, not on
        # whether MP4s existed: in --standalone mode every ffmpeg conversion can
        # fail (which only warns), leaving an empty gallery with no explanation.
        if not media_tags:
            media_tags = (
                "<p><em>No videos rendered for this cluster.</em></p>"
                if not mp4s
                else f"<p><em>{len(mp4s)} video(s) rendered but GIF conversion failed "
                "&mdash; check the ffmpeg warnings.</em></p>"
            )
        galleries += f"""
        <div class="cluster-gallery">
            <h3>{escape(cid)} &mdash; {s["count"]:,} samples, weight {s["weight"]:.3f},
                ~{s["repeats_per_sample"]:.1f}x repeats/sample/epoch</h3>
            <div class="video-grid">{media_tags}</div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Cluster Diagnostic Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       max-width: 1400px; margin: 0 auto; padding: 20px; background: #f8f9fa; color: #212529; }}
h1, h2, h3 {{ color: #1a1a2e; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
th, td {{ border: 1px solid #dee2e6; padding: 8px 12px; text-align: right; }}
th {{ background: #343a40; color: white; }}
tr:nth-child(even) {{ background: #f1f3f5; }}
td:first-child, th:first-child {{ text-align: left; }}
.chart {{ text-align: center; margin: 20px 0; }}
.chart img {{ max-width: 100%; }}
.info-box {{ background: #e7f5ff; border-left: 4px solid #339af0; padding: 16px; margin: 20px 0; }}
.cluster-gallery {{ margin: 24px 0; padding: 16px; background: white; border-radius: 8px;
                     box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.video-grid {{ display: flex; flex-wrap: wrap; gap: 12px; }}
video {{ border-radius: 4px; background: #1a1a1a; }}
.meta {{ color: #868e96; font-size: 0.9em; }}
</style>
</head>
<body>

<h1>Cluster Diagnostic Report</h1>
<p class="meta">Generated: {now} | Source: <code>{escape(cluster_json_path)}</code> |
   Total samples: {total:,} | Clusters: {n_clusters}</p>

<h2>Pipeline Overview</h2>
<p>Features extracted from each NPZ &rarr; Z-score normalization &rarr; PCA
   &rarr; Elbow KMeans &rarr; {n_clusters} clusters.</p>
<p class="meta">The specific feature set and PCA width depend on how
   <code>cluster.py</code> was invoked and are not recorded in the cluster JSON:
   <code>--mode trajectory</code> (the default) flattens the ego future trajectory
   only, while <code>--mode enriched</code> adds neighbor and ego-state blocks
   through a two-stage PCA. Check the <code>cluster.py</code> command line for the
   actual <code>--pca_components</code> / <code>--neighbor_pca_components</code>
   values used.</p>

<h2>Cluster Distribution</h2>
<div class="chart"><img src="{chart_uri}" alt="Cluster size distribution"></div>
<div style="overflow-x: auto;">
<table>
<tr>
  <th>Cluster ID</th><th>Sample Count</th><th>% of Total</th>
  <th>Weight</th><th>Sampling Rate</th><th>Draws/Epoch</th>
  <th>Repeats/Sample</th>
</tr>
{table_rows}
</table>
</div>

<h2>Sampling Behavior</h2>
<div class="info-box">
<p><strong>Does it oversample (repeat)?</strong> Yes. The sampler uses
   <code>torch.multinomial(weights, total, replacement=True)</code>.
   Samples from rare clusters appear multiple times per epoch.
   Samples from common clusters may be skipped in some epochs.</p>
<p><strong>What happens to unmatched samples?</strong> Samples not in any cluster
   receive the mean of matched weights (neutral rate). They are neither
   boosted nor starved. A warning is emitted.</p>
<p><strong>Weight exponent (alpha)</strong> = {alpha:.2f}. Each sample's weight is
   <code>(1 / cluster_frequency) ** alpha</code>, normalized to mean 1.0 &mdash; so the
   Weight column above is the oversampling multiplier. alpha=1.0 gives every
   cluster an equal share of draws; lower values soften it; 0.0 is uniform.</p>
<p><strong>Total draws per epoch</strong> = <code>len(data_list)</code>
   ({total:,}), same as vanilla DistributedSampler.</p>
</div>

<h2>Per-Cluster Video Examples</h2>
{galleries}

{_render_errors_section(errors)}

</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return str(output_path)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an HTML diagnostic report for cluster-weighted sampling"
    )
    parser.add_argument(
        "--cluster_json",
        type=str,
        required=True,
        help="Path to the clustering result JSON produced by cluster.py",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for report.html and videos/",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=3,
        help="Max BEV video examples to render per cluster (default: 3)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel video rendering workers (default: 1)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cluster_weight_alpha",
        type=float,
        default=1.0,
        help="Exponent on inverse-frequency weights in [0, 1], matching training's "
        "--cluster_weight_alpha (1.0 = full inverse, 0.0 = uniform)",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Produce a single self-contained HTML with embedded GIFs (240px, 3fps)",
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()

    # Preflight ffmpeg alongside render-video-txt. Without this, --standalone renders
    # every cluster's videos first and only then dies with FileNotFoundError inside
    # generate_html_report -- throwing away the entire expensive render pass.
    if args.standalone and not shutil.which("ffmpeg"):
        raise RuntimeError(
            "--standalone needs ffmpeg on PATH to convert MP4s to embedded GIFs, "
            "and it was not found. Install ffmpeg or drop --standalone."
        )

    print(f"Loading cluster JSON: {args.cluster_json}")
    clusters = load_cluster_json(args.cluster_json)
    total = sum(len(v) for v in clusters.values())
    print(f"  {len(clusters)} clusters, {total:,} total samples")

    print("Computing cluster statistics...")
    stats = compute_cluster_stats(clusters, alpha=args.cluster_weight_alpha)

    print("Rendering bar chart...")
    chart_uri = render_bar_chart(stats)

    print(f"Subsampling {args.max_videos} videos per cluster...")
    subsampled = subsample_cluster_paths(clusters, args.max_videos, args.seed)
    n_videos = sum(len(v) for v in subsampled.values())
    print(f"  {n_videos} videos to render across {len(subsampled)} clusters")

    print(f"Rendering videos (workers={args.workers})...")
    rendered, errors = render_cluster_videos(subsampled, args.output_dir, args.workers)
    if errors:
        print(f"  {len(errors)} file(s) failed to render")

    print("Generating HTML report...")
    report_path = generate_html_report(
        stats,
        chart_uri,
        rendered,
        errors,
        args.cluster_json,
        args.output_dir,
        standalone=args.standalone,
        alpha=args.cluster_weight_alpha,
    )
    print(f"Report saved to {report_path}")
    if args.standalone:
        size_mb = Path(report_path).stat().st_size / 1024 / 1024
        print(f"  Standalone mode: {size_mb:.1f}MB self-contained HTML")


if __name__ == "__main__":
    main()
