"""Stage 3: Render BEV overlays and self-contained HTML report."""

import argparse
import base64
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from human_match_prototype.route_projection import stitch_route_lanes
from human_match_prototype.sampler import TrajectorySampler

DEFAULT_MODEL_DIR = Path("/opt/autoware/mlmodels/diffusion_planner_for_x2")


def _b64(png_path: Path) -> str:
    return base64.b64encode(png_path.read_bytes()).decode()


def render_scene_overlay(
    sampler: TrajectorySampler,
    npz_path: str,
    out_png: Path,
    num_samples: int = 64,
    seed: int = 0,
    temperature: float = 1.0,
) -> None:
    """Render BEV overlay with route centerline and planner samples."""
    try:
        from src.visualization import PAST_FRAMES, precompute_static, render_frame
    except ImportError:
        raise ImportError(
            "clip-review-tool is required for BEV overlays. "
            "Install with: uv pip install -e ../clip-review-tool"
        )

    data = dict(np.load(npz_path, allow_pickle=True))
    r = sampler.sample(str(npz_path), num_samples=num_samples, seed=seed, temperature=temperature)

    fig, ax = plt.subplots(figsize=(14, 14))
    static = precompute_static(data)
    static["view_half"] = static["view_half"] * 0.7
    render_frame(fig, ax, data, static, t=PAST_FRAMES - 1, filename=Path(npz_path).name)

    # Planner samples
    for s in r.ego_samples:
        ax.plot(s[:, 0], s[:, 1], color="#E040FB", alpha=0.3, lw=0.5, zorder=40)

    # Human trajectory
    human = r.human_future[:, :2]
    ax.plot(human[:, 0], human[:, 1], color="#00E676", lw=2.5, zorder=45, label="Human")

    # Route centerline
    route_lanes = data["route_lanes"]
    route = stitch_route_lanes(np.asarray(route_lanes))
    if route.qa.route_valid and len(route.centerline) > 1:
        ax.plot(
            route.centerline[:, 0],
            route.centerline[:, 1],
            color="#00BCD4",
            lw=1.5,
            ls="--",
            alpha=0.7,
            zorder=38,
            label="Route",
        )

    ax.legend(loc="upper right", fontsize=9, framealpha=0.7)
    fig.savefig(out_png, dpi=120, facecolor="#1A1A1A", bbox_inches="tight")
    plt.close(fig)


def render_html_report(
    review_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
    overlay_pngs: list[Path],
    dist_png: Path,
    out_html: Path,
    metadata: dict,
) -> None:
    """Generate self-contained HTML report with embedded images."""
    n_total = len(ranked_df)
    n_route_valid = (
        int(ranked_df["route_valid"].sum()) if "route_valid" in ranked_df.columns else "N/A"
    )

    # Build review table
    review_cols = [
        c
        for c in [
            "npz_path",
            "R_overall",
            "R_lateral",
            "selection_reason",
            "es_2s",
            "es_4s",
            "es_8s",
            "es_lat_2s",
            "es_lat_4s",
            "es_lat_8s",
            "route_valid",
        ]
        if c in review_df.columns
    ]
    head = "".join(f"<th>{c}</th>" for c in review_cols)
    body = ""
    for _, row in review_df.iterrows():
        tds = "".join(
            f"<td>{row[c]:.3f}</td>"
            if isinstance(row.get(c), float) and not np.isnan(row.get(c, float("nan")))
            else f"<td>{row.get(c, '')}</td>"
            for c in review_cols
        )
        body += f"<tr>{tds}</tr>\n"

    overlays_html = "\n".join(
        f'<figure><img src="data:image/png;base64,{_b64(p)}" style="max-width:700px">'
        f"<figcaption>{p.stem}</figcaption></figure>"
        for p in overlay_pngs
        if p.exists()
    )

    dist_img = (
        f'<img src="data:image/png;base64,{_b64(dist_png)}" style="max-width:100%">'
        if dist_png.exists()
        else ""
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Per-Scene Evaluation Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 2rem; max-width: 1200px; color: #1a1a1a; }}
h1 {{ border-bottom: 2px solid #2a78d6; padding-bottom: 0.5em; }}
table {{ border-collapse: collapse; font-size: 13px; margin: 1em 0; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: right; }}
th {{ background: #f5f5f5; text-align: center; }}
.meta {{ background: #f0f4ff; padding: 1em; border-radius: 6px; margin: 1em 0; font-size: 13px; }}
figure {{ display: inline-block; margin: 8px; }}
figcaption {{ font-size: 12px; text-align: center; color: #666; }}
.warning {{ color: #d03b3b; font-weight: 600; }}
</style></head><body>
<h1>Per-Scene Evaluation Report</h1>
<div class="meta">
<strong>Temperature:</strong> {metadata.get("temperature", "?")}&emsp;
<strong>Seed:</strong> {metadata.get("seed", "?")}&emsp;
<strong>Samples:</strong> {metadata.get("num_samples", "?")}&emsp;
<strong>Scenes scored:</strong> {n_total}&emsp;
<strong>Valid routes:</strong> {n_route_valid}&emsp;
<p class="warning">Interpretation: Top-ranked scenes are highest-disagreement review candidates, not automatic planner failures.
T=1.0 evaluates the training-matched distribution, not deployed T=0.5.</p>
</div>

<h2>Score Distributions</h2>
{dist_img}

<h2>Review Candidates ({len(review_df)} scenes)</h2>
<p>Selection: Top5(R_overall) &cup; Top5(R_lateral), deduplicated.</p>
<table><tr>{head}</tr>{body}</table>

<h2>BEV Overlays</h2>
<p>Magenta: planner samples. Green: human trajectory. Cyan dashed: route centerline.</p>
{overlays_html}

</body></html>"""
    out_html.write_text(html)


def main():
    p = argparse.ArgumentParser(description="Stage 3: Render report with BEV overlays.")
    p.add_argument("--review_set", required=True)
    p.add_argument("--scores", required=True, help="ranked.csv from Stage 2")
    p.add_argument("--output", required=True, help="Output HTML path")
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model_dir", type=Path, default=None)
    args = p.parse_args()

    model_dir = args.model_dir or DEFAULT_MODEL_DIR
    sampler = TrajectorySampler(
        str(model_dir / "args.json"),
        str(model_dir / "diffusion_planner.onnx"),
        args.device,
    )

    review_df = pd.read_csv(args.review_set)
    ranked_df = pd.read_csv(args.scores)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_path.parent / "overlays"
    overlay_dir.mkdir(exist_ok=True)

    overlay_pngs = []
    for _, row in review_df.iterrows():
        npz = row["npz_path"]
        png = overlay_dir / f"{Path(npz).stem}.png"
        print(f"Rendering {Path(npz).name}...")
        render_scene_overlay(sampler, npz, png, args.num_samples, args.seed, args.temperature)
        overlay_pngs.append(png)

    dist_png = out_path.parent / "distributions.png"

    render_html_report(
        review_df,
        ranked_df,
        overlay_pngs,
        dist_png,
        out_path,
        {
            "temperature": args.temperature,
            "seed": args.seed,
            "num_samples": args.num_samples,
            "n_scenes": len(ranked_df),
        },
    )
    print(f"Report: {out_path}")


if __name__ == "__main__":
    main()
