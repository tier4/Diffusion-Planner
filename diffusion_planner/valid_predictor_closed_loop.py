"""Closed-loop validation of a Diffusion-Planner checkpoint.

Two modes (``--mode``):

* **full** (default) — roll out every route under ``--npz_root`` in ``--seg_len`` chunks;
  same behavior as the original mining validation CLI.
* **grouped** — one full-route closed-loop rollout per sequence, then per-map-area
  metrics and videos split by reproducer frame index (from ``scenario_classification_json``).

Open-loop counterpart: ``valid_predictor.py``.

Examples::

    # Full-route validation
    python diffusion_planner/valid_predictor_closed_loop.py \\
        --model_path ./best_model.pth \\
        --npz_root /path/to/valid/2026-01-15

    # Grouped-by-area validation (after classify_scenario_corpus)
    python diffusion_planner/valid_predictor_closed_loop.py \\
        --mode grouped \\
        --model_path ./best_model.pth \\
        --npz_root /path/to/x2_dev/2231_odaiba.../valid/2026-01-15 \\
        --classification_json_root ../Diffusion-Planner-Meta-Repository/dataset/scenario_classification_json \\
        --out_dir ./odaiba_cl_results \\
        --near_miss_thresh 0.3 \\
        --replan_interval 10 \\
        --draw_every 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scenario_generation.closed_loop_cli import (  # noqa: E402
    add_full_route_args,
    add_grouped_args,
    add_rollout_args,
    print_full_summary,
    run_full_route_eval,
    run_grouped_eval,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=("full", "grouped"),
        default="full",
        help="full: segment-wise rollout over all routes; grouped: per-area metrics from classification JSON",
    )
    add_rollout_args(p)
    add_full_route_args(p)
    add_grouped_args(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "grouped":
        return run_grouped_eval(args)

    summary = run_full_route_eval(args)
    out_dir = args.out_dir
    if out_dir is None:
        from datetime import datetime

        out_dir = args.model_path.parent / "closed_loop" / datetime.now().strftime("%Y%m%d_%H%M%S")
    print_full_summary(summary, args.near_miss_thresh, Path(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
