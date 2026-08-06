"""Single source of truth for closed-loop score/metric key names -- the small headline numbers
surfaced in W&B (per-group score trends + cross-group overview) and the local HTML report, split
into two groups:

- COMPARISON: meaningful in both objects and empty-world ("__noobj") ablation mode -- shown for
  every group/label, objects vs noobj overlaid as separate lines.
- OBJECTS_ONLY: structurally 0 with no traffic to react to (collision-with-other-agents counts)
  -- shown for objects-labeled groups only, since a noobj line here is always a meaningless flat 0.

Kept dependency-free (no torch/matplotlib/wandb) so wandb_closed_loop_workspace.py's standalone
CLI can import it without pulling in the heavier deps wandb_closed_loop.py/
closed_loop_html_report.py need.
"""

from __future__ import annotations

# Per-group score keys: closed_loop_scores/{key}/{label} in wandb; summary/episode columns in the
# HTML report.
COMPARISON_SCORE_KEYS = (
    "mean_route_completion",
    "total_curb_hits",
    "total_snaps",
    "total_red_light_violations",
    "total_strong_brakes",
)
OBJECTS_ONLY_SCORE_KEYS = ("total_collision_events",)
SCORE_KEYS = COMPARISON_SCORE_KEYS + OBJECTS_ONLY_SCORE_KEYS

# Cross-group overview sum keys: closed_loop_overview/{key} in wandb, summed across ALL groups
# (COMPARISON) or objects-labeled groups only (OBJECTS_ONLY). mean_route_completion is NOT here --
# the overview computes a segment-weighted average under its own "route_completion" key instead
# of a plain sum.
COMPARISON_OVERVIEW_SUM_KEYS = (
    "total_curb_hits",
    "total_snaps",
    "total_red_light_violations",
    "total_strong_brakes",
    "n_segments_diverged",
)
OBJECTS_ONLY_OVERVIEW_SUM_KEYS = ("total_collision_events",)

# Resolve each key from a (segment or summary) dict -- closed_loop_eval's nested-metrics
# categories (object/road_border/red_light_violation/strong_brake/reproducer) replaced the old
# flat keys. mean_route_completion/n_segments_diverged have no nested category and stay flat.
SCORE_EXTRACTORS = {
    "mean_route_completion": lambda d: d.get("mean_route_completion"),
    "n_segments_diverged": lambda d: d.get("n_segments_diverged"),
    "total_collision_events": lambda d: d.get("object", {}).get("collision_count"),
    "total_curb_hits": lambda d: d.get("road_border", {}).get("collision_count"),
    "total_snaps": lambda d: d.get("reproducer", {}).get("snap_count"),
    "total_red_light_violations": lambda d: d.get("red_light_violation", {}).get("count"),
    "total_strong_brakes": lambda d: d.get("strong_brake", {}).get("count"),
}


def extract_score(d: dict, key: str):
    """Resolve one of the small headline score keys from a segment or summary dict."""
    return SCORE_EXTRACTORS[key](d)
