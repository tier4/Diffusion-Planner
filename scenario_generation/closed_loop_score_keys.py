from __future__ import annotations

# Resolve each key from a (segment or summary) dict -- closed_loop_eval's nested-metrics
# categories (object/road_border/red_light_violation/strong_brake/reproducer) replaced the old
# flat keys. mean_route_completion/n_segments_diverged have no nested category and stay flat.
SCORE_EXTRACTORS = {
    "mean_route_completion": lambda d: d.get("mean_route_completion"),
    "n_segments_diverged": lambda d: d.get("n_segments_diverged"),
    "pass_rate": lambda d: d.get("pass_rate"),
    "fail_count": lambda d: d.get("fail_count"),
    "total_collision_events": lambda d: d.get("object", {}).get("collision_count"),
    "total_curb_hits": lambda d: d.get("road_border", {}).get("collision_count"),
    "total_snaps": lambda d: d.get("reproducer", {}).get("snap_count"),
    "total_red_light_violations": lambda d: d.get("red_light_violation", {}).get("count"),
    "total_strong_brakes": lambda d: d.get("strong_brake", {}).get("count"),
}


def extract_score(d: dict, key: str):
    """Resolve one of the small headline score keys from a segment or summary dict."""
    return SCORE_EXTRACTORS[key](d)
