from __future__ import annotations

from scenario_generation.turn_interval import CLASS_NAMES

# Resolve each key from a (segment or summary) dict -- closed_loop_eval's nested-metrics
# categories (object/road_border/red_light_violation/strong_brake/reproducer) replaced the old
# flat keys. mean_route_completion/n_segments_diverged have no nested category and stay flat.
SCORE_EXTRACTORS = {
    "mean_route_completion": lambda d: d.get("mean_route_completion"),
    "n_segments_diverged": lambda d: d.get("n_segments_diverged"),
    "pass_rate": lambda d: d.get("pass_rate"),
    "fail_count": lambda d: d.get("fail_count"),
    "total_collision_events": lambda d: d.get("object", {}).get("collision_count"),
    "total_rear_collision_events": lambda d: d.get("object_rear", {}).get("collision_count"),
    "total_curb_hits": lambda d: d.get("road_border", {}).get("collision_count"),
    "total_snaps": lambda d: d.get("reproducer", {}).get("snap_count"),
    "total_red_light_violations": lambda d: d.get("red_light_violation", {}).get("count"),
    "total_strong_brakes": lambda d: d.get("strong_brake", {}).get("count"),
    "turn_indicator_transition_accuracy": lambda d: d.get("turn_indicator", {}).get(
        "transition_accuracy"
    ),
    "turn_indicator_false_positive_rate": lambda d: d.get("turn_indicator", {}).get(
        "false_positive_rate"
    ),
}


def _turn_interval_extractor(name: str, field: str):
    # A per-group summary nests the stats under ``turn_indicator``; the cross-group aggregate
    # row carries them flat as ``turn_indicator_interval_stats``.
    def extract(d: dict):
        stats = d.get("turn_indicator", {}).get("interval_stats") or d.get(
            "turn_indicator_interval_stats"
        )
        return (stats or {}).get(name, {}).get(field)

    return extract


for _name in CLASS_NAMES:
    for _field in ("accuracy_mean", "ttc_mean_s", "zero_rate"):
        SCORE_EXTRACTORS[f"turn_interval_{_name}_{_field}"] = _turn_interval_extractor(
            _name, _field
        )


def extract_score(d: dict, key: str):
    """Resolve one of the small headline score keys from a segment or summary dict."""
    return SCORE_EXTRACTORS[key](d)
