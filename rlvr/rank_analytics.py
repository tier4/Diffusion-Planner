"""Rank analytics for ranked SFT: track which generation configs win and why."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from rlvr.reward import RewardBreakdown, RewardConfig


# ---------------------------------------------------------------------------
# Generation config labels
# ---------------------------------------------------------------------------

_FIXED_LABELS = [
    "det_pure",            # 0: no guidance, no noise
    "CL5_SPD5_det",        # 1
    "CL8_SPD5_det",        # 2
    "CL10_SPD8_det",       # 3
    "CL10_SPD10_det",      # 4
    "CL5_SPD5_noisy",      # 5
    "CL8_SPD8_noisy",      # 6
    "CL10_SPD8_noisy",     # 7
    "CL10_SPD10_noisy",    # 8
]

_CATEGORY_MAP = {
    "det_pure": "det_pure",
    "CL5_SPD5_det": "guided_det",
    "CL8_SPD5_det": "guided_det",
    "CL10_SPD8_det": "guided_det",
    "CL10_SPD10_det": "guided_det",
    "CL5_SPD5_noisy": "guided_noisy",
    "CL8_SPD8_noisy": "guided_noisy",
    "CL10_SPD8_noisy": "guided_noisy",
    "CL10_SPD10_noisy": "guided_noisy",
}


def _classify_label(label: str) -> str:
    """Classify any label (including experimental ones) into a category."""
    if label in _CATEGORY_MAP:
        return _CATEGORY_MAP[label]
    if label.startswith("random_") or label.startswith("explorer_"):
        return "random"
    if label == "det_pure":
        return "det_pure"
    # Experimental labels — classify by structure
    if label.startswith("noise_n"):
        return "noise_only_exp"
    if "_col" in label:
        return "collision_exp"
    if "lat" in label.lower():
        return "lateral_exp"
    if "str" in label.lower():
        return "stretched_exp"
    if "_only_" in label.lower():
        return "decoupled_exp"
    # Fall back to noisy/det based on naming
    if "_det" in label or label.endswith("_det"):
        return "guided_det"
    return "guided_noisy"


def get_generation_config_labels(K: int = 16) -> list[str]:
    """Return human-readable labels for the K generation config indices."""
    labels = list(_FIXED_LABELS)
    for i in range(len(labels), K):
        labels.append(f"random_{i}")
    return labels[:K]


def get_category(label: str) -> str:
    """Map a config label to its category."""
    return _classify_label(label)


# ---------------------------------------------------------------------------
# Dominant component analysis
# ---------------------------------------------------------------------------

def _breakdown_to_dict(r: RewardBreakdown) -> dict[str, float]:
    return {
        "safety": r.safety,
        "progress": r.progress,
        "smoothness": r.smoothness,
        "feasibility": r.feasibility,
        "centerline": r.centerline,
        "red_light": r.red_light,
        "total": r.total,
        "rb_near_penalty": r.rb_near_penalty,
        "rb_wide_penalty": r.rb_wide_penalty,
        "rb_crossing": bool(r.rb_crossing),
    }


def mean_breakdown_dict(rewards: list[RewardBreakdown]) -> dict[str, float]:
    """Compute mean of each breakdown field across K trajectories."""
    fields = [
        "safety", "progress", "smoothness", "feasibility",
        "centerline", "red_light", "rb_near_penalty", "rb_wide_penalty",
    ]
    K = len(rewards)
    if K == 0:
        return {f: 0.0 for f in fields}
    return {f: sum(getattr(r, f) for r in rewards) / K for f in fields}


def compute_dominant_component(
    winner: RewardBreakdown,
    mean_bd: dict[str, float],
    config: RewardConfig,
) -> tuple[str, float]:
    """Find which weighted component contributes most to the winner's advantage.

    Uses the quality_score formula from reward.py:2151-2158.
    Returns (component_name, weighted_delta).
    """
    deltas = {
        "progress": config.w_progress * (winner.progress - mean_bd["progress"]),
        "safety": config.w_safety * (winner.safety - mean_bd["safety"]),
        "smoothness": config.w_smooth * (winner.smoothness - mean_bd["smoothness"]),
        "centerline": config.w_centerline * (winner.centerline - mean_bd["centerline"]),
        # Negated because penalties are subtracted in quality_score
        "rb_near": -config.rb_near_scale * (winner.rb_near_penalty - mean_bd["rb_near_penalty"]),
        "rb_wide": -config.rb_wide_scale * (winner.rb_wide_penalty - mean_bd["rb_wide_penalty"]),
    }
    best_comp = max(deltas, key=lambda c: deltas[c])
    return best_comp, deltas[best_comp]


# ---------------------------------------------------------------------------
# Per-scene record
# ---------------------------------------------------------------------------

@dataclass
class SceneRankRecord:
    scene_path: str
    winner_idx: int
    winner_label: str
    winner_reward: float
    mean_reward: float
    det_reward: float
    winner_breakdown: dict[str, float]
    mean_breakdown: dict[str, float]
    dominant_component: str
    dominant_delta: float

    def to_dict(self) -> dict:
        return {
            "scene": self.scene_path,
            "winner_idx": self.winner_idx,
            "winner_label": self.winner_label,
            "winner_reward": round(self.winner_reward, 3),
            "mean_reward": round(self.mean_reward, 3),
            "det_reward": round(self.det_reward, 3),
            "dominant_component": self.dominant_component,
            "dominant_delta": round(self.dominant_delta, 3),
            "winner_breakdown": {k: round(v, 4) for k, v in self.winner_breakdown.items()},
            "mean_breakdown": {k: round(v, 4) for k, v in self.mean_breakdown.items()},
        }


# ---------------------------------------------------------------------------
# Epoch-level analytics
# ---------------------------------------------------------------------------

@dataclass
class EpochRankAnalytics:
    epoch: int
    n_scenes: int
    records: list[SceneRankRecord] = field(default_factory=list)
    # Derived (populated by finalize)
    win_counts: dict[str, int] = field(default_factory=dict)
    win_rates: dict[str, float] = field(default_factory=dict)
    category_rates: dict[str, float] = field(default_factory=dict)
    never_won: list[str] = field(default_factory=list)
    dominant_component_counts: dict[str, int] = field(default_factory=dict)
    mean_winner_reward: float = 0.0
    mean_det_reward: float = 0.0
    mean_improvement: float = 0.0

    def finalize(self, all_labels: list[str]) -> None:
        """Compute derived fields from records."""
        n = len(self.records)
        if n == 0:
            return

        # Win counts per label
        label_counter = Counter(r.winner_label for r in self.records)
        self.win_counts = {lbl: label_counter.get(lbl, 0) for lbl in all_labels}
        self.win_rates = {lbl: cnt / n for lbl, cnt in self.win_counts.items()}

        # Category rates
        cat_counter = Counter(get_category(r.winner_label) for r in self.records)
        # Include both standard and experimental categories
        all_cats = ["det_pure", "guided_det", "guided_noisy", "random",
                    "lateral_exp", "stretched_exp", "decoupled_exp", "noise_only_exp", "collision_exp"]
        for cat in all_cats:
            self.category_rates[cat] = cat_counter.get(cat, 0) / n

        # Never won
        self.never_won = [lbl for lbl, cnt in self.win_counts.items() if cnt == 0]

        # Dominant component counts
        self.dominant_component_counts = dict(Counter(r.dominant_component for r in self.records))

        # Reward stats
        self.mean_winner_reward = sum(r.winner_reward for r in self.records) / n
        self.mean_det_reward = sum(r.det_reward for r in self.records) / n
        self.mean_improvement = self.mean_winner_reward - self.mean_det_reward

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "n_scenes": self.n_scenes,
            "summary": {
                "win_counts": self.win_counts,
                "win_rates": {k: round(v, 4) for k, v in self.win_rates.items()},
                "category_rates": {k: round(v, 4) for k, v in self.category_rates.items()},
                "never_won": self.never_won,
                "dominant_components": self.dominant_component_counts,
                "mean_winner_reward": round(self.mean_winner_reward, 3),
                "mean_det_reward": round(self.mean_det_reward, 3),
                "mean_improvement": round(self.mean_improvement, 3),
            },
            "per_scene": [r.to_dict() for r in self.records],
        }


# ---------------------------------------------------------------------------
# Print / save
# ---------------------------------------------------------------------------

def print_epoch_summary(analytics: EpochRankAnalytics) -> None:
    """Print formatted rank analytics summary to stdout."""
    n = len(analytics.records)
    print(f"\n  === Rank Analytics (Epoch {analytics.epoch}) ===")
    print(f"  Config Win Rates ({n} scenes):")
    # Sort by win rate descending
    sorted_labels = sorted(analytics.win_rates.items(), key=lambda x: -x[1])
    for lbl, rate in sorted_labels:
        cnt = analytics.win_counts[lbl]
        if cnt > 0:
            print(f"    {lbl:<22s}: {rate*100:5.1f}% ({cnt} wins)")
    if analytics.never_won:
        print(f"  Never won: {', '.join(analytics.never_won)}")

    # Category rates
    cats = analytics.category_rates
    parts = [f"{cat} {rate*100:.0f}%" for cat, rate in sorted(cats.items(), key=lambda x: -x[1])]
    print(f"  Category Rates: {' | '.join(parts)}")

    # Dominant components
    dom = analytics.dominant_component_counts
    total_dom = sum(dom.values()) or 1
    dom_parts = [f"{comp} {cnt/total_dom*100:.0f}%" for comp, cnt in sorted(dom.items(), key=lambda x: -x[1])]
    print(f"  Dominant Components: {' | '.join(dom_parts)}")

    print(f"  Mean improvement over det: {analytics.mean_improvement:+.2f}")
    print()


def save_epoch_analytics(analytics: EpochRankAnalytics, run_dir: Path, epoch: int) -> None:
    """Save per-epoch analytics to JSON."""
    path = run_dir / f"rank_analytics_epoch_{epoch:03d}.json"
    with open(path, "w") as f:
        json.dump(analytics.to_dict(), f, indent=2)
    print(f"  Saved rank analytics to {path.name}")


# ---------------------------------------------------------------------------
# Cross-epoch summary (called once at end of training)
# ---------------------------------------------------------------------------

def save_cross_epoch_summary(run_dir: Path) -> None:
    """Read all per-epoch JSONs and produce a cross-epoch summary."""
    run_dir = Path(run_dir)
    epoch_files = sorted(run_dir.glob("rank_analytics_epoch_*.json"))
    if not epoch_files:
        return

    epochs_data = []
    for f in epoch_files:
        with open(f) as fh:
            epochs_data.append(json.load(fh))

    # Inter-epoch trends: category rates per epoch
    category_trends: dict[int, dict[str, float]] = {}
    config_trends: dict[int, dict[str, float]] = {}
    for ed in epochs_data:
        ep = ed["epoch"]
        category_trends[ep] = ed["summary"]["category_rates"]
        config_trends[ep] = ed["summary"]["win_rates"]

    # Per-scene evolution: for each scene, winner label at each epoch
    scene_evolution: dict[str, list[dict]] = {}
    for ed in epochs_data:
        ep = ed["epoch"]
        for rec in ed["per_scene"]:
            scene = rec["scene"]
            if scene not in scene_evolution:
                scene_evolution[scene] = []
            scene_evolution[scene].append({
                "epoch": ep,
                "winner_label": rec["winner_label"],
                "category": get_category(rec["winner_label"]),
                "winner_reward": rec["winner_reward"],
            })

    # Redundancy report: configs winning <5% across ALL epochs
    all_config_rates: dict[str, list[float]] = {}
    for ep_rates in config_trends.values():
        for lbl, rate in ep_rates.items():
            all_config_rates.setdefault(lbl, []).append(rate)

    redundant = []
    for lbl, rates in all_config_rates.items():
        if max(rates) < 0.05:
            redundant.append(lbl)

    summary = {
        "n_epochs": len(epochs_data),
        "category_trends": {str(k): v for k, v in category_trends.items()},
        "config_trends": {str(k): v for k, v in config_trends.items()},
        "scene_evolution": scene_evolution,
        "redundant_configs": redundant,
    }

    path = run_dir / "rank_analytics_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  === Cross-Epoch Rank Analytics Summary ===")
    print(f"  Saved to {path.name}")
    if redundant:
        print(f"  Redundant configs (<5% win rate across all epochs): {', '.join(redundant)}")
    else:
        print(f"  No configs were redundant (all had >5% win rate in at least one epoch)")

    # Print category trend table
    print(f"\n  Category rates by epoch:")
    print(f"  {'Epoch':>6s}  {'det_pure':>9s}  {'guided_det':>10s}  {'guided_noisy':>12s}  {'random':>7s}")
    for ep in sorted(category_trends.keys()):
        cats = category_trends[ep]
        print(f"  {ep:>6d}  {cats.get('det_pure',0)*100:>8.1f}%  {cats.get('guided_det',0)*100:>9.1f}%  {cats.get('guided_noisy',0)*100:>11.1f}%  {cats.get('random',0)*100:>6.1f}%")
    print()
