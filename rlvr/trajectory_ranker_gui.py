"""Trajectory Ranker GUI -- per-trajectory guidance tuning, reward debugging, and optional GRPO training.

Generates N trajectories per scene matching the batched GRPO sampler strategy,
scores them with the rule-based reward, and lets you edit each trajectory's
guidance config individually (noise, guidance types, scales) with single-traj
regeneration.

Launch
------
source .venv/bin/activate

# Visualization + per-trajectory tuning (default):
python rlvr/trajectory_ranker_gui.py \
  --model_path /path/to/model.pth \
  --npz_list   /path/to/train_or_valid.json \
  --no-training

# With GRPO training controls:
python rlvr/trajectory_ranker_gui.py \
  --model_path /path/to/model.pth \
  --npz_list   /path/to/train_or_valid.json \
  --use_lora
"""

from __future__ import annotations

import argparse
import functools
import json
import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusion_planner.model.guidance.composer import GuidanceComposer
from diffusion_planner.model.guidance.config import GuidanceConfig, GuidanceSetConfig
from diffusion_planner.utils.visualize_input import visualize_inputs
from matplotlib.figure import Figure

from guidance_gui.generate_samples import generate_samples
from guidance_gui.visualization import (
    _calculate_curvature,
    _calculate_velocities,
    _gt_curvature,
    _gt_velocities,
)
from preference_optimization.model_utils import load_model
from preference_optimization.utils import load_npz_data
from rlvr.grpo_sampler import SampledTrajectory
from rlvr.reward import (
    RewardBreakdown,
    RewardConfig,
    compute_group_advantages,
    compute_reward_batch,
)

_DIVERGING_CMAP = plt.get_cmap("RdYlGn")

_DEFAULT_PROTOTYPES_PATH = str(Path(__file__).parent / "prototypes_k16.npy")
_GENERATE_SCRIPT = Path(__file__).parent.parent / "guidance_gui" / "scripts" / "generate_prototypes.py"

ALL_GUIDANCE_NAMES = [
    "centerline_following",
    "speed",
    "lane_keeping",
    "road_border",
    "route_following",
    "collision",
    "anchor_following",
    "lateral",
    "longitudinal",
]

# Short display names for labels
_SHORT_NAMES = {
    "centerline_following": "cl",
    "speed": "spd",
    "lane_keeping": "lk",
    "road_border": "rb",
    "route_following": "rf",
    "collision": "col",
    "anchor_following": "anc",
    "lateral": "lat",
    "longitudinal": "lon",
}


# ---------------------------------------------------------------------------
# Per-trajectory config data model
# ---------------------------------------------------------------------------

@dataclass
class TrajectorySlotConfig:
    """Editable config for one trajectory slot."""
    noise_scale: float = 0.0
    global_guidance_scale: float = 1.0
    is_deterministic: bool = False
    # name -> (enabled, scale, params_dict)
    guidance: dict[str, tuple[bool, float, dict]] = field(default_factory=dict)

    def __post_init__(self):
        for name in ALL_GUIDANCE_NAMES:
            if name not in self.guidance:
                self.guidance[name] = (False, 1.0, {})


def _format_label(slot: TrajectorySlotConfig) -> str:
    if slot.is_deterministic and not any(en for en, _, _ in slot.guidance.values()):
        return "DET"
    parts = []
    if slot.is_deterministic:
        parts.append("det")
    elif slot.noise_scale > 0:
        parts.append(f"ns={slot.noise_scale:.1f}")
    for name in ALL_GUIDANCE_NAMES:
        enabled, scale, _ = slot.guidance.get(name, (False, 1.0, {}))
        if enabled:
            parts.append(f"{_SHORT_NAMES[name]}={scale:.0f}")
    return "+".join(parts) if parts else "none"


def _format_dropdown_choices(
    slot_configs: list[TrajectorySlotConfig],
    reward_breakdowns: list[RewardBreakdown],
) -> list[str]:
    choices = []
    for i, slot in enumerate(slot_configs):
        label = _format_label(slot)
        reward_str = ""
        if i < len(reward_breakdowns):
            reward_str = f" R={reward_breakdowns[i].total:.1f}"
        choices.append(f"#{i + 1} {label}{reward_str}")
    return choices


def slot_config_to_composer(
    slot: TrajectorySlotConfig,
    gt_max_speed: float | None,
    gt_min_speed: float,
    prototypes_path: str | None = None,
) -> tuple[GuidanceComposer | None, GuidanceSetConfig | None]:
    fns = []
    for name in ALL_GUIDANCE_NAMES:
        enabled, scale, params = slot.guidance.get(name, (False, 1.0, {}))
        if not enabled:
            continue
        p = dict(params)
        if name == "speed" and gt_max_speed is not None:
            p.setdefault("v_high", gt_max_speed)
            p.setdefault("v_low", gt_min_speed)
        if name == "anchor_following":
            if prototypes_path:
                p.setdefault("prototypes_path", prototypes_path)
            p.setdefault("anchor_index", 0)
        fns.append(GuidanceConfig(name=name, enabled=True, scale=scale, params=p))
    if not fns:
        return None, None
    set_cfg = GuidanceSetConfig(functions=fns, global_scale=slot.global_guidance_scale)
    return GuidanceComposer(set_cfg), set_cfg


def generate_batched_sampler_configs(
    gt_max_speed: float | None,
    gt_min_speed: float,
    k: int = 16,
) -> list[TrajectorySlotConfig]:
    """Generate K trajectory configs matching the batched GRPO sampler strategy."""
    configs: list[TrajectorySlotConfig] = []

    # Slot 0: deterministic, no guidance
    configs.append(TrajectorySlotConfig(is_deterministic=True))

    # Slots 1-4: CL+SPD guided, deterministic
    cl_spd_det = [
        (5.0, 5.0),
        (8.0, 5.0),
        (10.0, 8.0),
        (10.0, 10.0),
    ]
    for cl_s, spd_s in cl_spd_det:
        slot = TrajectorySlotConfig(is_deterministic=True, global_guidance_scale=1.0)
        slot.guidance["centerline_following"] = (True, cl_s, {})
        if gt_max_speed is not None:
            slot.guidance["speed"] = (True, spd_s, {"v_high": gt_max_speed, "v_low": gt_min_speed})
        configs.append(slot)

    # Slots 5-8: CL+SPD guided, with noise
    cl_spd_noisy = [
        (5.0, 5.0, 0.3, 0.8),
        (8.0, 8.0, 0.3, 0.8),
        (10.0, 8.0, 0.3, 0.8),
        (10.0, 10.0, 0.5, 1.0),
    ]
    for cl_s, spd_s, n_lo, n_hi in cl_spd_noisy:
        ns = round(random.uniform(n_lo, n_hi), 2)
        slot = TrajectorySlotConfig(noise_scale=ns, global_guidance_scale=1.0)
        slot.guidance["centerline_following"] = (True, cl_s, {})
        if gt_max_speed is not None:
            slot.guidance["speed"] = (True, spd_s, {"v_high": gt_max_speed, "v_low": gt_min_speed})
        configs.append(slot)

    # Slots 9-12: random CL + optional RB
    n_rand1 = max(0, min(4, k - len(configs) - 3))
    for _ in range(n_rand1):
        ns = round(random.uniform(0.5, 2.0), 2)
        gs = round(random.uniform(0.3, 1.5), 2)
        cl_s = round(random.uniform(2.0, 8.0), 1)
        slot = TrajectorySlotConfig(noise_scale=ns, global_guidance_scale=gs)
        slot.guidance["centerline_following"] = (True, cl_s, {})
        if random.random() < 0.5:
            rb_s = round(random.uniform(0.3, 1.5), 1)
            slot.guidance["road_border"] = (True, rb_s, {})
        configs.append(slot)

    # Slots 13-15: noise-only or light CL
    n_rand2 = k - len(configs)
    for _ in range(n_rand2):
        ns = round(random.uniform(0.5, 2.0), 2)
        slot = TrajectorySlotConfig(noise_scale=ns)
        if random.random() < 0.5:
            cl_s = round(random.uniform(1.0, 5.0), 1)
            gs = round(random.uniform(0.2, 1.0), 2)
            slot.guidance["centerline_following"] = (True, cl_s, {})
            slot.global_guidance_scale = gs
        configs.append(slot)

    return configs[:k]


# ---------------------------------------------------------------------------
# Prototypes helper (unchanged)
# ---------------------------------------------------------------------------

def ensure_prototypes(npz_list_path: str, prototypes_path: str, force: bool = False) -> str | None:
    if not force and Path(prototypes_path).exists():
        print(f"Using existing prototypes: {prototypes_path}")
        return prototypes_path
    if not _GENERATE_SCRIPT.exists():
        print(f"Warning: generate_prototypes.py not found at {_GENERATE_SCRIPT}")
        return prototypes_path
    print(f"Generating prototypes from {npz_list_path} -> {prototypes_path} ...")
    try:
        subprocess.run(
            [
                sys.executable, str(_GENERATE_SCRIPT),
                "--npz_list", npz_list_path,
                "--output", prototypes_path,
                "--k", "16",
                "--max_samples", "50000",
            ],
            check=True, timeout=600,
        )
        print(f"Prototypes saved to {prototypes_path}")
    except subprocess.CalledProcessError as e:
        print(f"Warning: prototype generation failed: {e}")
    except subprocess.TimeoutExpired:
        print("Warning: prototype generation timed out")
    if not Path(prototypes_path).exists():
        print(f"Warning: prototypes file not created at {prototypes_path}")
        return None
    return prototypes_path


# ---------------------------------------------------------------------------
# TrajectoryRanker
# ---------------------------------------------------------------------------

class TrajectoryRanker:
    """Generates and scores N diverse trajectories per scene with per-trajectory control."""

    def __init__(
        self,
        policy_model,
        model_args,
        npz_paths: list[str],
        npz_list_path: str,
        prototypes_path: str,
        n_trajectories: int = 16,
    ):
        self.policy_model = policy_model
        self.model_args = model_args
        self.npz_paths = npz_paths
        self.npz_list_path = npz_list_path
        self.prototypes_path = prototypes_path
        self.n_trajectories = n_trajectories
        self.current_index = 0
        self.device = next(policy_model.parameters()).device

        self.current_data: dict[str, torch.Tensor] | None = None
        self._norm_data: dict[str, torch.Tensor] | None = None
        self._gt_max_speed: float | None = None
        self._gt_min_speed: float = 0.0
        self._ref_trajectory: torch.Tensor | None = None  # [1, T, 4] from det traj

        self.slot_configs: list[TrajectorySlotConfig] = []
        self.sampled_trajectories: list[SampledTrajectory] = []
        self.reward_breakdowns: list[RewardBreakdown] = []
        self.advantages: np.ndarray = np.array([])
        self.saved_scenes: list[dict] = []

        self.reward_config = RewardConfig()
        self.accepted_groups: list[dict] = []

    def _compute_gt_speed_bounds(self) -> None:
        self._gt_max_speed = None
        self._gt_min_speed = 0.0
        if self.current_data is None or "ego_agent_future" not in self.current_data:
            return
        gt = self.current_data["ego_agent_future"]
        if gt.dim() == 3:
            gt = gt[0]
        gt_np = gt.cpu().numpy()
        valid = ~((gt_np[:, 0] == 0) & (gt_np[:, 1] == 0))
        if valid.sum() >= 10:
            vel = np.diff(gt_np[valid][:, :2], axis=0) / 0.1
            speeds = np.linalg.norm(vel, axis=-1)
            self._gt_max_speed = float(speeds.max())
            self._gt_min_speed = float(np.percentile(speeds, 10))

    def load_sample(self) -> None:
        if not self.npz_paths or self.current_index >= len(self.npz_paths):
            return

        self.current_data = load_npz_data(
            self.npz_paths[self.current_index], self.device
        )
        self._compute_gt_speed_bounds()

        self._norm_data = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in self.current_data.items()
        }
        self._norm_data = self.model_args.observation_normalizer(self._norm_data)

        self.slot_configs = generate_batched_sampler_configs(
            self._gt_max_speed, self._gt_min_speed, k=self.n_trajectories,
        )

        self.policy_model.eval()
        self.sampled_trajectories = []
        self._ref_trajectory = None
        for i, slot in enumerate(self.slot_configs):
            traj = self._generate_from_slot(slot)
            # Cache the first (deterministic) trajectory as reference for lat/lon guidance
            if i == 0 and slot.is_deterministic:
                self._ref_trajectory = torch.from_numpy(traj).unsqueeze(0).to(self.device)
                self._norm_data["reference_trajectory"] = self._ref_trajectory
            self.sampled_trajectories.append(SampledTrajectory(
                trajectory=traj,
                noise_scale=0.0 if slot.is_deterministic else slot.noise_scale,
                guidance_config=None,
                is_deterministic=slot.is_deterministic,
                label=_format_label(slot),
            ))

        self._score_trajectories()

    def _generate_from_slot(self, slot: TrajectorySlotConfig) -> np.ndarray:
        composer, set_cfg = slot_config_to_composer(
            slot, self._gt_max_speed, self._gt_min_speed, self.prototypes_path,
        )
        noise = 0.0 if slot.is_deterministic else slot.noise_scale
        samples = generate_samples(
            model=self.policy_model,
            model_args=self.model_args,
            data=self._norm_data,
            noise_scale=noise,
            n_samples=1,
            composer=composer,
            device=self.device,
        )
        return samples[0]

    def regenerate_single(self, index: int) -> None:
        if (
            self._norm_data is None
            or index >= len(self.slot_configs)
            or index >= len(self.sampled_trajectories)
        ):
            return
        slot = self.slot_configs[index]
        # Ensure reference_trajectory is in norm_data for lat/lon guidance
        if self._ref_trajectory is not None:
            self._norm_data["reference_trajectory"] = self._ref_trajectory
        self.policy_model.eval()
        traj = self._generate_from_slot(slot)
        if index == 0 and slot.is_deterministic:
            self._ref_trajectory = torch.from_numpy(traj).unsqueeze(0).to(self.device)
            self._norm_data["reference_trajectory"] = self._ref_trajectory
        self.sampled_trajectories[index] = SampledTrajectory(
            trajectory=traj,
            noise_scale=0.0 if slot.is_deterministic else slot.noise_scale,
            guidance_config=None,
            is_deterministic=slot.is_deterministic,
            label=_format_label(slot),
        )
        self._score_trajectories()

    def _score_trajectories(self) -> None:
        if not self.sampled_trajectories or self.current_data is None:
            return
        traj_batch = torch.tensor(
            np.stack([st.trajectory for st in self.sampled_trajectories]),
            device=self.device, dtype=torch.float32,
        )
        self.reward_breakdowns = compute_reward_batch(
            traj_batch, self.current_data, self.reward_config,
        )
        self.advantages = compute_group_advantages(self.reward_breakdowns)

    def _create_trajectory_plot(
        self, time_step: int = 40, view_range: float = 60.0, selected_idx: int = 0,
    ) -> Figure:
        fig = Figure(figsize=(10, 11.5))
        ax = fig.add_subplot(111)

        if self.current_data is None:
            return fig

        data_cpu = {k: v.cpu() for k, v in self.current_data.items()}
        visualize_inputs(data_cpu, save_path=None, ax=ax, view_ranges=[120])

        if not self.sampled_trajectories:
            return fig

        N = len(self.sampled_trajectories)
        advantages = self.advantages

        rank_order = np.argsort(advantages)
        ranks = np.empty(N, dtype=int)
        for rank_pos, idx in enumerate(rank_order):
            ranks[idx] = rank_pos

        for i, st in enumerate(self.sampled_trajectories):
            traj = st.trajectory
            rank_frac = ranks[i] / max(N - 1, 1)
            display_rank = N - ranks[i]
            is_selected = (i == selected_idx)

            has_guidance = (
                any(en for en, _, _ in self.slot_configs[i].guidance.values())
                if i < len(self.slot_configs) else False
            )
            is_det_baseline = st.is_deterministic and not has_guidance
            if is_det_baseline:
                color = "dodgerblue"
                lw = 3.0
                alpha = 1.0
                linestyle = "--"
                prefix = ">> " if is_selected else ""
                label = f"{prefix}#{i+1} R={self.reward_breakdowns[i].total:.1f} [DET]"
            else:
                color = _DIVERGING_CMAP(rank_frac)
                lw = 1.0 + 2.5 * rank_frac
                alpha = 0.3 + 0.7 * rank_frac
                linestyle = "-"
                prefix = ">> " if is_selected else ""
                label = f"{prefix}#{i+1} R={self.reward_breakdowns[i].total:.1f} ({st.label})"

            # Selected trajectory: white outline underneath
            if is_selected:
                ax.plot(
                    traj[:, 0], traj[:, 1],
                    color="white", linewidth=lw + 4, alpha=0.9,
                    linestyle=linestyle, zorder=8,
                )
                ax.plot(
                    traj[:, 0], traj[:, 1],
                    color="black", linewidth=lw + 2, alpha=0.7,
                    linestyle=linestyle, zorder=9,
                )

            ax.plot(
                traj[:, 0], traj[:, 1],
                color=color, linewidth=lw, alpha=alpha,
                linestyle=linestyle, label=label,
                zorder=10 if is_selected else 5,
            )

            # Selected trajectory: star marker at time_step
            if is_selected and 0 <= time_step < len(traj):
                ax.scatter(
                    [traj[time_step, 0]], [traj[time_step, 1]],
                    color=color, s=200, zorder=15,
                    edgecolors="black", linewidths=2, marker="*",
                )
            elif ranks[i] >= N - 3 and 0 <= time_step < len(traj):
                ax.scatter(
                    [traj[time_step, 0]], [traj[time_step, 1]],
                    color=color, s=80, zorder=10,
                    edgecolors="black", marker="D",
                )

            rb = self.reward_breakdowns[i]
            if rb.collision_step is not None and 0 <= rb.collision_step < len(traj):
                ct = rb.collision_step
                ax.scatter(
                    [traj[ct, 0]], [traj[ct, 1]],
                    color="red", s=100, zorder=12,
                    edgecolors="darkred", linewidths=1.5, marker="X",
                )

        if "ego_agent_future" in data_cpu:
            gt = data_cpu["ego_agent_future"]
            if hasattr(gt, "numpy"):
                gt = gt.numpy()
            gt = np.array(gt).reshape(-1, 3)
            valid = ~((gt[:, 0] == 0) & (gt[:, 1] == 0))
            if np.any(valid):
                ax.plot(
                    gt[valid, 0], gt[valid, 1],
                    "k--", linewidth=2, alpha=0.6, label="GT",
                )

        ax.legend(loc="upper left", fontsize=5.5, ncol=2)
        ax.set_title(f"Scene {self.current_index + 1} / {len(self.npz_paths)}")

        ref = self.sampled_trajectories[0].trajectory
        cx = (ref[0, 0] + ref[-1, 0]) / 2
        cy = (ref[0, 1] + ref[-1, 1]) / 2
        half = view_range / 2
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal")
        return fig

    def _create_speed_curvature_plot(self, selected_idx: int = 0) -> Figure:
        fig = Figure(figsize=(8, 6))
        ax_speed = fig.add_subplot(211)
        ax_curv = fig.add_subplot(212)

        if not self.sampled_trajectories or self.current_data is None:
            return fig

        data_cpu = {
            k: v.cpu().numpy() if hasattr(v, "cpu") else v
            for k, v in self.current_data.items()
        }
        ego_state = np.array(data_cpu["ego_current_state"]).reshape(-1)

        N = len(self.sampled_trajectories)
        rank_order = np.argsort(self.advantages)
        top3_indices = set(rank_order[-min(3, N):][::-1])

        # Always include selected trajectory
        plot_indices = [selected_idx] if selected_idx not in top3_indices else []
        plot_indices += [i for i in rank_order[-min(3, N):][::-1]]

        all_speeds = []
        all_curvs = []

        for plot_i, idx in enumerate(plot_indices):
            st = self.sampled_trajectories[idx]
            traj = st.trajectory
            is_sel = (idx == selected_idx)

            if is_sel:
                color = "blue"
                lw = 2.5
            else:
                rank_frac = (N - 1 - plot_i) / max(N - 1, 1)
                color = _DIVERGING_CMAP(rank_frac)
                lw = 1.8

            vel = _calculate_velocities(traj, ego_state) / 3.6  # km/h -> m/s
            curv = _calculate_curvature(traj, ego_state)
            all_speeds.append(vel)
            all_curvs.append(curv)
            t = np.arange(len(vel))
            sel_tag = " [SEL]" if is_sel else ""
            ax_speed.plot(
                t, vel, color=color, linewidth=lw, alpha=0.8,
                label=f"#{idx+1} {st.label}{sel_tag}",
            )
            ax_curv.plot(
                np.arange(len(curv)), curv,
                color=color, linewidth=lw, alpha=0.8,
            )

        if "ego_agent_future" in data_cpu:
            ego_future = np.array(data_cpu["ego_agent_future"]).reshape(-1, 3)
            gt_vel = _gt_velocities(ego_future, ego_state)
            gt_curv = _gt_curvature(ego_future, ego_state)
            if gt_vel is not None:
                gt_vel_ms = gt_vel / 3.6  # km/h -> m/s
                all_speeds.append(gt_vel_ms)
                ax_speed.plot(
                    np.arange(len(gt_vel_ms)), gt_vel_ms,
                    "k--", linewidth=2, alpha=0.7, label="GT",
                )
            if gt_curv is not None:
                all_curvs.append(gt_curv)
                ax_curv.plot(
                    np.arange(len(gt_curv)), gt_curv,
                    "k--", linewidth=2, alpha=0.7,
                )

        # Auto-scale y-axes with 10% padding
        if all_speeds:
            all_v = np.concatenate(all_speeds)
            v_min, v_max = float(all_v.min()), float(all_v.max())
            v_pad = max(0.5, (v_max - v_min) * 0.1)
            ax_speed.set_ylim(max(0, v_min - v_pad), v_max + v_pad)
        ax_speed.set_ylabel("Speed (m/s)")
        ax_speed.set_title("Speed (top-3 + selected)")
        ax_speed.legend(loc="upper right", fontsize=7)
        ax_speed.grid(True, alpha=0.3)

        if all_curvs:
            all_c = np.concatenate(all_curvs)
            c_min, c_max = float(all_c.min()), float(all_c.max())
            c_pad = max(0.01, (c_max - c_min) * 0.1)
            ax_curv.set_ylim(c_min - c_pad, c_max + c_pad)
        ax_curv.set_ylabel("Curvature (1/m)")
        ax_curv.set_xlabel("Time step")
        ax_curv.set_title("Curvature (top-3 + selected)")
        ax_curv.grid(True, alpha=0.3)
        ax_curv.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

        fig.tight_layout()
        return fig

    def _format_reward_table(self, selected_idx: int = 0) -> str:
        if not self.reward_breakdowns:
            return ""

        rows = list(zip(
            range(len(self.reward_breakdowns)),
            self.reward_breakdowns,
            self.advantages,
            self.sampled_trajectories,
        ))
        rows.sort(key=lambda r: r[1].total, reverse=True)

        cfg = self.reward_config
        # UI design: the user needs to see (a) what the reward weighting
        # adds up to per component, and (b) which hard gates fired / how
        # close we are to firing them. Cramming 20 columns into markdown
        # is unreadable, so:
        #   * Keep the component-breakdown columns (Safety..CL) compact.
        #   * Collapse all gate-fire flags into ONE "Gates" column that
        #     is empty when everything passes and otherwise shows just
        #     the firing tag(s): RB✗, Lane✗, Coll@t, Kin✗, SC✗.
        #   * Add rb_d (road-border) and sc_d (stopped-neighbor) as two
        #     dedicated numeric columns; hide values >= 10 m as "—" so
        #     irrelevant scenes don't pollute the view.
        # Result: 11 columns instead of 21.
        def _gates_tag(rb) -> str:
            tags = []
            if getattr(rb, "rb_crossing", False):
                tags.append("RB✗")
            if getattr(rb, "lane_crossing", False):
                tags.append("Lane✗")
            cs = getattr(rb, "collision_step", None)
            if cs is not None:
                tags.append(f"Coll@{int(cs)}")
            if not getattr(rb, "kinematic_gate", True):
                tags.append("Kin✗")
            if getattr(rb, "static_crossing", False):
                tags.append("SC✗")
            return " ".join(tags) if tags else "—"

        def _fmt_d(d: float, cutoff: float = 10.0) -> str:
            return f"{d:.2f}" if d < cutoff else "—"

        lines = [
            "| # | Rank | Saf | Prog | Smo | Feas | CL | Total | Adv | Gates | rb_d | sc_d | Config |",
            "|---|------|-----|------|-----|------|----|-------|-----|-------|------|------|--------|",
        ]
        for rank, (idx, rb, adv, st) in enumerate(rows, 1):
            is_sel = (idx == selected_idx)
            config_col = "**[DET]**" if st.is_deterministic and st.label == "DET" else st.label
            b = "**" if is_sel else ""
            ws = cfg.w_safety * rb.safety
            wp = cfg.w_progress * rb.progress
            wm = cfg.w_smooth * rb.smoothness
            wf = cfg.w_feasibility * rb.feasibility
            wc = cfg.w_centerline * rb.centerline
            gates = _gates_tag(rb)
            rb_d = _fmt_d(getattr(rb, "rb_min_dist", 99.0))
            sc_d = _fmt_d(getattr(rb, "sc_min_dist", 99.0))
            sel_marker = ">>" if is_sel else ""
            lines.append(
                f"| {b}{sel_marker}#{idx+1}{b} | {b}{rank}{b} | "
                f"{b}{ws:.1f}{b} | {b}{wp:.1f}{b} | {b}{wm:.1f}{b} | "
                f"{b}{wf:.1f}{b} | {b}{wc:.1f}{b} | "
                f"{b}{rb.total:.1f}{b} | {b}{adv:+.2f}{b} | "
                f"{b}{gates}{b} | {b}{rb_d}{b} | {b}{sc_d}{b} | {config_col} |"
            )
        return "\n".join(lines)

    def save_current_scene(self, save_dir: str, zoom: int = 5, time_step: int = 40) -> str:
        if not self.sampled_trajectories or self.current_data is None:
            return "Nothing to save"

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        save_idx = len(self.saved_scenes)
        view_range = 100 - (int(zoom) - 1) * 90 / 9
        fig = self._create_trajectory_plot(time_step=int(time_step), view_range=view_range)
        img_path = save_path / f"scene_{save_idx}_idx{self.current_index}.png"
        fig.savefig(str(img_path), dpi=150, bbox_inches="tight")

        scene_data = {
            "scene_index": self.current_index,
            "npz_path": self.npz_paths[self.current_index],
            "image": str(img_path),
            "trajectories": np.stack([st.trajectory for st in self.sampled_trajectories]),
            "labels": [st.label for st in self.sampled_trajectories],
            "noise_scales": [st.noise_scale for st in self.sampled_trajectories],
            "is_deterministic": [st.is_deterministic for st in self.sampled_trajectories],
            "slot_configs": [
                {
                    "noise_scale": s.noise_scale,
                    "global_guidance_scale": s.global_guidance_scale,
                    "is_deterministic": s.is_deterministic,
                    "guidance": {
                        name: {"enabled": en, "scale": sc}
                        for name, (en, sc, _) in s.guidance.items() if en
                    },
                }
                for s in self.slot_configs
            ],
            "rewards": {
                "safety": [rb.safety for rb in self.reward_breakdowns],
                "progress": [rb.progress for rb in self.reward_breakdowns],
                "smoothness": [rb.smoothness for rb in self.reward_breakdowns],
                "feasibility": [rb.feasibility for rb in self.reward_breakdowns],
                "centerline": [rb.centerline for rb in self.reward_breakdowns],
                "red_light": [rb.red_light for rb in self.reward_breakdowns],
                "total": [rb.total for rb in self.reward_breakdowns],
                "collision_step": [rb.collision_step for rb in self.reward_breakdowns],
                "off_road_fraction": [rb.off_road_fraction for rb in self.reward_breakdowns],
            },
            "advantages": self.advantages.tolist(),
        }
        self.saved_scenes.append(scene_data)

        dump_path = save_path / "ranker_dump.json"
        serializable = []
        for sc in self.saved_scenes:
            s = dict(sc)
            s["trajectories"] = sc["trajectories"].tolist()
            serializable.append(s)
        with open(dump_path, "w") as f:
            json.dump(serializable, f, indent=2)

        return f"Saved scene {self.current_index} ({len(self.saved_scenes)} total) -> {img_path.name}"

    def accept_current_group(self) -> str:
        if not self.sampled_trajectories or self.current_data is None:
            return "Nothing to accept (no trajectories loaded)"
        if np.all(self.advantages == 0):
            return "Skipped: all advantages are zero (no gradient signal)"
        self.accepted_groups.append({
            "npz_path": self.npz_paths[self.current_index],
            "data": self.current_data,
            "trajectories": [st.trajectory for st in self.sampled_trajectories],
            "reward_breakdowns": self.reward_breakdowns,
            "advantages": self.advantages,
        })
        return (
            f"Accepted scene {self.current_index} "
            f"({len(self.accepted_groups)} groups queued)"
        )

    def clear_accepted_groups(self) -> str:
        count = len(self.accepted_groups)
        self.accepted_groups.clear()
        return f"Cleared {count} groups"


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

def build_interface(
    ranker: TrajectoryRanker,
    trainer=None,
) -> gr.Blocks:
    training_enabled = trainer is not None
    title = "Trajectory Ranker + GRPO" if training_enabled else "Trajectory Ranker"

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")

        # Hidden state for selected trajectory index
        selected_state = gr.State(value=0)

        with gr.Row():
            # --- Left sidebar ---
            with gr.Column(scale=1):
                gr.Markdown("### Navigation")
                with gr.Row():
                    btn_m30 = gr.Button("<-30", size="sm")
                    btn_m10 = gr.Button("<-10", size="sm")
                    btn_m1 = gr.Button("<-1", size="sm")
                    btn_p1 = gr.Button("1->", size="sm")
                    btn_p10 = gr.Button("10->", size="sm")
                    btn_p30 = gr.Button("30->", size="sm")
                with gr.Row():
                    btn_shuffle = gr.Button("Shuffle", size="sm")
                    btn_regen_all = gr.Button("Re-do All", size="sm")
                jump_input = gr.Number(
                    label="Jump to index", value=0, minimum=0, precision=0,
                )
                n_traj_sl = gr.Slider(
                    4, 32, value=ranker.n_trajectories, step=1, label="N trajectories",
                )

                gr.Markdown("### Reward Weights")
                w_safety = gr.Slider(0.0, 20.0, value=5.0, step=0.5, label="w_safety")
                w_progress = gr.Slider(0.0, 10.0, value=2.0, step=0.1, label="w_progress")
                w_smooth = gr.Slider(0.0, 10.0, value=0.5, step=0.1, label="w_smooth")
                w_feasibility = gr.Slider(0.0, 10.0, value=5.0, step=0.1, label="w_feasibility")
                w_centerline = gr.Slider(0.0, 10.0, value=5.0, step=0.1, label="w_centerline")

                gr.Markdown("### Display")
                zoom_sl = gr.Slider(1, 10, value=5, step=1, label="Zoom")
                time_sl = gr.Slider(0, 79, value=40, step=1, label="Time step")

                gr.Markdown("### Prototypes")
                proto_path = gr.Textbox(
                    label="Prototypes path",
                    value=ranker.prototypes_path or "",
                )
                btn_regen_protos = gr.Button("Regen Protos", size="sm")

            # --- Center content ---
            with gr.Column(scale=2):
                traj_plot = gr.Plot(label="Trajectories")
                reward_table = gr.Markdown("")
                with gr.Row():
                    btn_save = gr.Button("Save Scene", size="sm")
                    _ts = __import__("datetime").datetime.now().strftime("%y-%m-%d-%H-%M-%S")
                    save_dir = gr.Textbox(
                        value=f".datasets/trajectory-dump-{_ts}",
                        label="Save directory", scale=3,
                    )
                    save_status = gr.Markdown("")
                with gr.Accordion("Speed & Curvature Plots", open=False):
                    speed_curv_plot = gr.Plot(label="Speed & Curvature")
                sample_info = gr.Markdown("Scene -- / --")

                # GRPO Training Controls
                if training_enabled:
                    with gr.Accordion("GRPO Training", open=True):
                        with gr.Row():
                            btn_accept = gr.Button("Accept Group", variant="primary", size="sm")
                            btn_skip = gr.Button("Skip", size="sm")
                            btn_clear_queue = gr.Button("Clear Queue", size="sm")
                        queue_status = gr.Markdown("0 groups queued")
                        gr.Markdown("### Training Parameters")
                        with gr.Row():
                            beta_sl = gr.Slider(0.0, 1.0, value=0.1, step=0.01, label="KL beta")
                            lr_sl = gr.Slider(1e-6, 1e-3, value=1e-5, step=1e-6, label="Learning rate")
                            accum_sl = gr.Slider(1, 16, value=4, step=1, label="Grad accum groups")
                        with gr.Row():
                            btn_train = gr.Button("Train on Queued Groups", variant="primary")
                            epoch_display = gr.Number(value=0, label="Current epoch", interactive=False)
                        train_log = gr.Markdown("No training yet.")

            # --- Right sidebar: per-trajectory editor ---
            with gr.Column(scale=1):
                gr.Markdown("### Trajectory Editor")
                traj_dropdown = gr.Dropdown(
                    choices=[f"#{i+1}" for i in range(ranker.n_trajectories)],
                    value="#1",
                    label="Select trajectory",
                    interactive=True,
                )
                config_summary = gr.Markdown("*Select a trajectory to edit*")

                gr.Markdown("---")
                noise_sl = gr.Slider(0.0, 2.0, value=0.0, step=0.05, label="Noise scale")
                global_g_sl = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="Global guidance scale")
                det_cb = gr.Checkbox(value=True, label="Deterministic (noise=0)")

                gr.Markdown("#### Guidance Functions")
                # 9 guidance types: each gets a checkbox + scale slider
                cb_cl = gr.Checkbox(value=False, label="Centerline following")
                sl_cl = gr.Slider(0.1, 15.0, value=5.0, step=0.1, label="CL scale")

                cb_spd = gr.Checkbox(value=False, label="Speed")
                sl_spd = gr.Slider(0.1, 15.0, value=5.0, step=0.1, label="SPD scale")
                spd_stretch = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="SPD stretch (1.0=keep, 1.3=30% faster)")

                cb_lk = gr.Checkbox(value=False, label="Lane keeping")
                sl_lk = gr.Slider(0.1, 15.0, value=5.0, step=0.1, label="LK scale")

                cb_rb = gr.Checkbox(value=False, label="Road border")
                sl_rb = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="RB scale")

                cb_rf = gr.Checkbox(value=False, label="Route following")
                sl_rf = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="RF scale")

                cb_col = gr.Checkbox(value=False, label="Collision")
                sl_col = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="COL scale")

                cb_anc = gr.Checkbox(value=False, label="Anchor following")
                sl_anc = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="ANC scale")

                cb_lat = gr.Checkbox(value=False, label="Lateral")
                sl_lat = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="LAT scale")
                eta_lat = gr.Slider(-1.0, 1.0, value=0.0, step=0.05, label="LAT eta (offset direction)")

                cb_lon = gr.Checkbox(value=False, label="Longitudinal")
                sl_lon = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="LON scale")
                eta_lon = gr.Slider(-1.0, 1.0, value=0.0, step=0.05, label="LON eta (speed direction)")

                gr.Markdown("---")
                btn_regen_single = gr.Button(
                    "Regenerate This Trajectory", variant="primary", size="sm",
                )

        # --- Component lists for callbacks ---
        editor_guidance_components = [
            cb_cl, sl_cl, cb_spd, sl_spd, spd_stretch, cb_lk, sl_lk,
            cb_rb, sl_rb, cb_rf, sl_rf, cb_col, sl_col,
            cb_anc, sl_anc, cb_lat, sl_lat, eta_lat, cb_lon, sl_lon, eta_lon,
        ]
        editor_components = [noise_sl, global_g_sl, det_cb] + editor_guidance_components
        reward_inputs = [w_safety, w_progress, w_smooth, w_feasibility, w_centerline]
        display_inputs = [zoom_sl, time_sl]
        main_outputs = [traj_plot, reward_table, speed_curv_plot, sample_info]

        # --- Helper: parse selected index from dropdown ---
        def _parse_dropdown(choice_str: str) -> int:
            try:
                return int(choice_str.split("#")[1].split(" ")[0]) - 1
            except (IndexError, ValueError):
                return 0

        # --- Helper: populate editor from slot config ---
        def _populate_editor(slot: TrajectorySlotConfig):
            g = slot.guidance
            lat_params = g.get("lateral", (False, 1.0, {}))[2]
            lon_params = g.get("longitudinal", (False, 1.0, {}))[2]
            return (
                slot.noise_scale,
                slot.global_guidance_scale,
                slot.is_deterministic,
                g.get("centerline_following", (False, 1.0, {}))[0],
                g.get("centerline_following", (False, 5.0, {}))[1],
                g.get("speed", (False, 1.0, {}))[0],
                g.get("speed", (False, 5.0, {}))[1],
                g.get("speed", (False, 1.0, {}))[2].get("stretch", 1.0),
                g.get("lane_keeping", (False, 1.0, {}))[0],
                g.get("lane_keeping", (False, 5.0, {}))[1],
                g.get("road_border", (False, 1.0, {}))[0],
                g.get("road_border", (False, 1.0, {}))[1],
                g.get("route_following", (False, 1.0, {}))[0],
                g.get("route_following", (False, 1.0, {}))[1],
                g.get("collision", (False, 1.0, {}))[0],
                g.get("collision", (False, 1.0, {}))[1],
                g.get("anchor_following", (False, 1.0, {}))[0],
                g.get("anchor_following", (False, 1.0, {}))[1],
                g.get("lateral", (False, 1.0, {}))[0],
                g.get("lateral", (False, 1.0, {}))[1],
                lat_params.get("eta_lat", 0.0),
                g.get("longitudinal", (False, 1.0, {}))[0],
                g.get("longitudinal", (False, 1.0, {}))[1],
                lon_params.get("eta_lon", 0.0),
            )

        # --- Helper: read editor into slot config ---
        def _read_editor_into_slot(
            noise_val, global_g_val, is_det,
            cl_on, cl_s, spd_on, spd_s, spd_stretch_val, lk_on, lk_s,
            rb_on, rb_s, rf_on, rf_s, col_on, col_s,
            anc_on, anc_s, lat_on, lat_s, eta_lat_val, lon_on, lon_s, eta_lon_val,
        ) -> TrajectorySlotConfig:
            slot = TrajectorySlotConfig(
                noise_scale=float(noise_val),
                global_guidance_scale=float(global_g_val),
                is_deterministic=bool(is_det),
            )
            slot.guidance["centerline_following"] = (bool(cl_on), float(cl_s), {})
            slot.guidance["speed"] = (bool(spd_on), float(spd_s), {
                "stretch": float(spd_stretch_val),
            })
            slot.guidance["lane_keeping"] = (bool(lk_on), float(lk_s), {})
            slot.guidance["road_border"] = (bool(rb_on), float(rb_s), {})
            slot.guidance["route_following"] = (bool(rf_on), float(rf_s), {})
            slot.guidance["collision"] = (bool(col_on), float(col_s), {})
            slot.guidance["anchor_following"] = (bool(anc_on), float(anc_s), {})
            slot.guidance["lateral"] = (bool(lat_on), float(lat_s), {
                "eta_lat": float(eta_lat_val), "lambda_lat": 3.0,
            })
            slot.guidance["longitudinal"] = (bool(lon_on), float(lon_s), {
                "eta_lon": float(eta_lon_val), "lambda_lon": 0.5,
            })
            return slot

        # --- Helper: format config summary ---
        def _config_summary_md(idx: int) -> str:
            if idx >= len(ranker.slot_configs):
                return ""
            slot = ranker.slot_configs[idx]
            lines = [f"**Trajectory #{idx + 1}**"]
            lines.append(f"- Noise: {slot.noise_scale:.2f}")
            lines.append(f"- Global scale: {slot.global_guidance_scale:.2f}")
            lines.append(f"- Deterministic: {slot.is_deterministic}")
            active = [(n, s) for n, (en, s, _) in slot.guidance.items() if en]
            if active:
                lines.append("- Active guidance:")
                for name, scale in active:
                    lines.append(f"  - {name}: {scale:.1f}")
            else:
                lines.append("- No guidance")
            if idx < len(ranker.reward_breakdowns):
                rb = ranker.reward_breakdowns[idx]
                lines.append(f"- **Reward: {rb.total:.1f}**")
            return "\n".join(lines)

        # --- Render helpers ---
        def _apply_reward_config(ws, wp, wm, wf, wc):
            ranker.reward_config = RewardConfig(
                w_safety=float(ws),
                w_progress=float(wp),
                w_smooth=float(wm),
                w_feasibility=float(wf),
                w_centerline=float(wc),
            )

        def _render(sel_idx, zoom, ts):
            view_range = 100 - (int(zoom) - 1) * 90 / 9
            traj_fig = ranker._create_trajectory_plot(
                time_step=int(ts), view_range=view_range, selected_idx=int(sel_idx),
            )
            table = ranker._format_reward_table(selected_idx=int(sel_idx))
            sc_fig = ranker._create_speed_curvature_plot(selected_idx=int(sel_idx))
            info = f"Scene {ranker.current_index + 1} / {len(ranker.npz_paths)}"
            return traj_fig, table, sc_fig, info

        def _dropdown_choices():
            return _format_dropdown_choices(ranker.slot_configs, ranker.reward_breakdowns)

        # --- Full run: load scene, generate all, render ---
        def _full_run(n_traj, ws, wp, wm, wf, wc, zoom, ts):
            ranker.n_trajectories = int(n_traj)
            _apply_reward_config(ws, wp, wm, wf, wc)
            ranker.load_sample()
            sel_idx = 0
            renders = _render(sel_idx, zoom, ts)
            choices = _dropdown_choices()
            editor_vals = _populate_editor(ranker.slot_configs[0]) if ranker.slot_configs else _populate_editor(TrajectorySlotConfig())
            summary = _config_summary_md(0)
            return (
                sel_idx,                                    # selected_state
                gr.update(choices=choices, value=choices[0] if choices else "#1"),  # dropdown
                summary,                                    # config_summary
                *editor_vals,                               # editor components (21 values)
                *renders,                                   # plot, table, speed_curv, info
            )

        full_run_inputs = [n_traj_sl] + reward_inputs + display_inputs
        full_run_outputs = (
            [selected_state, traj_dropdown, config_summary]
            + editor_components
            + main_outputs
        )

        # --- Navigation ---
        def _nav(delta, *args):
            ranker.current_index = max(
                0, min(len(ranker.npz_paths) - 1, ranker.current_index + delta),
            )
            return _full_run(*args)

        def _shuffle(*args):
            random.shuffle(ranker.npz_paths)
            ranker.current_index = 0
            return _full_run(*args)

        def _jump(idx, *args):
            ranker.current_index = max(
                0, min(len(ranker.npz_paths) - 1, int(idx)),
            )
            return _full_run(*args)

        for delta, btn in [
            (-30, btn_m30), (-10, btn_m10), (-1, btn_m1),
            (1, btn_p1), (10, btn_p10), (30, btn_p30),
        ]:
            btn.click(
                functools.partial(_nav, delta),
                inputs=full_run_inputs, outputs=full_run_outputs,
            )
        btn_shuffle.click(_shuffle, inputs=full_run_inputs, outputs=full_run_outputs)
        btn_regen_all.click(_full_run, inputs=full_run_inputs, outputs=full_run_outputs)
        jump_input.submit(
            _jump, inputs=[jump_input] + full_run_inputs, outputs=full_run_outputs,
        )

        # N trajectories slider -> full regen
        n_traj_sl.release(_full_run, inputs=full_run_inputs, outputs=full_run_outputs)

        # --- Select trajectory from dropdown ---
        def _on_select(choice_str, zoom, ts):
            idx = _parse_dropdown(choice_str)
            if idx >= len(ranker.slot_configs):
                idx = 0
            editor_vals = _populate_editor(ranker.slot_configs[idx])
            summary = _config_summary_md(idx)
            renders = _render(idx, zoom, ts)
            return (idx, summary, *editor_vals, *renders)

        select_outputs = (
            [selected_state, config_summary]
            + editor_components
            + main_outputs
        )
        traj_dropdown.change(
            _on_select,
            inputs=[traj_dropdown] + display_inputs,
            outputs=select_outputs,
        )

        # --- Regenerate single trajectory ---
        def _regen_single(sel_idx, zoom, ts, *editor_args):
            idx = int(sel_idx)
            slot = _read_editor_into_slot(*editor_args)
            if idx < len(ranker.slot_configs):
                ranker.slot_configs[idx] = slot
                ranker.regenerate_single(idx)
            # Update renders and dropdown
            renders = _render(idx, zoom, ts)
            choices = _dropdown_choices()
            summary = _config_summary_md(idx)
            return (
                gr.update(choices=choices, value=choices[idx] if idx < len(choices) else choices[0]),
                summary,
                *renders,
            )

        regen_single_outputs = [traj_dropdown, config_summary] + main_outputs
        btn_regen_single.click(
            _regen_single,
            inputs=[selected_state] + display_inputs + editor_components,
            outputs=regen_single_outputs,
        )

        # --- Reward weight changes -> rescore only ---
        def _rescore_and_render(sel_idx, ws, wp, wm, wf, wc, zoom, ts):
            _apply_reward_config(ws, wp, wm, wf, wc)
            ranker._score_trajectories()
            renders = _render(int(sel_idx), zoom, ts)
            choices = _dropdown_choices()
            current_idx = int(sel_idx)
            return (
                gr.update(choices=choices, value=choices[current_idx] if current_idx < len(choices) else choices[0]),
                *renders,
            )

        rescore_inputs = [selected_state] + reward_inputs + display_inputs
        rescore_outputs = [traj_dropdown] + main_outputs
        for sl in reward_inputs:
            sl.release(_rescore_and_render, inputs=rescore_inputs, outputs=rescore_outputs)
            sl.change(_rescore_and_render, inputs=rescore_inputs, outputs=rescore_outputs)

        # --- Display changes -> rerender only ---
        def _display_only(sel_idx, zoom, ts):
            return _render(int(sel_idx), zoom, ts)

        display_only_inputs = [selected_state] + display_inputs
        for sl in display_inputs:
            sl.release(_display_only, inputs=display_only_inputs, outputs=main_outputs)

        # --- Prototypes ---
        def _regen_protos(p_path):
            if not p_path:
                p_path = _DEFAULT_PROTOTYPES_PATH
            if not _GENERATE_SCRIPT.exists():
                return f"Script not found: {_GENERATE_SCRIPT}"
            try:
                subprocess.run(
                    [
                        sys.executable, str(_GENERATE_SCRIPT),
                        "--npz_list", ranker.npz_list_path,
                        "--output", p_path,
                        "--k", "16",
                        "--max_samples", "50000",
                    ],
                    check=True, capture_output=True, text=True, timeout=600,
                )
                return f"Regenerated prototypes at {p_path}"
            except subprocess.CalledProcessError as e:
                return f"Error: {e.stderr[:500]}"
            except subprocess.TimeoutExpired:
                return "Timeout generating prototypes"

        btn_regen_protos.click(
            _regen_protos, inputs=[proto_path], outputs=[sample_info],
        )

        btn_save.click(
            lambda d, z, ts: ranker.save_current_scene(d, zoom=z, time_step=ts),
            inputs=[save_dir, zoom_sl, time_sl], outputs=[save_status],
        )

        # --- GRPO training event wiring ---
        if training_enabled:
            _grpo_epoch_counter = [0]

            def _accept_and_advance(*args):
                # args = full_run_inputs
                msg = ranker.accept_current_group()
                ranker.current_index = min(
                    len(ranker.npz_paths) - 1, ranker.current_index + 1,
                )
                full_out = _full_run(*args)
                return (msg, *full_out)

            def _skip_and_advance(*args):
                ranker.current_index = min(
                    len(ranker.npz_paths) - 1, ranker.current_index + 1,
                )
                full_out = _full_run(*args)
                msg = f"Skipped. {len(ranker.accepted_groups)} groups queued"
                return (msg, *full_out)

            def _clear_queue():
                return ranker.clear_accepted_groups()

            def _train_epoch(beta_val, lr_val, accum_val, *args):
                if not ranker.accepted_groups:
                    return (
                        "No groups queued. Accept some scenes first.",
                        _grpo_epoch_counter[0],
                        f"0 groups queued",
                    )

                trainer.beta = float(beta_val)
                trainer.grad_accum_groups = int(accum_val)
                for pg in trainer.optimizer.param_groups:
                    pg["lr"] = float(lr_val)

                _grpo_epoch_counter[0] += 1
                epoch = _grpo_epoch_counter[0]

                if epoch == 1:
                    npz_paths = [g["npz_path"] for g in ranker.accepted_groups]
                    trainer.save_epoch1_baselines(npz_paths)

                groups = list(ranker.accepted_groups)
                metrics = trainer.train_on_groups(groups, epoch)
                drift = trainer.compute_trajectory_drift()
                trainer.log_metrics(epoch, metrics)
                trainer.save_checkpoint(epoch, {})
                ranker.accepted_groups.clear()

                log_lines = [
                    f"**Epoch {epoch}** -- trained on {len(groups)} groups",
                    f"- Loss: {metrics.get('loss', 0):.4f}",
                    f"- Policy loss: {metrics.get('policy_loss', 0):.4f}",
                    f"- KL loss: {metrics.get('kl_loss', 0):.4f}",
                ]
                if drift:
                    log_lines.append(f"- {drift}")
                log_lines.append(f"\nQueue cleared.")

                return ("\n".join(log_lines), epoch, "0 groups queued")

            btn_accept.click(
                _accept_and_advance,
                inputs=full_run_inputs,
                outputs=[queue_status] + full_run_outputs,
            )
            btn_skip.click(
                _skip_and_advance,
                inputs=full_run_inputs,
                outputs=[queue_status] + full_run_outputs,
            )
            btn_clear_queue.click(
                _clear_queue, inputs=[], outputs=[queue_status],
            )
            btn_train.click(
                _train_epoch,
                inputs=[beta_sl, lr_sl, accum_sl],
                outputs=[train_log, epoch_display, queue_status],
            )

        demo.load(_full_run, inputs=full_run_inputs, outputs=full_run_outputs)

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Trajectory Ranker GUI")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--npz_list", type=Path, required=True,
                        help="JSON file listing .npz scene paths")
    parser.add_argument("--prototypes", type=Path, default=None,
                        help="Path to prototypes .npy (auto-generated from npz_list if omitted)")
    parser.add_argument("--regen-prototypes", action="store_true",
                        help="Force regenerate prototypes from npz_list even if cached file exists")
    parser.add_argument("--n_trajectories", type=int, default=16)
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--share", action="store_true")

    parser.add_argument("--no-training", action="store_true",
                        help="Disable GRPO training controls (visualization-only mode)")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to GRPO config JSON. Required unless "
                             "--no-training is set (viz-only mode).")
    parser.add_argument("--use_lora", action="store_true", default=False,
                        help="Apply LoRA adapters for training")
    parser.add_argument("--exp_name", type=str, default="grpo_gui",
                        help="Experiment name for checkpoint directory")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, model_args = load_model(args.model_path, device)

    with open(args.npz_list) as f:
        npz_paths = json.load(f)
    print(f"Loaded {len(npz_paths)} samples")

    if args.prototypes:
        prototypes_path = str(args.prototypes)
    else:
        prototypes_path = _DEFAULT_PROTOTYPES_PATH

    prototypes_path = ensure_prototypes(
        npz_list_path=str(args.npz_list),
        prototypes_path=prototypes_path,
        force=args.regen_prototypes,
    )
    print(f"Prototypes: {prototypes_path}")

    grpo_trainer = None
    if not args.no_training:
        from rlvr.grpo_config import GRPOConfig

        if args.config is None:
            raise SystemExit(
                "--config is required unless --no-training is set."
            )
        if not args.config.exists():
            raise FileNotFoundError(f"GRPO config not found: {args.config}")
        grpo_cfg = GRPOConfig.from_json(args.config)
        print(f"Loaded GRPO config from {args.config}")

        if args.use_lora:
            grpo_cfg.use_lora = True

        if grpo_cfg.use_lora:
            from preference_optimization.lora_utils import apply_lora
            model = apply_lora(
                model,
                r=grpo_cfg.lora_rank,
                lora_alpha=grpo_cfg.lora_alpha,
                lora_dropout=grpo_cfg.lora_dropout,
            )

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            print("Warning: no trainable parameters found. Use --use_lora or ensure model is not frozen.")
        else:
            from datetime import datetime

            from torch import optim

            from rlvr.grpo_trainer import GRPOTrainer

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_dir = args.npz_list.parent / f"{timestamp}_{args.exp_name}"
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"Training output: {run_dir}")

            optimizer = optim.AdamW(trainable_params, lr=grpo_cfg.learning_rate)

            grpo_trainer = GRPOTrainer(
                policy_model=model,
                model_args=model_args,
                optimizer=optimizer,
                device=device,
                run_dir=run_dir,
                config=grpo_cfg,
                use_lora=grpo_cfg.use_lora,
            )
            mode_str = "multi-epoch" if grpo_cfg.uses_importance_sampling else "on-policy"
            print(f"GRPO training enabled [{mode_str}] "
                  f"(N={grpo_cfg.num_generations}, M={grpo_cfg.inner_epochs}, "
                  f"kl={grpo_cfg.kl_coef}, lr={grpo_cfg.learning_rate})")
    else:
        model.eval()
        print("Visualization-only mode (--no-training)")

    ranker = TrajectoryRanker(
        policy_model=model,
        model_args=model_args,
        npz_paths=npz_paths,
        npz_list_path=str(args.npz_list),
        prototypes_path=prototypes_path,
        n_trajectories=args.n_trajectories,
    )

    demo = build_interface(ranker, trainer=grpo_trainer)
    demo.launch(server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
