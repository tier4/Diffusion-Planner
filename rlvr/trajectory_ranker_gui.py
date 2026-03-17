"""Trajectory Ranker GUI -- visualization and reward debugging for diverse trajectory groups.

Generates N trajectories per scene with diverse noise/guidance configs, scores
them with the rule-based reward, computes group-relative advantages, and
visualizes everything with a red-to-green advantage colormap.

Launch
------
source .venv/bin/activate
python rlvr/trajectory_ranker_gui.py \\
  --model_path /path/to/model.pth \\
  --npz_list   /path/to/train_or_valid.json

Prototypes are auto-generated from the npz_list if not provided.
Use --prototypes to point to an existing file, or --regen-prototypes to force
regeneration even if a cached file exists.
"""

from __future__ import annotations

import argparse
import functools
import json
import random
import subprocess
import sys
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from diffusion_planner.utils.visualize_input import visualize_inputs
from guidance_gui.visualization import (
    _calculate_curvature,
    _calculate_lateral_acceleration,
    _calculate_velocities,
    _gt_curvature,
    _gt_velocities,
    _draw_vehicle_footprint,
)
from preference_optimization.model_utils import load_model
from preference_optimization.utils import load_npz_data

from rlvr.grpo_sampler import SamplerConfig, SampledTrajectory, generate_diverse_group
from rlvr.reward import RewardBreakdown, RewardConfig, compute_group_advantages, compute_reward_batch


_DIVERGING_CMAP = plt.get_cmap("RdYlGn")

_DEFAULT_PROTOTYPES_PATH = str(Path(__file__).parent / "prototypes_k16.npy")
_GENERATE_SCRIPT = Path(__file__).parent.parent / "guidance_gui" / "scripts" / "generate_prototypes.py"


def ensure_prototypes(npz_list_path: str, prototypes_path: str, force: bool = False) -> str:
    """Generate prototypes from npz_list if they don't exist (or force=True).

    Returns the path to the prototypes file.
    """
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

    return prototypes_path


class TrajectoryRanker:
    """Generates and scores N diverse trajectories per scene."""

    def __init__(
        self,
        policy_model,
        model_args,
        npz_paths: list[str],
        npz_list_path: str,
        prototypes_path: str,
    ):
        self.policy_model = policy_model
        self.model_args = model_args
        self.npz_paths = npz_paths
        self.npz_list_path = npz_list_path
        self.current_index = 0
        self.device = next(policy_model.parameters()).device

        self.sampler_config = SamplerConfig(prototypes_path=prototypes_path)
        self.reward_config = RewardConfig()

        self.current_data: dict[str, torch.Tensor] | None = None
        self.sampled_trajectories: list[SampledTrajectory] = []
        self.reward_breakdowns: list[RewardBreakdown] = []
        self.advantages: np.ndarray = np.array([])
        self.saved_scenes: list[dict] = []

    def load_sample(self) -> None:
        if not self.npz_paths or self.current_index >= len(self.npz_paths):
            return

        self.current_data = load_npz_data(
            self.npz_paths[self.current_index], self.device
        )
        self.sampled_trajectories = generate_diverse_group(
            model=self.policy_model,
            model_args=self.model_args,
            data=self.current_data,
            config=self.sampler_config,
            device=self.device,
        )
        self._score_trajectories()

    def _score_trajectories(self) -> None:
        if not self.sampled_trajectories or self.current_data is None:
            return

        # Stack all trajectories into (N, T, 4) and evaluate in one batched pass
        traj_batch = torch.tensor(
            np.stack([st.trajectory for st in self.sampled_trajectories]),
            device=self.device, dtype=torch.float32,
        )  # (N, T, 4)
        self.reward_breakdowns = compute_reward_batch(
            traj_batch, self.current_data, self.reward_config
        )
        self.advantages = compute_group_advantages(self.reward_breakdowns)

    def _create_trajectory_plot(
        self, time_step: int = 40, view_range: float = 60.0
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

            if st.is_deterministic:
                color = "dodgerblue"
                lw = 3.0
                alpha = 1.0
                linestyle = "--"
                label = f"#{display_rank} R={self.reward_breakdowns[i].total:.1f} [DET]"
            else:
                color = _DIVERGING_CMAP(rank_frac)
                lw = 1.0 + 2.5 * rank_frac
                alpha = 0.3 + 0.7 * rank_frac
                linestyle = "-"
                label = f"#{display_rank} R={self.reward_breakdowns[i].total:.1f} ({st.label})"

            ax.plot(
                traj[:, 0], traj[:, 1],
                color=color, linewidth=lw, alpha=alpha,
                linestyle=linestyle, label=label,
            )

            # Top-3 get diamond markers, deterministic gets a star
            if st.is_deterministic and 0 <= time_step < len(traj):
                ax.scatter(
                    [traj[time_step, 0]], [traj[time_step, 1]],
                    color=color, s=120, zorder=11,
                    edgecolors="black", marker="*",
                )
            elif ranks[i] >= N - 3 and 0 <= time_step < len(traj):
                ax.scatter(
                    [traj[time_step, 0]], [traj[time_step, 1]],
                    color=color, s=80, zorder=10,
                    edgecolors="black", marker="D",
                )

            # Collision point: red X at the collision timestep
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

        ax.legend(loc="upper left", fontsize=6, ncol=2)
        ax.set_title(f"Scene {self.current_index + 1} / {len(self.npz_paths)}")

        ref = self.sampled_trajectories[0].trajectory
        cx = (ref[0, 0] + ref[-1, 0]) / 2
        cy = (ref[0, 1] + ref[-1, 1]) / 2
        half = view_range / 2
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal")
        return fig

    def _create_speed_curvature_plot(self) -> Figure:
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
        top3_indices = rank_order[-min(3, N):][::-1]

        for plot_i, idx in enumerate(top3_indices):
            st = self.sampled_trajectories[idx]
            traj = st.trajectory
            rank_frac = (N - 1 - plot_i) / max(N - 1, 1)
            color = _DIVERGING_CMAP(rank_frac)

            vel = _calculate_velocities(traj, ego_state)
            curv = _calculate_curvature(traj, ego_state)
            t = np.arange(len(vel))

            display_rank = plot_i + 1
            ax_speed.plot(
                t, vel, color=color, linewidth=1.8, alpha=0.8,
                label=f"#{display_rank} {st.label}",
            )
            ax_curv.plot(
                np.arange(len(curv)), curv,
                color=color, linewidth=1.8, alpha=0.8,
            )

        if "ego_agent_future" in data_cpu:
            ego_future = np.array(data_cpu["ego_agent_future"]).reshape(-1, 3)
            gt_vel = _gt_velocities(ego_future, ego_state)
            gt_curv = _gt_curvature(ego_future, ego_state)
            if gt_vel is not None:
                ax_speed.plot(
                    np.arange(len(gt_vel)), gt_vel,
                    "k--", linewidth=2, alpha=0.7, label="GT",
                )
            if gt_curv is not None:
                ax_curv.plot(
                    np.arange(len(gt_curv)), gt_curv,
                    "k--", linewidth=2, alpha=0.7,
                )

        ax_speed.set_ylabel("Speed (km/h)")
        ax_speed.set_ylim(0, 80)
        ax_speed.set_title("Speed (top-3)")
        ax_speed.legend(loc="upper right", fontsize=7)
        ax_speed.grid(True, alpha=0.3)

        ax_curv.set_ylabel("Curvature (1/m)")
        ax_curv.set_xlabel("Time step")
        ax_curv.set_ylim(-0.2, 0.2)
        ax_curv.set_title("Curvature (top-3)")
        ax_curv.grid(True, alpha=0.3)
        ax_curv.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

        fig.tight_layout()
        return fig

    def _format_reward_table(self) -> str:
        if not self.reward_breakdowns:
            return ""

        rows = list(zip(
            range(len(self.reward_breakdowns)),
            self.reward_breakdowns,
            self.advantages,
            self.sampled_trajectories,
        ))
        rows.sort(key=lambda r: r[1].total, reverse=True)

        lines = [
            "| Rank | Safety | Progress | Smooth | Feasible | Centerline | Total | Adv | Config |",
            "|------|--------|----------|--------|----------|------------|-------|-----|--------|",
        ]
        for rank, (idx, rb, adv, st) in enumerate(rows, 1):
            config_col = f"**[DET]**" if st.is_deterministic else st.label
            b = "**" if st.is_deterministic else ""
            lines.append(
                f"| {b}{rank}{b} | {b}{rb.safety:.1f}{b} | {b}{rb.progress:.1f}{b} | "
                f"{b}{rb.smoothness:.1f}{b} | {b}{rb.feasibility:.2f}{b} | "
                f"{b}{rb.centerline:.2f}{b} | "
                f"{b}{rb.total:.1f}{b} | {b}{adv:+.2f}{b} | {config_col} |"
            )
        return "\n".join(lines)

    def save_current_scene(self, save_dir: str, zoom: int = 5, time_step: int = 40) -> str:
        """Append current scene trajectories + rewards to a dump file, and save plot image."""
        if not self.sampled_trajectories or self.current_data is None:
            return "Nothing to save"

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        save_idx = len(self.saved_scenes)

        # Save the trajectory plot as image
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
            "rewards": {
                "safety": [rb.safety for rb in self.reward_breakdowns],
                "progress": [rb.progress for rb in self.reward_breakdowns],
                "smoothness": [rb.smoothness for rb in self.reward_breakdowns],
                "feasibility": [rb.feasibility for rb in self.reward_breakdowns],
                "centerline": [rb.centerline for rb in self.reward_breakdowns],
                "total": [rb.total for rb in self.reward_breakdowns],
                "collision_step": [rb.collision_step for rb in self.reward_breakdowns],
                "off_road_fraction": [rb.off_road_fraction for rb in self.reward_breakdowns],
            },
            "advantages": self.advantages.tolist(),
            "reward_config": {
                "w_safety": self.reward_config.w_safety,
                "w_progress": self.reward_config.w_progress,
                "w_smooth": self.reward_config.w_smooth,
                "w_feasibility": self.reward_config.w_feasibility,
                "w_centerline": self.reward_config.w_centerline,
            },
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


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

def build_interface(ranker: TrajectoryRanker) -> gr.Blocks:
    with gr.Blocks(title="Trajectory Ranker") as demo:
        gr.Markdown("# Trajectory Ranker")

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
                    btn_regen = gr.Button("Re-do", size="sm")
                jump_input = gr.Number(
                    label="Jump to index", value=0, minimum=0, precision=0
                )

                gr.Markdown("### Noise")
                n_traj_sl = gr.Slider(
                    2, 64, value=8, step=1, label="N trajectories"
                )
                noise_lo = gr.Slider(
                    0.0, 5.0, value=0.5, step=0.1, label="Noise min"
                )
                noise_hi = gr.Slider(
                    0.0, 5.0, value=4.0, step=0.1, label="Noise max"
                )

                gr.Markdown("### Guidance")
                enable_guidance_cb = gr.Checkbox(
                    value=True, label="Enable guidance (random per trajectory)"
                )
                guidance_prob_sl = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.05,
                    label="Per-type inclusion probability",
                )
                guidance_scale_sl = gr.Slider(
                    0.1, 5.0, value=2.0, step=0.1,
                    label="Guidance scale max",
                )
                gr.Markdown("**Guidance types in random pool:**")
                cb_centerline = gr.Checkbox(value=True, label="Centerline following")
                cb_anchor = gr.Checkbox(value=True, label="Anchor following")
                cb_collision = gr.Checkbox(value=False, label="Collision")
                cb_route = gr.Checkbox(value=False, label="Route following")
                cb_lane = gr.Checkbox(value=False, label="Lane keeping")

                gr.Markdown("### Reward Weights")
                w_safety = gr.Slider(
                    0.0, 20.0, value=5.0, step=0.5, label="w_safety"
                )
                w_progress = gr.Slider(
                    0.0, 10.0, value=1.0, step=0.1, label="w_progress"
                )
                w_smooth = gr.Slider(
                    0.0, 10.0, value=0.5, step=0.1, label="w_smooth"
                )
                w_feasibility = gr.Slider(
                    0.0, 10.0, value=5.0, step=0.1, label="w_feasibility"
                )
                w_centerline = gr.Slider(
                    0.0, 10.0, value=5.0, step=0.1, label="w_centerline"
                )

                gr.Markdown("### Prototypes")
                proto_path = gr.Textbox(
                    label="Prototypes path",
                    value=ranker.sampler_config.prototypes_path or "",
                )
                btn_regen_protos = gr.Button("Regen Protos", size="sm")

                gr.Markdown("### Display")
                zoom_sl = gr.Slider(1, 10, value=5, step=1, label="Zoom")
                time_sl = gr.Slider(0, 79, value=40, step=1, label="Time step")

            # --- Main content ---
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

        # --- Input lists (order matters for positional unpacking) ---
        sampler_inputs = [
            n_traj_sl, noise_lo, noise_hi,                          # 0-2
            enable_guidance_cb, guidance_prob_sl, guidance_scale_sl, # 3-5
            cb_centerline, cb_anchor, cb_collision, cb_route, cb_lane,  # 6-10
            proto_path,                                              # 11
        ]
        reward_inputs = [w_safety, w_progress, w_smooth, w_feasibility, w_centerline]  # 12-16
        display_inputs = [zoom_sl, time_sl]                              # 16-17
        all_inputs = sampler_inputs + reward_inputs + display_inputs
        N_SAMPLER = len(sampler_inputs)
        N_REWARD = len(reward_inputs)
        outputs = [traj_plot, reward_table, speed_curv_plot, sample_info]

        def _apply_sampler_config(
            n_traj, ns_lo, ns_hi,
            enable_guidance, guidance_prob, gs_max,
            use_cl, use_anchor, use_col, use_route, use_lane,
            p_path,
        ):
            ranker.sampler_config = SamplerConfig(
                n_trajectories=int(n_traj),
                noise_scale_range=(float(ns_lo), float(ns_hi)),
                guidance_scale_range=(0.1, float(gs_max)),
                enable_guidance=bool(enable_guidance),
                guidance_prob=float(guidance_prob),
                enable_centerline=bool(use_cl),
                enable_anchor=bool(use_anchor),
                enable_collision=bool(use_col),
                enable_route_following=bool(use_route),
                enable_lane_keeping=bool(use_lane),
                prototypes_path=p_path if p_path else None,
            )

        def _apply_reward_config(ws, wp, wm, wf, wc):
            ranker.reward_config = RewardConfig(
                w_safety=float(ws),
                w_progress=float(wp),
                w_smooth=float(wm),
                w_feasibility=float(wf),
                w_centerline=float(wc),
            )

        def _render(zoom, ts):
            view_range = 100 - (int(zoom) - 1) * 90 / 9
            traj_fig = ranker._create_trajectory_plot(
                time_step=int(ts), view_range=view_range
            )
            table = ranker._format_reward_table()
            sc_fig = ranker._create_speed_curvature_plot()
            info = f"Scene {ranker.current_index + 1} / {len(ranker.npz_paths)}"
            return traj_fig, table, sc_fig, info

        def _full_run(*args):
            sampler_args = args[:N_SAMPLER]
            reward_args = args[N_SAMPLER:N_SAMPLER + N_REWARD]
            display_args = args[N_SAMPLER + N_REWARD:]
            _apply_sampler_config(*sampler_args)
            _apply_reward_config(*reward_args)
            ranker.load_sample()
            return _render(*display_args)

        def _rescore_and_render(*args):
            reward_args = args[N_SAMPLER:N_SAMPLER + N_REWARD]
            display_args = args[N_SAMPLER + N_REWARD:]
            _apply_reward_config(*reward_args)
            ranker._score_trajectories()
            return _render(*display_args)

        def _display_only(*args):
            display_args = args[N_SAMPLER + N_REWARD:]
            return _render(*display_args)

        def _nav(delta, *args):
            ranker.current_index = max(
                0, min(len(ranker.npz_paths) - 1, ranker.current_index + delta)
            )
            return _full_run(*args)

        def _shuffle(*args):
            random.shuffle(ranker.npz_paths)
            ranker.current_index = 0
            return _full_run(*args)

        def _jump(idx, *args):
            ranker.current_index = max(
                0, min(len(ranker.npz_paths) - 1, int(idx))
            )
            return _full_run(*args)

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

        # --- Wire events ---
        for delta, btn in [
            (-30, btn_m30), (-10, btn_m10), (-1, btn_m1),
            (1, btn_p1), (10, btn_p10), (30, btn_p30),
        ]:
            btn.click(
                functools.partial(_nav, delta),
                inputs=all_inputs, outputs=outputs,
            )

        btn_shuffle.click(_shuffle, inputs=all_inputs, outputs=outputs)
        btn_regen.click(_full_run, inputs=all_inputs, outputs=outputs)
        jump_input.submit(
            _jump, inputs=[jump_input] + all_inputs, outputs=outputs
        )

        # Sampler param changes -> full regeneration
        for sl in [n_traj_sl, noise_lo, noise_hi, guidance_prob_sl, guidance_scale_sl]:
            sl.release(_full_run, inputs=all_inputs, outputs=outputs)
        for cb in [enable_guidance_cb, cb_centerline, cb_anchor, cb_collision, cb_route, cb_lane]:
            cb.change(_full_run, inputs=all_inputs, outputs=outputs)

        # Reward weight changes -> rescore only (both release and change for responsiveness)
        for sl in [w_safety, w_progress, w_smooth, w_feasibility, w_centerline]:
            sl.release(_rescore_and_render, inputs=all_inputs, outputs=outputs)
            sl.change(_rescore_and_render, inputs=all_inputs, outputs=outputs)

        # Display changes -> rerender only
        for sl in [zoom_sl, time_sl]:
            sl.release(_display_only, inputs=all_inputs, outputs=outputs)

        btn_regen_protos.click(
            _regen_protos, inputs=[proto_path], outputs=[sample_info]
        )

        btn_save.click(
            lambda d, z, ts: ranker.save_current_scene(d, zoom=z, time_step=ts),
            inputs=[save_dir, zoom_sl, time_sl], outputs=[save_status],
        )

        demo.load(_full_run, inputs=all_inputs, outputs=outputs)

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
    parser.add_argument("--n_trajectories", type=int, default=8)
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, model_args = load_model(args.model_path, device)
    model.eval()

    with open(args.npz_list) as f:
        npz_paths = json.load(f)
    print(f"Loaded {len(npz_paths)} samples")

    # Resolve prototypes: explicit path > default cached > auto-generate
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

    ranker = TrajectoryRanker(
        policy_model=model,
        model_args=model_args,
        npz_paths=npz_paths,
        npz_list_path=str(args.npz_list),
        prototypes_path=prototypes_path,
    )
    ranker.sampler_config.n_trajectories = args.n_trajectories

    demo = build_interface(ranker)
    demo.launch(server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
