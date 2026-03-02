"""Guidance Playground — simplified version of the DPO annotation GUI.

Shows N independent samples under configurable noise + guidance instead of
a deterministic/stochastic pair for annotation.

Launch
------
source .venv/bin/activate
python guidance_playground/app.py \\
  --model_path /path/to/model.pth \\
  --npz_list   /path/to/train_or_valid.json
"""

import argparse
import json
import random
from pathlib import Path

import gradio as gr
import matplotlib.cm as cm
import numpy as np
import torch
from matplotlib.figure import Figure

from diffusion_planner.utils.visualize_input import visualize_inputs
from preference_optimization.annotation_gui import PreferenceAnnotator
from preference_optimization.model_utils import load_model
from preference_optimization.utils import load_npz_data

from guidance_playground.generate_samples import generate_samples
from guidance_playground.guidance_ui import build_guidance_panel, make_guidance_set_config


_CMAP = cm.get_cmap("tab10")




class PlaygroundAnnotator(PreferenceAnnotator):
    """PreferenceAnnotator with N-sample generation instead of DPO pairing."""

    def __init__(self, policy_model, model_args, npz_paths,
                 prototypes: np.ndarray | None = None,
                 prototypes_path: str | None = None,
                 prototype_counts: np.ndarray | None = None):
        # target_count=0 — no annotation quota needed
        super().__init__(policy_model, model_args, npz_paths, target_count=0)
        self.all_samples: np.ndarray | None = None  # (N, T, 4)
        self.prototypes = prototypes                 # (K, 80, 2) or None
        self.prototypes_path = prototypes_path       # str path for GuidanceConfig.params
        self.prototype_counts = prototype_counts     # (K,) or None

    # ------------------------------------------------------------------
    # Override: generate N samples instead of a deterministic/stochastic pair
    # ------------------------------------------------------------------
    def load_sample(
        self,
        noise_scale: float,
        fde_threshold: float,
        ade_threshold: float,
        max_retries: int,
        zoom_level: int = 5,
        gt_similarity_mode: bool = True,
        enable_initial_pruning: bool = True,
        initial_pos_threshold: float = 0.055,
        initial_yaw_threshold_deg: float = 0.55,
        guidance=None,
        time_step: int = 40,
        # Playground-specific (passed as extra kwargs from the Gradio wiring)
        n_samples: int = 4,
    ):
        if not self.npz_paths or self.current_index >= len(self.npz_paths):
            return None, None, None, "No samples", "", "", self.get_sidebar_state(), self.get_labeled_history_display()

        self.current_data = load_npz_data(self.npz_paths[self.current_index], self.device)
        self.gt_available = self._check_gt_available()

        # Normalize a copy for inference
        norm_data = {k: v.clone() if isinstance(v, torch.Tensor) else v
                     for k, v in self.current_data.items()}
        norm_data = self.model_args.observation_normalizer(norm_data)

        # Build composer if guidance is active
        composer = None
        if guidance is not None and guidance.active_functions():
            from diffusion_planner.model.guidance.composer import GuidanceComposer
            composer = GuidanceComposer(guidance)

        samples = generate_samples(
            model=self.policy_model,
            model_args=self.model_args,
            data=norm_data,
            noise_scale=float(noise_scale),
            n_samples=int(n_samples),
            composer=composer,
            device=self.device,
        )
        self.all_samples = samples
        self.ego_shape = self.current_data["ego_shape"].tolist()

        # Keep traj_1 / traj_2 pointing at samples 0 & 1 so velocity/lateral
        # plots (which read self.trajectory_1/2) still work unchanged.
        self.trajectory_1 = samples[0].tolist()
        self.trajectory_2 = samples[min(1, len(samples) - 1)].tolist()

        view_range = 100 - (int(zoom_level) - 1) * 90 / 9
        traj_plot = self._create_trajectory_plot(time_step=int(time_step), view_range=view_range)
        vel_plot   = self._create_velocity_plot(time_step=int(time_step))
        lat_plot   = self._create_lateral_curvature_plot(time_step=int(time_step))

        sample_info = f"Sample {self.current_index + 1} / {len(self.npz_paths)}"
        return traj_plot, vel_plot, lat_plot, sample_info, "", "", self.get_sidebar_state(), self.get_labeled_history_display()

    # ------------------------------------------------------------------
    # Override: N coloured trajectories instead of green / orange pair
    # ------------------------------------------------------------------
    def _create_trajectory_plot(self, time_step=None, view_range=60):
        fig = Figure(figsize=(10, 11.5))
        ax = fig.add_subplot(111)

        data_cpu = {k: v.cpu() for k, v in self.current_data.items()}
        visualize_inputs(data_cpu, save_path=None, ax=ax, view_ranges=[120])

        samples = self.all_samples
        if samples is None:
            return fig

        ref = samples[0]
        cx = (ref[0, 0] + ref[-1, 0]) / 2
        cy = (ref[0, 1] + ref[-1, 1]) / 2

        for i, traj in enumerate(samples):
            color = _CMAP(i % 10)
            ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2.5,
                    alpha=0.8, label=f"Sample {i}")
            if time_step is not None and 0 <= time_step < len(traj):
                ax.scatter([traj[time_step, 0]], [traj[time_step, 1]],
                           color=color, s=60, zorder=10, edgecolors="black")

        if "ego_agent_future" in self.current_data:
            gt = self.current_data["ego_agent_future"].cpu().numpy()[0]  # (80, 3)
            valid = ~((gt[:, 0] == 0) & (gt[:, 1] == 0))
            if np.any(valid):
                ax.plot(gt[valid, 0], gt[valid, 1], "k--", linewidth=2,
                        alpha=0.6, label="GT")

        ax.legend(loc="upper left", fontsize=8)
        ax.set_title(f"Sample {self.current_index + 1} / {len(self.npz_paths)}")
        half = view_range / 2
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal")
        return fig


# ---------------------------------------------------------------------------
# Build a minimal Gradio interface reusing PreferenceAnnotator's patterns
# ---------------------------------------------------------------------------

def build_playground_interface(annotator: PlaygroundAnnotator):
    with gr.Blocks(title="Guidance Playground") as demo:
        gr.Markdown("# Guidance Playground")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Navigation")
                with gr.Row():
                    btn_m30 = gr.Button("← 30", size="sm")
                    btn_m10 = gr.Button("← 10", size="sm")
                    btn_m1  = gr.Button("← 1",  size="sm")
                    btn_p1  = gr.Button("1 →",  size="sm")
                    btn_p10 = gr.Button("10 →", size="sm")
                    btn_p30 = gr.Button("30 →", size="sm")
                with gr.Row():
                    btn_shuffle  = gr.Button("Shuffle",  size="sm")
                    btn_resample = gr.Button("Resample", size="sm")
                jump_input = gr.Number(label="Jump to index", value=0, minimum=0, precision=0)

                gr.Markdown("### Generation")
                noise_scale   = gr.Slider(0.0, 5.0, value=2.5, step=0.1, label="Noise Scale")
                n_samples_sl  = gr.Slider(1, 8, value=4, step=1, label="N Samples")
                zoom_slider   = gr.Slider(1, 10, value=5, step=1, label="Zoom (1=100m, 10=10m)")
                time_slider   = gr.Slider(0, 79, value=40, step=1, label="Time Step")

                gr.Markdown("### Guidance")
                panel = build_guidance_panel(
                    annotator.prototypes_path or "guidance_playground/prototypes_k16.npy"
                )

            with gr.Column(scale=2):
                traj_plot = gr.Plot(label="Trajectories")
                with gr.Accordion("Speed & Curvature Plots", open=False):
                    with gr.Row():
                        vel_plot = gr.Plot(label="Speed & Acceleration")
                        lat_plot = gr.Plot(label="Lateral Curvature")
                sample_info = gr.Markdown("Sample — / —")

        # ---- helpers ----
        # panel.inputs order: [enable_cb, collision_cb, collision_scale, route_cb, route_scale,
        #                       lane_cb, lane_scale, centerline_cb, centerline_scale,
        #                       anchor_cb, anchor_scale, anchor_index, anchor_path, global_scale]
        _gen_inputs = [noise_scale, n_samples_sl, zoom_slider, time_slider] + panel.inputs
        _outputs = [traj_plot, vel_plot, lat_plot, sample_info]

        def _run(ns, n, zl, ts, eg, uc, ucs, urf, urfs, ulk, ulks, ucf, ucfs, ua, uas, ai, ap, gs):
            result = annotator.load_sample(
                noise_scale=ns, fde_threshold=2.0, ade_threshold=1.0, max_retries=1,
                zoom_level=zl,
                guidance=make_guidance_set_config(eg, uc, ucs, urf, urfs, ulk, ulks, ucf, ucfs, ua, uas, ai, ap, gs),
                time_step=int(ts), n_samples=int(n),
            )
            return result[0], result[1], result[2], result[3]

        def _nav(delta, *args):
            annotator.current_index = max(0, min(len(annotator.npz_paths) - 1,
                                                  annotator.current_index + delta))
            return _run(*args)

        def _shuffle(*args):
            random.shuffle(annotator.npz_paths)
            annotator.current_index = 0
            return _run(*args)

        def _jump(idx, *args):
            annotator.current_index = max(0, min(len(annotator.npz_paths) - 1, int(idx)))
            return _run(*args)

        import functools
        for delta, btn in [(-30, btn_m30), (-10, btn_m10), (-1, btn_m1),
                            (1, btn_p1), (10, btn_p10), (30, btn_p30)]:
            btn.click(functools.partial(_nav, delta), inputs=_gen_inputs, outputs=_outputs)

        btn_shuffle.click(_shuffle,  inputs=_gen_inputs,              outputs=_outputs)
        btn_resample.click(_run,     inputs=_gen_inputs,              outputs=_outputs)
        jump_input.submit(_jump,     inputs=[jump_input] + _gen_inputs, outputs=_outputs)

        for slider in [noise_scale, n_samples_sl, zoom_slider,
                       panel.global_scale,
                       panel.collision_scale, panel.route_scale,
                       panel.lane_scale, panel.centerline_scale, panel.anchor_scale]:
            slider.release(_run, inputs=_gen_inputs, outputs=_outputs)

        time_slider.release(_run, inputs=_gen_inputs, outputs=_outputs)

        for cb in [panel.enable_cb, panel.collision_cb, panel.route_cb, panel.lane_cb,
                   panel.centerline_cb, panel.anchor_cb]:
            cb.change(_run, inputs=_gen_inputs, outputs=_outputs)

        # Gallery click: inject evt.index directly to bypass potentially stale component value,
        # then regenerate immediately with the new anchor.
        # anchor_index is at position 15 in _gen_inputs (4 app-level + 11 in panel.inputs).
        _ANCHOR_IDX_POS = 4 + 11  # = 15

        def _select_anchor(evt: gr.SelectData, *gen_args):
            gen_list = list(gen_args)
            gen_list[_ANCHOR_IDX_POS] = evt.index
            return _run(*gen_list)

        panel.gallery.select(_select_anchor, inputs=_gen_inputs, outputs=_outputs)

        demo.load(_run, inputs=_gen_inputs, outputs=_outputs)

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  type=Path, required=True)
    parser.add_argument("--npz_list",    type=Path, required=True)
    parser.add_argument("--prototypes",  type=Path, default=None,
                        help="Path to prototypes_k*.npy (optional, enables anchor guidance gallery)")
    parser.add_argument("--port",        type=int,  default=7860)
    parser.add_argument("--share",       action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, model_args = load_model(args.model_path, device)
    model.eval()

    with open(args.npz_list) as f:
        npz_paths = json.load(f)
    print(f"Loaded {len(npz_paths)} samples")

    prototypes = prototype_counts = prototypes_path = None
    if args.prototypes and args.prototypes.exists():
        prototypes = np.load(str(args.prototypes))          # (K, 80, 2)
        prototypes_path = str(args.prototypes)
        counts_path = Path(str(args.prototypes).replace(".npy", "_counts.npy"))
        if counts_path.exists():
            prototype_counts = np.load(str(counts_path))
        print(f"Loaded prototypes: {prototypes.shape}")
    else:
        print("No prototypes — anchor guidance disabled.")

    annotator = PlaygroundAnnotator(model, model_args, npz_paths,
                                    prototypes=prototypes,
                                    prototypes_path=prototypes_path,
                                    prototype_counts=prototype_counts)

    demo = build_playground_interface(annotator)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
