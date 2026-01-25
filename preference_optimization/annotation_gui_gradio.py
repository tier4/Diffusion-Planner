"""
Gradio-based preference annotation UI for DPO training.
Browser-compatible interface for trajectory preference annotation.
"""

import json
import random
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusion_planner.utils.visualize_input import visualize_inputs
from matplotlib.figure import Figure
from utils import generate_trajectory_pair_with_retry, load_npz_data


class AnnotationState:
    """State management for Gradio interface."""

    def __init__(self, policy_model, model_args, npz_paths, target_count):
        self.policy_model = policy_model
        self.model_args = model_args
        self.device = next(policy_model.parameters()).device
        self.npz_paths = list(npz_paths)
        self.target_count = target_count
        self.preferences = []
        self.current_index = 0
        self.current_data = None
        self.trajectory_1 = None
        self.trajectory_2 = None
        self.current_fde = 0.0
        self.current_attempts = 0

        seed = random.randint(0, 2**32 - 1)
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32))
        print(f"Gradio annotation seed: {seed}")


def calculate_velocities(trajectory: list, ego_current_state: torch.Tensor) -> np.ndarray:
    """Calculate velocities in km/h from trajectory points and initial ego state.

    Assumes 1 time step = 0.1 seconds.
    """
    traj_np = np.array(trajectory)
    ego_state = ego_current_state.cpu().numpy()[0]

    positions = np.vstack([ego_state[:2], traj_np[:, :2]])
    velocities = []

    for i in range(len(positions) - 1):
        dx = positions[i + 1, 0] - positions[i, 0]
        dy = positions[i + 1, 1] - positions[i, 1]
        # Distance in meters per 0.1 second -> m/s -> km/h
        velocity_m_per_step = np.sqrt(dx**2 + dy**2)
        velocity_m_per_s = velocity_m_per_step / 0.1
        velocity_km_per_h = velocity_m_per_s * 3.6
        velocities.append(velocity_km_per_h)

    return np.array(velocities)


def create_trajectory_plot(state: AnnotationState) -> Figure:
    """Create trajectory visualization plot."""
    fig = Figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    traj_1_np = np.array(state.trajectory_1)
    traj_2_np = np.array(state.trajectory_2)

    data_cpu = {k: v.cpu() for k, v in state.current_data.items()}
    visualize_inputs(data_cpu, save_path=None, ax=ax, view_ranges=[60])

    ax.plot(
        traj_1_np[:, 0],
        traj_1_np[:, 1],
        "g-",
        linewidth=3,
        alpha=0.7,
        label="Trajectory 1 (Temp=0)",
    )
    ax.plot(
        traj_2_np[:, 0],
        traj_2_np[:, 1],
        color="orange",
        linewidth=3,
        alpha=0.7,
        label="Trajectory 2",
    )
    ax.legend(loc="upper left")
    ax.set_title("Trajectory Comparison")

    return fig


def create_velocity_plot(state: AnnotationState) -> Figure:
    """Create velocity comparison plot."""
    fig = Figure(figsize=(6, 8))
    ax = fig.add_subplot(111)

    vel_1 = calculate_velocities(state.trajectory_1, state.current_data["ego_current_state"])
    vel_2 = calculate_velocities(state.trajectory_2, state.current_data["ego_current_state"])

    time_steps = np.arange(len(vel_1))
    ax.plot(time_steps, vel_1, "g-", linewidth=2, alpha=0.7, label="Trajectory 1 Velocity")
    ax.plot(time_steps, vel_2, color="orange", linewidth=2, alpha=0.7, label="Trajectory 2 Velocity")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Velocity (km/h)")
    ax.set_ylim(0, 60)
    ax.set_title("Velocity Comparison")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    return fig


def load_and_generate(
    state: AnnotationState, noise_scale: float, fde_threshold: float, max_retries: int
):
    """Load current sample and generate trajectory pair."""
    if not state.npz_paths or state.current_index >= len(state.npz_paths):
        return None, None, "No samples available", "Complete"

    npz_path = state.npz_paths[state.current_index]
    state.current_data = load_npz_data(npz_path, state.device)

    traj_1, traj_2, fde, attempts = generate_trajectory_pair_with_retry(
        state.policy_model,
        state.model_args,
        state.current_data,
        noise_scale=noise_scale,
        fde_threshold=fde_threshold,
        max_retries=int(max_retries),
        device=state.device,
    )

    state.trajectory_1 = traj_1.tolist()
    state.trajectory_2 = traj_2.tolist()
    state.current_fde = fde
    state.current_attempts = attempts

    traj_plot = create_trajectory_plot(state)
    vel_plot = create_velocity_plot(state)

    progress_text = (
        f"Sample {state.current_index + 1}/{len(state.npz_paths)} - {npz_path}\n"
        f"Preferences Collected: {len(state.preferences)}/{state.target_count}"
    )
    fde_text = f"FDE: {fde:.2f}m (Attempts: {attempts})"

    return traj_plot, vel_plot, fde_text, progress_text


def regenerate_pair(
    state: AnnotationState, noise_scale: float, fde_threshold: float, max_retries: int
):
    """Regenerate trajectory pair with current parameters."""
    if state.current_data is None:
        return None, None, "No data loaded", gr.update()

    traj_1, traj_2, fde, attempts = generate_trajectory_pair_with_retry(
        state.policy_model,
        state.model_args,
        state.current_data,
        noise_scale=noise_scale,
        fde_threshold=fde_threshold,
        max_retries=int(max_retries),
        device=state.device,
    )

    state.trajectory_1 = traj_1.tolist()
    state.trajectory_2 = traj_2.tolist()
    state.current_fde = fde
    state.current_attempts = attempts

    traj_plot = create_trajectory_plot(state)
    vel_plot = create_velocity_plot(state)
    fde_text = f"FDE: {fde:.2f}m (Attempts: {attempts})"

    print(f"Regenerated pair. FDE: {fde:.2f}m, Attempts: {attempts}")

    return traj_plot, vel_plot, fde_text, gr.update()


def select_trajectory(
    state: AnnotationState,
    winner: str,
    noise_scale: float,
    fde_threshold: float,
    max_retries: int,
):
    """Record preference and move to next sample."""
    if state.current_data is None:
        return None, None, "No data loaded", gr.update()

    npz_path = state.npz_paths[state.current_index]

    if winner == "trajectory_1":
        traj_w, traj_l = state.trajectory_1, state.trajectory_2
    else:
        traj_w, traj_l = state.trajectory_2, state.trajectory_1

    state.preferences.append(
        {
            "npz_path": npz_path,
            "trajectory_w": traj_w,
            "trajectory_l": traj_l,
        }
    )

    print(f"Recorded preference for {npz_path} (Winner: {winner})")

    # Check if annotation is complete
    if len(state.preferences) >= state.target_count:
        return (
            None,
            None,
            f"Annotation complete! Collected {len(state.preferences)} samples.",
            f"Complete: {len(state.preferences)}/{state.target_count}",
        )

    # Move to next sample
    state.current_index = (state.current_index + 1) % len(state.npz_paths)
    return load_and_generate(state, noise_scale, fde_threshold, max_retries)


def jump_samples(
    state: AnnotationState, delta: int, noise_scale: float, fde_threshold: float, max_retries: int
):
    """Jump forward or backward by delta samples."""
    if not state.npz_paths:
        return None, None, "No samples available", gr.update()

    state.current_index = max(0, min(state.current_index + delta, len(state.npz_paths) - 1))
    return load_and_generate(state, noise_scale, fde_threshold, max_retries)


def create_gradio_interface(
    policy_model, model_args, npz_list: Path, target_count: int
) -> tuple[gr.Blocks, AnnotationState]:
    """Create and return Gradio interface."""
    with open(npz_list, "r") as f:
        npz_paths = json.load(f)

    state = AnnotationState(policy_model, model_args, npz_paths, target_count)

    with gr.Blocks(title="Trajectory Preference Annotation", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Trajectory Preference Annotation System")
        gr.Markdown("Select which trajectory is better, or regenerate to get a new pair.")

        with gr.Row():
            progress_text = gr.Textbox(
                label="Progress",
                value="Click a button to start",
                interactive=False,
                scale=2,
            )
            fde_text = gr.Textbox(
                label="Current FDE",
                value="FDE: 0.00m (Attempts: 0)",
                interactive=False,
                scale=1,
            )

        # Parameters
        gr.Markdown("## Parameters")
        with gr.Row():
            noise_scale = gr.Slider(
                minimum=0.5,
                maximum=5.0,
                value=2.5,
                step=0.1,
                label="Noise Scale",
                info="Controls diversity of second trajectory",
            )
            fde_threshold = gr.Slider(
                minimum=0.5,
                maximum=10.0,
                value=2.0,
                step=0.1,
                label="FDE Threshold (m)",
                info="Minimum distance between trajectory endpoints",
            )
            max_retries = gr.Slider(
                minimum=10,
                maximum=200,
                value=50,
                step=10,
                label="Max Retries",
                info="Maximum attempts to meet FDE threshold",
            )

        # Visualizations
        gr.Markdown("## Trajectory Visualization")
        with gr.Row():
            trajectory_plot = gr.Plot(label="Trajectory Comparison")
            velocity_plot = gr.Plot(label="Velocity Comparison")

        # Action buttons
        gr.Markdown("## Select Preferred Trajectory")
        with gr.Row():
            select_1_btn = gr.Button(
                "✓ Trajectory 1 (Green) is Better",
                variant="primary",
                size="lg",
            )
            select_2_btn = gr.Button(
                "✓ Trajectory 2 (Orange) is Better",
                variant="primary",
                size="lg",
            )
            regenerate_btn = gr.Button(
                "🔄 Regenerate Pair",
                variant="secondary",
                size="lg",
            )

        # Navigation
        gr.Markdown("## Navigation")
        with gr.Row():
            prev_30_btn = gr.Button("← 30")
            prev_10_btn = gr.Button("← 10")
            prev_1_btn = gr.Button("← 1")
            next_1_btn = gr.Button("1 →")
            next_10_btn = gr.Button("10 →")
            next_30_btn = gr.Button("30 →")

        # Event handlers
        select_1_btn.click(
            fn=lambda ns, ft, mr: select_trajectory(state, "trajectory_1", ns, ft, mr),
            inputs=[noise_scale, fde_threshold, max_retries],
            outputs=[trajectory_plot, velocity_plot, fde_text, progress_text],
        )

        select_2_btn.click(
            fn=lambda ns, ft, mr: select_trajectory(state, "trajectory_2", ns, ft, mr),
            inputs=[noise_scale, fde_threshold, max_retries],
            outputs=[trajectory_plot, velocity_plot, fde_text, progress_text],
        )

        regenerate_btn.click(
            fn=lambda ns, ft, mr: regenerate_pair(state, ns, ft, mr),
            inputs=[noise_scale, fde_threshold, max_retries],
            outputs=[trajectory_plot, velocity_plot, fde_text, progress_text],
        )

        # Navigation handlers
        prev_30_btn.click(
            fn=lambda ns, ft, mr: jump_samples(state, -30, ns, ft, mr),
            inputs=[noise_scale, fde_threshold, max_retries],
            outputs=[trajectory_plot, velocity_plot, fde_text, progress_text],
        )

        prev_10_btn.click(
            fn=lambda ns, ft, mr: jump_samples(state, -10, ns, ft, mr),
            inputs=[noise_scale, fde_threshold, max_retries],
            outputs=[trajectory_plot, velocity_plot, fde_text, progress_text],
        )

        prev_1_btn.click(
            fn=lambda ns, ft, mr: jump_samples(state, -1, ns, ft, mr),
            inputs=[noise_scale, fde_threshold, max_retries],
            outputs=[trajectory_plot, velocity_plot, fde_text, progress_text],
        )

        next_1_btn.click(
            fn=lambda ns, ft, mr: jump_samples(state, 1, ns, ft, mr),
            inputs=[noise_scale, fde_threshold, max_retries],
            outputs=[trajectory_plot, velocity_plot, fde_text, progress_text],
        )

        next_10_btn.click(
            fn=lambda ns, ft, mr: jump_samples(state, 10, ns, ft, mr),
            inputs=[noise_scale, fde_threshold, max_retries],
            outputs=[trajectory_plot, velocity_plot, fde_text, progress_text],
        )

        next_30_btn.click(
            fn=lambda ns, ft, mr: jump_samples(state, 30, ns, ft, mr),
            inputs=[noise_scale, fde_threshold, max_retries],
            outputs=[trajectory_plot, velocity_plot, fde_text, progress_text],
        )

        # Load first sample on startup
        demo.load(
            fn=lambda ns, ft, mr: load_and_generate(state, ns, ft, mr),
            inputs=[noise_scale, fde_threshold, max_retries],
            outputs=[trajectory_plot, velocity_plot, fde_text, progress_text],
        )

    return demo, state


def collect_preferences_gui_gradio(
    policy_model,
    model_args,
    npz_list: Path,
    target_count: int,
    noise_scale: float = 2.5,
    fde_threshold: float = 2.0,
    max_retries: int = 50,
) -> list[dict]:
    """Run Gradio-based GUI preference collection and return annotations.

    Note: This function will block until the user completes annotation or closes the interface.
    """
    was_training = policy_model.training
    policy_model.eval()

    demo, state = create_gradio_interface(policy_model, model_args, npz_list, target_count)

    print("Starting Gradio interface...")
    print("The interface will open in your browser. Complete the annotation and close the tab when done.")

    # Launch with share=False for local use
    demo.launch(share=False, inbrowser=True)

    # When the interface is closed, return the collected preferences
    prefs = state.preferences

    if was_training:
        policy_model.train()

    return prefs
