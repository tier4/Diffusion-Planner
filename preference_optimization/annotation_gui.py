"""Gradio-based preference annotation UI for trajectory comparison."""

import json
import random
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from diffusion_planner.utils.visualize_input import visualize_inputs
from matplotlib.figure import Figure

from preference_optimization.utils import generate_trajectory_pair, load_npz_data


class PreferenceAnnotator:
    """Manages state and logic for trajectory preference annotation."""

    def __init__(self, policy_model, model_args, npz_paths: list[str], target_count: int):
        """Initialize the annotator.

        Args:
            policy_model: The diffusion planner model
            model_args: Model configuration arguments
            npz_paths: List of paths to NPZ observation files
            target_count: Target number of preferences to collect
        """
        self.policy_model = policy_model
        self.model_args = model_args
        self.device = next(policy_model.parameters()).device
        self.npz_paths = npz_paths
        self.target_count = target_count

        # Annotation state
        self.preferences: list[dict] = []
        self.current_index = 0
        self.current_data = None
        self.trajectory_1 = None
        self.trajectory_2 = None
        self.current_fde = 0.0
        self.current_attempts = 0
        self.annotation_complete = False

        # Set random seed for reproducibility
        seed = random.randint(0, 2**32 - 1)
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32))
        print(f"Annotation seed: {seed}")

    def load_sample(
        self, noise_scale: float, fde_threshold: float, max_retries: int
    ) -> tuple[Figure, Figure, str, str]:
        """Load current sample and generate trajectory pair.

        Args:
            noise_scale: Noise scale for trajectory generation
            fde_threshold: Minimum FDE threshold
            max_retries: Maximum retry attempts

        Returns:
            Tuple of (trajectory_plot, velocity_plot, fde_text, progress_text)
        """
        if self.annotation_complete:
            return None, None, "Annotation complete!", self._format_progress()

        if not self.npz_paths or self.current_index >= len(self.npz_paths):
            return None, None, "No samples available", "Complete"

        npz_path = self.npz_paths[self.current_index]
        self.current_data = load_npz_data(npz_path, self.device)

        # Generate trajectory pair
        traj_1, traj_2, fde, attempts = generate_trajectory_pair(
            self.policy_model,
            self.model_args,
            self.current_data,
            noise_scale=float(noise_scale),
            fde_threshold=float(fde_threshold),
            max_retries=int(max_retries),
            device=self.device,
        )

        self.trajectory_1 = traj_1.tolist()
        self.trajectory_2 = traj_2.tolist()
        self.current_fde = fde
        self.current_attempts = attempts

        # Create visualizations
        traj_plot = self._create_trajectory_plot()
        vel_plot = self._create_velocity_plot()

        # Create status text
        fde_text = f"FDE: {fde:.2f}m (Attempts: {attempts})"
        progress_text = self._format_progress()

        return traj_plot, vel_plot, fde_text, progress_text

    def regenerate(
        self, noise_scale: float, fde_threshold: float, max_retries: int
    ) -> tuple[Figure, Figure, str, str]:
        """Regenerate trajectory pair with current parameters.

        Args:
            noise_scale: Noise scale for trajectory generation
            fde_threshold: Minimum FDE threshold
            max_retries: Maximum retry attempts

        Returns:
            Tuple of (trajectory_plot, velocity_plot, fde_text, progress_text)
        """
        if self.current_data is None:
            return None, None, "No data loaded", self._format_progress()

        # Generate new pair
        traj_1, traj_2, fde, attempts = generate_trajectory_pair(
            self.policy_model,
            self.model_args,
            self.current_data,
            noise_scale=float(noise_scale),
            fde_threshold=float(fde_threshold),
            max_retries=int(max_retries),
            device=self.device,
        )

        self.trajectory_1 = traj_1.tolist()
        self.trajectory_2 = traj_2.tolist()
        self.current_fde = fde
        self.current_attempts = attempts

        traj_plot = self._create_trajectory_plot()
        vel_plot = self._create_velocity_plot()
        fde_text = f"FDE: {fde:.2f}m (Attempts: {attempts})"
        progress_text = self._format_progress()

        print(f"Regenerated pair. FDE: {fde:.2f}m, Attempts: {attempts}")

        return traj_plot, vel_plot, fde_text, progress_text

    def select_winner(
        self, winner: str, noise_scale: float, fde_threshold: float, max_retries: int
    ) -> tuple[Figure | None, Figure | None, str, str]:
        """Record preference and move to next sample.

        Args:
            winner: Either "trajectory_1" or "trajectory_2"
            noise_scale: Noise scale for next sample
            fde_threshold: Minimum FDE threshold
            max_retries: Maximum retry attempts

        Returns:
            Tuple of (trajectory_plot, velocity_plot, fde_text, progress_text)
        """
        if self.current_data is None:
            return None, None, "No data loaded", "Error"

        npz_path = self.npz_paths[self.current_index]

        # Determine winner and loser
        if winner == "trajectory_1":
            traj_w, traj_l = self.trajectory_1, self.trajectory_2
        else:
            traj_w, traj_l = self.trajectory_2, self.trajectory_1

        # Record preference
        self.preferences.append(
            {"npz_path": npz_path, "trajectory_w": traj_w, "trajectory_l": traj_l}
        )

        print(f"Recorded preference for {npz_path} (Winner: {winner})")

        # Check if complete
        if len(self.preferences) >= self.target_count:
            self.annotation_complete = True
            complete_msg = (
                f"✅ Annotation complete!\n\n"
                f"Collected {len(self.preferences)} preferences.\n\n"
                f"The server will close automatically in 3 seconds.\n"
                f"Training will start automatically."
            )
            print(f"\n{'='*60}")
            print("ANNOTATION COMPLETE!")
            print(f"Collected {len(self.preferences)} preferences")
            print(f"{'='*60}\n")

            # Schedule server shutdown
            import threading
            def shutdown_server():
                import time
                time.sleep(3)
                if hasattr(self, '_demo') and self._demo is not None:
                    print("Closing Gradio server...")
                    self._demo.close()

            threading.Thread(target=shutdown_server, daemon=True).start()

            return None, None, complete_msg, f"Complete: {len(self.preferences)}/{self.target_count}"

        # Move to next sample
        self.current_index = (self.current_index + 1) % len(self.npz_paths)
        return self.load_sample(noise_scale, fde_threshold, max_retries)

    def jump(
        self, delta: int, noise_scale: float, fde_threshold: float, max_retries: int
    ) -> tuple[Figure | None, Figure | None, str, str]:
        """Jump to a different sample.

        Args:
            delta: Number of samples to jump (positive or negative)
            noise_scale: Noise scale for trajectory generation
            fde_threshold: Minimum FDE threshold
            max_retries: Maximum retry attempts

        Returns:
            Tuple of (trajectory_plot, velocity_plot, fde_text, progress_text)
        """
        if not self.npz_paths:
            return None, None, "No samples available", "Error"

        # Ensure delta is integer
        delta = int(delta)
        self.current_index = max(0, min(self.current_index + delta, len(self.npz_paths) - 1))
        return self.load_sample(noise_scale, fde_threshold, max_retries)

    def _format_progress(self) -> str:
        """Format progress text."""
        if not self.npz_paths:
            return "No samples"

        npz_path = self.npz_paths[self.current_index] if self.current_index < len(self.npz_paths) else ""
        return (
            f"Sample: {self.current_index + 1}/{len(self.npz_paths)}\n"
            f"File: {npz_path}\n"
            f"Preferences collected: {len(self.preferences)}/{self.target_count}"
        )

    def _create_trajectory_plot(self) -> Figure:
        """Create trajectory visualization plot."""
        fig = Figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        traj_1_np = np.array(self.trajectory_1)
        traj_2_np = np.array(self.trajectory_2)

        # Visualize map and context
        data_cpu = {k: v.cpu() for k, v in self.current_data.items()}
        visualize_inputs(data_cpu, save_path=None, ax=ax, view_ranges=[60])

        # Plot trajectories
        ax.plot(
            traj_1_np[:, 0],
            traj_1_np[:, 1],
            "g-",
            linewidth=3,
            alpha=0.7,
            label="Trajectory 1 (Deterministic)",
        )
        ax.plot(
            traj_2_np[:, 0],
            traj_2_np[:, 1],
            color="orange",
            linewidth=3,
            alpha=0.7,
            label="Trajectory 2 (Stochastic)",
        )
        ax.legend(loc="upper left")
        ax.set_title("Trajectory Comparison")

        return fig

    def _create_velocity_plot(self) -> Figure:
        """Create velocity comparison plot."""
        fig = Figure(figsize=(6, 8))
        ax = fig.add_subplot(111)

        vel_1 = self._calculate_velocities(self.trajectory_1)
        vel_2 = self._calculate_velocities(self.trajectory_2)

        time_steps = np.arange(len(vel_1))
        ax.plot(time_steps, vel_1, "g-", linewidth=2, alpha=0.7, label="Trajectory 1")
        ax.plot(time_steps, vel_2, color="orange", linewidth=2, alpha=0.7, label="Trajectory 2")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Velocity (km/h)")
        ax.set_ylim(0, 60)
        ax.set_title("Velocity Comparison")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        return fig

    def _calculate_velocities(self, trajectory: list) -> np.ndarray:
        """Calculate velocities from trajectory points.

        Assumes 1 time step = 0.1 seconds.

        Args:
            trajectory: List of trajectory points [[x, y, heading, velocity], ...]

        Returns:
            Array of velocities in km/h
        """
        traj_np = np.array(trajectory)
        ego_state = self.current_data["ego_current_state"].cpu().numpy()[0]

        # Include initial ego position
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


def create_interface(
    policy_model, model_args, npz_list: Path, target_count: int
) -> tuple[gr.Blocks, PreferenceAnnotator]:
    """Create Gradio interface for preference annotation.

    Args:
        policy_model: The diffusion planner model
        model_args: Model configuration arguments
        npz_list: Path to JSON file containing list of NPZ paths
        target_count: Target number of preferences to collect

    Returns:
        Tuple of (gradio_interface, annotator_instance)
    """
    with open(npz_list, "r") as f:
        npz_paths = json.load(f)

    annotator = PreferenceAnnotator(policy_model, model_args, npz_paths, target_count)

    with gr.Blocks(title="Trajectory Preference Annotation") as demo:
        gr.Markdown("# 🚗 Trajectory Preference Annotation")
        gr.Markdown(
            "**Instructions:** Compare two trajectories and select which one is better. "
            "Consider safety, comfort, and efficiency."
        )

        # Status display
        with gr.Row():
            progress_text = gr.Textbox(
                label="📊 Progress",
                value="Loading...",
                interactive=False,
                scale=2,
                lines=3,
            )
            fde_text = gr.Textbox(
                label="📏 Final Displacement Error",
                value="FDE: 0.00m",
                interactive=False,
                scale=1,
                lines=3,
            )

        # Parameter controls
        gr.Markdown("## ⚙️ Generation Parameters")
        gr.Markdown(
            "**Noise Scale:** Controls diversity of the stochastic trajectory\n\n"
            "**FDE Threshold:** Minimum distance between trajectory endpoints (meters)\n\n"
            "**Max Retries:** Maximum attempts to meet FDE threshold"
        )
        with gr.Row():
            noise_scale = gr.Slider(
                minimum=0.5,
                maximum=5.0,
                value=2.5,
                step=0.1,
                label="Noise Scale",
            )
            fde_threshold = gr.Slider(
                minimum=0.5,
                maximum=10.0,
                value=2.0,
                step=0.1,
                label="FDE Threshold (m)",
            )
            max_retries = gr.Slider(
                minimum=10,
                maximum=200,
                value=50,
                step=10,
                label="Max Retries",
            )

        # Visualizations
        gr.Markdown("## 🎨 Trajectory Visualization")
        with gr.Row():
            trajectory_plot = gr.Plot(label="Trajectory Comparison")
            velocity_plot = gr.Plot(label="Velocity Comparison")

        # Action buttons
        gr.Markdown("## ✅ Select Preferred Trajectory")
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
        gr.Markdown("## 🧭 Navigation (Jump between samples)")
        with gr.Row():
            prev_30_btn = gr.Button("← 30", size="sm")
            prev_10_btn = gr.Button("← 10", size="sm")
            prev_1_btn = gr.Button("← 1", size="sm")
            next_1_btn = gr.Button("1 →", size="sm")
            next_10_btn = gr.Button("10 →", size="sm")
            next_30_btn = gr.Button("30 →", size="sm")

        # Event handlers
        select_1_btn.click(
            fn=lambda ns, ft, mr: annotator.select_winner("trajectory_1", ns, ft, mr),
            inputs=[noise_scale, fde_threshold, max_retries],
            outputs=[trajectory_plot, velocity_plot, fde_text, progress_text],
        )

        select_2_btn.click(
            fn=lambda ns, ft, mr: annotator.select_winner("trajectory_2", ns, ft, mr),
            inputs=[noise_scale, fde_threshold, max_retries],
            outputs=[trajectory_plot, velocity_plot, fde_text, progress_text],
        )

        regenerate_btn.click(
            fn=lambda ns, ft, mr: annotator.regenerate(ns, ft, mr),
            inputs=[noise_scale, fde_threshold, max_retries],
            outputs=[trajectory_plot, velocity_plot, fde_text, progress_text],
        )

        # Navigation handlers - fix lambda closure issue
        def make_jump_fn(delta_val):
            return lambda ns, ft, mr: annotator.jump(delta_val, ns, ft, mr)

        prev_30_btn.click(
            fn=make_jump_fn(-30),
            inputs=[noise_scale, fde_threshold, max_retries],
            outputs=[trajectory_plot, velocity_plot, fde_text, progress_text],
        )
        prev_10_btn.click(
            fn=make_jump_fn(-10),
            inputs=[noise_scale, fde_threshold, max_retries],
            outputs=[trajectory_plot, velocity_plot, fde_text, progress_text],
        )
        prev_1_btn.click(
            fn=make_jump_fn(-1),
            inputs=[noise_scale, fde_threshold, max_retries],
            outputs=[trajectory_plot, velocity_plot, fde_text, progress_text],
        )
        next_1_btn.click(
            fn=make_jump_fn(1),
            inputs=[noise_scale, fde_threshold, max_retries],
            outputs=[trajectory_plot, velocity_plot, fde_text, progress_text],
        )
        next_10_btn.click(
            fn=make_jump_fn(10),
            inputs=[noise_scale, fde_threshold, max_retries],
            outputs=[trajectory_plot, velocity_plot, fde_text, progress_text],
        )
        next_30_btn.click(
            fn=make_jump_fn(30),
            inputs=[noise_scale, fde_threshold, max_retries],
            outputs=[trajectory_plot, velocity_plot, fde_text, progress_text],
        )

        # Load first sample on startup
        demo.load(
            fn=lambda ns, ft, mr: annotator.load_sample(ns, ft, mr),
            inputs=[noise_scale, fde_threshold, max_retries],
            outputs=[trajectory_plot, velocity_plot, fde_text, progress_text],
        )

    return demo, annotator


def collect_preferences(
    policy_model, model_args, npz_list: Path, target_count: int
) -> list[dict]:
    """Collect trajectory preferences via Gradio GUI.

    This function launches a web-based interface for annotating trajectory preferences.
    The interface will block until annotation is complete.

    Args:
        policy_model: The diffusion planner model
        model_args: Model configuration arguments
        npz_list: Path to JSON file containing list of NPZ observation files
        target_count: Target number of preferences to collect

    Returns:
        List of preference dictionaries
    """
    import time

    was_training = policy_model.training
    policy_model.eval()

    demo, annotator = create_interface(policy_model, model_args, npz_list, target_count)

    # Store demo reference in annotator for shutdown
    annotator._demo = demo

    print("Launching Gradio interface...")
    print("Complete the annotation in your browser.")
    print("The server will close automatically when annotation is complete.")
    print("(Or press Ctrl+C to interrupt)")

    try:
        # Launch without blocking
        demo.launch(
            share=False,
            inbrowser=True,
            prevent_thread_lock=True,  # Don't block
            show_error=True,
        )

        # Poll for completion
        print("\nWaiting for annotation to complete...")
        while not annotator.annotation_complete:
            time.sleep(1)

        # Wait a bit for the final message to display
        time.sleep(3)

    except KeyboardInterrupt:
        print("\nAnnotation interrupted by user.")
    finally:
        try:
            demo.close()
            print("Gradio server closed.")
        except:
            pass

    preferences = annotator.preferences

    if was_training:
        policy_model.train()

    print(f"Collected {len(preferences)} preferences.")
    return preferences
