"""
GUI-based preference annotation utilities for DPO training.
"""

import json
import random
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
import torch
from diffusion_planner.utils.visualize_input import visualize_inputs
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from utils import generate_trajectory_pair, load_npz_data

matplotlib.use("TkAgg")

import tkinter as tk
from tkinter import messagebox


class AnnotationGUI:
    """Tkinter GUI for annotating trajectory preferences."""

    def __init__(
        self,
        policy_model,
        model_args,
        npz_paths: Sequence[str],
        target_count: int,
        noise_scale: float = 2.5,
        fde_threshold: float = 2.0,
        max_retries: int = 50,
    ):
        self.policy_model = policy_model
        self.model_args = model_args
        self.device = next(policy_model.parameters()).device
        self.npz_paths = list(npz_paths)
        self.preferences: list[dict] = []
        self.target_count = target_count
        self.noise_scale = noise_scale
        self.fde_threshold = fde_threshold
        self.max_retries = max_retries
        self.current_fde = 0.0
        self.current_attempts = 0

        seed = random.randint(0, 2**32 - 1)
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32))
        print(f"GUI annotation seed: {seed}")

        self.current_index = 0
        self.current_data = None
        self.trajectory_1 = None
        self.trajectory_2 = None

        self.root = tk.Tk()
        self.root.title("Trajectory Annotation")
        self.root.geometry("1400x1200")
        self.root.minsize(1400, 1200)

        self._build_ui()

        if self.npz_paths:
            self._show_current_sample()
        else:
            messagebox.showinfo("Complete", "No samples to annotate!")
            self.root.destroy()

    def _build_ui(self):
        info_frame = tk.Frame(self.root)
        info_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.info_label = tk.Label(info_frame, text="Loading...", font=("Arial", 12), anchor="w")
        self.info_label.pack(side=tk.LEFT)

        viz_frame = tk.Frame(self.root)
        viz_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig = Figure(figsize=(14, 9))
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.pref_label = tk.Label(
            info_frame, text="Preferences: 0", font=("Arial", 12), anchor="e"
        )
        self.pref_label.pack(side=tk.RIGHT)

        # Parameter controls frame
        param_frame = tk.Frame(self.root)
        param_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Noise scale
        tk.Label(param_frame, text="Noise Scale:", font=("Arial", 10)).grid(
            row=0, column=0, padx=5
        )
        self.noise_scale_var = tk.DoubleVar(value=self.noise_scale)
        tk.Scale(
            param_frame,
            from_=0.5,
            to=5.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.noise_scale_var,
            length=200,
        ).grid(row=0, column=1, padx=5)

        # FDE threshold
        tk.Label(param_frame, text="FDE Threshold:", font=("Arial", 10)).grid(
            row=1, column=0, padx=5
        )
        self.fde_threshold_var = tk.DoubleVar(value=self.fde_threshold)
        tk.Scale(
            param_frame,
            from_=0.5,
            to=10.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.fde_threshold_var,
            length=200,
        ).grid(row=1, column=1, padx=5)

        # Max retries
        tk.Label(param_frame, text="Max Retries:", font=("Arial", 10)).grid(
            row=2, column=0, padx=5
        )
        self.max_retries_var = tk.IntVar(value=self.max_retries)
        tk.Scale(
            param_frame,
            from_=10,
            to=200,
            resolution=10,
            orient=tk.HORIZONTAL,
            variable=self.max_retries_var,
            length=200,
        ).grid(row=2, column=1, padx=5)

        # FDE display
        self.fde_label = tk.Label(
            param_frame,
            text=f"Current FDE: 0.00 (Attempts: 0)",
            font=("Arial", 11, "bold"),
            fg="blue",
        )
        self.fde_label.grid(row=3, column=0, columnspan=2, pady=5)

        control_frame = tk.Frame(self.root, height=120)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        control_frame.pack_propagate(False)

        back_buttons = tk.Frame(control_frame)
        back_buttons.pack(side=tk.LEFT, padx=10)

        for step in [1, 10, 30]:
            tk.Button(
                back_buttons,
                text=f"← {step}",
                command=lambda s=step: self._jump(-s),
                width=8,
            ).pack(side=tk.LEFT, padx=5)

        forward_buttons = tk.Frame(control_frame)
        forward_buttons.pack(side=tk.RIGHT, padx=10)
        for step in [1, 10, 30]:
            tk.Button(
                forward_buttons,
                text=f"{step} →",
                command=lambda s=step: self._jump(s),
                width=8,
            ).pack(side=tk.LEFT, padx=5)

        button_frame = tk.Frame(self.root, height=240)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=20)
        button_frame.pack_propagate(False)

        buttons = [
            ("Trajectory 1 (Green) is Better", self._select_1, "green"),
            ("Trajectory 2 (Orange) is Better", self._select_2, "orange"),
            ("Regenerate Pair", self._regenerate_pair, "purple"),
        ]
        for text, command, color in buttons:
            btn = tk.Button(
                button_frame,
                text=text,
                command=command,
                font=("Arial", 14, "bold"),
                bg=color,
                fg="white",
                activebackground=color,
                activeforeground="white",
                width=25,
                height=3,
            )
            btn.pack(side=tk.LEFT, expand=True, padx=5)

    def _load_next(self):
        self.current_index = min(self.current_index + 1, len(self.npz_paths) - 1)
        self._show_current_sample()

    def _calculate_velocities(
        self, trajectory: list, ego_current_state: torch.Tensor
    ) -> np.ndarray:
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

    def _visualize(self):
        self.fig.clear()
        gs = self.fig.add_gridspec(1, 2, width_ratios=[2, 1])
        ax_traj = self.fig.add_subplot(gs[0])
        ax_vel = self.fig.add_subplot(gs[1])

        traj_1_np = np.array(self.trajectory_1)
        traj_2_np = np.array(self.trajectory_2)

        data_cpu = {k: v.cpu() for k, v in self.current_data.items()}
        visualize_inputs(data_cpu, save_path=None, ax=ax_traj, view_ranges=[60])
        ax_traj.plot(
            traj_1_np[:, 0],
            traj_1_np[:, 1],
            "g-",
            linewidth=3,
            alpha=0.7,
            label="Trajectory 1 (Temp=0)",
        )
        ax_traj.plot(
            traj_2_np[:, 0],
            traj_2_np[:, 1],
            color="orange",
            linewidth=3,
            alpha=0.7,
            label="Trajectory 2",
        )
        ax_traj.legend(loc="upper left")
        ax_traj.set_title("Trajectories")

        vel_1 = self._calculate_velocities(
            self.trajectory_1, self.current_data["ego_current_state"]
        )
        vel_2 = self._calculate_velocities(
            self.trajectory_2, self.current_data["ego_current_state"]
        )

        time_steps = np.arange(len(vel_1))
        ax_vel.plot(time_steps, vel_1, "g-", linewidth=2, alpha=0.7, label="Trajectory 1 Velocity")
        ax_vel.plot(
            time_steps, vel_2, color="orange", linewidth=2, alpha=0.7, label="Trajectory 2 Velocity"
        )
        ax_vel.set_xlabel("Time Step")
        ax_vel.set_ylabel("Velocity (km/h)")
        ax_vel.set_ylim(0, 60)
        ax_vel.set_title("Velocity Comparison")
        ax_vel.legend(loc="upper right")
        ax_vel.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw()

    def _select_1(self):
        self._record("trajectory_1")

    def _select_2(self):
        self._record("trajectory_2")

    def _record(self, winner: str):
        npz_path = self.npz_paths[self.current_index]
        if winner == "trajectory_1":
            traj_w, traj_l = self.trajectory_1, self.trajectory_2
        else:
            traj_w, traj_l = self.trajectory_2, self.trajectory_1

        self.preferences.append(
            {
                "npz_path": npz_path,
                "trajectory_w": traj_w,
                "trajectory_l": traj_l,
            }
        )
        print(f"Recorded preference for {npz_path}")

        if len(self.preferences) >= self.target_count:
            messagebox.showinfo(
                "Complete", f"Annotation complete! Collected {len(self.preferences)} samples."
            )
            self.root.destroy()
            return

        self.current_index = (self.current_index + 1) % len(self.npz_paths)
        self._show_current_sample()

    def _regenerate_pair(self, update_index: bool = True):
        if self.current_data is None:
            return

        # Get current parameter values
        noise_scale = self.noise_scale_var.get()
        fde_threshold = self.fde_threshold_var.get()
        max_retries = self.max_retries_var.get()

        # Import the new function
        from utils import generate_trajectory_pair_with_retry

        traj_1, traj_2, fde, attempts = generate_trajectory_pair_with_retry(
            self.policy_model,
            self.model_args,
            self.current_data,
            noise_scale=noise_scale,
            fde_threshold=fde_threshold,
            max_retries=max_retries,
            device=self.device,
        )

        self.trajectory_1 = traj_1.tolist()
        self.trajectory_2 = traj_2.tolist()
        self.current_fde = fde
        self.current_attempts = attempts

        # Update FDE display
        self.fde_label.config(text=f"Current FDE: {fde:.2f}m (Attempts: {attempts})")

        self._visualize()
        if update_index:
            print(f"Regenerated trajectory pair. FDE: {fde:.2f}m, Attempts: {attempts}")

    def run(self):
        self.root.mainloop()

    def _update_status(self, npz_path: str | None = None):
        total = max(len(self.npz_paths), 1)
        current = min(self.current_index + 1, total)
        path = npz_path or (self.npz_paths[self.current_index] if self.npz_paths else "")
        self.info_label.config(text=f"Sample {current}/{total} - {path}")
        self.pref_label.config(text=f"Preferences: {len(self.preferences)}")

    def _jump(self, delta: int):
        if not self.npz_paths:
            return
        self.current_index = max(0, min(self.current_index + delta, len(self.npz_paths) - 1))
        self._show_current_sample()

    def _show_current_sample(self):
        if not self.npz_paths:
            messagebox.showinfo("Complete", "No samples to annotate!")
            self.root.destroy()
            return

        self.current_index = max(0, min(self.current_index, len(self.npz_paths) - 1))
        npz_path = self.npz_paths[self.current_index]
        self._update_status(npz_path)

        try:
            self.current_data = load_npz_data(npz_path, self.device)
            self._regenerate_pair(update_index=False)
        except Exception as exc:  # pragma: no cover - GUI path
            messagebox.showerror("Error", f"Failed to load sample:\n{str(exc)}")
            print(f"Error loading {npz_path}: {exc}")
            self.current_index = min(self.current_index + 1, len(self.npz_paths) - 1)
            self._show_current_sample()


def collect_preferences_gui(
    policy_model,
    model_args,
    npz_list: Path,
    target_count: int,
    noise_scale: float = 2.5,
    fde_threshold: float = 2.0,
    max_retries: int = 50,
) -> list[dict]:
    """Run GUI preference collection and return annotations."""
    with open(npz_list, "r") as f:
        npz_paths = json.load(f)

    was_training = policy_model.training
    policy_model.eval()

    gui = AnnotationGUI(
        policy_model,
        model_args,
        npz_paths,
        target_count=target_count,
        noise_scale=noise_scale,
        fde_threshold=fde_threshold,
        max_retries=max_retries,
    )
    gui.run()
    prefs = gui.preferences

    if was_training:
        policy_model.train()

    return prefs
