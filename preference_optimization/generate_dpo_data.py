"""
GUI-based preference annotation utilities for DPO training.
"""

import json
import random
from pathlib import Path
from typing import Sequence

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

    def __init__(self, policy_model, model_args, npz_paths: Sequence[str]):
        self.policy_model = policy_model
        self.model_args = model_args
        self.device = next(policy_model.parameters()).device
        self.npz_paths = list(npz_paths)
        self.preferences: list[dict] = []

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
        self.root.geometry("1400x1100")

        self._build_ui()

        if self.npz_paths:
            self._load_next()
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

        button_frame = tk.Frame(self.root, height=200)
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
        if self.current_index >= len(self.npz_paths):
            messagebox.showinfo("Complete", "Annotation complete!")
            self.root.destroy()
            return

        npz_path = self.npz_paths[self.current_index]
        self.info_label.config(
            text=f"Sample {self.current_index + 1}/{len(self.npz_paths)} - {npz_path}"
        )

        try:
            self.current_data = load_npz_data(npz_path, self.device)
            self._regenerate_pair(update_index=False)
        except Exception as exc:  # pragma: no cover - GUI path
            messagebox.showerror("Error", f"Failed to load sample:\n{str(exc)}")
            print(f"Error loading {npz_path}: {exc}")
            self.current_index += 1
            self._load_next()

    def _visualize(self):
        self.fig.clear()
        ax = self.fig.add_subplot(1, 1, 1)

        traj_1_np = np.array(self.trajectory_1)
        traj_2_np = np.array(self.trajectory_2)

        data_cpu = {k: v.cpu() for k, v in self.current_data.items()}
        visualize_inputs(data_cpu, save_path=None, ax=ax, view_ranges=[60])
        ax.plot(traj_1_np[:, 0], traj_1_np[:, 1], "g-", linewidth=3, alpha=0.7, label="Trajectory 1")
        ax.plot(traj_2_np[:, 0], traj_2_np[:, 1], color="orange", linewidth=3, alpha=0.7, label="Trajectory 2")
        ax.legend(loc="upper left")

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

        self.current_index += 1
        self._load_next()

    def _regenerate_pair(self, update_index: bool = True):
        if self.current_data is None:
            return
        traj_1, traj_2 = generate_trajectory_pair(
            self.policy_model, self.model_args, self.current_data, device=self.device
        )
        self.trajectory_1 = traj_1.tolist()
        self.trajectory_2 = traj_2.tolist()
        self._visualize()
        if update_index:
            print("Regenerated trajectory pair for current sample.")

    def run(self):
        self.root.mainloop()


def collect_preferences_gui(policy_model, model_args, npz_list: Path) -> list[dict]:
    """Run GUI preference collection and return annotations."""
    with open(npz_list, "r") as f:
        npz_paths = json.load(f)

    was_training = policy_model.training
    policy_model.eval()

    gui = AnnotationGUI(policy_model, model_args, npz_paths)
    gui.run()
    prefs = gui.preferences

    if was_training:
        policy_model.train()

    return prefs
