"""
DPO Data Generation Program with GUI for Preference Annotation

This program generates paired trajectory samples from Diffusion Planner and allows
users to annotate which trajectory is preferred through a GUI interface.
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib
import numpy as np
import torch
from diffusion_planner.dimensions import *
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils.config import Config
from diffusion_planner.utils.visualize_input import visualize_inputs
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Use TkAgg backend for interactive GUI
matplotlib.use("TkAgg")

try:
    import tkinter as tk
    from tkinter import messagebox
except ImportError:
    print("Error: tkinter is not available. Please install python3-tk")
    exit(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--npz_list", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, default=Path("dpo_preferences.json"))
    parser.add_argument("--excluded_json", type=Path, default=Path("dpo_excluded.json"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume_from", type=Path, default=None)
    return parser.parse_args()


class DPODataGenerator:
    """Generates trajectory pairs and manages the annotation process."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # Set random seed once for the entire session
        seed = random.randint(0, 2**32 - 1)
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32))
        print(f"Random seed: {seed}")

        # Load model
        print(f"Loading model from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=self.device)

        # Load args from checkpoint using Config class
        model_dir = args.model_path.parent
        args_path = model_dir / "args.json"

        # Use Config class to load configuration (handles normalizers automatically)
        model_args = Config(str(args_path), guidance_fn=None)

        # Initialize model
        self.model = Diffusion_Planner(model_args)

        # Load checkpoint weights (following diffusion_planner_node.py)
        if "model" in checkpoint:
            # Handle DDP checkpoint
            state_dict = checkpoint["model"]
            # Remove 'module.' prefix if present
            state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items()
            }
            self.model.load_state_dict(state_dict, strict=False)
        elif "ema_state_dict" in checkpoint:
            print("Loading EMA weights")
            self.model.load_state_dict(checkpoint["ema_state_dict"], strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)

        self.model.to(self.device)
        self.model.eval()

        self.model_args = model_args

        # Load NPZ file list
        with open(args.npz_list, "r") as f:
            self.npz_paths = json.load(f)

        # Load existing annotations if resuming
        self.preferences = []
        self.excluded = []

        if args.resume_from and args.resume_from.exists():
            print(f"Resuming from {args.resume_from}")
            with open(args.resume_from, "r") as f:
                self.preferences = json.load(f)

        if args.excluded_json.exists():
            print(f"Loading excluded list from {args.excluded_json}")
            with open(args.excluded_json, "r") as f:
                self.excluded = json.load(f)

        # Filter out already annotated and excluded files
        annotated_paths = {pref["npz_path"] for pref in self.preferences}
        excluded_paths = set(self.excluded)
        self.npz_paths = [
            p for p in self.npz_paths
            if p not in annotated_paths and p not in excluded_paths
        ]

        print(f"Total NPZ files to annotate: {len(self.npz_paths)}")

    def load_npz_data(self, npz_path: str | Path) -> dict[str, torch.Tensor]:
        """Load and preprocess NPZ file."""
        loaded = np.load(str(npz_path))
        data = {}

        for key, value in loaded.items():
            if key == "map_name" or key == "token":
                continue
            # Add batch dimension
            data[key] = torch.tensor(np.expand_dims(value, axis=0)).to(self.device)

        # Convert heading to cos/sin
        if "goal_pose" in data:
            data["goal_pose"] = heading_to_cos_sin(data["goal_pose"])
        if "ego_agent_past" in data:
            data["ego_agent_past"] = heading_to_cos_sin(data["ego_agent_past"])

        return data

    @torch.no_grad()
    def generate_trajectory_pair(
        self, data: dict[str, torch.Tensor]
    ) -> tuple[list, list]:
        """
        Generate two different trajectories with different random noise.

        Returns:
            tuple: (trajectory_1, trajectory_2)
                Each is a list of shape [T, 4] representing the predicted trajectory
        """
        # Add ego_shape if not present (following diffusion_planner_node.py:365-367)
        if "ego_shape" not in data:
            # Default values for ego vehicle shape
            wheel_base = 2.79
            ego_length = 4.34
            ego_width = 1.70
            data["ego_shape"] = torch.tensor(
                [[wheel_base, ego_length, ego_width]],
                dtype=torch.float32,
                device=self.device
            )

        # Normalize inputs once
        data = self.model_args.observation_normalizer(data)

        # Generate trajectory parameters
        B = data["ego_current_state"].shape[0]
        P = 1 + self.model_args.predicted_neighbor_num
        future_len = self.model_args.future_len

        trajectories = []

        for seed_idx in range(2):
            # Generate random noise as initial state
            # Different noise will be generated automatically for each iteration
            data["sampled_trajectories"] = 2.5 * torch.randn(B, P, future_len + 1, 4).to(self.device)

            # Run inference
            _, outputs = self.model(data)

            # Extract ego prediction
            prediction = outputs["prediction"]  # [B, P, T, 4]
            ego_prediction = prediction[0, 0].cpu().numpy()  # [T, 4]

            trajectories.append(ego_prediction.tolist())

        return trajectories[0], trajectories[1]

    def save_preferences(self):
        """Save current preferences and excluded list to JSON."""
        with open(self.args.output_json, "w") as f:
            json.dump(self.preferences, f, indent=2)

        with open(self.args.excluded_json, "w") as f:
            json.dump(self.excluded, f, indent=2)

        print(f"\nSaved {len(self.preferences)} preferences to {self.args.output_json}")
        print(f"Saved {len(self.excluded)} excluded samples to {self.args.excluded_json}")


class AnnotationGUI:
    """GUI for annotating trajectory preferences."""

    def __init__(self, generator: DPODataGenerator):
        self.generator = generator
        self.current_index = 0
        self.current_data = None
        self.trajectory_1 = None
        self.trajectory_2 = None

        # Create main window
        self.root = tk.Tk()
        self.root.title("Trajectory Annotation")
        self.root.geometry("1400x1100")

        # Create UI elements
        self.create_ui()

        # Load first sample
        if len(self.generator.npz_paths) > 0:
            self.load_next_sample()
        else:
            messagebox.showinfo("Complete", "No more samples to annotate!")
            self.root.destroy()

    def create_ui(self):
        """Create UI elements."""
        # Top info panel
        info_frame = tk.Frame(self.root)
        info_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.info_label = tk.Label(
            info_frame,
            text="Loading...",
            font=("Arial", 12),
            anchor="w",
        )
        self.info_label.pack(side=tk.LEFT)

        # Visualization frame
        viz_frame = tk.Frame(self.root)
        viz_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create matplotlib figure (single plot)
        self.fig = Figure(figsize=(14, 9))
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Button frame
        button_frame = tk.Frame(self.root, height=200)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=20)
        button_frame.pack_propagate(False)

        # Create buttons
        buttons = [
            ("Trajectory 1 (Green) is Better", self.select_trajectory_1, "green"),
            ("Trajectory 2 (Orange) is Better", self.select_trajectory_2, "orange"),
            ("判断不能 (Cannot Judge)", self.select_cannot_judge, "blue"),
            ("Exclude this Sample", self.exclude_sample, "red"),
            ("Save & Quit", self.save_and_quit, "gray"),
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

    def load_next_sample(self):
        """Load next NPZ sample and generate trajectory pair."""
        if self.current_index >= len(self.generator.npz_paths):
            messagebox.showinfo("Complete", f"Annotation complete! Annotated {len(self.generator.preferences)} samples.")
            self.save_and_quit()
            return

        npz_path = self.generator.npz_paths[self.current_index]

        # Update info label
        self.info_label.config(
            text=f"Sample {self.current_index + 1}/{len(self.generator.npz_paths)} - {npz_path}"
        )

        try:
            # Load data
            self.current_data = self.generator.load_npz_data(npz_path)

            # Generate trajectory pair
            self.trajectory_1, self.trajectory_2 = self.generator.generate_trajectory_pair(
                self.current_data
            )

            # Visualize
            self.visualize_trajectories()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load sample:\n{str(e)}")
            print(f"Error loading {npz_path}: {e}")
            # Skip this sample
            self.current_index += 1
            self.load_next_sample()

    def visualize_trajectories(self):
        """Visualize input and both trajectories."""
        self.fig.clear()

        # Create single plot: both trajectories for comparison
        ax = self.fig.add_subplot(1, 1, 1)

        # Convert trajectories to numpy for visualization
        traj_1_np = np.array(self.trajectory_1)  # [T, 4]
        traj_2_np = np.array(self.trajectory_2)

        # Plot: Both trajectories for comparison
        data_cpu = {k: v.cpu() for k, v in self.current_data.items()}
        visualize_inputs(data_cpu, save_path=None, ax=ax, view_ranges=[60])
        ax.plot(traj_1_np[:, 0], traj_1_np[:, 1], 'g-', linewidth=3, alpha=0.7, label="Trajectory 1 (Green)")
        ax.plot(traj_2_np[:, 0], traj_2_np[:, 1], color='orange', linewidth=3, alpha=0.7, label="Trajectory 2 (Orange)")
        ax.legend(loc='upper left')

        self.fig.tight_layout()
        self.canvas.draw()

    def select_trajectory_1(self):
        """User prefers trajectory 1."""
        self.record_preference("trajectory_1")

    def select_trajectory_2(self):
        """User prefers trajectory 2."""
        self.record_preference("trajectory_2")

    def select_cannot_judge(self):
        """User cannot judge which trajectory is better."""
        self.record_preference("cannot_judge")

    def exclude_sample(self):
        """Exclude this sample from DPO training."""
        npz_path = self.generator.npz_paths[self.current_index]
        self.generator.excluded.append(npz_path)
        print(f"Excluded: {npz_path}")

        # Auto-save
        self.generator.save_preferences()

        # Move to next sample
        self.current_index += 1
        self.load_next_sample()

    def record_preference(self, preference: str):
        """Record the user's preference and move to next sample."""
        npz_path = self.generator.npz_paths[self.current_index]

        preference_data = {
            "npz_path": npz_path,
            "trajectory_1": self.trajectory_1,
            "trajectory_2": self.trajectory_2,
            "preference": preference,  # "trajectory_1", "trajectory_2", or "equal"
        }

        self.generator.preferences.append(preference_data)
        print(f"Recorded preference: {preference} for {npz_path}")

        # Auto-save every 10 samples
        if len(self.generator.preferences) % 10 == 0:
            self.generator.save_preferences()

        # Move to next sample
        self.current_index += 1
        self.load_next_sample()

    def save_and_quit(self):
        """Save all preferences and quit."""
        self.generator.save_preferences()
        self.root.destroy()

    def run(self):
        """Start the GUI event loop."""
        self.root.mainloop()


def main():
    args = parse_args()

    # Create data generator
    generator = DPODataGenerator(args)

    # Create and run GUI
    gui = AnnotationGUI(generator)
    gui.run()


if __name__ == "__main__":
    main()
