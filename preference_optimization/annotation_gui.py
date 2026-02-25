"""Gradio-based preference annotation UI for trajectory comparison."""

import json
import random
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from diffusion_planner.utils.visualize_input import visualize_inputs
from matplotlib.figure import Figure

from preference_optimization.utils import (
    calculate_ade,
    generate_trajectory_pair,
    load_npz_data,
    should_prune_by_initial_pose,
)


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

        # Initial pose pruning state
        self.initial_displacement: float = 0.0
        self.initial_yaw_diff: float = 0.0
        self.is_pruned: bool = False
        self.enable_initial_pruning: bool = True

        # GT availability state
        self.gt_available: bool = False

        # Navigation and tracking state
        self.current_jump_size = 1  # Tracks the last navigation button click
        self.labeled_indices: set[int] = set()  # Tracks which samples have been labeled
        self.labeled_history: list[int] = []  # Tracks last 10 labeled sample indices (most recent first)
        self.auto_skip_labeled = False  # Toggle for auto-skipping labeled samples
        self.original_npz_paths = npz_paths  # Keep original list for filtering
        self.current_filter = "All"  # Current filter: "All", "Finished", "Unfinished"

        # Set random seed for reproducibility
        seed = random.randint(0, 2**32 - 1)
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32))
        print(f"Annotation seed: {seed}")

    def load_sample(
        self, noise_scale: float, fde_threshold: float, ade_threshold: float, max_retries: int, zoom_level: int = 5,
        gt_similarity_mode: bool = True, enable_initial_pruning: bool = True,
        initial_pos_threshold: float = 0.055, initial_yaw_threshold_deg: float = 0.55,
        enable_guidance: bool = False,
        use_collision: bool = True,
        use_route_following: bool = False,
        use_lane_keeping: bool = False,
        use_centerline_following: bool = False,
        guidance_scale: float = 0.5,
        time_step: int = 40,
    ) -> tuple[Figure, Figure, Figure, str, str, str, str, str]:
        """Load current sample and generate trajectory pair.

        Args:
            noise_scale: Noise scale for trajectory generation
            fde_threshold: FDE threshold - min between trajectories (diversity mode)
            ade_threshold: ADE threshold - max to GT (GT-similarity mode)
            max_retries: Maximum retry attempts
            zoom_level: Zoom level 1-10 (1=zoomed out, 10=zoomed in)
            gt_similarity_mode: If True (default), find stochastic trajectory close to GT using ADE

        Returns:
            Tuple of (trajectory_plot, velocity_plot, lateral_plot, metric_text, progress_text, metrics_text, sidebar_status, history_display)
        """
        if self.annotation_complete:
            return None, None, None, "Annotation complete!", self._format_progress(), "", self.get_sidebar_state(), self.get_labeled_history_display()

        if not self.npz_paths or self.current_index >= len(self.npz_paths):
            return None, None, None, "No samples available", "Complete", "", self.get_sidebar_state(), self.get_labeled_history_display()

        npz_path = self.npz_paths[self.current_index]
        self.current_data = load_npz_data(npz_path, self.device)

        # Get ground truth trajectory for GT-similarity mode
        gt_trajectory = None
        if gt_similarity_mode and "ego_agent_future" in self.current_data:
            gt_trajectory = self.current_data["ego_agent_future"][0].cpu().numpy()

        # Store mode for status display
        self.gt_similarity_mode = gt_similarity_mode

        # Generate trajectory pair
        traj_1, traj_2, metric, attempts, ego_shape, initial_disp, initial_yaw_diff, is_pruned = generate_trajectory_pair(
            self.policy_model,
            self.model_args,
            self.current_data,
            noise_scale=float(noise_scale),
            fde_threshold=float(fde_threshold),
            ade_threshold=float(ade_threshold),
            max_retries=int(max_retries),
            device=self.device,
            gt_similarity_mode=gt_similarity_mode,
            gt_trajectory=gt_trajectory,
            enable_initial_pruning=enable_initial_pruning,
            initial_pos_threshold=float(initial_pos_threshold),
            initial_yaw_threshold_deg=float(initial_yaw_threshold_deg),
            enable_guidance=bool(enable_guidance),
            use_collision=bool(use_collision),
            use_route_following=bool(use_route_following),
            use_lane_keeping=bool(use_lane_keeping),
            use_centerline_following=bool(use_centerline_following),
            guidance_scale=float(guidance_scale),
        )

        self.ego_shape = ego_shape.tolist() # [1, 3] wheel_base length, width
        self.trajectory_1 = traj_1.tolist()
        self.trajectory_2 = traj_2.tolist()
        self.current_metric = metric
        self.current_attempts = attempts
        self.initial_displacement = initial_disp
        self.initial_yaw_diff = initial_yaw_diff
        self.is_pruned = is_pruned
        self.enable_initial_pruning = enable_initial_pruning
        self.gt_available = self._check_gt_available()

        # Convert zoom_level to view_range: level 1 = 100m, level 10 = 10m
        view_range = 100 - (int(zoom_level) - 1) * 90 / 9
        traj_plot = self._create_trajectory_plot(time_step=int(time_step), view_range=view_range)
        vel_plot = self._create_velocity_plot(time_step=int(time_step))
        lat_plot = self._create_lateral_curvature_plot(time_step=int(time_step))

        # Create status text based on mode
        if gt_similarity_mode:
            metric_text = f"ADE (vs GT): {metric:.2f}m (Attempts: {attempts})"
        else:
            metric_text = f"FDE (vs Det.): {metric:.2f}m (Attempts: {attempts})"
        metric_text += f" | pos: {initial_disp:.3f}m, yaw: {initial_yaw_diff:.2f}°"
        if is_pruned:
            metric_text += " ⚠️ PRUNED"
        progress_text = self._format_progress()
        metrics_text = self._format_metrics_comparison()

        return traj_plot, vel_plot, lat_plot, metric_text, progress_text, metrics_text, self.get_sidebar_state(), self.get_labeled_history_display()

    def regenerate(
        self, noise_scale: float, fde_threshold: float, ade_threshold: float, max_retries: int, zoom_level: int = 5,
        gt_similarity_mode: bool = True, enable_initial_pruning: bool = True,
        initial_pos_threshold: float = 0.055, initial_yaw_threshold_deg: float = 0.55,
        enable_guidance: bool = False,
        use_collision: bool = True,
        use_route_following: bool = False,
        use_lane_keeping: bool = False,
        use_centerline_following: bool = False,
        guidance_scale: float = 0.5,
        time_step: int = 40,
    ) -> tuple[Figure, Figure, Figure, str, str, str, str, str]:
        """Regenerate trajectory pair with current parameters.

        Args:
            noise_scale: Noise scale for trajectory generation
            fde_threshold: FDE threshold - min between trajectories (diversity mode)
            ade_threshold: ADE threshold - max to GT (GT-similarity mode)
            max_retries: Maximum retry attempts
            zoom_level: Zoom level 1-10 (1=zoomed out, 10=zoomed in)
            gt_similarity_mode: If True (default), find stochastic trajectory close to GT using ADE

        Returns:
            Tuple of (trajectory_plot, velocity_plot, lateral_plot, metric_text, progress_text, metrics_text, sidebar_status, history_display)
        """
        if self.current_data is None:
            return None, None, None, "No data loaded", self._format_progress(), "", self.get_sidebar_state(), self.get_labeled_history_display()

        # Get ground truth trajectory for GT-similarity mode
        gt_trajectory = None
        if gt_similarity_mode and "ego_agent_future" in self.current_data:
            gt_trajectory = self.current_data["ego_agent_future"][0].cpu().numpy()

        # Store mode for status display
        self.gt_similarity_mode = gt_similarity_mode

        # Generate new pair
        traj_1, traj_2, metric, attempts, ego_shape, initial_disp, initial_yaw_diff, is_pruned = generate_trajectory_pair(
            self.policy_model,
            self.model_args,
            self.current_data,
            noise_scale=float(noise_scale),
            fde_threshold=float(fde_threshold),
            ade_threshold=float(ade_threshold),
            max_retries=int(max_retries),
            device=self.device,
            gt_similarity_mode=gt_similarity_mode,
            gt_trajectory=gt_trajectory,
            enable_initial_pruning=enable_initial_pruning,
            initial_pos_threshold=float(initial_pos_threshold),
            initial_yaw_threshold_deg=float(initial_yaw_threshold_deg),
            enable_guidance=bool(enable_guidance),
            use_collision=bool(use_collision),
            use_route_following=bool(use_route_following),
            use_lane_keeping=bool(use_lane_keeping),
            use_centerline_following=bool(use_centerline_following),
            guidance_scale=float(guidance_scale),
        )

        self.ego_shape = ego_shape.tolist()
        self.trajectory_1 = traj_1.tolist()
        self.trajectory_2 = traj_2.tolist()
        self.current_metric = metric
        self.current_attempts = attempts
        self.initial_displacement = initial_disp
        self.initial_yaw_diff = initial_yaw_diff
        self.is_pruned = is_pruned
        self.enable_initial_pruning = enable_initial_pruning

        # Convert zoom_level to view_range: level 1 = 100m, level 10 = 10m
        view_range = 100 - (int(zoom_level) - 1) * 90 / 9
        traj_plot = self._create_trajectory_plot(time_step=int(time_step), view_range=view_range)
        vel_plot = self._create_velocity_plot(time_step=int(time_step))
        lat_plot = self._create_lateral_curvature_plot(time_step=int(time_step))

        # Create status text based on mode
        if gt_similarity_mode:
            metric_text = f"ADE (vs GT): {metric:.2f}m (Attempts: {attempts})"
        else:
            metric_text = f"FDE (vs Det.): {metric:.2f}m (Attempts: {attempts})"
        metric_text += f" | pos: {initial_disp:.3f}m, yaw: {initial_yaw_diff:.2f}°"
        if is_pruned:
            metric_text += " ⚠️ PRUNED"
        progress_text = self._format_progress()
        metrics_text = self._format_metrics_comparison()

        print(f"Regenerated pair. Metric: {metric:.2f}m, Attempts: {attempts}")

        return traj_plot, vel_plot, lat_plot, metric_text, progress_text, metrics_text, self.get_sidebar_state(), self.get_labeled_history_display()

    def _self_shutdown(self) -> None:
        """Shutdown the server.
        The server will close automatically in 3 seconds.
        Training will start automatically.
        """
        # Schedule server shutdown
        import threading
        def shutdown_server():
            import time
            time.sleep(3)
            if hasattr(self, '_demo') and self._demo is not None:
                print("Closing Gradio server...")
                self._demo.close()

        threading.Thread(target=shutdown_server, daemon=True).start()


    def launch_training(self) -> tuple[Figure | None, Figure | None, Figure | None, str, str, str, str, str]:
        """Launch training."""
        print("Launching training...")
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
        self._self_shutdown()
        return None, None, None, "Training launched", "Training launched", "", self.get_sidebar_state(), self.get_labeled_history_display()

    def select_winner(
        self, winner: str, noise_scale: float, fde_threshold: float, ade_threshold: float, max_retries: int, zoom_level: int = 5,
        gt_similarity_mode: bool = True, enable_initial_pruning: bool = True,
        initial_pos_threshold: float = 0.055, initial_yaw_threshold_deg: float = 0.55,
        enable_guidance: bool = False,
        use_collision: bool = True,
        use_route_following: bool = False,
        use_lane_keeping: bool = False,
        use_centerline_following: bool = False,
        guidance_scale: float = 0.5,
        time_step: int = 40,
    ) -> tuple[Figure | None, Figure | None, Figure | None, str, str, str, str, str]:
        """Record preference and move to next sample.

        Args:
            winner: Either "trajectory_1" or "trajectory_2"
            noise_scale: Noise scale for next sample
            fde_threshold: FDE threshold - min between trajectories (diversity mode)
            ade_threshold: ADE threshold - max to GT (GT-similarity mode)
            max_retries: Maximum retry attempts
            zoom_level: Zoom level 1-10 (1=zoomed out, 10=zoomed in)
            gt_similarity_mode: If True (default), find stochastic trajectory close to GT using ADE

        Returns:
            Tuple of (trajectory_plot, velocity_plot, lateral_plot, metric_text, progress_text, metrics_text, sidebar_status, history_display)
        """
        if self.current_data is None:
            return None, None, None, "No data loaded", "Error", "", self.get_sidebar_state(), self.get_labeled_history_display()

        # Mark current sample as labeled
        self.mark_as_labeled(self.current_index)

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

            self._self_shutdown()

            return None, None, None, complete_msg, f"Complete: {len(self.preferences)}/{self.target_count}", "", self.get_sidebar_state(), self.get_labeled_history_display()

        # Move to next sample (with auto-skip if enabled)
        if self.auto_skip_labeled:
            self.current_index = self.get_next_unlabeled()
        else:
            self.current_index = (self.current_index + self.current_jump_size) % len(self.npz_paths)
        
        return self.load_sample(noise_scale, fde_threshold, ade_threshold, max_retries, zoom_level, gt_similarity_mode, enable_initial_pruning, initial_pos_threshold, initial_yaw_threshold_deg, enable_guidance, use_collision, use_route_following, use_lane_keeping, use_centerline_following, guidance_scale, time_step)

    def select_gt_as_winner(
        self, noise_scale: float, fde_threshold: float, ade_threshold: float, max_retries: int, zoom_level: int = 5,
        gt_similarity_mode: bool = True, enable_initial_pruning: bool = True,
        initial_pos_threshold: float = 0.055, initial_yaw_threshold_deg: float = 0.55,
        enable_guidance: bool = False,
        use_collision: bool = True,
        use_route_following: bool = False,
        use_lane_keeping: bool = False,
        use_centerline_following: bool = False,
        guidance_scale: float = 0.5,
        time_step: int = 40,
    ) -> tuple[Figure | None, Figure | None, Figure | None, str, str, str, str, str]:
        """Record GT as winner (against the deterministic trajectory) and advance to next sample.

        The GT trajectory is smoothed using the same unicycle kinematic smoother applied
        during training, ensuring distribution consistency with the base model's training data.

        Returns:
            Tuple of (trajectory_plot, velocity_plot, lateral_plot, metric_text, progress_text, metrics_text, sidebar_status, history_display)
        """
        if self.current_data is None:
            return None, None, None, "No data loaded", "Error", "", self.get_sidebar_state(), self.get_labeled_history_display()

        if not self.gt_available:
            return None, None, None, "GT not available for this sample", self._format_progress(), "", self.get_sidebar_state(), self.get_labeled_history_display()

        gt_trajectory = self._get_gt_trajectory()
        if gt_trajectory is None:
            return None, None, None, "GT conversion failed", self._format_progress(), "", self.get_sidebar_state(), self.get_labeled_history_display()

        self.mark_as_labeled(self.current_index)
        npz_path = self.npz_paths[self.current_index]

        self.preferences.append({
            "npz_path": npz_path,
            "trajectory_w": gt_trajectory.tolist(),
            "trajectory_l": self.trajectory_1,   # deterministic (green) is the loser
        })

        print(f"Recorded GT as winner for {npz_path}")

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
            self._self_shutdown()
            return None, None, None, complete_msg, f"Complete: {len(self.preferences)}/{self.target_count}", "", self.get_sidebar_state(), self.get_labeled_history_display()

        if self.auto_skip_labeled:
            self.current_index = self.get_next_unlabeled()
        else:
            self.current_index = (self.current_index + self.current_jump_size) % len(self.npz_paths)

        return self.load_sample(noise_scale, fde_threshold, ade_threshold, max_retries, zoom_level, gt_similarity_mode, enable_initial_pruning, initial_pos_threshold, initial_yaw_threshold_deg, enable_guidance, use_collision, use_route_following, use_lane_keeping, use_centerline_following, guidance_scale, time_step)

    def _check_gt_available(self) -> bool:
        """Return True if a valid GT trajectory exists for the current sample."""
        if self.current_data is None or "ego_agent_future" not in self.current_data:
            return False
        gt = self.current_data["ego_agent_future"][0].cpu().numpy()
        valid = ~((gt[:, 0] == 0) & (gt[:, 1] == 0))
        return bool(valid.mean() >= 0.8)

    def _get_gt_trajectory(self) -> np.ndarray | None:
        """Return the GT trajectory as [T, 4] (x, y, cos, sin).

        The NPZ ego_agent_future is stored as [T, 3] (x, y, heading in radians) and comes
        directly from Autoware's EKF localization with no additional smoothing applied during
        rosbag-to-NPZ conversion (verified in both the Python and C++ converter paths). The
        model trains on this raw EKF output as GT, so no further smoothing is applied here.

        Returns:
            GT trajectory as [T, 4] numpy array, or None on error.
        """
        try:
            gt_raw = self.current_data["ego_agent_future"][0].cpu().numpy()  # [T, 3]
            cos_yaw = np.cos(gt_raw[:, 2:3])
            sin_yaw = np.sin(gt_raw[:, 2:3])
            return np.concatenate([gt_raw[:, :2], cos_yaw, sin_yaw], axis=1).astype(np.float32)
        except Exception as e:
            print(f"GT conversion failed: {e}")
            return None

    def jump(
        self, delta: int, noise_scale: float, fde_threshold: float, ade_threshold: float, max_retries: int, zoom_level: int = 5,
        gt_similarity_mode: bool = True, enable_initial_pruning: bool = True,
        initial_pos_threshold: float = 0.055, initial_yaw_threshold_deg: float = 0.55,
        enable_guidance: bool = False,
        use_collision: bool = True,
        use_route_following: bool = False,
        use_lane_keeping: bool = False,
        use_centerline_following: bool = False,
        guidance_scale: float = 0.5,
        time_step: int = 40,
    ) -> tuple[Figure | None, Figure | None, Figure | None, str, str, str, str, str]:
        """Jump to a different sample.

        Args:
            delta: Number of samples to jump (positive or negative)
            noise_scale: Noise scale for trajectory generation
            fde_threshold: FDE threshold - min between trajectories (diversity mode)
            ade_threshold: ADE threshold - max to GT (GT-similarity mode)
            max_retries: Maximum retry attempts
            zoom_level: Zoom level 1-10 (1=zoomed out, 10=zoomed in)
            gt_similarity_mode: If True (default), find stochastic trajectory close to GT using ADE

        Returns:
            Tuple of (trajectory_plot, velocity_plot, lateral_plot, metric_text, progress_text, metrics_text, sidebar_status, history_display)
        """
        if not self.npz_paths:
            return None, None, None, "No samples available", "Error", "", self.get_sidebar_state(), self.get_labeled_history_display()

        # Ensure delta is integer
        delta = int(delta)
        self.current_index = max(0, min(self.current_index + delta, len(self.npz_paths) - 1))
        return self.load_sample(noise_scale, fde_threshold, ade_threshold, max_retries, zoom_level, gt_similarity_mode, enable_initial_pruning, initial_pos_threshold, initial_yaw_threshold_deg, enable_guidance, use_collision, use_route_following, use_lane_keeping, use_centerline_following, guidance_scale, time_step)

    def handle_keyboard_navigation(
        self, direction: str, noise_scale: float, fde_threshold: float, ade_threshold: float,
        max_retries: int, zoom_level: int = 5, gt_similarity_mode: bool = True,
        enable_initial_pruning: bool = True, initial_pos_threshold: float = 0.055,
        initial_yaw_threshold_deg: float = 0.55,
        enable_guidance: bool = False, use_collision: bool = True,
        use_route_following: bool = False, use_lane_keeping: bool = False,
        use_centerline_following: bool = False,
        guidance_scale: float = 0.5, time_step: int = 40,
    ) -> tuple[Figure | None, Figure | None, Figure | None, str, str, str, str, str]:
        """Handle keyboard navigation (left/right arrow keys).

        Args:
            direction: "left" or "right"
            noise_scale: Noise scale for trajectory generation
            fde_threshold: FDE threshold
            ade_threshold: ADE threshold
            max_retries: Maximum retry attempts
            zoom_level: Zoom level 1-10
            gt_similarity_mode: GT similarity mode flag

        Returns:
            Tuple of all visualization outputs plus sidebar states
        """
        # Determine delta based on direction and current jump size
        delta = -self.current_jump_size if direction == "left" else self.current_jump_size
        return self.jump(delta, noise_scale, fde_threshold, ade_threshold, max_retries, zoom_level, gt_similarity_mode, enable_initial_pruning, initial_pos_threshold, initial_yaw_threshold_deg, enable_guidance, use_collision, use_route_following, use_lane_keeping, use_centerline_following, guidance_scale, time_step)

    def jump_to_next_unlabeled(
        self, noise_scale: float, fde_threshold: float, ade_threshold: float, max_retries: int,
        zoom_level: int = 5, gt_similarity_mode: bool = True, enable_initial_pruning: bool = True,
        initial_pos_threshold: float = 0.055, initial_yaw_threshold_deg: float = 0.55,
        enable_guidance: bool = False,
        use_collision: bool = True,
        use_route_following: bool = False,
        use_lane_keeping: bool = False,
        use_centerline_following: bool = False,
        guidance_scale: float = 0.5,
        time_step: int = 40,
    ) -> tuple[Figure | None, Figure | None, Figure | None, str, str, str, str, str]:
        """Jump to the next unlabeled sample.

        Args:
            noise_scale: Noise scale for trajectory generation
            fde_threshold: FDE threshold
            ade_threshold: ADE threshold
            max_retries: Maximum retry attempts
            zoom_level: Zoom level 1-10
            gt_similarity_mode: GT similarity mode flag

        Returns:
            Tuple of all visualization outputs plus sidebar states
        """
        next_idx = self.get_next_unlabeled()
        if next_idx == self.current_index:
            # All samples are labeled
            return None, None, None, "All samples labeled!", self._format_progress(), "", self.get_sidebar_state(), self.get_labeled_history_display()
        
        self.current_index = next_idx
        return self.load_sample(noise_scale, fde_threshold, ade_threshold, max_retries, zoom_level, gt_similarity_mode, enable_initial_pruning, initial_pos_threshold, initial_yaw_threshold_deg, enable_guidance, use_collision, use_route_following, use_lane_keeping, use_centerline_following, guidance_scale, time_step)

    def update_time_display(
        self, time_step: int, zoom_level: int = 5
    ) -> tuple[Figure | None, Figure | None, Figure | None]:
        """Redraw visualization for selected time step.

        This method only redraws the plots using cached trajectory data.
        It does NOT call the diffusion planner model - no new trajectory generation.

        Args:
            time_step: Time step index (0 to T-1) for footprint display
            zoom_level: Zoom level 1-10 (1=zoomed out 120m, 10=zoomed in 20m)

        Returns:
            Tuple of (trajectory_plot, velocity_plot, lateral_plot)
        """
        if self.trajectory_1 is None or self.trajectory_2 is None:
            return None, None, None

        if self.current_data is None:
            return None, None, None

        time_step = int(time_step)
        zoom_level = int(zoom_level)
        # Convert zoom_level to view_range: level 1 = 100m, level 10 = 10m
        # Linear interpolation: view_range = 100 - (zoom_level - 1) * (100 - 10) / 9
        view_range = 100 - (zoom_level - 1) * 90 / 9
        traj_plot = self._create_trajectory_plot(time_step=time_step, view_range=view_range)
        vel_plot = self._create_velocity_plot(time_step=time_step)
        lat_plot = self._create_lateral_curvature_plot(time_step=time_step)
        return traj_plot, vel_plot, lat_plot

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

    def update_jump_size(self, delta: int) -> None:
        """Update the current jump size based on navigation button click.

        Args:
            delta: The jump size (can be negative)
        """
        self.current_jump_size = abs(delta)

    def mark_as_labeled(self, index: int) -> None:
        """Mark a sample as labeled and update history.

        Args:
            index: Sample index to mark as labeled
        """
        self.labeled_indices.add(index)
        
        # Update history (most recent first, max 10 items)
        if index in self.labeled_history:
            self.labeled_history.remove(index)
        self.labeled_history.insert(0, index)
        if len(self.labeled_history) > 10:
            self.labeled_history = self.labeled_history[:10]

    def get_next_unlabeled(self) -> int:
        """Find the next unlabeled sample index.

        Returns:
            Next unlabeled sample index (wraps around if needed)
        """
        if not self.npz_paths:
            return 0

        # Start from current_index + self.current_jump_size
        for offset in range(self.current_jump_size, len(self.npz_paths) + 1):
            candidate = (self.current_index + offset) % len(self.npz_paths)
            if candidate not in self.labeled_indices:
                return candidate
        
        # All samples are labeled, return current
        return self.current_index

    def get_sidebar_state(self) -> str:
        """Get formatted sidebar state display.

        Returns:
            Markdown formatted string with progress and status
        """
        if not self.npz_paths:
            return "**No samples loaded**"

        status = "Labeled" if self.current_index in self.labeled_indices else "Unlabeled"
        total_labeled = len(self.labeled_indices)
        total_samples = len(self.original_npz_paths)
        
        lines = [
            f"### Current Sample",
            f"**Index:** {self.current_index + 1} / {len(self.npz_paths)}",
            f"**Status:** {status}",
            "",
            f"### Progress",
            f"**Collected:** {len(self.preferences)} / {self.target_count}",
            f"**Labeled:** {total_labeled} / {total_samples}",
            "",
            f"### Navigation",
            f"**Jump Size:** {self.current_jump_size}",
            f"**Filter:** {self.current_filter}",
        ]
        
        return "\n".join(lines)

    def get_labeled_history_display(self) -> str:
        """Get formatted labeled history display.

        Returns:
            Markdown formatted string with clickable history items
        """
        if not self.labeled_history:
            return "*No labels yet*"

        lines = []
        for idx in self.labeled_history:
            # Show 1-indexed for user display
            lines.append(f"- Sample #{idx + 1}")
        
        return "\n".join(lines)

    def jump_to_index(
        self, target_index: int, noise_scale: float, fde_threshold: float, ade_threshold: float,
        max_retries: int, zoom_level: int = 5, gt_similarity_mode: bool = True,
        enable_initial_pruning: bool = True, initial_pos_threshold: float = 0.055,
        initial_yaw_threshold_deg: float = 0.55,
        enable_guidance: bool = False, use_collision: bool = True,
        use_route_following: bool = False, use_lane_keeping: bool = False,
        use_centerline_following: bool = False,
        guidance_scale: float = 0.5, time_step: int = 40,
    ) -> tuple[Figure | None, Figure | None, Figure | None, str, str, str, str, str]:
        """Jump directly to a specific sample index.

        Args:
            target_index: Target sample index (1-indexed from user input)
            noise_scale: Noise scale for trajectory generation
            fde_threshold: FDE threshold
            ade_threshold: ADE threshold
            max_retries: Maximum retry attempts
            zoom_level: Zoom level 1-10
            gt_similarity_mode: GT similarity mode flag

        Returns:
            Tuple of all visualization outputs plus sidebar states
        """
        if not self.npz_paths:
            return None, None, None, "No samples available", "Error", "", self.get_sidebar_state(), self.get_labeled_history_display()

        # Convert from 1-indexed to 0-indexed and clamp to valid range
        target_index = int(target_index) - 1
        target_index = max(0, min(target_index, len(self.npz_paths) - 1))
        
        self.current_index = target_index
        return self.load_sample(noise_scale, fde_threshold, ade_threshold, max_retries, zoom_level, gt_similarity_mode, enable_initial_pruning, initial_pos_threshold, initial_yaw_threshold_deg, enable_guidance, use_collision, use_route_following, use_lane_keeping, use_centerline_following, guidance_scale, time_step)

    def toggle_filter(
        self, filter_mode: str, noise_scale: float, fde_threshold: float, ade_threshold: float,
        max_retries: int, zoom_level: int = 5, gt_similarity_mode: bool = True,
        enable_initial_pruning: bool = True, initial_pos_threshold: float = 0.055,
        initial_yaw_threshold_deg: float = 0.55,
        enable_guidance: bool = False, use_collision: bool = True,
        use_route_following: bool = False, use_lane_keeping: bool = False,
        use_centerline_following: bool = False,
        guidance_scale: float = 0.5, time_step: int = 40,
    ) -> tuple[Figure | None, Figure | None, Figure | None, str, str, str, str, str]:
        """Toggle sample filter between All/Finished/Unfinished.

        Args:
            filter_mode: "All", "Finished", or "Unfinished"
            noise_scale: Noise scale for trajectory generation
            fde_threshold: FDE threshold
            ade_threshold: ADE threshold
            max_retries: Maximum retry attempts
            zoom_level: Zoom level 1-10
            gt_similarity_mode: GT similarity mode flag

        Returns:
            Tuple of all visualization outputs plus sidebar states
        """
        self.current_filter = filter_mode
        
        # Apply filter to npz_paths
        if filter_mode == "Finished":
            self.npz_paths = [path for i, path in enumerate(self.original_npz_paths) if i in self.labeled_indices]
        elif filter_mode == "Unfinished":
            self.npz_paths = [path for i, path in enumerate(self.original_npz_paths) if i not in self.labeled_indices]
        else:  # "All"
            self.npz_paths = self.original_npz_paths
        
        # Reset to first sample in filtered list
        self.current_index = 0
        
        if not self.npz_paths:
            return None, None, None, "No samples match filter", "No samples", "", self.get_sidebar_state(), self.get_labeled_history_display()

        return self.load_sample(noise_scale, fde_threshold, ade_threshold, max_retries, zoom_level, gt_similarity_mode, enable_initial_pruning, initial_pos_threshold, initial_yaw_threshold_deg, enable_guidance, use_collision, use_route_following, use_lane_keeping, use_centerline_following, guidance_scale, time_step)

    def _draw_vehicle_footprint(
        self, ax, x: float, y: float, heading: float, color: str, alpha: float = 0.8
    ) -> None:
        """Draw vehicle footprint as a rotated rectangle.

        The trajectory point (x, y) represents the rear axle center.
        The vehicle center is computed by offsetting forward by wheel_base/2.

        Args:
            ax: Matplotlib axis to draw on
            x: Rear axle center x position (trajectory point)
            y: Rear axle center y position (trajectory point)
            heading: Vehicle heading in radians
            color: Fill color for the rectangle
            alpha: Transparency (0-1)
        """
        import matplotlib.patches as patches

        wheel_base = self.ego_shape[0][0]
        length = self.ego_shape[0][1]
        width = self.ego_shape[0][2]

        cos_h = np.cos(heading)
        sin_h = np.sin(heading)

        # Compute vehicle center from rear axle
        # Vehicle center is wheel_base/2 forward from rear axle
        center_x = x + (wheel_base / 2) * cos_h
        center_y = y + (wheel_base / 2) * sin_h

        # Compute 4 corners of the rectangle relative to vehicle center
        # then rotate and translate
        half_len = length / 2
        half_wid = width / 2

        # Corners in local frame (before rotation): front-left, front-right, rear-right, rear-left
        local_corners = [
            (+half_len, +half_wid),
            (+half_len, -half_wid),
            (-half_len, -half_wid),
            (-half_len, +half_wid),
        ]

        # Rotate and translate corners to world frame
        world_corners = []
        for lx, ly in local_corners:
            # Rotate by heading
            wx = center_x + lx * cos_h - ly * sin_h
            wy = center_y + lx * sin_h + ly * cos_h
            world_corners.append((wx, wy))

        # Draw filled polygon
        polygon = patches.Polygon(
            world_corners, closed=True, facecolor=color, edgecolor="black", alpha=alpha, linewidth=1.5
        )
        ax.add_patch(polygon)

        # Draw heading indicator (small line from center pointing forward)
        indicator_len = length * 0.3
        ax.plot(
            [center_x, center_x + indicator_len * cos_h],
            [center_y, center_y + indicator_len * sin_h],
            color="black",
            linewidth=2,
            alpha=alpha,
        )

    def _create_trajectory_plot(self, time_step: int | None = None, view_range: int = 60) -> Figure:
        """Create trajectory visualization plot.

        Args:
            time_step: Optional time step index to show vehicle footprints.
                       If None, shows full trajectories without footprints.
            view_range: View range in meters for the plot (default: 60m)

        Returns:
            Matplotlib Figure with trajectory visualization
        """
        fig = Figure(figsize=(10, 11.5))
        ax = fig.add_subplot(111)

        traj_1_np = np.array(self.trajectory_1)
        traj_2_np = np.array(self.trajectory_2)

        # Calculate center as midpoint between start and end of green trajectory
        start_pos = traj_1_np[0, :2]
        end_pos = traj_1_np[-1, :2]
        center_x = (start_pos[0] + end_pos[0]) / 2
        center_y = (start_pos[1] + end_pos[1]) / 2

        # Visualize map and context (use large view range to get all context)
        data_cpu = {k: v.cpu() for k, v in self.current_data.items()}
        visualize_inputs(data_cpu, save_path=None, ax=ax, view_ranges=[120])

        # Stochastic trajectory color:
        #   orange  — normal
        #   grey    — would be pruned but pruning is OFF (visual indicator only)
        #   red     — actively pruned (pruning is ON, trajectory was flagged)
        if self.is_pruned and self.enable_initial_pruning:
            stochastic_color = "red"
            stochastic_label = "Trajectory 2 (Stochastic) ⚠️ PRUNED"
        elif self.is_pruned and not self.enable_initial_pruning:
            stochastic_color = "gray"
            stochastic_label = "Trajectory 2 (Stochastic) ⚠️ would be pruned"
        else:
            stochastic_color = "orange"
            stochastic_label = "Trajectory 2 (Stochastic)"

        # Plot full trajectories
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
            color=stochastic_color,
            linewidth=3,
            alpha=0.7,
            label=stochastic_label,
        )

        # Draw vehicle footprints at selected time step
        if time_step is not None and 0 <= time_step < len(traj_1_np):
            # Trajectory format is [x, y, cos(heading), sin(heading)]
            # Need to use arctan2(sin, cos) to get the heading angle

            # Trajectory 1 footprint (green)
            x1, y1 = traj_1_np[time_step, 0], traj_1_np[time_step, 1]
            cos1, sin1 = traj_1_np[time_step, 2], traj_1_np[time_step, 3]
            heading1 = np.arctan2(sin1, cos1)
            self._draw_vehicle_footprint(ax, x1, y1, heading1, color="green", alpha=0.6)

            # Trajectory 2 footprint
            x2, y2 = traj_2_np[time_step, 0], traj_2_np[time_step, 1]
            cos2, sin2 = traj_2_np[time_step, 2], traj_2_np[time_step, 3]
            heading2 = np.arctan2(sin2, cos2)
            self._draw_vehicle_footprint(ax, x2, y2, heading2, color=stochastic_color, alpha=0.6)

            # Mark the positions with dots
            ax.scatter([x1], [y1], c="green", s=50, zorder=10, edgecolors="black")
            ax.scatter([x2], [y2], c=stochastic_color, s=50, zorder=10, edgecolors="black")

            # Update title with time info
            time_sec = time_step * 0.1
            total_time = (len(traj_1_np) - 1) * 0.1
            title = f"Trajectory Comparison - Time: {time_sec:.1f}s / {total_time:.1f}s"
            if self.is_pruned:
                title += f"  ⚠️ PRUNED (pos:{self.initial_displacement:.3f}m, yaw:{self.initial_yaw_diff:.2f}°)"
            ax.set_title(title)
        else:
            title = "Trajectory Comparison"
            if self.is_pruned:
                title += f"  ⚠️ PRUNED (pos:{self.initial_displacement:.3f}m, yaw:{self.initial_yaw_diff:.2f}°)"
            ax.set_title(title)

        ax.legend(loc="upper left")

        # Set axis limits centered on midpoint of green trajectory
        half_range = view_range / 2
        ax.set_xlim(center_x - half_range, center_x + half_range)
        ax.set_ylim(center_y - half_range, center_y + half_range)
        ax.set_aspect('equal')

        return fig

    def _create_velocity_plot(self, time_step: int | None = None) -> Figure:
        """Create velocity and acceleration comparison plot.

        Args:
            time_step: Optional time step index to show time marker.
                       If None, no time marker is shown.

        Returns:
            Matplotlib Figure with velocity and acceleration subplots
        """
        fig = Figure(figsize=(6, 8))

        # Create two subplots: velocity (top) and acceleration (bottom)
        ax_vel = fig.add_subplot(211)
        ax_acc = fig.add_subplot(212)

        vel_1 = self._calculate_velocities(self.trajectory_1)
        vel_2 = self._calculate_velocities(self.trajectory_2)
        vel_gt = self._calculate_gt_velocities()

        # Calculate accelerations (derivative of velocity)
        acc_1 = self._calculate_accelerations(vel_1)
        acc_2 = self._calculate_accelerations(vel_2)

        time_steps_vel = np.arange(len(vel_1))
        time_steps_acc = np.arange(len(acc_1))

        # Velocity subplot
        ax_vel.plot(time_steps_vel, vel_1, "g-", linewidth=2, alpha=0.7, label="Green (Det.)")
        ax_vel.plot(time_steps_vel, vel_2, color="orange", linewidth=2, alpha=0.7, label="Orange (Stoch.)")
        # Add ground truth velocity if available
        if vel_gt is not None:
            time_steps_gt = np.arange(len(vel_gt))
            ax_vel.plot(time_steps_gt, vel_gt, color="red", linewidth=2, linestyle="--", alpha=0.7, label="Ground Truth")
        ax_vel.set_ylabel("Velocity (km/h)")
        ax_vel.set_ylim(0, 60)
        ax_vel.set_title("Velocity Comparison")
        ax_vel.legend(loc="upper right")
        ax_vel.grid(True, alpha=0.3)

        # Acceleration subplot
        ax_acc.plot(time_steps_acc, acc_1, "g-", linewidth=2, alpha=0.7, label="Trajectory 1")
        ax_acc.plot(time_steps_acc, acc_2, color="orange", linewidth=2, alpha=0.7, label="Trajectory 2")
        ax_acc.set_xlabel("Time Step")
        ax_acc.set_ylabel("Acceleration (m/s²)")
        ax_acc.set_ylim(-2.5, 2.5)
        ax_acc.set_title("Acceleration Comparison")
        ax_acc.legend(loc="upper right")
        ax_acc.grid(True, alpha=0.3)
        ax_acc.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

        # Draw time marker if time_step is provided
        if time_step is not None:
            # Velocity marker
            if 0 <= time_step < len(vel_1):
                ax_vel.axvline(x=time_step, color="blue", linestyle="--", linewidth=2, alpha=0.8)
                # Show current values
                v1_val = vel_1[time_step]
                v2_val = vel_2[time_step]
                ax_vel.scatter([time_step], [v1_val], c="green", s=80, zorder=10, edgecolors="black")
                ax_vel.scatter([time_step], [v2_val], c="orange", s=80, zorder=10, edgecolors="black")
                # Add ground truth marker if available
                vgt_val = None
                if vel_gt is not None and 0 <= time_step < len(vel_gt):
                    vgt_val = vel_gt[time_step]
                    ax_vel.scatter([time_step], [vgt_val], c="red", s=80, zorder=10, edgecolors="black", marker="D")

            # Acceleration marker
            if 0 <= time_step < len(acc_1):
                ax_acc.axvline(x=time_step, color="blue", linestyle="--", linewidth=2, alpha=0.8)
                # Show current values
                a1_val = acc_1[time_step]
                a2_val = acc_2[time_step]
                ax_acc.scatter([time_step], [a1_val], c="green", s=80, zorder=10, edgecolors="black")
                ax_acc.scatter([time_step], [a2_val], c="orange", s=80, zorder=10, edgecolors="black")

                # Add text annotations for current values
                time_sec = time_step * 0.1
                if vgt_val is not None:
                    ax_vel.set_title(f"Velocity @ {time_sec:.1f}s: G={v1_val:.1f}, O={v2_val:.1f}, GT={vgt_val:.1f} km/h")
                else:
                    ax_vel.set_title(f"Velocity @ {time_sec:.1f}s: G={v1_val:.1f} km/h, O={v2_val:.1f} km/h")
                ax_acc.set_title(f"Acceleration @ {time_sec:.1f}s: G={a1_val:.2f} m/s², O={a2_val:.2f} m/s²")

        fig.tight_layout()
        return fig

    def _create_lateral_curvature_plot(self, time_step: int | None = None) -> Figure:
        """Create lateral acceleration and curvature comparison plot.

        Args:
            time_step: Optional time step index to show time marker.
                       If None, no time marker is shown.

        Returns:
            Matplotlib Figure with lateral acceleration and curvature subplots
        """
        fig = Figure(figsize=(6, 8))

        # Create two subplots: lateral acceleration (top) and curvature (bottom)
        ax_lat = fig.add_subplot(211)
        ax_curv = fig.add_subplot(212)

        # Calculate velocities and curvatures
        vel_1 = self._calculate_velocities(self.trajectory_1)
        vel_2 = self._calculate_velocities(self.trajectory_2)
        vel_gt = self._calculate_gt_velocities()

        curv_1 = self._calculate_curvature(self.trajectory_1)
        curv_2 = self._calculate_curvature(self.trajectory_2)
        curv_gt = self._calculate_gt_curvature()

        lat_acc_1 = self._calculate_lateral_acceleration(curv_1, vel_1)
        lat_acc_2 = self._calculate_lateral_acceleration(curv_2, vel_2)
        lat_acc_gt = None
        if curv_gt is not None and vel_gt is not None:
            lat_acc_gt = self._calculate_lateral_acceleration(curv_gt, vel_gt)

        time_steps = np.arange(len(curv_1))

        # Lateral acceleration subplot
        ax_lat.plot(time_steps, lat_acc_1, "g-", linewidth=2, alpha=0.7, label="Green (Det.)")
        ax_lat.plot(time_steps, lat_acc_2, color="orange", linewidth=2, alpha=0.7, label="Orange (Stoch.)")
        # Add ground truth lateral acceleration if available
        if lat_acc_gt is not None:
            time_steps_gt = np.arange(len(lat_acc_gt))
            ax_lat.plot(time_steps_gt, lat_acc_gt, color="red", linewidth=2, linestyle="--", alpha=0.7, label="Ground Truth")
        ax_lat.set_ylabel("Lateral Accel (m/s²)")
        ax_lat.set_ylim(0, 8)
        ax_lat.set_title("Lateral Acceleration Comparison")
        ax_lat.legend(loc="upper right")
        ax_lat.grid(True, alpha=0.3)
        # Add comfort threshold line (typically 3-4 m/s² is uncomfortable)
        ax_lat.axhline(y=3.0, color="purple", linestyle=":", linewidth=1, alpha=0.5)

        # Curvature subplot
        ax_curv.plot(time_steps, curv_1, "g-", linewidth=2, alpha=0.7, label="Green (Det.)")
        ax_curv.plot(time_steps, curv_2, color="orange", linewidth=2, alpha=0.7, label="Orange (Stoch.)")
        # Add ground truth curvature if available
        if curv_gt is not None:
            time_steps_gt = np.arange(len(curv_gt))
            ax_curv.plot(time_steps_gt, curv_gt, color="red", linewidth=2, linestyle="--", alpha=0.7, label="Ground Truth")
        ax_curv.set_xlabel("Time Step")
        ax_curv.set_ylabel("Curvature (1/m)")
        ax_curv.set_ylim(-0.2, 0.2)
        ax_curv.set_title("Curvature Comparison")
        ax_curv.legend(loc="upper right")
        ax_curv.grid(True, alpha=0.3)
        ax_curv.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

        # Draw time marker if time_step is provided
        if time_step is not None:
            # Lateral acceleration marker
            lat_gt_val = None
            if 0 <= time_step < len(lat_acc_1):
                ax_lat.axvline(x=time_step, color="blue", linestyle="--", linewidth=2, alpha=0.8)
                lat1_val = lat_acc_1[time_step]
                lat2_val = lat_acc_2[time_step]
                ax_lat.scatter([time_step], [lat1_val], c="green", s=80, zorder=10, edgecolors="black")
                ax_lat.scatter([time_step], [lat2_val], c="orange", s=80, zorder=10, edgecolors="black")
                # Add ground truth marker if available
                if lat_acc_gt is not None and 0 <= time_step < len(lat_acc_gt):
                    lat_gt_val = lat_acc_gt[time_step]
                    ax_lat.scatter([time_step], [lat_gt_val], c="red", s=80, zorder=10, edgecolors="black", marker="D")

            # Curvature marker
            curv_gt_val = None
            if 0 <= time_step < len(curv_1):
                ax_curv.axvline(x=time_step, color="blue", linestyle="--", linewidth=2, alpha=0.8)
                c1_val = curv_1[time_step]
                c2_val = curv_2[time_step]
                ax_curv.scatter([time_step], [c1_val], c="green", s=80, zorder=10, edgecolors="black")
                ax_curv.scatter([time_step], [c2_val], c="orange", s=80, zorder=10, edgecolors="black")
                # Add ground truth marker if available
                if curv_gt is not None and 0 <= time_step < len(curv_gt):
                    curv_gt_val = curv_gt[time_step]
                    ax_curv.scatter([time_step], [curv_gt_val], c="red", s=80, zorder=10, edgecolors="black", marker="D")

                # Add text annotations for current values
                time_sec = time_step * 0.1
                if lat_gt_val is not None:
                    ax_lat.set_title(f"Lat.Accel @ {time_sec:.1f}s: G={lat1_val:.2f}, O={lat2_val:.2f}, GT={lat_gt_val:.2f} m/s²")
                else:
                    ax_lat.set_title(f"Lat.Accel @ {time_sec:.1f}s: G={lat1_val:.2f} m/s², O={lat2_val:.2f} m/s²")
                if curv_gt_val is not None:
                    ax_curv.set_title(f"Curvature @ {time_sec:.1f}s: G={c1_val:.3f}, O={c2_val:.3f}, GT={curv_gt_val:.3f} 1/m")
                else:
                    ax_curv.set_title(f"Curvature @ {time_sec:.1f}s: G={c1_val:.3f} 1/m, O={c2_val:.3f} 1/m")

        fig.tight_layout()
        return fig

    def _calculate_accelerations(self, velocities: np.ndarray) -> np.ndarray:
        """Calculate accelerations from velocity array.

        Assumes 1 time step = 0.1 seconds.

        Args:
            velocities: Array of velocities in km/h

        Returns:
            Array of accelerations in m/s²
        """
        # Convert km/h to m/s
        vel_m_per_s = velocities / 3.6

        # Calculate acceleration (dv/dt)
        # dt = 0.1 seconds
        accelerations = np.diff(vel_m_per_s) / 0.1

        # Pad with 0 at the end to match length
        accelerations = np.append(accelerations, 0.0)

        return accelerations

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

    def _calculate_curvature(self, trajectory: list) -> np.ndarray:
        """Calculate path curvature from heading changes.

        Curvature = d(heading) / d(arc_length)

        Args:
            trajectory: List of trajectory points [[x, y, cos(heading), sin(heading)], ...]

        Returns:
            Array of curvatures in 1/m (positive = left turn, negative = right turn)
            Length matches trajectory length (same as velocities)
        """
        traj_np = np.array(trajectory)
        ego_state = self.current_data["ego_current_state"].cpu().numpy()[0]

        # Get headings from cos/sin - trajectory format is [x, y, cos, sin]
        # Include ego state to compute curvature for first trajectory point
        cos_vals = np.concatenate([[ego_state[2]], traj_np[:, 2]])
        sin_vals = np.concatenate([[ego_state[3]], traj_np[:, 3]])
        headings = np.arctan2(sin_vals, cos_vals)

        # Get positions (ego + trajectory = 81 points)
        positions = np.vstack([ego_state[:2], traj_np[:, :2]])

        # Calculate arc length (distance traveled between points) - 80 values
        diffs = np.diff(positions, axis=0)
        arc_lengths = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)

        # Calculate heading changes (unwrap to handle angle wrapping) - 80 values
        heading_diffs = np.diff(np.unwrap(headings))

        # Curvature = d(heading) / d(arc_length)
        # Avoid division by zero
        # Result has 80 values, matching velocity array length
        curvatures = np.zeros(len(heading_diffs))
        valid_mask = arc_lengths > 1e-6
        curvatures[valid_mask] = heading_diffs[valid_mask] / arc_lengths[valid_mask]

        return curvatures

    def _calculate_lateral_acceleration(
        self, curvatures: np.ndarray, velocities: np.ndarray
    ) -> np.ndarray:
        """Calculate lateral acceleration from curvature and velocity.

        Lateral acceleration = velocity^2 * curvature

        Args:
            curvatures: Array of curvatures in 1/m
            velocities: Array of velocities in km/h

        Returns:
            Array of lateral accelerations in m/s²
        """
        # Convert velocity from km/h to m/s
        vel_m_per_s = velocities / 3.6

        # Lateral acceleration = v^2 * curvature
        # Use absolute curvature for magnitude
        lateral_acc = vel_m_per_s**2 * np.abs(curvatures)

        return lateral_acc

    def _calculate_gt_velocities(self) -> np.ndarray | None:
        """Calculate velocities from ground truth trajectory (ego_agent_future).

        Ground truth format is [x, y, heading] (80 points).

        Returns:
            Array of velocities in km/h, or None if no ground truth available
        """
        if self.current_data is None or "ego_agent_future" not in self.current_data:
            return None

        ego_future = self.current_data["ego_agent_future"].cpu().numpy()[0]  # [80, 3]
        ego_state = self.current_data["ego_current_state"].cpu().numpy()[0]

        # Filter out invalid points (where x=0 and y=0)
        valid_mask = ~((ego_future[:, 0] == 0) & (ego_future[:, 1] == 0))
        if not np.any(valid_mask):
            return None

        # Include initial ego position
        positions = np.vstack([ego_state[:2], ego_future[:, :2]])
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

    def _calculate_gt_curvature(self) -> np.ndarray | None:
        """Calculate curvature from ground truth trajectory (ego_agent_future).

        Ground truth format is [x, y, heading] where heading is direct angle in radians.

        Returns:
            Array of curvatures in 1/m, or None if no ground truth available
        """
        if self.current_data is None or "ego_agent_future" not in self.current_data:
            return None

        ego_future = self.current_data["ego_agent_future"].cpu().numpy()[0]  # [80, 3]
        ego_state = self.current_data["ego_current_state"].cpu().numpy()[0]

        # Filter out invalid points
        valid_mask = ~((ego_future[:, 0] == 0) & (ego_future[:, 1] == 0))
        if not np.any(valid_mask):
            return None

        # Get headings - ground truth has direct heading angle in radians
        # Include ego state heading (computed from cos/sin)
        ego_heading = np.arctan2(ego_state[3], ego_state[2])
        headings = np.concatenate([[ego_heading], ego_future[:, 2]])

        # Get positions (ego + trajectory = 81 points)
        positions = np.vstack([ego_state[:2], ego_future[:, :2]])

        # Calculate arc length (distance traveled between points) - 80 values
        diffs = np.diff(positions, axis=0)
        arc_lengths = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)

        # Calculate heading changes (unwrap to handle angle wrapping) - 80 values
        heading_diffs = np.diff(np.unwrap(headings))

        # Curvature = d(heading) / d(arc_length)
        curvatures = np.zeros(len(heading_diffs))
        valid_arc_mask = arc_lengths > 1e-6
        curvatures[valid_arc_mask] = heading_diffs[valid_arc_mask] / arc_lengths[valid_arc_mask]

        return curvatures

    def _calculate_ade_fde(self, trajectory: list) -> tuple[float | None, float | None]:
        """Calculate ADE and FDE for a trajectory compared to ground truth.

        ADE (Average Displacement Error): Mean Euclidean distance across all timesteps.
        FDE (Final Displacement Error): Euclidean distance at the final timestep.

        Args:
            trajectory: List of trajectory points [[x, y, cos, sin], ...]

        Returns:
            Tuple of (ADE, FDE) in meters, or (None, None) if no ground truth available
        """
        if self.current_data is None or "ego_agent_future" not in self.current_data:
            return None, None

        ego_future = self.current_data["ego_agent_future"].cpu().numpy()[0]  # [80, 3]
        traj_np = np.array(trajectory)

        # Check for valid ground truth (not all zeros)
        valid_mask = ~((ego_future[:, 0] == 0) & (ego_future[:, 1] == 0))
        if not np.any(valid_mask):
            return None, None

        # Get predicted positions (trajectory has 80 points)
        pred_positions = traj_np[:, :2]  # [80, 2]

        # Get ground truth positions
        gt_positions = ego_future[:, :2]  # [80, 2]

        # Calculate displacement errors for each timestep
        displacements = np.sqrt(np.sum((pred_positions - gt_positions) ** 2, axis=1))

        # ADE: Average of all displacement errors
        ade = np.mean(displacements)

        # FDE: Displacement error at final timestep
        fde = displacements[-1]

        return ade, fde

    def _calculate_summary_metrics(self, trajectory: list) -> dict:
        """Calculate summary metrics for a trajectory.

        Args:
            trajectory: List of trajectory points

        Returns:
            Dictionary of summary metrics
        """
        traj_np = np.array(trajectory)

        # Calculate all derived quantities
        velocities = self._calculate_velocities(trajectory)
        accelerations = self._calculate_accelerations(velocities)
        curvatures = self._calculate_curvature(trajectory)
        lateral_acc = self._calculate_lateral_acceleration(curvatures, velocities)

        # Path length
        diffs = np.diff(traj_np[:, :2], axis=0)
        path_length = np.sum(np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2))

        return {
            "path_length": path_length,
            "avg_speed": np.mean(velocities),
            "max_speed": np.max(velocities),
            "max_long_acc": np.max(np.abs(accelerations)),
            "max_lat_acc": np.max(lateral_acc),
            "max_curvature": np.max(np.abs(curvatures)),
            "rms_long_acc": np.sqrt(np.mean(accelerations**2)),
            "rms_lat_acc": np.sqrt(np.mean(lateral_acc**2)),
        }

    def _format_metrics_comparison(self) -> str:
        """Format summary metrics comparison between two trajectories.

        Returns:
            Formatted string comparing T1 vs T2 metrics
        """
        if self.trajectory_1 is None or self.trajectory_2 is None:
            return "No trajectories loaded"

        m1 = self._calculate_summary_metrics(self.trajectory_1)
        m2 = self._calculate_summary_metrics(self.trajectory_2)

        def better_indicator(v1, v2, lower_is_better=True):
            """Return indicator showing which value is better."""
            if lower_is_better:
                if v1 < v2:
                    return "✓", ""
                elif v2 < v1:
                    return "", "✓"
            else:
                if v1 > v2:
                    return "✓", ""
                elif v2 > v1:
                    return "", "✓"
            return "", ""

        # Calculate ADE and FDE for both trajectories
        ade_1, fde_1 = self._calculate_ade_fde(self.trajectory_1)
        ade_2, fde_2 = self._calculate_ade_fde(self.trajectory_2)

        lines = [
            "## 📊 Trajectory Metrics",
            "",
            "| Metric | Green (Det.) | Orange (Stoch.) | Better |",
            "|--------|--------------|-----------------|--------|",
        ]

        # ADE (lower is better - closer to ground truth)
        if ade_1 is not None and ade_2 is not None:
            b1, b2 = better_indicator(ade_1, ade_2, lower_is_better=True)
            lines.append(f"| ADE (vs GT) | {ade_1:.2f} m {b1} | {ade_2:.2f} m {b2} | Lower |")

        # FDE (lower is better - closer to ground truth)
        if fde_1 is not None and fde_2 is not None:
            b1, b2 = better_indicator(fde_1, fde_2, lower_is_better=True)
            lines.append(f"| FDE (vs GT) | {fde_1:.2f} m {b1} | {fde_2:.2f} m {b2} | Lower |")

        # Max longitudinal acceleration (lower is more comfortable)
        b1, b2 = better_indicator(m1["max_long_acc"], m2["max_long_acc"], lower_is_better=True)
        lines.append(f"| Max Long. Accel | {m1['max_long_acc']:.2f} m/s² {b1} | {m2['max_long_acc']:.2f} m/s² {b2} | Lower |")

        # Max lateral acceleration (lower is more comfortable)
        b1, b2 = better_indicator(m1["max_lat_acc"], m2["max_lat_acc"], lower_is_better=True)
        lines.append(f"| Max Lat. Accel | {m1['max_lat_acc']:.2f} m/s² {b1} | {m2['max_lat_acc']:.2f} m/s² {b2} | Lower |")

        # RMS accelerations (lower is smoother)
        b1, b2 = better_indicator(m1["rms_long_acc"], m2["rms_long_acc"], lower_is_better=True)
        lines.append(f"| RMS Long. Accel | {m1['rms_long_acc']:.2f} m/s² {b1} | {m2['rms_long_acc']:.2f} m/s² {b2} | Lower |")

        b1, b2 = better_indicator(m1["rms_lat_acc"], m2["rms_lat_acc"], lower_is_better=True)
        lines.append(f"| RMS Lat. Accel | {m1['rms_lat_acc']:.2f} m/s² {b1} | {m2['rms_lat_acc']:.2f} m/s² {b2} | Lower |")

        return "\n".join(lines)


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

    # Load external assets
    css_path = Path(__file__).parent / "static" / "annotation_styles.css"
    with open(css_path, "r") as f:
        css_content = f.read()

    js_path = Path(__file__).parent / "static" / "keyboard_handler.js"
    with open(js_path, "r") as f:
        js_content = f.read()

    # Note: css parameter will move to launch() in future Gradio versions
    # For now, keeping it here for compatibility
    with gr.Blocks(title="Trajectory Preference Annotation", css=css_content) as demo:
        # Hidden textbox to capture keyboard events - must be visible in DOM but hidden via CSS
        keyboard_input = gr.Textbox(value="", label="Keyboard Input", elem_id="keyboard_capture", container=False)
        
        with gr.Row(elem_classes="main-row"):
            # LEFT SIDEBAR (scale=1) - Fixed position
            with gr.Column(scale=1, elem_classes="sidebar-column"):
                gr.Markdown("# 🚗 Annotation")
                gr.Markdown(
                    "**Instructions:** Compare trajectories and select the better one. "
                    "Consider safety, comfort, and efficiency."
                )
                
                gr.Markdown("---")
                
                # Integrated Progress and Metric display
                gr.Markdown("## 📊 Status")
                progress_text = gr.Textbox(
                    label="Progress",
                    value="Loading...",
                    interactive=False,
                    lines=2,
                )
                fde_text = gr.Textbox(
                    label="Metric",
                    value="FDE: 0.00m",
                    interactive=False,
                    lines=1,
                )
                sidebar_status = gr.Markdown(value="Loading...")
                
                gr.Markdown("---")
                
                # Jump-to input
                gr.Markdown("## Quick Jump")
                jump_to_input = gr.Number(label="Jump to Sample #", precision=0, minimum=1, value=1)
                jump_to_btn = gr.Button("Go to Sample", size="sm")
                
                gr.Markdown("---")
                
                # Labeled history
                gr.Markdown("## Recent Labeled")
                history_display = gr.Markdown(value="*No labels yet*")
                
                gr.Markdown("---")
                
                # Filter controls
                gr.Markdown("## Filters")
                show_finished_radio = gr.Radio(
                    choices=["All", "Finished", "Unfinished"],
                    value="All",
                    label="Show Samples"
                )
                
                # Next unlabeled navigation
                next_unlabeled_btn = gr.Button("⏭️ Next Unlabeled", variant="primary", size="sm")
                auto_skip_checkbox = gr.Checkbox(label="Auto-skip labeled", value=False)
                
                gr.Markdown("---")
                gr.Markdown("**Keyboard:** ⬅️ ➡️ to navigate")

                # Launch Training
                launch_training_btn = gr.Button("🚀 Launch Training", variant="primary", size="lg")
            
            # MAIN CONTENT (scale=4)
            with gr.Column(scale=4, elem_classes="main-column"):
                gr.Markdown("# 🚗 Trajectory Comparison")

                # Parameter controls
                gr.Markdown("## ⚙️ Generation Parameters")
                with gr.Row():
                    noise_scale = gr.Slider(
                        minimum=0.5,
                        maximum=5.0,
                        value=2.5,
                        step=0.1,
                        label="Noise Scale",
                        info="Controls diversity of the stochastic trajectory"
                    )
                    fde_threshold = gr.Slider(
                        minimum=0.5,
                        maximum=10.0,
                        value=2.0,
                        step=0.1,
                        label="FDE Threshold (m)",
                        info="Min FDE against green (diversity mode)"
                    )
                    ade_threshold = gr.Slider(
                        minimum=0.1,
                        maximum=5.0,
                        value=1.0,
                        step=0.1,
                        label="ADE Threshold (m)",
                        info="Max ADE to ground truth (GT mode)"
                    )
                    max_retries = gr.Slider(
                        minimum=10,
                        maximum=200,
                        value=50,
                        step=10,
                        label="Max Retries",
                        info="Maximum attempts to meet threshold"
                    )
                
                # GT Similarity Mode toggle - DEFAULT TO TRUE
                with gr.Row():
                    gt_similarity_checkbox = gr.Checkbox(
                        value=True,
                        label="🎯 GT Similarity Mode (find stochastic trajectory close to ground truth)",
                        info="When enabled: retry until ADE(stochastic, GT) <= ADE threshold. When disabled: retry until FDE(det., stochastic) >= FDE threshold."
                    )

                # Initial pose pruning controls
                gr.Markdown("## ✂️ Initial Pose Pruning")
                with gr.Row():
                    enable_initial_pruning_checkbox = gr.Checkbox(
                        value=True,
                        label="Enable Initial Pose Pruning (visual indicator only when disabled)",
                        info="Disabled: highlights would-be-pruned trajectories in red. Enabled: retries until initial pose thresholds are met."
                    )
                with gr.Row():
                    initial_pos_threshold_slider = gr.Slider(
                        minimum=0.01,
                        maximum=0.1,
                        value=0.055,
                        step=0.005,
                        label="Initial Position Threshold (m)",
                        info="Max displacement between first poses of det. and stoch. trajectories"
                    )
                    initial_yaw_threshold_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.55,
                        step=0.05,
                        label="Initial Yaw Threshold (°)",
                        info="Max absolute yaw difference between first poses"
                    )
                # Guidance controls
                gr.Markdown("## 🧭 Guidance")
                with gr.Row():
                    enable_guidance_checkbox = gr.Checkbox(
                        value=False,
                        label="Enable Guidance",
                        info="When enabled, uses zeros as starting noise and guidance shapes the stochastic trajectory"
                    )
                with gr.Row():
                    use_collision_checkbox = gr.Checkbox(
                        value=True,
                        label="Collision Avoidance",
                        info="Penalise trajectories that collide with neighbouring agents"
                    )
                    use_route_following_checkbox = gr.Checkbox(
                        value=False,
                        label="Route Following",
                        info="Penalise trajectories that stray from the planned route"
                    )
                    use_lane_keeping_checkbox = gr.Checkbox(
                        value=False,
                        label="Lane Keeping",
                        info="Penalise trajectories where the vehicle protrudes beyond lane boundaries"
                    )
                    use_centerline_following_checkbox = gr.Checkbox(
                        value=False,
                        label="Centerline Following",
                        info="Continuously attract the trajectory toward the nearest lane centerline (quadratic cost)"
                    )
                with gr.Row():
                    guidance_scale_slider = gr.Slider(
                        minimum=0.0,
                        maximum=5.0,
                        value=0.5,
                        step=0.1,
                        label="Guidance Scale",
                        info="Strength of classifier guidance during DPM-Solver sampling"
                    )

                # Visualizations - Trajectory with zoom slider in one column
                gr.Markdown("## 🎨 Trajectory Visualization")
                with gr.Row():
                    # Trajectory plot with zoom slider below it
                    with gr.Column():
                        trajectory_plot = gr.Plot(label="Trajectory Comparison")
                        zoom_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="🔍 Zoom (1=100m, 10=10m)",
                        )
                    velocity_plot = gr.Plot(label="Velocity & Long. Acceleration")
                    lateral_plot = gr.Plot(label="Lat. Acceleration & Curvature")

                # Time slider for visualization control
                gr.Markdown("## ⏱️ Time Control")
                gr.Markdown(
                    "*Drag to view vehicle footprints at different times. "
                    "Also shows sampling points on velocity/acceleration plots.*"
                )
                time_slider = gr.Slider(
                    minimum=0,
                    maximum=79,
                    value=40,
                    step=1,
                    label="Time Step (0.1s per step, total 8.0s) - Updates footprint & plot markers",
                )

                # Metrics and Selection in parallel columns
                with gr.Row():
                    # Left column: Metrics display
                    with gr.Column(scale=2):
                        metrics_display = gr.Markdown(
                            value="*Metrics will appear after loading trajectories*",
                            label="📊 Metrics",
                        )

                    # Right column: Selection buttons
                    with gr.Column(scale=1):
                        gr.Markdown("## ✅ Selection")
                        gr.Markdown(
                            "**Green** = Deterministic (baseline)\n\n"
                            "**Orange** = Stochastic (candidate)"
                        )
                        select_orange_btn = gr.Button(
                            "✓ Orange (Stochastic) is Better",
                            variant="primary",
                            size="lg",
                        )
                        select_gt_btn = gr.Button(
                            "🎯 GT is Best",
                            variant="stop",
                            size="lg",
                        )
                        regenerate_btn = gr.Button(
                            "🔄 Regenerate Stochastic",
                            variant="secondary",
                            size="lg",
                        )

                # Navigation
                gr.Markdown("## 🧭 Navigation (Jump between samples)")
                
                # Define navigation buttons configuration
                nav_buttons_config = [
                    {"delta": -30, "label": "← 30"},
                    {"delta": -10, "label": "← 10"},
                    {"delta": -1, "label": "← 1"},
                    {"delta": 1, "label": "1 →"},
                    {"delta": 10, "label": "10 →"},
                    {"delta": 30, "label": "30 →"},
                ]
                
                # Create navigation buttons dynamically
                nav_buttons = {}
                with gr.Row():
                    for config in nav_buttons_config:
                        nav_buttons[config["delta"]] = gr.Button(config["label"], size="sm")

        # Event handlers
        # Helper: append button interactivity states to any annotator result tuple.
        # select_orange_btn is disabled when the stochastic trajectory is pruned.
        # select_gt_btn is disabled when GT is not available for the current sample.
        def _with_btn(result):
            # Disable orange button only when pruning is actively ON and trajectory failed.
            # When pruning is OFF it's a visual indicator only — user can still annotate.
            orange_blocked = annotator.is_pruned and annotator.enable_initial_pruning
            return (
                *result,
                gr.update(interactive=not orange_blocked),
                gr.update(interactive=annotator.gt_available),
            )

        # Common input lists (pruning controls + time_step appended to existing params)
        _std_inputs = [noise_scale, fde_threshold, ade_threshold, max_retries, zoom_slider, gt_similarity_checkbox]
        _pruning_inputs = [enable_initial_pruning_checkbox, initial_pos_threshold_slider, initial_yaw_threshold_slider]
        _guidance_inputs = [enable_guidance_checkbox, use_collision_checkbox, use_route_following_checkbox, use_lane_keeping_checkbox, use_centerline_following_checkbox, guidance_scale_slider]
        _full_inputs = _std_inputs + _pruning_inputs + _guidance_inputs + [time_slider]

        # Common output list (selection buttons appended so interactivity can be controlled)
        _std_outputs = [trajectory_plot, velocity_plot, lateral_plot, fde_text, progress_text, metrics_display, sidebar_status, history_display]
        _full_outputs = _std_outputs + [select_orange_btn, select_gt_btn]

        # Orange (stochastic) is selected as winner, green (deterministic) as loser
        select_orange_btn.click(
            fn=lambda ns, ft, at, mr, zl, gt, eip, ipt, iyt, eg, uc, urf, ulk, ucf, gs, ts: _with_btn(
                annotator.select_winner("trajectory_2", ns, ft, at, mr, zl, gt, eip, ipt, iyt, eg, uc, urf, ulk, ucf, gs, ts)
            ),
            inputs=_full_inputs,
            outputs=_full_outputs,
        )

        # GT is selected as winner, deterministic (green) as loser
        select_gt_btn.click(
            fn=lambda ns, ft, at, mr, zl, gt, eip, ipt, iyt, eg, uc, urf, ulk, ucf, gs, ts: _with_btn(
                annotator.select_gt_as_winner(ns, ft, at, mr, zl, gt, eip, ipt, iyt, eg, uc, urf, ulk, ucf, gs, ts)
            ),
            inputs=_full_inputs,
            outputs=_full_outputs,
        )

        regenerate_btn.click(
            fn=lambda ns, ft, at, mr, zl, gt, eip, ipt, iyt, eg, uc, urf, ulk, ucf, gs, ts: _with_btn(
                annotator.regenerate(ns, ft, at, mr, zl, gt, eip, ipt, iyt, eg, uc, urf, ulk, ucf, gs, ts)
            ),
            inputs=_full_inputs,
            outputs=_full_outputs,
        )

        # Time slider handler - only redraws, does NOT regenerate trajectories
        time_slider.change(
            fn=lambda t, z: annotator.update_time_display(int(t), int(z)),
            inputs=[time_slider, zoom_slider],
            outputs=[trajectory_plot, velocity_plot, lateral_plot],
        )

        # Zoom slider handler - only redraws trajectory plot
        zoom_slider.change(
            fn=lambda t, z: annotator.update_time_display(int(t), int(z)),
            inputs=[time_slider, zoom_slider],
            outputs=[trajectory_plot, velocity_plot, lateral_plot],
        )

        # Navigation handlers - fix lambda closure issue and update jump size
        def make_jump_fn(delta_val):
            def jump_and_update(ns, ft, at, mr, zl, gt, eip, ipt, iyt, eg, uc, urf, ulk, ucf, gs, ts):
                annotator.update_jump_size(delta_val)
                return _with_btn(annotator.jump(delta_val, ns, ft, at, mr, zl, gt, eip, ipt, iyt, eg, uc, urf, ulk, ucf, gs, ts))
            return jump_and_update

        # Wire up navigation button handlers dynamically
        for delta, button in nav_buttons.items():
            button.click(
                fn=make_jump_fn(delta),
                inputs=_full_inputs,
                outputs=_full_outputs,
            )

        # Sidebar event handlers
        # Jump-to button - only trigger on button click, not on input change
        jump_to_btn.click(
            fn=lambda idx, ns, ft, at, mr, zl, gt, eip, ipt, iyt, eg, uc, urf, ulk, ucf, gs, ts: _with_btn(
                annotator.jump_to_index(idx, ns, ft, at, mr, zl, gt, eip, ipt, iyt, eg, uc, urf, ulk, ucf, gs, ts)
            ),
            inputs=[jump_to_input] + _full_inputs,
            outputs=_full_outputs,
        )

        # Filter radio
        show_finished_radio.change(
            fn=lambda filter_mode, ns, ft, at, mr, zl, gt, eip, ipt, iyt, eg, uc, urf, ulk, ucf, gs, ts: _with_btn(
                annotator.toggle_filter(filter_mode, ns, ft, at, mr, zl, gt, eip, ipt, iyt, eg, uc, urf, ulk, ucf, gs, ts)
            ),
            inputs=[show_finished_radio] + _full_inputs,
            outputs=_full_outputs,
        )

        # Next unlabeled button
        next_unlabeled_btn.click(
            fn=lambda ns, ft, at, mr, zl, gt, eip, ipt, iyt, eg, uc, urf, ulk, ucf, gs, ts: _with_btn(
                annotator.jump_to_next_unlabeled(ns, ft, at, mr, zl, gt, eip, ipt, iyt, eg, uc, urf, ulk, ucf, gs, ts)
            ),
            inputs=_full_inputs,
            outputs=_full_outputs,
        )

        # Auto-skip checkbox
        auto_skip_checkbox.change(
            fn=lambda checked: setattr(annotator, 'auto_skip_labeled', checked),
            inputs=[auto_skip_checkbox],
            outputs=[],
        )

        # Launch Training button
        launch_training_btn.click(
            fn=lambda: annotator.launch_training(),
            inputs=None,
            outputs=[trajectory_plot, velocity_plot, lateral_plot, fde_text, progress_text, metrics_display, sidebar_status, history_display],
        )

        # Keyboard navigation handler - only trigger on valid arrow keys
        def handle_keyboard_event(key, ns, ft, at, mr, zl, gt, eip, ipt, iyt, eg, uc, urf, ulk, ucf, gs, ts):
            print(f"Python handler received key: '{key}'")
            # Parse the key (format is "ArrowLeft:123" or "ArrowRight:123")
            actual_key = key.split(':')[0] if ':' in key else key
            print(f"Parsed key: '{actual_key}'")

            if actual_key in ["ArrowLeft", "ArrowRight"]:
                direction = "left" if actual_key == "ArrowLeft" else "right"
                print(f"Processing {direction} navigation")
                result = _with_btn(annotator.handle_keyboard_navigation(direction, ns, ft, at, mr, zl, gt, eip, ipt, iyt, eg, uc, urf, ulk, ucf, gs, ts))
                print(f"Navigation complete")
                return result
            else:
                print(f"Ignoring key: '{actual_key}'")
                # Return gr.update() to not change anything
                return [gr.update()] * 10

        keyboard_input.change(
            fn=handle_keyboard_event,
            inputs=[keyboard_input] + _full_inputs,
            outputs=_full_outputs,
        )

        # Setup keyboard capture on load (load from external JS file)
        # Wrap the IIFE in a function for Gradio
        demo.load(
            fn=None,
            inputs=None,
            outputs=None,
            js=f"function() {{ {js_content} }}"
        )

        # Load first sample on startup (separate from JS)
        demo.load(
            fn=lambda ns, ft, at, mr, zl, gt, eip, ipt, iyt, eg, uc, urf, ulk, ucf, gs, ts: _with_btn(
                annotator.load_sample(ns, ft, at, mr, zl, gt, eip, ipt, iyt, eg, uc, urf, ulk, ucf, gs, ts)
            ),
            inputs=_full_inputs,
            outputs=_full_outputs,
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
