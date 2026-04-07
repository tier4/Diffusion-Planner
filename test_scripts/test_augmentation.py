import argparse
import time
from copy import deepcopy
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.data_augmentation_bridge import (
    StatePerturbation as BridgeStatePerturbation,
)
from diffusion_planner.utils.visualize_input import visualize_inputs

parser = argparse.ArgumentParser()
parser.add_argument("target_npz", type=Path)
parser.add_argument("save_dir", type=Path)
parser.add_argument("--augment_type", choices=["quintic", "bridge"], default="quintic")
args = parser.parse_args()

target_npz = args.target_npz

save_dir = args.save_dir
save_dir.mkdir(parents=True, exist_ok=True)

loaded = np.load(target_npz)
data = {}
for key, value in loaded.items():
    if key == "token":
        continue
    data[key] = torch.tensor(value).unsqueeze(0)
    if key == "goal_pose" or key == "ego_agent_past":
        data[key] = heading_to_cos_sin(data[key])

# Load future trajectories separately
ego_future = torch.tensor(loaded["ego_agent_future"]).unsqueeze(0)
neighbors_future = torch.tensor(loaded["neighbor_agents_future"]).unsqueeze(0)

if args.augment_type == "quintic":
    aug = StatePerturbation(augment_prob=1.0, num_refine=20, device="cpu")
else:
    aug = BridgeStatePerturbation(augment_prob=1.0, device="cpu")

# Save original data visualization with augmentation range rectangle
original_save_path = save_dir / "original.png"
fig, ax = plt.subplots(figsize=(10, 10))

# Visualize inputs on the ax
view_range = 30
visualize_inputs(deepcopy(data), save_path=None, ax=ax, view_ranges=[view_range])

# Get augmentation ranges from the aug object
lo = aug._low.cpu().numpy()[0]  # Extract from tuple
hi = aug._high.cpu().numpy()[0]  # Extract from tuple
x_min, y_min = lo[0], lo[1]
x_max, y_max = hi[0], hi[1]

# Draw the augmentation range rectangle
rect = patches.Rectangle(
    (x_min, y_min),
    x_max - x_min,
    y_max - y_min,
    linewidth=2,
    edgecolor="red",
    facecolor="none",
    linestyle="--",
    label="Augmentation Range",
)
ax.add_patch(rect)
ax.legend()

plt.tight_layout()
plt.savefig(original_save_path, dpi=100)
plt.close()

trial_num = 10
elapsed_times = []
for i in range(trial_num):
    t0 = time.perf_counter()
    aug_data, aug_ego_future, aug_neighbors_future = aug(
        deepcopy(data), ego_future.clone(), neighbors_future.clone()
    )
    elapsed_times.append(time.perf_counter() - t0)

    # Save augmented data to npz file
    data_dict = {}
    for key, value in aug_data.items():
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.squeeze(0).detach().cpu().numpy()
        else:
            data_dict[key] = value

    # Add future trajectories with consistent naming
    data_dict["ego_agent_future"] = aug_ego_future.squeeze(0).detach().cpu().numpy()
    data_dict["neighbor_agents_future"] = aug_neighbors_future.squeeze(0).detach().cpu().numpy()
    aug_data["ego_agent_future"] = aug_ego_future
    aug_data["neighbor_agents_future"] = aug_neighbors_future

    # Save to npz file
    output_path = save_dir / f"augmented_{i:08d}.npz"
    np.savez(output_path, **data_dict)

    # Use deepcopy to avoid side effects from visualize_inputs
    visualize_inputs(
        deepcopy(aug_data), save_dir / f"augmented_{i:08d}.png", view_ranges=[view_range]
    )

print(f"Augmented data saved: {trial_num} files to {save_dir}")
print(
    f"Augmentation time: mean={sum(elapsed_times) / len(elapsed_times) * 1000:.1f}ms  min={min(elapsed_times) * 1000:.1f}ms  max={max(elapsed_times) * 1000:.1f}ms"
)
