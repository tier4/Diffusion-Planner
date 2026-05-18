"""Scene branch tree — data model and JSON persistence for the scene branch editor.

A SceneTree tracks a base NPZ replay directory and a tree of branches.
Each branch can fork from a parent at a specific timestep, carry obstacle
placements and crop ranges, and store resimulated NPZ output.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from glob import glob
from pathlib import Path


@dataclass
class ObstaclePlacement:
    """A stopped vehicle placed by the user at a specific timestep."""

    label: str
    timestep: int
    x: float
    y: float
    yaw_deg: float
    length: float = 4.5
    width: float = 1.8
    history_steps: int = 30

    @property
    def yaw_rad(self) -> float:
        return math.radians(self.yaw_deg)

    def snapped(self) -> ObstaclePlacement:
        """Return a copy with position snapped to 10cm and yaw to 5 degrees."""
        return ObstaclePlacement(
            label=self.label,
            timestep=self.timestep,
            x=round(self.x, 1),
            y=round(self.y, 1),
            yaw_deg=round(self.yaw_deg / 5.0) * 5.0,
            length=self.length,
            width=self.width,
            history_steps=self.history_steps,
        )


@dataclass
class BranchNode:
    """A single branch in the scene tree."""

    id: str
    parent_id: str | None = None
    fork_timestep: int | None = None
    modifications: list[ObstaclePlacement] = field(default_factory=list)
    crop_range: tuple[int, int] | None = None
    resim_steps: int | None = None
    resim_advance_mode: str = "perfect"
    resim_model_path: str | None = None
    npz_dir: str | None = None

    def obstacles_at_or_before(self, timestep: int) -> list[ObstaclePlacement]:
        """Return obstacles that exist at the given timestep (placed at or before it)."""
        return [o for o in self.modifications if o.timestep <= timestep]


@dataclass
class SceneTree:
    """Tree of branching scene modifications with JSON persistence."""

    base_npz_dir: str
    ego_shape: tuple[float, float, float] = (2.925, 4.5, 1.9)
    branches: dict[str, BranchNode] = field(default_factory=dict)
    active_branch: str = "root"
    version: int = 1

    # ── Construction ──────────────────────────────────────────────────

    @classmethod
    def create_from_npz_dir(cls, npz_dir: str | Path) -> SceneTree:
        """Create a new tree with a root branch pointing at the given NPZ directory."""
        npz_dir = str(Path(npz_dir).resolve())
        npz_files = _scan_npz_dir(npz_dir)
        if not npz_files:
            raise ValueError(f"No replay_step_*.npz or step_*.npz files found in {npz_dir}")

        import numpy as np
        with np.load(npz_files[0]) as first:
            ego_shape = tuple(float(v) for v in first["ego_shape"]) if "ego_shape" in first else (2.925, 4.5, 1.9)

        tree = cls(
            base_npz_dir=npz_dir,
            ego_shape=ego_shape,
            branches={
                "root": BranchNode(id="root", npz_dir=npz_dir),
            },
        )
        return tree

    # ── Tree operations ───────────────────────────────────────────────

    def fork_branch(self, parent_id: str, timestep: int, new_id: str | None = None) -> str:
        """Create a new branch forking from parent_id at the given timestep.

        Returns the new branch ID.
        """
        if parent_id not in self.branches:
            raise KeyError(f"Parent branch '{parent_id}' not found")
        parent = self.branches[parent_id]
        seq = self.get_npz_sequence(parent_id)
        if timestep < 0 or timestep >= len(seq):
            raise IndexError(
                f"Timestep {timestep} out of range [0, {len(seq) - 1}] for branch '{parent_id}'"
            )

        if new_id is None:
            suffix = 1
            while f"{parent_id}_{suffix:03d}" in self.branches:
                suffix += 1
            new_id = f"{parent_id}_{suffix:03d}"

        if new_id in self.branches:
            raise ValueError(f"Branch '{new_id}' already exists")

        self.branches[new_id] = BranchNode(
            id=new_id,
            parent_id=parent_id,
            fork_timestep=timestep,
        )
        return new_id

    def add_obstacle(self, branch_id: str, placement: ObstaclePlacement) -> None:
        if branch_id not in self.branches:
            raise KeyError(f"Branch '{branch_id}' not found")
        snapped = placement.snapped()
        self.branches[branch_id].modifications.append(snapped)

    def remove_obstacle(self, branch_id: str, label: str) -> bool:
        if branch_id not in self.branches:
            raise KeyError(f"Branch '{branch_id}' not found")
        branch = self.branches[branch_id]
        before = len(branch.modifications)
        branch.modifications = [o for o in branch.modifications if o.label != label]
        return len(branch.modifications) < before

    def set_crop(self, branch_id: str, start: int, end: int) -> None:
        if branch_id not in self.branches:
            raise KeyError(f"Branch '{branch_id}' not found")
        if start < 0 or end < start:
            raise ValueError(f"Invalid crop range [{start}, {end}]")
        self.branches[branch_id].crop_range = (start, end)

    def clear_crop(self, branch_id: str) -> None:
        if branch_id not in self.branches:
            raise KeyError(f"Branch '{branch_id}' not found")
        self.branches[branch_id].crop_range = None

    def delete_branch(self, branch_id: str) -> list[str]:
        """Delete a branch and all its descendants. Returns list of deleted IDs."""
        if branch_id == "root":
            raise ValueError("Cannot delete root branch")
        if branch_id not in self.branches:
            raise KeyError(f"Branch '{branch_id}' not found")

        to_delete = []
        queue = [branch_id]
        while queue:
            bid = queue.pop(0)
            to_delete.append(bid)
            queue.extend(b.id for b in self.branches.values() if b.parent_id == bid)

        for bid in to_delete:
            del self.branches[bid]

        if self.active_branch in to_delete:
            self.active_branch = "root"

        return to_delete

    def get_children(self, branch_id: str) -> list[str]:
        return [b.id for b in self.branches.values() if b.parent_id == branch_id]

    def get_npz_sequence(self, branch_id: str) -> list[str]:
        """Return the sorted list of NPZ paths for a branch.

        For a forked branch without its own resim output, returns the
        parent's sequence starting from fork_timestep onward. If the
        branch has its own npz_dir (from resimulation), returns those
        files instead. Crop is applied last.
        """
        if branch_id not in self.branches:
            raise KeyError(f"Branch '{branch_id}' not found")
        branch = self.branches[branch_id]

        if branch.npz_dir is not None:
            files = _scan_npz_dir(branch.npz_dir)
        elif branch.parent_id is not None:
            parent_seq = self.get_npz_sequence(branch.parent_id)
            fork = branch.fork_timestep or 0
            files = parent_seq[fork:]
        else:
            return []

        if branch.crop_range is not None:
            start, end = branch.crop_range
            end = min(end, len(files) - 1)
            files = files[start : end + 1]
        return files

    def get_all_obstacles(self, branch_id: str) -> list[ObstaclePlacement]:
        """Get all obstacles for a branch, including inherited from non-resimulated ancestors.

        Stops climbing at any ancestor that has been resimulated (npz_dir set),
        because its obstacles are already baked into the NPZ as neighbors.
        """
        obstacles = []
        bid = branch_id
        while bid is not None:
            branch = self.branches.get(bid)
            if branch is None:
                break
            obstacles.extend(branch.modifications)
            # Stop inheriting from resimulated ancestors — their obstacles
            # are already in the NPZ neighbor data
            parent = self.branches.get(branch.parent_id) if branch.parent_id else None
            if parent is not None and parent.npz_dir is not None:
                break
            bid = branch.parent_id
        return obstacles

    def next_obstacle_label(self, branch_id: str) -> str:
        """Generate a unique label for the next obstacle in this branch."""
        existing = self.get_all_obstacles(branch_id)
        idx = len(existing) + 1
        while any(o.label == f"obstacle_{idx}" for o in existing):
            idx += 1
        return f"obstacle_{idx}"

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": self.version,
            "base_npz_dir": self.base_npz_dir,
            "ego_shape": list(self.ego_shape),
            "active_branch": self.active_branch,
            "branches": {
                bid: _branch_to_dict(b) for bid, b in self.branches.items()
            },
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> SceneTree:
        path = Path(path)
        data = json.loads(path.read_text())
        branches = {}
        for bid, bdict in data["branches"].items():
            mods = [ObstaclePlacement(**m) for m in bdict.pop("modifications", [])]
            crop = bdict.pop("crop_range", None)
            if crop is not None:
                crop = tuple(crop)
            branches[bid] = BranchNode(modifications=mods, crop_range=crop, **bdict)
        active = data.get("active_branch", "root")
        if active not in branches:
            active = "root" if "root" in branches else next(iter(branches), "root")
        return cls(
            version=data.get("version", 1),
            base_npz_dir=data["base_npz_dir"],
            ego_shape=tuple(data.get("ego_shape", [2.925, 4.5, 1.9])),
            active_branch=active,
            branches=branches,
        )


# ── Helpers ───────────────────────────────────────────────────────────


def _scan_npz_dir(npz_dir: str) -> list[str]:
    """Find and sort NPZ files in a directory.

    Supports replay_step_NNNN.npz, step_NNNN.npz, and generic *.npz patterns.
    """
    patterns = ["replay_step_*.npz", "step_*.npz"]
    files: list[str] = []
    for pat in patterns:
        files.extend(glob(str(Path(npz_dir) / pat)))
    if not files:
        all_npz = sorted(glob(str(Path(npz_dir) / "*.npz")))
        files = [f for f in all_npz if not f.endswith(".json")]
    return sorted(set(files))


def _branch_to_dict(b: BranchNode) -> dict:
    d = asdict(b)
    d["modifications"] = [asdict(m) for m in b.modifications]
    if b.crop_range is not None:
        d["crop_range"] = list(b.crop_range)
    return d
