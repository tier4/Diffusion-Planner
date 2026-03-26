"""Batch thumbnail generation using visualize_inputs().

Renders every Nth scene in a batch as a small matplotlib figure for the GUI.
"""

import io
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image

from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils.visualize_input import visualize_inputs

from scene_search.batch_search import Batch


def _load_npz_as_viz_data(npz_path: str) -> dict[str, torch.Tensor]:
    """Load an NPZ file into the dict format expected by visualize_inputs().

    Same logic as preference_optimization/utils.py:load_npz_data but CPU-only
    and without model-specific keys (delay, etc.).
    """
    loaded = np.load(str(npz_path))
    data: dict[str, torch.Tensor] = {}

    for key, value in loaded.items():
        if key in {"map_name", "token", "delay", "version"}:
            continue
        data[key] = torch.tensor(np.expand_dims(value, axis=0))

    if "goal_pose" in data:
        data["goal_pose"] = heading_to_cos_sin(data["goal_pose"])
    if "ego_agent_past" in data:
        data["ego_agent_past"] = heading_to_cos_sin(data["ego_agent_past"])

    if "ego_shape" not in data:
        data["ego_shape"] = torch.tensor([[2.79, 4.34, 1.70]], dtype=torch.float32)

    return data


def render_single_thumbnail(npz_path: str, view_range: float = 60.0, figsize: tuple = (5, 5), dpi: int = 90) -> Figure:
    """Render a single scene thumbnail using visualize_inputs().

    Returns a matplotlib Figure.
    """
    data = _load_npz_as_viz_data(npz_path)
    fig = Figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    visualize_inputs(data, save_path=None, ax=ax, view_ranges=[view_range])
    ax.set_aspect("equal")
    fig.tight_layout(pad=0.5)
    return fig


# Size presets for thumbnails
THUMB_SIZE = (5, 5, 90)  # (figsize_w, figsize_h, dpi)


def _render_thumbnail_to_bytes(args: tuple) -> tuple[int, bytes, str]:
    """Worker function for parallel thumbnail rendering.

    Args: (index, npz_path, label, view_range, figsize_w, figsize_h, dpi)
    Returns: (index, png_bytes, label)
    """
    if len(args) == 7:
        idx, npz_path, label, view_range, fw, fh, dpi = args
    else:
        idx, npz_path, label, view_range = args
        fw, fh, dpi = THUMB_SIZE
    try:
        fig = render_single_thumbnail(npz_path, view_range=view_range, figsize=(fw, fh), dpi=dpi)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        buf.seek(0)
        return (idx, buf.read(), label)
    except Exception as e:
        return (idx, None, f"{label} (error: {e})")


def render_batch_thumbnails(
    batch: Batch,
    every_nth: int = 10,
    view_range: float = 60.0,
    max_workers: int = 4,
) -> list[tuple[bytes | None, str]]:
    """Render every Nth scene in a batch as PNG thumbnails.

    Args:
        batch: The Batch to preview.
        every_nth: Sample every Nth scene for thumbnails.
        view_range: View range in meters for visualize_inputs().
        max_workers: Parallel workers for rendering.

    Returns:
        List of (png_bytes, label) tuples. png_bytes is None on error.
        Labels show frame offset relative to first central match:
        "t-30", "t-20", ..., "t=0*", "t+10", ...
        Central scenes get a * suffix.
    """
    if not batch.scenes:
        return []

    # Determine which scenes to render
    central_set = set(batch.central_indices)
    # Always include the first central scene in the sample
    first_central = batch.central_indices[0] if batch.central_indices else 0

    indices_to_render = list(range(0, len(batch.scenes), every_nth))
    # Ensure the first central scene is included
    if first_central not in indices_to_render:
        indices_to_render.append(first_central)
        indices_to_render.sort()

    # Build render tasks with labels
    tasks = []
    for scene_idx in indices_to_render:
        offset = scene_idx - first_central
        if offset == 0:
            label = "t=0*"
        elif offset < 0:
            label = f"t{offset}"
        else:
            label = f"t+{offset}"
        if scene_idx in central_set and offset != 0:
            label += "*"

        fw, fh, dpi = THUMB_SIZE
        tasks.append((len(tasks), batch.scenes[scene_idx], label, view_range, fw, fh, dpi))

    # Render in parallel
    results = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for idx, png_bytes, label in executor.map(_render_thumbnail_to_bytes, tasks):
            results[idx] = (png_bytes, label)

    return results


def get_central_task_index(batch: Batch, every_nth: int = 10) -> tuple[list[tuple], int]:
    """Return the render tasks list and the index of the central scene task.

    Used for progressive rendering: render central first, then the rest.
    """
    if not batch.scenes:
        return [], 0

    central_set = set(batch.central_indices)
    first_central = batch.central_indices[0] if batch.central_indices else 0

    indices_to_render = list(range(0, len(batch.scenes), every_nth))
    if first_central not in indices_to_render:
        indices_to_render.append(first_central)
        indices_to_render.sort()

    tasks = []
    central_task_idx = 0
    for scene_idx in indices_to_render:
        offset = scene_idx - first_central
        if offset == 0:
            label = "t=0*"
            central_task_idx = len(tasks)
        elif offset < 0:
            label = f"t{offset}"
        else:
            label = f"t+{offset}"
        if scene_idx in central_set and offset != 0:
            label += "*"
        tasks.append((len(tasks), batch.scenes[scene_idx], label, 60.0))

    return tasks, central_task_idx


def render_central_thumbnail(batch: Batch, every_nth: int = 10, view_range: float = 60.0) -> tuple[list[tuple], int]:
    """Render only the central scene thumbnail immediately.

    Returns (pil_images_with_placeholders, n_total) where non-central entries
    are grey placeholder images with "Loading..." text.
    """
    from PIL import ImageDraw

    tasks, central_task_idx = get_central_task_index(batch, every_nth)
    if not tasks:
        return [], 0

    # Render central thumbnail
    _, png_bytes, central_label = _render_thumbnail_to_bytes(tasks[central_task_idx])

    # Build result with placeholders for non-central
    result = []
    for i, (_, _, label, _) in enumerate(tasks):
        if i == central_task_idx and png_bytes is not None:
            img = Image.open(io.BytesIO(png_bytes))
            result.append((img, label))
        else:
            # Grey placeholder
            placeholder = Image.new("RGB", (400, 400), color=(220, 220, 220))
            draw = ImageDraw.Draw(placeholder)
            draw.text((150, 190), "Loading...", fill=(150, 150, 150))
            draw.text((160, 210), label, fill=(120, 120, 120))
            result.append((placeholder, label))

    return result, len(tasks)


def render_remaining_thumbnails(
    batch: Batch,
    every_nth: int = 10,
    view_range: float = 60.0,
    max_workers: int = 4,
) -> list[tuple]:
    """Render all thumbnails (used after the central one is already shown).

    Returns full list of (PIL.Image, label) tuples.
    """
    tasks, central_task_idx = get_central_task_index(batch, every_nth)
    if not tasks:
        return []

    # Update view_range in tasks
    tasks = [(idx, path, label, view_range) for idx, path, label, _ in tasks]

    results = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for idx, png_bytes, label in executor.map(_render_thumbnail_to_bytes, tasks):
            if png_bytes is not None:
                results[idx] = (Image.open(io.BytesIO(png_bytes)), label)
            else:
                results[idx] = (Image.new("RGB", (400, 400), (220, 220, 220)), label)

    return results


def thumbnails_to_pil_images(thumbnails: list[tuple[bytes | None, str]]) -> list[tuple]:
    """Convert PNG byte thumbnails to PIL Images for Gradio Gallery.

    Returns list of (PIL.Image, label) tuples.
    """
    result = []
    for png_bytes, label in thumbnails:
        if png_bytes is not None:
            img = Image.open(io.BytesIO(png_bytes))
            result.append((img, label))
    return result
