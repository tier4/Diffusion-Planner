"""Top-level dashboard views."""

from .attention import render_attention
from .data_augmentation import render_data_augmentation
from .frame_browser import render_frame_browser
from .home import render_home
from .training_results import render_training_results

__all__ = [
    "render_attention",
    "render_data_augmentation",
    "render_frame_browser",
    "render_home",
    "render_training_results",
]
