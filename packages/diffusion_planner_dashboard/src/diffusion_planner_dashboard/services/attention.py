"""Run a planner over one frame and report what its fusion attention looked at."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from diffusion_planner.analysis import (
    SceneTokenLayout,
    capture_fusion_attention,
    class_summary,
    ego_query_attention,
    token_records,
)
from diffusion_planner.data.transforms import PlannerDataNormalizer
from diffusion_planner.models.diffusion_planner import DiffusionPlanner

__all__ = ["AttentionReading", "run_attention"]


@dataclass(frozen=True)
class AttentionReading:
    """Ego-query attention over one frame's scene tokens."""

    attention: NDArray[np.float32]
    """Share of the ego token's attention per scene token, shape ``(K,)``."""
    records: list[dict[str, Any]]
    """One entry per valid token, most-attended first."""
    layout: SceneTokenLayout
    layer_count: int
    head_count: int

    def by_block(self) -> dict[str, dict[str, float]]:
        """Attention aggregated per token block."""
        return class_summary(self.records)

    def by_agent_class(self) -> dict[str, dict[str, float]]:
        """Attention aggregated per neighbor agent class."""
        return class_summary(self.records, key="agent_class")


def run_attention(
    model: DiffusionPlanner,
    frame_data: Mapping[str, Any],
    *,
    device: str,
    layer: int | str = "mean",
    head: int | None = None,
) -> AttentionReading:
    """Encode one raw frame with capture enabled and summarize the result.

    The frame is normalized here exactly as inference normalizes it, so an
    edited frame is read the same way a recorded one is. Only the scene encoder
    runs: fusion attention is settled before any denoising step, so sampling
    would cost time without changing the reading.

    Args:
        model: A planner already on ``device``.
        frame_data: One raw, unnormalized frame.
        device: Torch device string.
        layer: ``"mean"``, ``"last"``, or a fusion layer index.
        head: An attention head, or ``None`` to average them.

    Returns:
        The reading, including per-token records for display.
    """
    torch_device = torch.device(device)
    normalizer = PlannerDataNormalizer()
    normalized = normalizer(
        {key: np.asarray(value) for key, value in frame_data.items()}
    )
    input_data = {
        key: torch.as_tensor(value, device=torch_device).unsqueeze(0)
        for key, value in normalized.items()
    }

    with capture_fusion_attention(model) as capture, torch.no_grad():
        model.scene_encoder(input_data)

    layout = SceneTokenLayout.from_dimensions()
    attention = ego_query_attention(capture, layout, layer=layer, head=head)
    scores = attention[0].detach().float().cpu().numpy()
    head_count = int(capture.by_layer()[0].weights.shape[1])
    return AttentionReading(
        attention=scores,
        records=token_records(frame_data, scores, layout),
        layout=layout,
        layer_count=capture.layer_count,
        head_count=head_count,
    )
