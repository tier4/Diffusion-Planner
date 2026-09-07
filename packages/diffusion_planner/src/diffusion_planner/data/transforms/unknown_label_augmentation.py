"""NumPy unknown-class augmentation for planner dataset frames."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..dimensions import AGENT_LABEL_DIM
from .base import Frame, FrameLike

UNKNOWN_LABEL_INDEX = AGENT_LABEL_DIM - 1
KNOWN_LABEL_DIM = AGENT_LABEL_DIM - 1


class PlannerUnknownLabelAugmentation:
    """Widen ``agent_label`` to the unknown class and relabel some agents unknown.

    Shards written before the unknown class store a three-column one-hot, so a
    zero column is appended to feed the four-class model. Rows that were all
    zero stay all zero: an empty neighbor slot must not become an unknown agent.

    Valid agents labelled vehicle, pedestrian, or bicycle are then relabelled
    unknown with a probability that starts from the per-class rate, grows
    linearly with distance from the ego up to ``distance_scale_max`` at
    ``distance_scale_range_m``, and is clipped by ``probability_cap``. At most
    ``max_renamed_fraction`` of the labelled agents in a frame are relabelled,
    so no frame loses every known class.
    """

    def __init__(
        self,
        probability: float = 0.0,
        probability_vehicle: float | None = None,
        probability_pedestrian: float | None = None,
        probability_bicycle: float | None = None,
        distance_scale_max: float = 1.0,
        distance_scale_range_m: float = 50.0,
        probability_cap: float = 1.0,
        max_renamed_fraction: float = 1.0,
    ) -> None:
        self.probability = probability
        self.probability_vehicle = probability_vehicle
        self.probability_pedestrian = probability_pedestrian
        self.probability_bicycle = probability_bicycle
        self.distance_scale_max = distance_scale_max
        self.distance_scale_range_m = distance_scale_range_m
        self.probability_cap = probability_cap
        self.max_renamed_fraction = max_renamed_fraction

    @property
    def class_probabilities(self) -> NDArray[np.float32]:
        """Per-class rename rate, falling back to the base probability."""
        overrides = (
            self.probability_vehicle,
            self.probability_pedestrian,
            self.probability_bicycle,
        )
        return np.asarray(
            [self.probability if value is None else value for value in overrides],
            dtype=np.float32,
        )

    def __call__(self, input_data: FrameLike) -> Frame:
        output = dict(input_data)
        labels = _widen_labels(input_data["agent_label"])
        output["agent_label"] = labels

        class_probabilities = self.class_probabilities
        if not np.any(class_probabilities > 0.0):
            return output

        candidates = _known_class_agents(labels, input_data["neighbor_agents_past"])
        if candidates.size == 0:
            return output

        classes = np.argmax(labels[candidates, :KNOWN_LABEL_DIM], axis=-1)
        distances = np.hypot(
            input_data["neighbor_agents_past"][candidates, -1, 0],
            input_data["neighbor_agents_past"][candidates, -1, 1],
        )
        probabilities = self._rename_probabilities(classes, distances)
        selected = candidates[np.random.random(candidates.size) < probabilities]
        selected = self._apply_frame_cap(selected, candidates.size)
        if selected.size == 0:
            return output

        labels[selected] = 0.0
        labels[selected, UNKNOWN_LABEL_INDEX] = 1.0
        return output

    def _rename_probabilities(
        self, classes: NDArray[np.intp], distances: NDArray[np.floating]
    ) -> NDArray[np.float32]:
        """Scale each per-class rate by distance and clip it to the cap."""
        base = self.class_probabilities[classes]
        if self.distance_scale_range_m > 0.0:
            reach = np.minimum(distances / self.distance_scale_range_m, 1.0)
        else:
            reach = np.ones_like(distances)
        scale = 1.0 + (self.distance_scale_max - 1.0) * reach
        return np.clip(base * scale, 0.0, self.probability_cap).astype(np.float32)

    def _apply_frame_cap(
        self, selected: NDArray[np.intp], num_candidates: int
    ) -> NDArray[np.intp]:
        """Keep at most ``max_renamed_fraction`` of a frame's labelled agents."""
        limit = int(np.floor(self.max_renamed_fraction * num_candidates))
        if selected.size <= limit:
            return selected
        return np.random.permutation(selected)[:limit]


def _widen_labels(agent_label: NDArray[np.generic]) -> NDArray[np.generic]:
    """Return a copy of ``agent_label`` with an unknown column appended."""
    missing = AGENT_LABEL_DIM - agent_label.shape[-1]
    if missing < 0:
        raise ValueError(
            f"agent_label has {agent_label.shape[-1]} columns, "
            f"more than the expected {AGENT_LABEL_DIM}"
        )
    if missing == 0:
        return agent_label.copy()
    padding = np.zeros((*agent_label.shape[:-1], missing), dtype=agent_label.dtype)
    return np.concatenate((agent_label, padding), axis=-1)


def _known_class_agents(
    labels: NDArray[np.generic], neighbor_agents_past: NDArray[np.generic]
) -> NDArray[np.intp]:
    """Return indices of valid agents that still carry a known class."""
    valid = np.abs(neighbor_agents_past).sum(axis=(-2, -1)) > 0.0
    known = labels[..., :KNOWN_LABEL_DIM].sum(axis=-1) > 0.0
    return np.flatnonzero(valid & known)
