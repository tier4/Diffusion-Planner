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
    ``max_renamed_fraction`` of the labelled agents in a frame are relabelled.

    ``max_renamed_fraction`` bounds the frame as a whole, which does not stop a
    single class from disappearing. Measured on 400 real frames at the settings
    used for the 2026-09-08 runs, 41 frames lost every pedestrian and 45 lost
    every bicycle, because renaming is drawn per agent and bicycles are rare.
    ``preserve_last_of_class`` keeps one agent of each class present in the
    frame, so the scene never silently stops containing pedestrians.
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
        preserve_last_of_class: bool = False,
        record_true_label: bool = False,
    ) -> None:
        self.probability = probability
        self.probability_vehicle = probability_vehicle
        self.probability_pedestrian = probability_pedestrian
        self.probability_bicycle = probability_bicycle
        self.distance_scale_max = distance_scale_max
        self.distance_scale_range_m = distance_scale_range_m
        self.probability_cap = probability_cap
        self.max_renamed_fraction = max_renamed_fraction
        self.preserve_last_of_class = preserve_last_of_class
        self.record_true_label = record_true_label

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
        if self.record_true_label:
            # The class that was hidden, kept so it can be used as a target. Renaming throws
            # it away otherwise, which makes it impossible to ask the model to recover it or
            # to check afterwards which agents were renamed.
            output["agent_label_true"] = labels.copy()

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
        if self.preserve_last_of_class:
            selected = self._keep_one_per_class(selected, candidates, classes)
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

    def _keep_one_per_class(
        self,
        selected: NDArray[np.intp],
        candidates: NDArray[np.intp],
        classes: NDArray[np.intp],
    ) -> NDArray[np.intp]:
        """Drop one rename per class that would otherwise be emptied from the frame.

        A frame that contained pedestrians should still contain a pedestrian afterwards.
        Without this the label is removed from every example of a class in the frame, which
        is the one case where the augmentation stops being label noise and becomes a
        different scene.
        """
        if selected.size == 0:
            return selected
        keep = np.ones(selected.size, dtype=bool)
        selected_set = set(selected.tolist())
        for class_index in np.unique(classes):
            in_class = candidates[classes == class_index]
            if not in_class.size:
                continue
            if any(agent not in selected_set for agent in in_class.tolist()):
                continue  # at least one survivor already
            # Every agent of this class was picked: spare one at random.
            spared = int(np.random.permutation(in_class)[0])
            keep &= selected != spared
        return selected[keep]

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
