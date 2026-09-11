"""Capture and summarize the scene encoder's fusion attention.

The fusion encoder is a stock ``nn.TransformerEncoder``, whose layers never
return attention weights during an ordinary forward pass. Wrapping each layer's
``self_attn`` and forcing ``need_weights`` exposes them.

Capturing perturbs the output slightly, and it is worth being precise about why,
because the numbers below are the budget any caller inherits. Two kernel choices
are involved, neither of which changes the mathematics:

- PyTorch's multi-head fast path replaces the layer body with a fused kernel and
  never calls ``self_attn``, so with it enabled **nothing is captured at all**.
  It is disabled for the duration and restored on exit. Fused against reference
  kernel accounts for about ``2.2e-6`` on the fused scene tokens.
- ``need_weights=True`` takes the explicit softmax path rather than the fused
  attention kernel, worth a further ``7.9e-7`` on the same tensors.

Measured end to end on a trained checkpoint through the full ten-step sampler,
that comes to ``1.2e-6`` maximum on trajectories of scale ``1.6`` — a relative
``7.7e-7`` — and leaves the turn-indicator decision unchanged. So a captured run
is numerically equivalent to an ordinary one at float32 rounding, not identical
to it. Treat the attention as a faithful reading of the model, and do not source
predictions from a captured pass when bit-exactness matters.

Attention is indexed by scene token, and the scene is one concatenated sequence
whose block order is fixed by ``SceneEncoder.forward``. :class:`SceneTokenLayout`
derives that order from the shared dimensions instead of hardcoding it, so it
follows the input schema rather than drifting from it.

"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from ..data.dimensions import (
    MAX_NUM_NEIGHBORS,
    NUM_INTERSECTION_AREAS,
    NUM_LANE_SEGMENTS,
    NUM_ROAD_BORDERS,
    NUM_ROUTE_SEGMENTS,
    NUM_STOP_LINES,
)
from ..visualizer.schema import AgentLabelIndex, NeighborIndex, PoseIndex

__all__ = [
    "AGENT_CLASS_NAMES",
    "EMPTY_SLOT_NAME",
    "UNLABELED_NAME",
    "AttentionCapture",
    "LayerAttention",
    "SceneTokenLayout",
    "capture_fusion_attention",
    "class_summary",
    "ego_query_attention",
    "fusion_attention",
    "fusion_dimensions",
    "neighbor_classes",
    "neighbor_valid",
    "token_records",
]

AGENT_CLASS_NAMES: tuple[str, ...] = tuple(
    member.name.removeprefix("IS_").lower() for member in AgentLabelIndex
)
"""Agent class names in ``agent_label`` column order, unknown included."""

UNLABELED_NAME = "unlabeled"
"""A valid agent whose label row is all zero, distinct from an explicit unknown."""

EMPTY_SLOT_NAME = "empty"
"""An unoccupied neighbor slot. Never a class: ``argmax`` on zeros would say vehicle."""

_NEIGHBOR_BLOCK = "neighbors"
_EGO_HISTORY_BLOCK = "ego_history"
_POSITIONLESS_BLOCKS = frozenset({"ego_shape"})

_BLOCK_SIZES: tuple[tuple[str, int], ...] = (
    (_NEIGHBOR_BLOCK, MAX_NUM_NEIGHBORS),
    ("lanes", NUM_LANE_SEGMENTS),
    ("route_lanes", NUM_ROUTE_SEGMENTS),
    ("intersection_area", NUM_INTERSECTION_AREAS),
    ("stop_lines", NUM_STOP_LINES),
    ("road_borders", NUM_ROAD_BORDERS),
    (_EGO_HISTORY_BLOCK, 1),
    ("goal_pose", 1),
    ("ego_shape", 1),
)
"""Block order and size, mirroring ``SceneEncoder.forward``'s concatenation."""


@dataclass(frozen=True)
class TokenBlock:
    """One contiguous run of scene tokens sharing a source tensor."""

    name: str
    start: int
    stop: int

    def __len__(self) -> int:
        return self.stop - self.start

    @property
    def indices(self) -> slice:
        """Slice selecting this block out of the token axis."""
        return slice(self.start, self.stop)


@dataclass(frozen=True)
class SceneTokenLayout:
    """Names and index ranges of the fused scene token sequence."""

    blocks: tuple[TokenBlock, ...]

    @classmethod
    def from_dimensions(cls) -> SceneTokenLayout:
        """Build the layout implied by the current input schema."""
        blocks: list[TokenBlock] = []
        cursor = 0
        for name, size in _BLOCK_SIZES:
            blocks.append(TokenBlock(name, cursor, cursor + size))
            cursor += size
        return cls(tuple(blocks))

    @property
    def total(self) -> int:
        """Number of tokens the scene encoder emits."""
        return self.blocks[-1].stop

    @property
    def names(self) -> tuple[str, ...]:
        """Block names in token order."""
        return tuple(block.name for block in self.blocks)

    @property
    def ego_query_index(self) -> int:
        """Index of the ego history token, the query the reports are built on."""
        return self.block(_EGO_HISTORY_BLOCK).start

    def block(self, name: str) -> TokenBlock:
        """Return the block called ``name``."""
        for candidate in self.blocks:
            if candidate.name == name:
                return candidate
        raise KeyError(f"no scene token block named {name!r}; have {self.names}")

    def slice_for(self, name: str) -> slice:
        """Slice selecting the named block out of the token axis."""
        return self.block(name).indices

    def class_of(self, index: int) -> tuple[str, int]:
        """Return the block name and block-local index of a token."""
        if not 0 <= index < self.total:
            raise IndexError(f"token {index} outside 0..{self.total - 1}")
        for block in self.blocks:
            if block.start <= index < block.stop:
                return block.name, index - block.start
        raise AssertionError("token index fell outside every block")


@dataclass(frozen=True)
class LayerAttention:
    """Attention weights and key input recorded from one fusion layer."""

    layer: int
    weights: torch.Tensor
    """Per-head attention with shape ``(B, heads, Q, K)``."""
    keys: torch.Tensor
    """The tensor actually passed as keys and values, for value-norm work."""


@dataclass
class AttentionCapture:
    """Attention recorded from the fusion encoder while the context is open."""

    records: list[LayerAttention] = field(default_factory=list)

    def clear(self) -> None:
        """Drop everything recorded so far, before another forward pass."""
        self.records.clear()

    def by_layer(self) -> dict[int, LayerAttention]:
        """Latest record per layer, so a second forward pass supersedes the first."""
        latest: dict[int, LayerAttention] = {}
        for record in self.records:
            latest[record.layer] = record
        return latest

    @property
    def layer_count(self) -> int:
        """Number of distinct fusion layers recorded."""
        return len(self.by_layer())

    def stacked(self) -> torch.Tensor:
        """Every layer's weights stacked as ``(layers, B, heads, Q, K)``."""
        latest = self.by_layer()
        if not latest:
            raise RuntimeError(
                "no attention was captured; run the model inside "
                "capture_fusion_attention() before reading it"
            )
        return torch.stack([latest[index].weights for index in sorted(latest)])


def _find_fusion_encoder(model: nn.Module) -> nn.Module:
    """Locate the scene encoder's fusion stack inside a planner."""
    for module in model.modules():
        if type(module).__name__ == "FusionEncoder":
            return module
    raise RuntimeError("no FusionEncoder found in the model")


def _fusion_layers(fusion: nn.Module) -> list[nn.Module]:
    """Return the fusion encoder's transformer layers."""
    transformer = getattr(fusion, "transformer", None)
    layers = getattr(transformer, "layers", None)
    if layers is None:
        raise RuntimeError(
            "FusionEncoder has no transformer.layers; the encoder structure "
            "changed and the attention capture needs updating"
        )
    return list(layers)


def fusion_dimensions(model: nn.Module) -> tuple[int, int]:
    """Return the fusion stack's layer and head counts without running it.

    Lets a caller build layer and head selectors before deciding what to
    capture, instead of running the model once just to discover its shape.
    """
    layers = _fusion_layers(_find_fusion_encoder(model))
    heads = int(cast(nn.MultiheadAttention, layers[0].self_attn).num_heads)
    return len(layers), heads


@contextmanager
def capture_fusion_attention(model: nn.Module) -> Iterator[AttentionCapture]:
    """Record fusion attention for every forward pass inside the block.

    Each layer's ``self_attn`` is temporarily wrapped to request per-head
    weights, and PyTorch's multi-head fast path is disabled for the duration
    because it replaces the layer body with a fused kernel that never calls
    ``self_attn``. Both are restored on exit. See the module docstring for what
    that costs numerically.

    Args:
        model: A planner, or any module containing the scene encoder.

    Yields:
        The capture, populated as the model runs.

    Raises:
        RuntimeError: If the block completed without recording anything, which
            means the forward pass routed around the wrapper. Failing here is
            deliberate: silently returning no attention reads as "this scene has
            none" rather than "the capture broke".
    """
    fusion = _find_fusion_encoder(model)
    layers = _fusion_layers(fusion)
    capture = AttentionCapture()

    fastpath = torch.backends.mha.get_fastpath_enabled()
    originals: list[tuple[nn.MultiheadAttention, Any]] = []
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        for index, layer in enumerate(layers):
            attention = cast(nn.MultiheadAttention, layer.self_attn)
            original = attention.forward
            originals.append((attention, original))
            attention.forward = _make_recorder(original, capture, index)
        yield capture
        if not capture.records:
            raise RuntimeError(
                f"the fusion encoder ran no captured attention across "
                f"{len(layers)} layers; the forward pass bypassed self_attn, so "
                "the capture wrapper needs updating for this PyTorch version"
            )
    finally:
        for attention, original in originals:
            attention.forward = original
        torch.backends.mha.set_fastpath_enabled(fastpath)


def _make_recorder(original: Any, capture: AttentionCapture, index: int) -> Any:
    """Wrap one ``MultiheadAttention.forward`` so it also records its weights."""

    def forward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        output, weights = original(query, key, value, **kwargs)
        if weights is not None:
            capture.records.append(
                LayerAttention(index, weights.detach(), key.detach())
            )
        return output, weights

    return forward


def _selected_layers(capture: AttentionCapture, layer: int | str) -> list[int]:
    """Resolve a ``mean`` / ``last`` / index selector to concrete layer indices."""
    available = sorted(capture.by_layer())
    if not available:
        raise RuntimeError(
            "no attention was captured; run the model inside "
            "capture_fusion_attention() before reading it"
        )
    if layer == "mean":
        return available
    if layer == "last":
        return [available[-1]]
    index = int(layer)
    if index not in available:
        raise ValueError(f"layer must be mean, last, or one of {available}")
    return [index]


def fusion_attention(
    capture: AttentionCapture,
    layer: int | str = "mean",
    head: int | None = None,
) -> torch.Tensor:
    """Average the captured attention over the selected layers and heads.

    Args:
        capture: A populated capture.
        layer: ``"mean"`` for every layer, ``"last"``, or a layer index.
        head: A head index, or ``None`` to average the heads.

    Returns:
        Attention with shape ``(B, Q, K)``.
    """
    latest = capture.by_layer()
    selected = _selected_layers(capture, layer)
    weights = torch.stack([latest[index].weights for index in selected]).mean(dim=0)
    if head is None:
        return weights.mean(dim=1)
    if not 0 <= head < weights.shape[1]:
        raise ValueError(f"head must be 0..{weights.shape[1] - 1}")
    return weights[:, head]


def ego_query_attention(
    capture: AttentionCapture,
    layout: SceneTokenLayout | None = None,
    layer: int | str = "mean",
    head: int | None = None,
) -> torch.Tensor:
    """Attention paid by the ego history token to every scene token.

    Returns:
        Attention with shape ``(B, K)``, summing to one over valid keys.
    """
    layout = layout or SceneTokenLayout.from_dimensions()
    weights = fusion_attention(capture, layer=layer, head=head)
    return weights[:, layout.ego_query_index]


def _as_numpy(values: Any) -> NDArray[np.float32]:
    """Return a detached float32 NumPy view of a tensor or array."""
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(values, dtype=np.float32)


def neighbor_valid(neighbor_agents_past: Any) -> NDArray[np.bool_]:
    """Occupancy of each neighbor slot, matching the encoder's own test.

    Validity comes from the pose history and never from the label, because an
    occupied slot may legitimately carry an all-zero label row.
    """
    poses = _as_numpy(neighbor_agents_past)
    return np.abs(poses).sum(axis=(-2, -1)) > 0.0


def neighbor_classes(agent_label: Any, neighbor_agents_past: Any) -> NDArray[np.str_]:
    """Name the class of every neighbor slot.

    Empty slots become :data:`EMPTY_SLOT_NAME` rather than a class: their label
    row is all zero, and ``argmax`` of zeros would silently report the first
    class for every unoccupied slot. An occupied slot whose label is all zero
    becomes :data:`UNLABELED_NAME`, which is a different statement from an
    explicit unknown one-hot.

    Returns:
        Names with shape ``(N,)`` for a single frame, or ``(B, N)`` batched.
    """
    labels = _as_numpy(agent_label)
    valid = neighbor_valid(neighbor_agents_past)
    labelled = labels.sum(axis=-1) > 0.0
    classes = np.argmax(labels, axis=-1)

    names = np.asarray(AGENT_CLASS_NAMES, dtype=np.str_)[classes]
    names = np.where(labelled, names, UNLABELED_NAME)
    return np.where(valid, names, EMPTY_SLOT_NAME).astype(np.str_)


def _polyline_positions(values: NDArray[np.float32]) -> NDArray[np.float32]:
    """Representative point per polyline: the valid vertex nearest the ego."""
    points = values[..., : PoseIndex.COS_YAW]
    occupied = np.abs(values).sum(axis=-1) > 0.0
    distance = np.where(occupied, np.linalg.norm(points, axis=-1), np.inf)
    nearest = np.argmin(distance, axis=-1)
    chosen = np.take_along_axis(points, nearest[..., None, None], axis=-2)[..., 0, :]
    empty = ~occupied.any(axis=-1)
    return np.where(empty[..., None], np.nan, chosen)


def _block_positions(
    block: str, frame: Mapping[str, Any]
) -> NDArray[np.float32] | None:
    """Representative ego-frame position of every token in a block."""
    if block == _NEIGHBOR_BLOCK:
        poses = _as_numpy(frame["neighbor_agents_past"])
        positions = poses[:, -1, : NeighborIndex.COS_YAW]
        return np.where(neighbor_valid(poses)[:, None], positions, np.nan)
    if block == _EGO_HISTORY_BLOCK:
        ego = _as_numpy(frame["ego_agent_past"])
        return ego[-1, : PoseIndex.COS_YAW][None, :]
    if block == "goal_pose":
        goal = _as_numpy(frame["goal_pose"])
        return goal[: PoseIndex.COS_YAW][None, :]
    if block == "ego_shape":
        return None
    values = frame.get(block)
    if values is None:
        return None
    return _polyline_positions(_as_numpy(values))


def token_records(
    frame: Mapping[str, Any],
    attention: Any,
    layout: SceneTokenLayout | None = None,
) -> list[dict[str, Any]]:
    """Describe every valid scene token and the attention it received.

    Args:
        frame: One unbatched frame of planner input tensors.
        attention: Attention over tokens with shape ``(K,)``.
        layout: Token layout, derived from the schema when omitted.

    Returns:
        Records sorted by attention, most-attended first.
    """
    layout = layout or SceneTokenLayout.from_dimensions()
    scores = _as_numpy(attention).reshape(-1)
    if scores.shape[0] != layout.total:
        raise ValueError(
            f"attention has {scores.shape[0]} tokens, layout expects {layout.total}"
        )

    classes = neighbor_classes(frame["agent_label"], frame["neighbor_agents_past"])
    records: list[dict[str, Any]] = []
    for block in layout.blocks:
        positions = _block_positions(block.name, frame)
        for local_index in range(len(block)):
            token_index = block.start + local_index
            position = None if positions is None else positions[local_index]
            if position is not None and bool(np.isnan(position).any()):
                position = None
            if (
                block.name == _NEIGHBOR_BLOCK
                and classes[local_index] == EMPTY_SLOT_NAME
            ):
                continue
            if position is None and block.name not in _POSITIONLESS_BLOCKS:
                continue
            record: dict[str, Any] = {
                "token_index": token_index,
                "block": block.name,
                "block_index": local_index,
                "attention": float(scores[token_index]),
                "attention_pct": float(scores[token_index] * 100.0),
                "x_m": None if position is None else float(position[0]),
                "y_m": None if position is None else float(position[1]),
                "distance_m": (
                    None
                    if position is None
                    else float(np.hypot(float(position[0]), float(position[1])))
                ),
            }
            if block.name == _NEIGHBOR_BLOCK:
                record["agent_class"] = str(classes[local_index])
            records.append(record)
    return sorted(records, key=lambda item: item["attention"], reverse=True)


def class_summary(
    records: list[dict[str, Any]], key: str = "block"
) -> dict[str, dict[str, float]]:
    """Aggregate records by token block, or by ``agent_class``.

    Selectivity is the share of attention divided by the share of valid tokens:
    ``1.0`` means the class receives exactly what count-proportional dilution
    would give it, and above ``1.0`` means the model prefers it.
    """
    grouped: dict[str, list[float]] = {}
    for record in records:
        name = record.get(key)
        if name is None:
            continue
        grouped.setdefault(str(name), []).append(record["attention"])

    total_attention = sum(sum(values) for values in grouped.values())
    total_tokens = sum(len(values) for values in grouped.values())
    summary: dict[str, dict[str, float]] = {}
    for name, values in sorted(grouped.items()):
        attention = sum(values)
        share = attention / total_attention if total_attention > 0.0 else 0.0
        token_share = len(values) / total_tokens if total_tokens > 0 else 0.0
        summary[name] = {
            "tokens": float(len(values)),
            "attention": attention,
            "share": share,
            "token_share": token_share,
            "selectivity": share / token_share if token_share > 0.0 else float("nan"),
        }
    return summary
