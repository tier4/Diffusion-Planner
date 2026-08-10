"""Input-side augmentations for the route.

Implements the high- and medium-priority proposals from
docs/route_augmentation_proposals.md:

- Route tail truncation (high): randomly shorten the forward extent of
  ``route_lanes`` so the model sees the short routes that occur near the goal /
  map edge at deployment time.
- Speed-limit unknown dropout (high): drop ``*_has_speed_limit`` to False
  (scene-wide, lanes and route together) so the encoder's ``unknown_speed_emb``
  path is actually trained.
- Route geometry noise (medium): per-segment constant lateral offset of the
  route centerline, simulating map survey error / map version drift and
  loosening centerline snapping.
- Route head trim (medium): drop the first route segment and shift the rest
  forward, simulating nearest-lanelet selection jitter at the route start.

(Traffic-light unknown dropout and scene flip live on separate branches.)

All transforms preserve the all-zero padding convention and never fabricate a
state that contradicts the ground truth trajectory. Applied in train_epoch
before StatePerturbation, on raw (un-normalized) ego-centric inputs.
"""

import torch

# Geometry channels used for validity checks (x, y, dx, dy, LB, RB), matching
# the masks in data_augmentation.StatePerturbation.
_GEOM_DIM = 8

_ROUTE_KEYS = ("route_lanes", "route_lanes_speed_limit", "route_lanes_has_speed_limit")


class RouteAugmentation:
    """Randomized route / speed-limit augmentation.

    Each component is applied independently per sample with its own
    probability. Probabilities of 0 disable a component.
    """

    def __init__(
        self,
        device,
        truncation_prob: float = 0.0,
        truncation_min_m: float = 60.0,
        truncation_max_m: float = 200.0,
        speed_limit_unknown_prob: float = 0.0,
        geometry_noise_prob: float = 0.0,
        geometry_noise_std_m: float = 0.15,
        head_trim_prob: float = 0.0,
    ):
        assert 0.0 < truncation_min_m <= truncation_max_m
        assert geometry_noise_std_m >= 0.0
        self._device = device
        self._truncation_prob = truncation_prob
        self._truncation_min_m = truncation_min_m
        self._truncation_max_m = truncation_max_m
        self._speed_limit_unknown_prob = speed_limit_unknown_prob
        self._geometry_noise_prob = geometry_noise_prob
        self._geometry_noise_std_m = geometry_noise_std_m
        self._head_trim_prob = head_trim_prob

    @torch.no_grad()
    def __call__(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Head trim first (it moves the route start), then the tail truncation
        # measures its arc-length budget from the new start, matching how a
        # shifted nearest-lanelet pick would look at deployment time.
        if self._head_trim_prob > 0.0:
            self._trim_route_head(inputs)
        if self._truncation_prob > 0.0:
            self._truncate_route_tail(inputs)
        if self._geometry_noise_prob > 0.0:
            self._jitter_route_geometry(inputs)
        if self._speed_limit_unknown_prob > 0.0:
            self._drop_speed_limit(inputs)
        return inputs

    def _trim_route_head(self, inputs: dict[str, torch.Tensor]) -> None:
        """Drop the first route segment and shift the rest one slot forward.

        Simulates the nearest-lanelet selection jitter at the route start
        (replan timing, lane changes, inside intersections). Shifting keeps the
        "slot 0 = current lanelet" convention of the ordered positional
        embedding; leaving a zero row at slot 0 would be a pattern that never
        occurs at deployment time. Only applied when at least 3 segments are
        valid, so at least 2 segments of forward intent always survive.
        """
        route = inputs["route_lanes"]  # [B, P, V, D]
        B = route.shape[0]
        seg_valid = route[..., :_GEOM_DIM].abs().sum(dim=(2, 3)) > 0  # [B, P]
        n_valid = seg_valid.sum(dim=1)  # [B]
        apply = (torch.rand(B, device=route.device) < self._head_trim_prob) & (n_valid >= 3)
        for key in _ROUTE_KEYS:
            t = inputs[key]
            # In-place masked shift: the advanced-indexing gather materializes
            # only the selected rows, so the transient allocation scales with
            # the applied subset instead of the whole batch, and an all-False
            # mask degenerates to an empty copy (no host sync needed to skip).
            t[apply, :-1] = t[apply, 1:]
            t[apply, -1] = 0

    def _truncate_route_tail(self, inputs: dict[str, torch.Tensor]) -> None:
        """Zero out route segments starting beyond a sampled arc-length budget.

        Distance is accumulated along the route polyline from the first route
        segment (the ego's current lanelet), which approximates distance from
        ego. The first segment is always kept so the immediate intent is never
        destroyed.
        """
        route = inputs["route_lanes"]  # [B, P, V, D]
        B = route.shape[0]

        valid_pt = route[..., :_GEOM_DIM].abs().sum(-1) > 0  # [B, P, V]
        xy = route[..., :2]
        step = (xy[:, :, 1:] - xy[:, :, :-1]).norm(dim=-1)  # [B, P, V-1]
        pair_valid = (valid_pt[:, :, 1:] & valid_pt[:, :, :-1]).to(step.dtype)
        seg_len = (step * pair_valid).sum(-1)  # [B, P]

        # Arc length from the route start to each segment's start.
        cum_before = torch.cumsum(seg_len, dim=1) - seg_len  # [B, P]

        budget = self._truncation_min_m + (
            self._truncation_max_m - self._truncation_min_m
        ) * torch.rand(B, 1, device=route.device)
        apply = torch.rand(B, 1, device=route.device) < self._truncation_prob

        drop = (cum_before >= budget) & apply  # [B, P]
        drop[:, 0] = False

        inputs["route_lanes"][drop] = 0.0
        inputs["route_lanes_speed_limit"][drop] = 0.0
        inputs["route_lanes_has_speed_limit"][drop] = False

    def _jitter_route_geometry(self, inputs: dict[str, torch.Tensor]) -> None:
        """Shift each route segment laterally by a constant per-segment offset.

        The offset is constant within a segment, so the per-point direction
        channels (dx, dy) and the relative boundary offsets stay exactly
        consistent -- only the absolute placement moves, like a map survey
        error. Point-wise independent noise would corrupt the direction
        channels and is deliberately not done (see the proposal doc).
        """
        route = inputs["route_lanes"]  # [B, P, V, D]
        B, P, _, _ = route.shape

        valid_pt = route[..., :_GEOM_DIM].abs().sum(-1) > 0  # [B, P, V]
        mean_dir = (route[..., 2:4] * valid_pt.unsqueeze(-1)).sum(dim=2)  # [B, P, 2]
        unit = mean_dir / mean_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        normal = torch.stack([-unit[..., 1], unit[..., 0]], dim=-1)  # [B, P, 2]

        offset = torch.randn(B, P, 1, device=route.device) * self._geometry_noise_std_m
        apply = (torch.rand(B, 1, 1, device=route.device) < self._geometry_noise_prob).to(
            route.dtype
        )
        shift = normal * offset * apply  # [B, P, 2]

        route[..., :2] += shift.unsqueeze(2) * valid_pt.unsqueeze(-1)

    def _drop_speed_limit(self, inputs: dict[str, torch.Tensor]) -> None:
        """Scene-wide speed-limit unknown dropout on lanes and route together.

        Scene-wide (not per-segment) so the same lanelet never carries a limit
        in ``lanes`` while missing it in ``route_lanes``.
        """
        route = inputs["route_lanes"]
        # Mask on the tensors' own device: an all-False mask makes the fills
        # no-ops, so no host-device sync is spent on an early-out check.
        scene = torch.rand(route.shape[0], device=route.device) < self._speed_limit_unknown_prob
        for prefix in ("lanes", "route_lanes"):
            inputs[f"{prefix}_speed_limit"][scene] = 0.0
            inputs[f"{prefix}_has_speed_limit"][scene] = False
