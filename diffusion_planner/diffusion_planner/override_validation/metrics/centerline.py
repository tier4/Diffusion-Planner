"""Centerline-ADE/FDE metric for Override Open-loop validation."""

import torch

_PREDICTION_TIMESTEP_SECONDS = 0.1


def _horizon_seconds(parameters: dict) -> float:
    horizon_seconds = float(parameters.get("horizon_seconds", 8.0))
    if horizon_seconds <= 0:
        raise ValueError("centerline horizon_seconds must be positive")
    return horizon_seconds


def _lane_batch(
    inputs: dict[str, torch.Tensor], batch_size: int, device: torch.device
) -> torch.Tensor:
    """Extract route-centerline tensors as ``(B, lane, point, feature)``."""
    lanes = inputs.get("route_lanes")
    if lanes is None:
        lanes = inputs.get("lanes")
    if lanes is None:
        raise ValueError("centerline metric requires route_lanes or lanes in the NPZ")

    lanes = lanes.to(device)
    # NPZ samples store a singleton context axis: [1, lane, point, feature].
    # DataLoader collates it to [B, 1, lane, point, feature].
    if lanes.ndim == 5:
        if lanes.shape[1] != 1:
            raise ValueError(
                f"expected singleton route_lanes context axis, got {tuple(lanes.shape)}"
            )
        lanes = lanes[:, 0]
    elif lanes.ndim == 3:
        lanes = lanes.unsqueeze(0)
    if lanes.ndim != 4:
        raise ValueError(f"route_lanes must have 3, 4, or 5 dimensions, got {tuple(lanes.shape)}")
    if lanes.shape[0] != batch_size or lanes.shape[-1] < 4:
        raise ValueError(
            "route_lanes batch/features do not match prediction: "
            f"lanes={tuple(lanes.shape)}, prediction_batch={batch_size}"
        )
    return lanes


def _point_to_polylines_min_dist(
    points: torch.Tensor,
    polylines: torch.Tensor,
    valid_points: torch.Tensor,
) -> torch.Tensor:
    """Return each point's nearest distance to any valid segment in ``polylines``."""
    valid_segments = valid_points[:, :-1] & valid_points[:, 1:]
    if not valid_segments.any():
        raise ValueError("centerline metric found no valid route-centerline segments")

    seg_p1 = polylines[:, :-1, :][valid_segments]
    seg_p2 = polylines[:, 1:, :][valid_segments]
    segment = seg_p2 - seg_p1
    segment_length_squared = (segment**2).sum(dim=-1).clamp_min(1e-10)
    offset = points[:, None, :] - seg_p1[None, :, :]
    projection = (
        (offset * segment[None, :, :]).sum(dim=-1) / segment_length_squared[None, :]
    ).clamp(0.0, 1.0)
    closest = seg_p1[None, :, :] + projection[:, :, None] * segment[None, :, :]
    return (points[:, None, :] - closest).norm(dim=-1).min(dim=1).values


def evaluate_centerline(
    prediction: torch.Tensor, inputs: dict[str, torch.Tensor], parameters: dict
) -> dict[str, torch.Tensor]:
    """Return distance-to-centerline ADE and FDE in metres for each prediction.

    Each ego point is projected onto the nearest valid segment of the route
    centerlines. ``ade_m`` is the mean distance through the requested horizon;
    ``fde_m`` is the final distance at that horizon.
    """
    if prediction.ndim != 3 or prediction.shape[-1] < 2:
        raise ValueError(f"prediction must have shape (B, T, D>=2), got {tuple(prediction.shape)}")

    horizon_seconds = _horizon_seconds(parameters)
    batch_size, available_steps, _ = prediction.shape
    horizon_steps = int(round(horizon_seconds / _PREDICTION_TIMESTEP_SECONDS))
    if horizon_steps < 1:
        raise ValueError("centerline horizon selects zero prediction steps")
    horizon_steps = min(horizon_steps, available_steps)

    lanes = _lane_batch(inputs, batch_size, prediction.device)
    prediction_xy = prediction[:, :horizon_steps, :2]
    distances = []
    for batch_index in range(batch_size):
        lane_features = lanes[batch_index]
        centerlines = lane_features[..., :2].to(dtype=prediction.dtype)
        # Padded rows are all zero. Direction makes an actual centerline point
        # at the ego origin valid, unlike an x/y-only nonzero check.
        valid_points = lane_features[..., :4].abs().sum(dim=-1) > 1e-6
        distances.append(
            _point_to_polylines_min_dist(prediction_xy[batch_index], centerlines, valid_points)
        )
    per_timestep_distance = torch.stack(distances, dim=0)
    return {
        "ade_m": per_timestep_distance.mean(dim=1),
        "fde_m": per_timestep_distance[:, -1],
    }
