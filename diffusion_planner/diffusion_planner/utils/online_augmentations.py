"""Online GPU augmentations integrated with repeat-draw force masks."""

import torch

from diffusion_planner.dimensions import (
    LB_X,
    LB_Y,
    LINE_TYPE_LEFT_START,
    LINE_TYPE_NUM,
    LINE_TYPE_RIGHT_START,
    RB_X,
    RB_Y,
    TURN_INDICATOR_OUTPUT_ENABLE_LEFT,
    TURN_INDICATOR_OUTPUT_ENABLE_RIGHT,
    Y,
    dY,
)


def _force_mask(force: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
    if force is None:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)
    return force.to(device=device, dtype=torch.bool)


class FlipAugmentation:
    """Mirror complete scenes across the ego longitudinal axis."""

    def __init__(self, flip_prob: float, device: torch.device | str) -> None:
        if not 0.0 <= flip_prob <= 1.0:
            raise ValueError(f"flip_prob must be in [0, 1], got {flip_prob}")
        self._flip_prob = flip_prob
        self._device = torch.device(device)

    @staticmethod
    def _negate_(x, channels, flip):
        signs = torch.ones((x.shape[0], x.shape[-1]), dtype=x.dtype, device=x.device)
        signs[:, channels] = (1 - 2 * flip.to(dtype=x.dtype))[:, None]
        x.mul_(signs.reshape(x.shape[0], *([1] * (x.ndim - 2)), x.shape[-1]))
        return x

    @staticmethod
    def _swap_slices_(x, left, right, flip):
        left_values = x[..., left]
        right_values = x[..., right]
        saved_left = left_values.clone()
        cond = flip.reshape(-1, *([1] * (x.ndim - 1)))
        left_values.copy_(torch.where(cond, right_values, left_values))
        right_values.copy_(torch.where(cond, saved_left, right_values))

    def _flip_lanes(self, lanes, flip):
        self._negate_(lanes, [Y, dY, LB_Y, RB_Y], flip)
        self._swap_slices_(lanes, slice(LB_X, LB_Y + 1), slice(RB_X, RB_Y + 1), flip)
        left = slice(LINE_TYPE_LEFT_START, LINE_TYPE_LEFT_START + LINE_TYPE_NUM)
        right = slice(LINE_TYPE_RIGHT_START, LINE_TYPE_RIGHT_START + LINE_TYPE_NUM)
        self._swap_slices_(lanes, left, right, flip)
        return lanes

    @staticmethod
    def _future_flip_channels(future):
        if future.shape[-1] == 3:
            return [1, 2]
        if future.shape[-1] == 4:
            return [1, 3]
        raise ValueError(f"unsupported future shape {tuple(future.shape)}")

    def __call__(self, inputs, ego_future, neighbors_future, force=None):
        batch_size = ego_future.shape[0]
        flip = torch.rand(batch_size, device=ego_future.device) < self._flip_prob
        flip |= _force_mask(force, batch_size, ego_future.device)

        self._negate_(inputs["ego_current_state"], [1, 3, 5, 7, 8, 9], flip)
        self._negate_(inputs["ego_agent_past"], [1, 3], flip)
        self._negate_(inputs["goal_pose"], [1, 3], flip)
        self._negate_(inputs["neighbor_agents_past"], [1, 3, 5], flip)
        self._negate_(inputs["static_objects"], [1, 3], flip)
        self._negate_(inputs["polygons"], [1], flip)
        self._negate_(inputs["line_strings"], [1], flip)
        inputs["lanes"] = self._flip_lanes(inputs["lanes"], flip)
        inputs["route_lanes"] = self._flip_lanes(inputs["route_lanes"], flip)

        indicators = inputs["turn_indicators"]
        swapped = torch.where(
            indicators == TURN_INDICATOR_OUTPUT_ENABLE_LEFT,
            torch.full_like(indicators, TURN_INDICATOR_OUTPUT_ENABLE_RIGHT),
            torch.where(
                indicators == TURN_INDICATOR_OUTPUT_ENABLE_RIGHT,
                torch.full_like(indicators, TURN_INDICATOR_OUTPUT_ENABLE_LEFT),
                indicators,
            ),
        )
        indicators.copy_(torch.where(flip[:, None], swapped, indicators))

        self._negate_(ego_future, self._future_flip_channels(ego_future), flip)
        self._negate_(neighbors_future, self._future_flip_channels(neighbors_future), flip)
        inputs["ego_agent_future"] = ego_future
        inputs["neighbor_agents_future"] = neighbors_future
        return inputs, ego_future, neighbors_future


class NeighborDropoutAugmentation:
    """Drop complete neighbor tracks from both inputs and prediction targets."""

    def __init__(self, dropout_prob: float, device: torch.device | str) -> None:
        if not 0.0 <= dropout_prob <= 1.0:
            raise ValueError(f"dropout_prob must be in [0, 1], got {dropout_prob}")
        self._dropout_prob = dropout_prob
        self._device = torch.device(device)

    def __call__(self, inputs, ego_future, neighbors_future, force=None):
        past = inputs["neighbor_agents_past"]
        batch_size, neighbor_count = past.shape[:2]
        valid = torch.count_nonzero(past, dim=(-2, -1)).ne(0)
        drop = valid & (
            torch.rand(batch_size, neighbor_count, device=past.device) < self._dropout_prob
        )

        # A forced row drops one valid neighbor when its normal roll dropped none.
        forced = _force_mask(force, batch_size, past.device)
        needs_drop = forced & valid.any(dim=1) & ~drop.any(dim=1)
        rows = torch.nonzero(needs_drop, as_tuple=False).flatten()
        if rows.numel():
            first_valid = valid[rows].to(torch.int64).argmax(dim=1)
            drop[rows, first_valid] = True

        drop = drop.view(batch_size, neighbor_count, 1, 1)
        past.masked_fill_(drop, 0.0)
        neighbors_future.masked_fill_(drop, 0.0)
        return inputs, ego_future, neighbors_future


class NeighborNoiseAugmentation:
    """Add tracking-like Gaussian noise to valid neighbor-history frames."""

    def __init__(
        self,
        pos_noise_std: float,
        vel_noise_std: float,
        heading_noise_std: float,
        device: torch.device | str,
    ) -> None:
        self._pos_noise_std = pos_noise_std
        self._vel_noise_std = vel_noise_std
        self._heading_noise_std = heading_noise_std
        self._device = torch.device(device)

    def __call__(self, inputs, ego_future, neighbors_future, force=None):
        del force  # Noise already runs on every enabled sample with independent draws.
        past = inputs["neighbor_agents_past"]
        batch_size, neighbor_count, time_count, _ = past.shape
        valid = torch.ne(past[..., 0], 0)
        for feature_idx in range(1, 8):
            valid.logical_or_(torch.ne(past[..., feature_idx], 0))
        invalid = ~valid

        noise = torch.randn(
            batch_size,
            neighbor_count,
            time_count,
            2,
            device=past.device,
            dtype=past.dtype,
        )
        noise.masked_fill_(invalid.unsqueeze(-1), 0.0)
        past[..., 0:2].add_(noise, alpha=self._pos_noise_std)
        noise.normal_()
        noise.masked_fill_(invalid.unsqueeze(-1), 0.0)
        past[..., 4:6].add_(noise, alpha=self._vel_noise_std)

        heading = torch.atan2(past[..., 3], past[..., 2])
        heading.add_(
            torch.randn_like(heading),
            alpha=self._heading_noise_std,
        )
        past[..., 2].copy_(torch.cos(heading)).masked_fill_(invalid, 0.0)
        past[..., 3].copy_(torch.sin(heading)).masked_fill_(invalid, 0.0)
        return inputs, ego_future, neighbors_future


class TurnIndicatorDropoutAugmentation:
    """Drop indicator inputs per sample while preserving classification GT."""

    def __init__(self, dropout_prob: float, device: torch.device | str) -> None:
        if not 0.0 <= dropout_prob <= 1.0:
            raise ValueError(f"dropout_prob must be in [0, 1], got {dropout_prob}")
        self._dropout_prob = dropout_prob
        self._device = torch.device(device)

    def __call__(self, inputs, ego_future, neighbors_future, force=None):
        indicators = inputs["turn_indicators"]
        batch_size = indicators.shape[0]
        inputs["turn_indicators_gt_source"] = indicators.clone()
        drop = torch.rand(batch_size, device=indicators.device) < self._dropout_prob
        drop |= _force_mask(force, batch_size, indicators.device)
        inputs["turn_indicators"] = torch.where(
            drop[:, None], torch.zeros_like(indicators), indicators
        )
        return inputs, ego_future, neighbors_future
