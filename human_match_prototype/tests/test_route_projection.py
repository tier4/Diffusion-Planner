import numpy as np
import pytest

from human_match_prototype.route_projection import (
    frenet_energy_scores,
    project_to_route,
    stitch_route_lanes,
)


def _make_route_lanes(
    segments: list[np.ndarray], total_slots: int = 25, pts_per_seg: int = 20, dim: int = 33
) -> np.ndarray:
    """Build a (total_slots, pts_per_seg, dim) route_lanes array from a list of (K, 2) xy segments."""
    out = np.zeros((total_slots, pts_per_seg, dim), dtype=np.float32)
    for i, seg_xy in enumerate(segments):
        n = min(len(seg_xy), pts_per_seg)
        out[i, :n, :2] = seg_xy[:n]
    return out


def _straight_segments(n_segs: int = 3, pts: int = 20, spacing: float = 1.0) -> list[np.ndarray]:
    """Create n_segs straight segments along +x with exact endpoint matching."""
    segs = []
    for i in range(n_segs):
        x0 = i * (pts - 1) * spacing
        xs = x0 + np.arange(pts) * spacing
        ys = np.zeros(pts)
        segs.append(np.stack([xs, ys], axis=-1))
    return segs


class TestStitchRouteLanes:
    def test_straight_continuous(self):
        """Three straight segments with exact joins produce a valid route."""
        segs = _straight_segments(3)
        rl = _make_route_lanes(segs)
        route = stitch_route_lanes(rl)
        assert route.qa.route_valid is True
        assert route.qa.n_valid_segments == 3
        assert route.qa.max_segment_gap < 0.01
        assert route.centerline.shape[1] == 2
        assert len(route.centerline) > 0
        # Arc length should be monotonically increasing
        assert np.all(np.diff(route.arc_length) >= 0)

    def test_empty_segments_skipped(self):
        """Segments that are all-zero are ignored."""
        segs = _straight_segments(2)
        rl = _make_route_lanes(segs)
        # Slots 2-24 are already zero
        route = stitch_route_lanes(rl)
        assert route.qa.n_valid_segments == 2
        assert route.qa.route_valid is True

    def test_small_gap_deduplicated(self):
        """Gap <= 0.5m between segments is deduplicated."""
        segs = _straight_segments(2)
        # Shift segment 1 start by 0.3m (below dedup threshold)
        segs[1][:, 0] += 0.3
        rl = _make_route_lanes(segs)
        route = stitch_route_lanes(rl)
        assert route.qa.route_valid is True
        assert route.qa.max_segment_gap < 0.5

    def test_moderate_gap_interpolated(self):
        """Gap between 0.5m and 3.0m is linearly interpolated."""
        segs = _straight_segments(2)
        segs[1][:, 0] += 2.0  # 2m gap
        rl = _make_route_lanes(segs)
        route = stitch_route_lanes(rl)
        assert route.qa.route_valid is True
        assert route.qa.total_interpolated_gap > 1.5

    def test_large_gap_invalid(self):
        """Gap > 3.0m marks route as invalid."""
        segs = _straight_segments(2)
        segs[1][:, 0] += 5.0  # 5m gap
        rl = _make_route_lanes(segs)
        route = stitch_route_lanes(rl)
        assert route.qa.route_valid is False
        assert route.qa.max_segment_gap > 3.0

    def test_batch_dim_squeezed(self):
        """(1, 25, 20, 33) shape is handled by squeezing batch dim."""
        segs = _straight_segments(2)
        rl = _make_route_lanes(segs)
        rl_batched = rl[np.newaxis, ...]  # (1, 25, 20, 33)
        route = stitch_route_lanes(rl_batched)
        assert route.qa.route_valid is True
        assert route.qa.n_valid_segments == 2


class TestProjectToRoute:
    def test_on_route_zero_lateral(self):
        """Points exactly on a straight route have d ≈ 0."""
        segs = _straight_segments(3)
        rl = _make_route_lanes(segs)
        route = stitch_route_lanes(rl)
        # Query points along the route
        points = np.array([[5.0, 0.0], [15.0, 0.0], [30.0, 0.0]])
        s, d, proj_dist = project_to_route(route, points)
        np.testing.assert_allclose(d, 0.0, atol=1e-4)
        np.testing.assert_allclose(proj_dist, 0.0, atol=1e-4)
        # Arc lengths should be increasing
        assert s[0] < s[1] < s[2]

    def test_lateral_offset_sign(self):
        """Point left of route (positive y for +x route) has d > 0."""
        segs = _straight_segments(3)
        rl = _make_route_lanes(segs)
        route = stitch_route_lanes(rl)
        left_point = np.array([[10.0, 2.0]])
        right_point = np.array([[10.0, -2.0]])
        _, d_left, _ = project_to_route(route, left_point)
        _, d_right, _ = project_to_route(route, right_point)
        assert d_left[0] > 0  # left of route
        assert d_right[0] < 0  # right of route

    def test_batch_projection(self):
        """(N, T, 2) input returns (N, T) shaped outputs."""
        segs = _straight_segments(3)
        rl = _make_route_lanes(segs)
        route = stitch_route_lanes(rl)
        points = np.zeros((4, 20, 2))
        points[:, :, 0] = np.linspace(0, 30, 20)
        s, d, proj_dist = project_to_route(route, points)
        assert s.shape == (4, 20)
        assert d.shape == (4, 20)


class TestFrenetEnergyScores:
    def test_identical_projections_zero(self):
        """When all samples have same s,d as human, ES ≈ 0."""
        T = 80
        human_sd = np.stack([np.linspace(0, 50, T), np.zeros(T)], axis=-1)
        samples_sd = np.tile(human_sd, (64, 1, 1))
        result = frenet_energy_scores(human_sd, samples_sd, {"4s": 40})
        assert result["es_lon_4s"] == pytest.approx(0.0, abs=1e-6)
        assert result["es_lat_4s"] == pytest.approx(0.0, abs=1e-6)

    def test_lateral_offset_detected(self):
        """Samples with lateral offset produce positive es_lat."""
        T = 80
        human_sd = np.stack([np.linspace(0, 50, T), np.zeros(T)], axis=-1)
        samples_sd = np.tile(human_sd, (64, 1, 1))
        samples_sd[:, :, 1] += 1.5  # lateral offset
        result = frenet_energy_scores(human_sd, samples_sd, {"4s": 40})
        assert result["es_lat_4s"] > 0
        assert result["es_lon_4s"] == pytest.approx(0.0, abs=1e-6)
