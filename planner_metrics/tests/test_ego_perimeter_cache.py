"""The cached ego perimeter must be the tensor the inlined loop used to build.

It feeds the per-tick road-border distance, so a different tensor here moves a metric.
"""

import torch

from planner_metrics.subscores import _ego_perimeter_points


def _inline(ego_shape, device, dtype):
    """The construction exactly as it was written inside compute_road_border_penalty."""
    wb, length, width = (ego_shape[i].item() for i in range(3))
    ro = (length - wb) / 2
    pts = []
    for j in range(20):
        f = j / 19
        pts.append((-ro + f * length, -width / 2))
        pts.append((-ro + f * length, width / 2))
        pts.append((-ro, -width / 2 + f * width))
        pts.append((length - ro, -width / 2 + f * width))
    return torch.tensor(pts, device=device, dtype=dtype)


def test_matches_the_inlined_construction():
    for shape in ([2.8, 4.8, 1.9], [2.75, 5.0, 1.85], [3.1, 4.6, 2.0]):
        for dtype in (torch.float32, torch.float64):
            ego_shape = torch.tensor(shape, dtype=dtype)
            got = _ego_perimeter_points(ego_shape, "cpu", dtype)
            assert torch.equal(got, _inline(ego_shape, "cpu", dtype)), (shape, dtype)


def test_a_host_side_shape_gives_the_same_points():
    """Supplying the shape on the host must be equivalent to reading it off the tensor --
    the two branches must not transform it differently."""
    ego_shape = torch.tensor([2.75, 5.0, 1.85], dtype=torch.float32)
    host = tuple(float(v) for v in ego_shape[:3].tolist())
    assert torch.equal(
        _ego_perimeter_points(ego_shape, "cpu", torch.float32),
        _ego_perimeter_points(ego_shape, "cpu", torch.float32, host),
    )
