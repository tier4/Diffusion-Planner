"""Signed in-cut geometry: is the ego biased toward the INSIDE of a bend?

The recovery harness reports an UNSIGNED lane usage (the scorer computes signed
``ego_lat`` then returns ``ego_lat.abs() / side_hw``, and the harness then takes
``sqrt(abs(...))``). An in-cut of 0.4 m and an out-swing of 0.4 m are therefore the
same number, so a directional question -- does this recipe bias the planner toward
the inside of a bend -- is invisible to it.

Convention, stated once and unit-tested below:

  signed_lat_m  : + = ego is LEFT of the route direction   (from
                  ``lat_offset_and_naive_score``, the scorer's own convention)
  kappa         : + = route turns LEFT (counter-clockwise, yaw increasing)
  incut_m       = sign(kappa) * signed_lat_m
                  + = ego is toward the INSIDE of the bend  (cutting the corner)
                  - = ego is toward the OUTSIDE (swinging wide)

On a left-hand bend the inside of the curve is to the left, so an ego that cuts
the corner is left of the centreline: same sign as kappa, not opposite.
"""

import numpy as np


def signed_curvature_from_poses(poses: np.ndarray) -> float:
    """Mean signed curvature (1/m) over a window of (x, y, yaw) poses.

    + = left turn. Uses unwrapped yaw over travelled arc length, which is robust
    to the pose sampling rate; returns 0.0 for a window with no travel.
    """
    poses = np.asarray(poses, dtype=float)
    if poses.shape[0] < 3:
        return 0.0
    seg = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)
    arc = float(seg.sum())
    if arc < 1e-3:
        return 0.0
    yaw = np.unwrap(poses[:, 2])
    return float((yaw[-1] - yaw[0]) / arc)


def incut_from_signed_lat(signed_lat_m: np.ndarray, kappa: float) -> np.ndarray:
    """Project signed lateral offset onto the inside of the bend.

    Returns metres, + = toward the inside. A straight window (kappa == 0) has no
    inside, so it returns all-NaN rather than silently reporting zeros that would
    dilute a curve-only average.
    """
    signed_lat_m = np.asarray(signed_lat_m, dtype=float)
    if kappa == 0.0:
        return np.full_like(signed_lat_m, np.nan)
    return np.sign(kappa) * signed_lat_m
