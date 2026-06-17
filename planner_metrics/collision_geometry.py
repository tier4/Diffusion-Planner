"""Oriented-bounding-box geometry primitives (SAT signed distance + corners).

Pure torch helpers shared by the subscores and by the planner's
collision-avoidance guidance. Moved here so both can use them without a
dependency cycle; ``diffusion_planner.model.guidance.collision`` re-exports them
for backward compatibility.
"""

import torch


def batch_signed_distance_rect(rect1, rect2):
    """
    rect1: [B, 4, 2]
    rect2: [B, 4, 2]

    return [B] (signed distance between two rectangles)
    """
    B, _, _ = rect1.shape
    norm_vec = torch.stack(
        [
            rect1[:, 0] - rect1[:, 1],
            rect1[:, 1] - rect1[:, 2],
            rect2[:, 0] - rect2[:, 1],
            rect2[:, 1] - rect2[:, 2],
        ],
        dim=1,
    )  # [B, 4, 2]
    norm_vec = norm_vec / torch.norm(norm_vec, dim=2, keepdim=True)

    proj1 = torch.einsum("bij,bkj->bik", norm_vec, rect1)  # [B, 4, 2] * [B, 4, 2] -> [B, 4, 4]
    proj1_min, proj1_max = proj1.min(dim=2)[0], proj1.max(dim=2)[0]  # [B, 4] [B, 4]

    proj2 = torch.einsum("bij,bkj->bik", norm_vec, rect2)  # [B, 4, 2] * [B, 4, 2] -> [B, 4, 4]
    proj2_min, proj2_max = proj2.min(dim=2)[0], proj2.max(dim=2)[0]  # [B, 4] [B, 4]

    overlap = torch.cat([proj1_min - proj2_max, proj2_min - proj1_max], dim=1)  # [B, 8]

    positive_distance = torch.where(overlap < 0, 1e5, overlap)

    is_overlap = (overlap < 0).all(dim=1)
    distance = torch.where(
        is_overlap, overlap.max(dim=1).values, positive_distance.min(dim=1).values
    )

    return distance


def center_rect_to_points(rect):
    """
    rect: [B, 6] (x, y, cos_h, sin_h, l, w)

    return [B, 4, 2] (4 points of the rectangle)
    """

    B, _ = rect.shape
    xy, cos_h, sin_h, lw = rect[:, :2], rect[:, 2], rect[:, 3], rect[:, 4:]

    rot = torch.stack([cos_h, -sin_h, sin_h, cos_h], dim=1).reshape(-1, 2, 2)  # [B, 2, 2]
    lw = torch.einsum(
        "bj,ij->bij", lw, torch.tensor([[1.0, 1], [-1, 1], [-1, -1], [1, -1]], device=lw.device) / 2
    )  # [B, 2] * [4, 2] -> [B, 4, 2]
    lw = torch.einsum("bij,bkj->bik", lw, rot)  # [B, 4, 2] * [B, 2, 2] -> [B, 4, 2]

    rect = xy[:, None, :] + lw  # [B, 4, 2]

    return rect
