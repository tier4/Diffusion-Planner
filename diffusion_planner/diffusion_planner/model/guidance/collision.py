import torch
import torch.nn.functional as F

ego_size = [5.0, 3.0]  # [length, width]

COG_TO_REAR = 1.67
CLIP_DISTANCE = 1.0
INFLATION = 1.0


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


def collision_guidance_fn(x, t, cond, inputs, *args, **kwargs) -> torch.Tensor:
    """
    x: [B * Pn+1, T + 1, 4]
    t: [B, 1],
    inputs: Dict[str, torch.Tensor]
    """
    B, P, T, _ = x.shape
    neighbor_current_mask = inputs["neighbor_current_mask"]  # [B, Pn]

    x: torch.Tensor = x.reshape(B, P, -1, 4)
    mask_diffusion_time = (t < 0.1) * (t > 0.005)
    mask_diffusion_time = mask_diffusion_time.view(B, 1, 1, 1)
    x = torch.where(mask_diffusion_time, x, x.detach())

    cos_sin = x[:, :, :, 2:].detach()
    cos_sin_norm = torch.norm(cos_sin, dim=-1, keepdim=True).clamp(min=1e-6)
    x = torch.cat(
        [
            x[:, :, :, :2],
            cos_sin / cos_sin_norm,
        ],
        dim=-1,
    )  # [B, P + 1, T, 4]

    ego_pred = x[:, :1, 1:, :]  # [B, 1, T, 4]
    cos_h, sin_h = ego_pred[..., 2:3], ego_pred[..., 3:4]
    ego_pred = torch.cat(
        [
            ego_pred[..., 0:1] + cos_h * COG_TO_REAR,
            ego_pred[..., 1:2] + sin_h * COG_TO_REAR,
            ego_pred[..., 2:],
        ],
        dim=-1,
    )

    neighbors_pred = x[:, 1:, 1:, :]  # [B, P, T, 4]

    B, Pn, T, _ = neighbors_pred.shape

    predictions = torch.cat([ego_pred, neighbors_pred.detach()], dim=1)  # [B, P + 1, T, 4]

    lw = torch.cat(
        [
            torch.tensor(ego_size, device=predictions.device)[None, None, :].repeat(B, 1, 1),
            inputs["neighbor_agents_past"][:, :Pn, -1, [7, 6]],
        ],
        dim=1,
    )  # [B, P, 2]

    bbox = torch.cat(
        [predictions, lw.unsqueeze(2).expand(-1, -1, T, -1) + INFLATION], dim=-1
    )  # [B, P, T, 6]

    bbox = center_rect_to_points(bbox.reshape(-1, 6)).reshape(B, Pn + 1, T, 4, 2)

    ego_bbox = (
        bbox[:, :1, :, :, :].expand(-1, Pn, -1, -1, -1)[~neighbor_current_mask].reshape(-1, 4, 2)
    )
    neighbor_bbox = bbox[:, 1:, :, :, :][~neighbor_current_mask].reshape(-1, 4, 2)

    distances = batch_signed_distance_rect(ego_bbox, neighbor_bbox)
    clip_distances = torch.maximum(
        1 - distances / CLIP_DISTANCE, torch.tensor(0.0, device=distances.device)
    )

    reward = -(
        torch.sum(clip_distances[clip_distances > 1])
        / (torch.sum((clip_distances[clip_distances > 1].detach() > 0).float()) + 1e-5)
        + torch.sum(clip_distances[clip_distances <= 1])
        / (torch.sum((clip_distances[clip_distances <= 1].detach() > 0).float()) + 1e-5)
    ).exp()

    x_aux = torch.autograd.grad(reward.sum(), x, retain_graph=True, allow_unused=True)[0][
        :, 0, :, :2
    ]  # [B, T, 2]

    T += 1
    x_mat = torch.einsum(
        "btd,nd->btn",
        x[:, 0, :, 2:],
        torch.tensor([[1.0, 0], [0, 1], [0, -1], [1, 0]], device=x.device),
    ).reshape(B, T, 2, 2)

    x_aux = torch.einsum("btij,btj->bti", x_mat, x_aux)
    # x_aux = torch.cat([x_aux[:, :5], torch.zeros_like(x_aux[:, 5:])], dim=1)

    x_aux = torch.stack(
        [
            torch.einsum(
                "bt,it->bi",
                x_aux[..., 0],
                torch.tril(
                    (-torch.linspace(0, 1, T, device=x.device)).exp().unsqueeze(0).repeat(T, 1)
                ),
            )
            * 0,
            F.conv1d(
                F.pad(x_aux[:, None, :, 1], (10, 10), mode="replicate"),
                torch.ones(1, 1, 21, device=x.device)
                * (-(torch.linspace(-2, 2, 21, device=x.device) ** 2) / 4).exp(),
            )[:, 0]
            * 1.0,
        ],
        dim=2,
    )
    x_aux = torch.einsum("btji,btj->bti", x_mat, x_aux)  # [B, T, 2]

    reward = torch.sum(x_aux.detach() * x[:, 0, :, :2], dim=(1, 2))

    return 300.0 * reward
