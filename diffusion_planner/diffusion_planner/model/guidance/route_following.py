import torch
import torch.nn.functional as F


def route_following_fn(x, t, cond, inputs, *args, **kwargs) -> torch.Tensor:
    """
    x: [B, P, T + 1, 4]
    t: [B],
    inputs: Dict[str, torch.Tensor]
    """
    B, P, T, _ = x.shape
    route_lanes = inputs["route_lanes"]  # [B, SegNum=25, PointNum=20, SEGMENT_POINT_DIM=33]
    route_lanes = route_lanes.reshape(B, 25 * 20, route_lanes.shape[-1])  # [B, 500, 33]
    route_lanes = route_lanes[:, :, :2]  # [B, 500, 2] - centerline XY only

    x: torch.Tensor = x.reshape(B, P, -1, 4)
    mask_diffusion_time = (t < 0.1) * (t > 0.005)
    mask_diffusion_time = mask_diffusion_time.view(B, 1, 1, 1)
    x = torch.where(mask_diffusion_time, x, x.detach())

    predictions = x[:, 0, :, :2]  # [B, T + 1, 2]

    pred_points = predictions[:, :T]
    # route_lanesを拡張 [B, 1, SegNum*PointNum, 2]
    expanded_routes = route_lanes.unsqueeze(1)
    # 予測点を拡張 [B, T, 1, 2]
    expanded_preds = pred_points.unsqueeze(2)
    # 全ての距離を一度に計算 [B, T, SegNum*PointNum]
    distances = torch.norm(expanded_preds - expanded_routes, dim=-1)
    # Per-timestep minimum distance to any route point [B, T]
    min_distances = torch.min(distances, dim=2)[0]
    raw_reward = -torch.sum(min_distances, dim=1)  # [B]

    # Compute and smooth the gradient (same pattern as collision guidance)
    # to bound the per-timestep correction and avoid abrupt steering.
    raw_grad = torch.autograd.grad(
        raw_reward.sum(), x, retain_graph=True, allow_unused=True
    )[0]

    if raw_grad is None:
        return torch.zeros(B, device=x.device)

    x_grad = raw_grad[:, 0, :T, :2]   # [B, T, 2]  ego XY only

    half = 10
    kernel_1d = (-(torch.linspace(-2.0, 2.0, 2 * half + 1, device=x.device) ** 2) / 4.0).exp()
    kernel_1d = kernel_1d / kernel_1d.sum()

    x_grad_smooth = F.conv1d(
        F.pad(x_grad.permute(0, 2, 1), (half, half), mode="replicate"),
        kernel_1d.unsqueeze(0).unsqueeze(0).expand(2, 1, -1),
        groups=2,
    ).permute(0, 2, 1)   # [B, T, 2]

    reward = torch.sum(x_grad_smooth.detach() * x[:, 0, :T, :2], dim=(1, 2))  # [B]

    return 100.0 * reward
