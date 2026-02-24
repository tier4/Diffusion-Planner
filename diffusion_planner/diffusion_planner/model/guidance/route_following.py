import torch


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
    # Negative sum: reward is higher when trajectory is close to the route.
    # Scale matches lane_keeping (0.05) so both guidance functions have
    # comparable influence; adjust with the guidance_scale UI slider.
    reward = -torch.sum(min_distances, dim=1)  # [B]

    return 0.05 * reward
