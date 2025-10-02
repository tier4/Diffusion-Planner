import torch
from torch import nn
from tqdm import tqdm

from diffusion_planner.loss import diffusion_loss_func
from diffusion_planner.utils import ddp
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.train_utils import get_epoch_mean_loss


def heading_to_cos_sin(x):
    """
    Convert heading angle to cosine and sine.
    Args:
        x: [B, T, 3] where last dimension is (x, y, heading)
    Output:
        x: [B, T, 4] where last dimension is (x, y, cos(heading), sin(heading))
    """
    return torch.cat(
        [
            x[..., :2],
            x[..., 2:3].cos(),
            x[..., 2:3].sin(),
        ],
        dim=-1,
    )


def train_epoch(data_loader, model, optimizer, args, ema, aug: StatePerturbation = None):
    epoch_loss = []

    model.train()

    if args.ddp:
        torch.cuda.synchronize()

    with tqdm(data_loader, desc="Training", unit="batch") as data_epoch:
        for batch in data_epoch:
            """
            data structure in batch: Tuple(Tensor)

            ego_current_state,
            ego_future_gt,

            neighbor_agents_past,
            neighbors_future_gt,

            lanes,
            lanes_speed_limit,
            lanes_has_speed_limit,

            route_lanes,
            route_lanes_speed_limit,
            route_lanes_has_speed_limit,

            static_objects,

            """

            # prepare data
            inputs = {
                "ego_agent_past": batch[0].to(args.device),
                "ego_current_state": batch[1].to(args.device),
                "neighbor_agents_past": batch[3].to(args.device),
                "lanes": batch[5].to(args.device),
                "lanes_speed_limit": batch[6].to(args.device),
                "lanes_has_speed_limit": batch[7].to(args.device),
                "route_lanes": batch[8].to(args.device),
                "route_lanes_speed_limit": batch[9].to(args.device),
                "route_lanes_has_speed_limit": batch[10].to(args.device),
                "polygons": batch[11].to(args.device),
                "line_strings": batch[12].to(args.device),
                "static_objects": batch[13].to(args.device),
                "turn_indicator": batch[14].to(args.device),
                "goal_pose": batch[15].to(args.device),
                "ego_shape": batch[16].to(args.device),
            }

            inputs["ego_agent_past"] = heading_to_cos_sin(inputs["ego_agent_past"])
            inputs["goal_pose"] = heading_to_cos_sin(inputs["goal_pose"])

            ego_future = batch[2].to(args.device)
            neighbors_future = batch[4].to(args.device)
            # Normalize to ego-centric
            if aug is not None:
                inputs, ego_future, neighbors_future = aug(inputs, ego_future, neighbors_future)

            # heading to cos sin
            ego_future = heading_to_cos_sin(ego_future)

            mask = torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0
            neighbors_future = heading_to_cos_sin(neighbors_future)
            neighbors_future[mask] = 0.0
            inputs = args.observation_normalizer(inputs)

            # call the mdoel
            optimizer.zero_grad()

            loss = diffusion_loss_func(
                model,
                inputs,
                ddp.get_model(model, args.ddp).sde.marginal_prob,
                (ego_future, neighbors_future, mask),
                args,
            )

            loss["loss"] = (
                loss["neighbor_prediction_loss"]
                + args.alpha_planning_loss * loss["ego_planning_loss"]
                + loss["turn_indicator_loss"]
            )

            # loss backward
            loss["loss"].backward()

            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            ema.update(model)

            if args.ddp:
                torch.cuda.synchronize()

            data_epoch.set_postfix(loss="{:.4f}".format(loss["loss"].item()))
            epoch_loss.append(loss)

    epoch_mean_loss = get_epoch_mean_loss(epoch_loss)

    if args.ddp:
        epoch_mean_loss = ddp.reduce_and_average_losses(epoch_mean_loss, torch.device(args.device))

    if ddp.get_rank() == 0:
        print(f"{epoch_mean_loss['loss']=:.4f}")
        print(f"{epoch_mean_loss['turn_indicator_accuracy']=:.4f}")

    return epoch_mean_loss, epoch_mean_loss["loss"]
