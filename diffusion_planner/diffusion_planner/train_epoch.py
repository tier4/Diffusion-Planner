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
        for inputs in data_epoch:
            inputs = {key: value.to(args.device) for key, value in inputs.items()}
            inputs["ego_agent_past"] = heading_to_cos_sin(inputs["ego_agent_past"])
            inputs["goal_pose"] = heading_to_cos_sin(inputs["goal_pose"])

            ego_future = inputs["ego_agent_future"]
            neighbors_future = inputs["neighbor_agents_future"]
            # Normalize to ego-centric
            if aug is not None:
                inputs, ego_future, neighbors_future = aug(inputs, ego_future, neighbors_future)

            # heading to cos sin
            ego_future = heading_to_cos_sin(ego_future)

            mask = torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0
            neighbors_future = heading_to_cos_sin(neighbors_future)
            neighbors_future[mask] = 0.0
            inputs = args.observation_normalizer(inputs)

            # call the model
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
