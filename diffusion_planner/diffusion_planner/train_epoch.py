import torch
from torch import nn
from tqdm import tqdm

from diffusion_planner.model.module.decoder import compute_training_loss
from diffusion_planner.utils import ddp
from diffusion_planner.utils.forced_augmentation import ForcedAugmentationSelector
from diffusion_planner.utils.train_utils import compute_grad_stats, get_epoch_mean_loss


def heading_to_cos_sin(x):
    """
    Convert heading angle to cosine and sine.
    Args:
        x: [B, T, 3] where last dimension is (x, y, heading)
    Output:
        x: [B, T, 4] where last dimension is (x, y, cos(heading), sin(heading))

    Idempotent: a [..., 4] input that is already (x, y, cos, sin) is returned
    unchanged. This guards against double-conversion (cos(cos)) now that scene-gen
    emits 4-col futures — callers can hand it either layout safely.
    """
    if x.shape[-1] == 4:
        return x
    return torch.cat(
        [
            x[..., :2],
            x[..., 2:3].cos(),
            x[..., 2:3].sin(),
        ],
        dim=-1,
    )


def train_epoch(
    data_loader,
    model,
    optimizer,
    args,
    ema,
    aug_pipeline: list | None = None,
    aug_selector: ForcedAugmentationSelector | None = None,
):
    if len(data_loader) == 0:
        empty = {"loss": 0.0, "turn_indicator_accuracy": 0.0}
        return empty, 0.0

    epoch_loss = []

    model.train()

    if args.ddp:
        torch.cuda.synchronize()

    # Captured before the tqdm wrap: tqdm does not forward attribute access to the
    # iterable it wraps, so reading .sampler off the wrapper would return None.
    train_sampler = getattr(data_loader, "sampler", None)

    if ddp.get_rank() == 0:
        data_loader = tqdm(data_loader, desc="Training", unit="batch")

    for inputs in data_loader:
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        inputs["ego_agent_past"] = heading_to_cos_sin(inputs["ego_agent_past"])
        inputs["goal_pose"] = heading_to_cos_sin(inputs["goal_pose"])

        ego_future = inputs["ego_agent_future"]
        neighbors_future = inputs["neighbor_agents_future"]
        # Bind this epoch's repeat flags on the first batch, never before. repeat_flags
        # is regenerated inside the sampler's __iter__, which the DataLoader only calls
        # once iteration starts -- binding in train.py would read the previous epoch's.
        if aug_selector is not None and not aug_selector.is_bound:
            if train_sampler is None:
                raise RuntimeError(
                    "forced augmentation needs a sampler exposing repeat_flags, but the "
                    "data_loader has no .sampler attribute"
                )
            aug_selector.start_epoch(
                args.current_epoch,
                getattr(train_sampler, "repeat_flags", None),
                getattr(train_sampler, "repeat_flags_epoch", None),
            )

        # One force mask per pool member for this batch. batch_size comes off the batch
        # itself rather than args.batch_size // world_size: that is a floor division and
        # a computed offset would drift on any partial batch.
        if aug_selector is not None:
            force_masks = aug_selector.masks_for_batch(
                inputs["ego_current_state"].shape[0], args.device
            )
        else:
            force_masks = {}

        # Normalize to ego-centric. Canonical order; state_perturbation runs last
        # because it ends with centric_transform, which the others assume has not yet
        # happened.
        for aug_name, aug in aug_pipeline or []:
            inputs, ego_future, neighbors_future = aug(
                inputs, ego_future, neighbors_future, force=force_masks.get(aug_name)
            )

        # heading to cos sin
        ego_future = heading_to_cos_sin(ego_future)

        mask = torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0
        neighbors_future = heading_to_cos_sin(neighbors_future)
        neighbors_future[mask] = 0.0
        inputs = args.observation_normalizer(inputs)

        # call the model
        optimizer.zero_grad()

        loss = compute_training_loss(model, inputs, (ego_future, neighbors_future, mask), args)

        loss["loss"] = (
            args.alpha_neighbor_loss * loss["neighbor_prediction_loss"]
            + args.alpha_planning_loss * loss["ego_planning_loss"]
            + loss["turn_indicator_loss"]
            + args.coeff_road_border_loss * loss["road_border_loss"]
            + args.coeff_neighbor_collision_loss * loss["neighbor_collision_loss"]
        )

        # loss backward
        loss["loss"].backward()

        # Gradient statistics (computed before clipping so that exploding
        # gradients are not masked by clip_grad_norm_).
        loss.update(compute_grad_stats(model.parameters()))

        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        ema.update(model)

        if args.ddp:
            torch.cuda.synchronize()
        epoch_loss.append(loss)

    epoch_mean_loss = get_epoch_mean_loss(epoch_loss)

    if args.ddp:
        epoch_mean_loss = ddp.reduce_and_average_losses(epoch_mean_loss, torch.device(args.device))

    if ddp.get_rank() == 0:
        print(f"{epoch_mean_loss['loss']=:.4f}")
        print(f"{epoch_mean_loss['turn_indicator_accuracy']=:.4f}")

    return epoch_mean_loss, epoch_mean_loss["loss"]
