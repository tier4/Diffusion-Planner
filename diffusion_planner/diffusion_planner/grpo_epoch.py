"""GRPO training epoch - the reinforcement-learning counterpart of ``train_epoch.py``.

Each batch is handled by one of two step types, chosen stochastically (``--sft_prob``):

  * **Supervised (SFT) step** - the ordinary ``compute_training_loss`` on the real
    ground-truth trajectories (identical to ``train_epoch``). Keeps the policy anchored to
    realistic behaviour and prevents reward-hacking / collapse.
  * **GRPO step** - expand each scene into a group of ``N`` samples, run a single
    multi-batch inference pass to draw ``N`` diverse ego trajectories, score them with a
    collision-based reward, convert rewards to group-relative advantages, and take an
    advantage-weighted diffusion-loss gradient step.

Each step does exactly one forward + one backward (DDP-safe).
"""

import random

import torch
from torch import nn
from tqdm import tqdm

from diffusion_planner.grpo_utils import (
    compute_collision_reward,
    compute_group_advantages,
    compute_grpo_loss,
    compute_gt_l2_distance,
    compute_kinematic_consistency_penalty,
    expand_batch,
    sample_group,
)
from diffusion_planner.model.module.decoder import compute_training_loss
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils import ddp
from diffusion_planner.utils.train_utils import get_epoch_mean_loss


def _neighbor_future_world(neighbor_future_raw: torch.Tensor):
    """Convert raw neighbor future (x, y, heading) to world-frame (x, y, cos, sin) + mask."""
    mask = torch.sum(torch.ne(neighbor_future_raw[..., :3], 0), dim=-1) == 0  # [B, Pn, T]
    neighbors_future = heading_to_cos_sin(neighbor_future_raw)  # [B, Pn, T, 4]
    neighbors_future[mask] = 0.0
    return neighbors_future, mask


def _sft_step(raw_inputs, model, optimizer, args, ema, aug):
    """A standard supervised training step on the real GT (mirrors ``train_epoch``).

    ``aug`` is a ``StatePerturbation`` (or ``None``); when given it perturbs the ego
    history/future exactly as in the supervised trainer, before the cos/sin + normalization.
    """
    inputs = dict(raw_inputs)
    inputs["ego_agent_past"] = heading_to_cos_sin(inputs["ego_agent_past"])
    inputs["goal_pose"] = heading_to_cos_sin(inputs["goal_pose"])

    ego_future = inputs["ego_agent_future"]
    neighbors_future_raw = inputs["neighbor_agents_future"]
    if aug is not None:
        inputs, ego_future, neighbors_future_raw = aug(inputs, ego_future, neighbors_future_raw)

    ego_future = heading_to_cos_sin(ego_future)
    neighbors_future, neighbor_future_mask = _neighbor_future_world(neighbors_future_raw)
    inputs = args.observation_normalizer(inputs)

    optimizer.zero_grad()
    loss = compute_training_loss(
        model, inputs, (ego_future, neighbors_future, neighbor_future_mask), args
    )
    loss["loss"] = (
        args.alpha_neighbor_loss * loss["neighbor_prediction_loss"]
        + args.alpha_planning_loss * loss["ego_planning_loss"]
        + loss["turn_indicator_loss"]
        + args.coeff_road_border_loss * loss["road_border_loss"]
        + args.coeff_neighbor_collision_loss * loss["neighbor_collision_loss"]
    )
    loss["loss"].backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    ema.update(model)

    return {
        "loss": loss["loss"].detach(),
        "sft_ego_planning_loss": loss["ego_planning_loss"].detach(),
        "is_grpo": torch.tensor(0.0),
    }


def _grpo_step(raw_inputs, model, optimizer, args, ema, collider_injector, aug):
    """A GRPO step: sample a group per scene, reward, advantage, policy-gradient update.

    ``aug`` is a ``StatePerturbation`` (or ``None``). When given it perturbs the scene's ego
    history/future first (recentering to the perturbed ego pose); collider injection then targets
    that perturbed future and the policy is sampled / gt-L2-scored on the perturbed scene.
    """
    n = args.num_generations

    raw_inputs = dict(raw_inputs)
    # StatePerturbation expects the ego past / goal in cos/sin form; it returns the (recentered)
    # inputs plus the perturbed ego + neighbor futures (still raw heading form).
    past_is_cos_sin = aug is not None
    if aug is not None:
        raw_inputs["ego_agent_past"] = heading_to_cos_sin(raw_inputs["ego_agent_past"])
        raw_inputs["goal_pose"] = heading_to_cos_sin(raw_inputs["goal_pose"])
        raw_inputs, ego_future_aug, neighbors_future_aug = aug(
            raw_inputs, raw_inputs["ego_agent_future"], raw_inputs["neighbor_agents_future"]
        )
        raw_inputs["ego_agent_future"] = ego_future_aug
        raw_inputs["neighbor_agents_future"] = neighbors_future_aug

    # Adversarial neighbor augmentation on the *scene* batch so every sample in a group faces an
    # identical scene (a prerequisite for comparable group advantages). Injectors only touch the
    # neighbor / ego-future columns, so they are unaffected by the ego-past cos/sin form above.
    if collider_injector is not None:
        raw_inputs = collider_injector.inject(
            raw_inputs, args.neighbor_inject_max, args.neighbor_inject_prob
        )

    exp = expand_batch(raw_inputs, n)
    B = exp["ego_current_state"].shape[0]  # == num_scenes * N
    num_scenes = B // n

    # aug already converted the ego past / goal to cos/sin; otherwise do it here.
    if not past_is_cos_sin:
        exp["ego_agent_past"] = heading_to_cos_sin(exp["ego_agent_past"])
        exp["goal_pose"] = heading_to_cos_sin(exp["goal_pose"])

    neighbors_future, neighbor_future_mask = _neighbor_future_world(exp["neighbor_agents_future"])
    neighbors_future_valid = ~neighbor_future_mask

    norm_exp = args.observation_normalizer(exp)

    # Multi-batch inference: draw one trajectory per row (group of N per scene).
    ego_world = sample_group(model, norm_exp, args.grpo_noise_scale, args.device)

    # Collision-based reward -> group-relative advantages.
    reward, nc_penalty, rb_penalty = compute_collision_reward(
        ego_world, norm_exp, neighbors_future, neighbors_future_valid, args
    )

    # Optional realism term: penalise L2 distance to the scene's own GT ego future.
    gt_l2_dist = torch.zeros_like(reward)
    if args.w_gt_l2 > 0.0:
        gt_l2_dist = compute_gt_l2_distance(ego_world, exp["ego_agent_future"])
        reward = reward - args.w_gt_l2 * gt_l2_dist

    # Optional kinematic-feasibility term: penalise the drift incurred when the generated
    # trajectory is converted to an (accel, curvature) action sequence and integrated back.
    kinematic_drift = torch.zeros_like(reward)
    if args.w_kinematic > 0.0:
        kinematic_drift = compute_kinematic_consistency_penalty(
            ego_world, exp["ego_agent_past"], exp["ego_current_state"]
        )
        reward = reward - args.w_kinematic * kinematic_drift

    advantages = compute_group_advantages(reward, num_scenes, n, args.advantage_eps)

    optimizer.zero_grad()
    loss_dict = compute_grpo_loss(
        model, norm_exp, ego_world, neighbors_future, neighbor_future_mask, advantages, args
    )
    loss_dict["loss"].backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    ema.update(model)

    return {
        "loss": loss_dict["loss"].detach(),
        "grpo_loss": loss_dict["loss"].detach(),
        "reward_mean": reward.mean().detach(),
        "reward_max": reward.view(num_scenes, n).max(dim=1).values.mean().detach(),
        "neighbor_collision_penalty": nc_penalty.sum(dim=-1).mean().detach(),
        "road_border_penalty": rb_penalty.sum(dim=-1).mean().detach(),
        "gt_l2_distance": gt_l2_dist.mean().detach(),
        "kinematic_drift": kinematic_drift.mean().detach(),
        "abs_advantage": loss_dict["abs_advantage"],
        "is_grpo": torch.tensor(1.0),
    }


def train_grpo_epoch(data_loader, model, optimizer, args, ema, collider_injector, aug):
    epoch_loss = []

    model.train()

    if args.ddp:
        torch.cuda.synchronize()

    if ddp.get_rank() == 0:
        data_loader = tqdm(data_loader, desc="GRPO", unit="batch")

    # Synchronize the SFT/GRPO choice across ranks (args.seed is identical on every rank) so
    # all ranks emit the same metric key set each epoch -- required for the keyed all-reduce.
    step_rng = random.Random(args.seed)

    for raw_inputs in data_loader:
        raw_inputs = {key: value.to(args.device) for key, value in raw_inputs.items()}

        if step_rng.random() < args.sft_prob:
            step_loss = _sft_step(raw_inputs, model, optimizer, args, ema, aug)
        else:
            step_loss = _grpo_step(raw_inputs, model, optimizer, args, ema, collider_injector, aug)

        if args.ddp:
            torch.cuda.synchronize()
        epoch_loss.append(step_loss)

    epoch_mean_loss = get_epoch_mean_loss(epoch_loss)

    if args.ddp:
        epoch_mean_loss = ddp.reduce_and_average_losses(epoch_mean_loss, torch.device(args.device))

    if ddp.get_rank() == 0:
        print(f"{epoch_mean_loss['loss']=:.4f}")
        if "reward_mean" in epoch_mean_loss:
            print(f"{epoch_mean_loss['reward_mean']=:.4f}")
            print(f"{epoch_mean_loss['reward_max']=:.4f}")

    return epoch_mean_loss, epoch_mean_loss["loss"]
