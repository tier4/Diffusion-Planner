"""Quick smoke test: verify direct_best and diffusion_low_t loss modes work end-to-end."""

import json
import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import torch

from preference_optimization.model_utils import load_model
from preference_optimization.utils import load_npz_data
from rlvr.grpo_config import GRPOConfig
from rlvr.grpo_loss import compute_direct_best_loss, compute_grpo_loss, _sample_t_for_mode
from rlvr.grpo_sampler import SamplerConfig, generate_diverse_group
from rlvr.reward import RewardConfig, compute_reward_batch, compute_group_advantages

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SSD = Path("/media/danielsanchez/2fb4af16-188c-4b7d-8ebb-4a7d0c90d207")
MODEL_PATH = SSD / "xx1-best-model/v3.0/best_model.pth"


def test_direct_best():
    print("Loading model...")
    model, model_args = load_model(MODEL_PATH, DEVICE)

    # Apply LoRA
    from preference_optimization.lora_utils import apply_lora
    model = apply_lora(model, r=16, lora_alpha=16, lora_dropout=0.05)

    with open(SSD / "path_lists/merged_20260216_20260224/path_list.json") as f:
        prob_scenes = json.load(f)[:3]

    config = GRPOConfig(
        loss_mode="direct_best",
        num_generations=4,
        kl_coef=0.1,
    )
    sampler_config = SamplerConfig(n_trajectories=4, enable_guidance=False)
    reward_config = RewardConfig()

    print(f"\nTest 1: direct_best loss on {len(prob_scenes)} scenes")
    for path in prob_scenes:
        data = load_npz_data(path, DEVICE)
        model.eval()
        with torch.no_grad():
            sampled = generate_diverse_group(model, model_args, data, sampler_config, DEVICE)
        trajectories = [s.trajectory for s in sampled]
        traj_batch = torch.tensor(
            __import__("numpy").stack(trajectories), device=DEVICE, dtype=torch.float32
        )
        rewards = compute_reward_batch(traj_batch, data, reward_config)
        best_idx = max(range(len(rewards)), key=lambda i: rewards[i].total)

        model.train()
        loss, metrics = compute_direct_best_loss(
            model, trajectories[best_idx], data, model_args, DEVICE, config,
        )
        print(f"  direct_best loss={loss.item():.6f}, direct_mse={metrics['direct_mse']:.6f}")

        # Verify gradient flows
        loss.backward()
        grad_count = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
        print(f"  Parameters with nonzero grad: {grad_count}")
        assert grad_count > 0, "No gradients flowing!"
        model.zero_grad()

    print("\nTest 2: diffusion_low_t loss")
    config2 = GRPOConfig(
        loss_mode="diffusion_low_t",
        diffusion_t_range=[0.001, 0.1],
        num_generations=4,
        kl_coef=0.1,
    )
    data = load_npz_data(prob_scenes[0], DEVICE)
    model.eval()
    with torch.no_grad():
        sampled = generate_diverse_group(model, model_args, data, sampler_config, DEVICE)
    trajectories = [s.trajectory for s in sampled]
    traj_batch = torch.tensor(
        __import__("numpy").stack(trajectories), device=DEVICE, dtype=torch.float32
    )
    rewards = compute_reward_batch(traj_batch, data, reward_config)
    advantages = compute_group_advantages(rewards)

    model.train()
    loss, metrics = compute_grpo_loss(
        model, trajectories, advantages, data, model_args, config2, DEVICE,
    )
    print(f"  diffusion_low_t loss={loss.item():.6f}")
    loss.backward()
    grad_count = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  Parameters with nonzero grad: {grad_count}")
    assert grad_count > 0, "No gradients flowing!"
    model.zero_grad()

    print("\nTest 3: diffusion_multistep loss")
    config3 = GRPOConfig(
        loss_mode="diffusion_multistep",
        diffusion_k_steps=2,
        num_generations=4,
        kl_coef=0.1,
    )
    loss, metrics = compute_grpo_loss(
        model, trajectories, advantages, data, model_args, config3, DEVICE,
    )
    print(f"  diffusion_multistep loss={loss.item():.6f}")
    loss.backward()
    grad_count = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  Parameters with nonzero grad: {grad_count}")
    assert grad_count > 0, "No gradients flowing!"

    print("\nALL SMOKE TESTS PASSED!")


if __name__ == "__main__":
    test_direct_best()
