"""Shared model/input helpers for the token-analysis command-line tools."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from diffusion_planner.dimensions import MAX_NUM_AGENTS, OUTPUT_T, POSE_DIM
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils.config import Config


def init_distributed(requested_device: str):
    """Initialize torchrun data parallelism and return device/rank metadata."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = requested_device
    if world_size > 1:
        if requested_device.startswith("cuda"):
            device = f"cuda:{local_rank}"
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"
        dist.init_process_group(backend=backend)
    return device, rank, local_rank, world_size


def prepare_inputs(inputs: dict, cfg, device: str, *, include_future: bool = False):
    """Apply the validation preprocessing used by the planner."""
    inputs = {k: v.to(device) for k, v in inputs.items()}
    batch_size = inputs["ego_current_state"].shape[0]
    inputs["sampled_trajectories"] = torch.zeros(
        batch_size, MAX_NUM_AGENTS, OUTPUT_T + 1, POSE_DIM, dtype=torch.float32, device=device
    )
    inputs["delay"] = torch.zeros(batch_size, dtype=torch.float32, device=device)
    inputs["ego_agent_past"] = heading_to_cos_sin(inputs["ego_agent_past"])
    inputs["goal_pose"] = heading_to_cos_sin(inputs["goal_pose"])
    ego_future = heading_to_cos_sin(inputs["ego_agent_future"]) if include_future else None
    inputs = cfg.observation_normalizer(inputs)
    return (inputs, ego_future) if include_future else inputs


def latest_ckpt(run_dir: Path) -> Path:
    if (run_dir / "best_model.pth").exists():
        return run_dir / "best_model.pth"
    epoch_dirs = sorted(
        (d for d in run_dir.iterdir() if re.fullmatch(r"epoch\d+", d.name)),
        key=lambda d: int(d.name[5:]),
    )
    if epoch_dirs:
        return epoch_dirs[-1] / "best_model.pth"
    return run_dir / "best_model" / "best_model.pth"


def load_model(run_dir: Path, device: str):
    cfg = Config(str(run_dir / "args.json"))
    cfg.device = device
    cfg.ddp = False
    model = Diffusion_Planner(cfg).to(device)
    ckpt_path = latest_ckpt(run_dir)
    state = torch.load(ckpt_path, map_location=device)
    state = state["model"] if "model" in state else state
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model, cfg, ckpt_path


def find_fusion(encoder):
    for module in encoder.modules():
        if type(module).__name__ == "FusionEncoder":
            return module
    raise RuntimeError("FusionEncoder not found")


def neighbor_dist(neighbors: torch.Tensor) -> torch.Tensor:
    valid = (neighbors[:, :, -6:, :8] != 0).any(dim=(2, 3))
    distance = neighbors[:, :, -1, :2].norm(dim=-1)
    return torch.where(valid, distance, torch.full_like(distance, float("inf")))


def polyline_dist(values: torch.Tensor, geom_dims: int | None = None) -> torch.Tensor:
    valid = (
        (values != 0).any(dim=-1)
        if geom_dims is None
        else (values[..., :geom_dims] != 0).any(dim=-1)
    )
    distance = values[..., :2].norm(dim=-1)
    distance = torch.where(valid, distance, torch.full_like(distance, float("inf")))
    return distance.min(dim=-1).values


def patch_fusion(fusion, store):
    """Capture the model's pre-norm attention weights, inputs, and mask."""
    for layer_index, block in enumerate(fusion.blocks):

        def make_forward(layer, index):
            def forward(x, mask):
                query = layer.norm1(x)
                attention_output, weights = layer.attn(
                    query,
                    x,
                    x,
                    key_padding_mask=mask,
                    need_weights=True,
                    average_attn_weights=True,
                )
                store.append(
                    {
                        "layer": index,
                        "weights": weights.detach(),
                        "w": weights.detach(),
                        "kv": x.detach(),
                        "mask": mask.detach(),
                    }
                )
                x = x + layer.drop_path(attention_output)
                return x + layer.drop_path(layer.mlp(layer.norm2(x)))

            return forward

        block.forward = make_forward(block, layer_index)
