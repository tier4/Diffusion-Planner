import json
import random
from typing import Any

import numpy as np
import torch


def openjson(path):
    """Load a json file; transparently handles zstd-compressed ``*.zst`` files."""
    if str(path).endswith(".zst"):
        import io

        import zstandard

        with open(path, "rb") as f:
            reader = zstandard.ZstdDecompressor().stream_reader(f)
            return json.load(io.TextIOWrapper(reader, encoding="utf-8"))
    with open(path, "r", encoding="utf-8") as f:
        dict = json.load(f)
    return dict


def set_seed(CUR_SEED):
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_grad_stats(parameters, prefix="grad"):
    """
    Compute global-gradient statistics (l1/l2/linf/mean/std) WITHOUT
    materializing a single concatenated copy of every gradient.

    torch.cat([...all grads...]) allocated one huge contiguous block at peak
    memory (right after backward), which was a primary driver of caching-allocator
    fragmentation. Here we accumulate the reductions per-parameter instead, and
    sync to host only once (a single 5-element .tolist()).
    """
    l1 = sq = smax = ssum = None
    n = 0
    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad.detach()
        a = g.abs()
        g_l1 = a.sum()
        g_sq = (g * g).sum()
        g_max = a.max()
        g_sum = g.sum()
        l1 = g_l1 if l1 is None else l1 + g_l1
        sq = g_sq if sq is None else sq + g_sq
        smax = g_max if smax is None else torch.maximum(smax, g_max)
        ssum = g_sum if ssum is None else ssum + g_sum
        n += g.numel()

    if n == 0:
        return {}

    mean = ssum / n
    var = (sq / n - mean * mean).clamp_min(0)
    # Single device->host sync for all five scalars.
    l1_v, l2_v, linf_v, mean_v, std_v = torch.stack(
        [l1, sq.sqrt(), smax, mean, var.sqrt()]
    ).tolist()
    return {
        f"{prefix}/l1_norm": l1_v,
        f"{prefix}/l2_norm": l2_v,
        f"{prefix}/linf_norm": linf_v,
        f"{prefix}/mean": mean_v,
        f"{prefix}/std": std_v,
    }


def get_epoch_mean_loss(epoch_loss):
    epoch_mean_loss = {}
    for current_loss in epoch_loss:
        for key, value in current_loss.items():
            if key in epoch_mean_loss:
                epoch_mean_loss[key].append(
                    value if isinstance(value, (int, float)) else value.item()
                )
            else:
                epoch_mean_loss[key] = [value if isinstance(value, (int, float)) else value.item()]

    for key, values in epoch_mean_loss.items():
        epoch_mean_loss[key] = np.mean(np.array(values))

    return epoch_mean_loss


def strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Remove DDP ``module.`` prefix from checkpoint keys."""
    return {
        k.replace("module.", "", 1) if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }


def resume_model(path: str, model, optimizer, scheduler, ema, device, use_ddp: bool = False):
    """
    load ckpt from path
    """
    ckpt = torch.load(path, map_location=device)

    # load model
    if use_ddp:
        try:
            model.load_state_dict(ckpt["model"])
        except:
            model.load_state_dict(ckpt)
    else:
        try:
            model.load_state_dict(strip_module_prefix(ckpt["model"]))
        except:
            model.load_state_dict(strip_module_prefix(ckpt))
    print("Model load done")

    # load optimizer
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
        print("Optimizer load done")
    except:
        print("no pretrained optimizer found")

    # load schedule
    try:
        scheduler.load_state_dict(ckpt["schedule"])
        print("Schedule load done")
    except:
        print("no schedule found,")

    # load step
    try:
        init_epoch = ckpt["epoch"]
        print("Step load done")
    except:
        init_epoch = 0

    # Load wandb id
    try:
        wandb_id = ckpt["wandb_id"]
        print("wandb id load done")
    except:
        wandb_id = None

    try:
        ema_state = ckpt["ema_state_dict"]
        if not use_ddp:
            ema_state = strip_module_prefix(ema_state)
        ema.ema.load_state_dict(ema_state)
        ema.ema.eval()
        for p in ema.ema.parameters():
            p.requires_grad_(False)

        print("ema load done")
    except:
        print("no ema shadow found")

    return model, optimizer, scheduler, init_epoch, wandb_id, ema
