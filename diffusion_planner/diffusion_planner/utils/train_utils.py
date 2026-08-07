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
