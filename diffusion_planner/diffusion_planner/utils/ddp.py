import hashlib
import os
import subprocess
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed import init_process_group


def ddp_setup_universal(verbose=False, args=None):
    if args.ddp == False:
        print(f"do not use ddp, train on GPU 0")
        return 0, 0, 1

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ.get("LOCAL_RANK", 0))
        os.environ.setdefault("MASTER_PORT", str(getattr(args, "port", "29529")))
        os.environ.setdefault("MASTER_ADDR", "localhost")
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        gpu = rank % torch.cuda.device_count()
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        os.environ["MASTER_PORT"] = str(args.port)
        os.environ["MASTER_ADDR"] = addr
    else:
        print("Not using DDP mode")
        # A direct invocation may retain the default --ddp=True. Keep the
        # caller's later DDP wrapping and collective calls consistent.
        args.ddp = False
        return 0, 0, 1

    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(gpu)
    os.environ["RANK"] = str(rank)

    torch.cuda.set_device(gpu)
    dist_backend = "nccl"
    dist_url = "env://"
    print("| distributed init (rank {}): {}, gpu {}".format(rank, dist_url, gpu), flush=True)
    init_process_group(
        init_method=dist_url,
        backend=dist_backend,
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=1000),
        device_id=torch.device("cuda", gpu),
    )
    torch.distributed.barrier()
    if verbose:
        setup_for_distributed(rank == 0)
    return rank, gpu, world_size


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_model(model, use_ddp):
    if use_ddp:
        return model.module
    else:
        return model


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def all_reduce_sum(value, device):
    """Sum a python scalar across all DDP ranks.

    Collective: must be called by every rank. Returns ``value`` unchanged when DDP
    is not initialized (single-process run), so callers work in both modes.
    """
    if not is_dist_avail_and_initialized():
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


def all_reduce_min(value, device):
    """Minimum of a python scalar across all DDP ranks.

    Collective: must be called by every rank. Returns ``value`` unchanged when DDP
    is not initialized (single-process run).
    """
    if not is_dist_avail_and_initialized():
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return t.item()


def reduce_and_average_losses(loss_dict, device):
    if not is_dist_avail_and_initialized():
        return loss_dict

    world_size = dist.get_world_size()
    keys = list(loss_dict)
    # Validate the dictionary contract collectively before reducing values.
    # This prevents a rank-divergent diagnostic from silently pairing metrics
    # by different insertion orders.
    key_digest = int.from_bytes(
        hashlib.sha256("\0".join(keys).encode("utf-8")).digest()[:8],
        byteorder="little",
        signed=True,
    )
    key_meta = torch.tensor([len(keys), key_digest], dtype=torch.int64, device=device)
    key_min = key_meta.clone()
    key_max = key_meta.clone()
    dist.all_reduce(key_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(key_max, op=dist.ReduceOp.MAX)
    if not torch.equal(key_min, key_max):
        raise RuntimeError("Distributed loss dictionaries have different keys or ordering")
    if not keys:
        return loss_dict

    values = []
    for key in keys:
        value = loss_dict[key]
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f"Distributed metric {key!r} must be scalar, got {value.shape}")
            values.append(value.detach().to(device=device, dtype=torch.float64).reshape(()))
        else:
            values.append(torch.tensor(float(value), dtype=torch.float64, device=device))
    packed = torch.stack(values)
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    loss_dict.update(zip(keys, (packed / world_size).cpu().tolist(), strict=True))
    return loss_dict
