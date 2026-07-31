import os
import subprocess
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed import init_process_group


def dist_init_file() -> str:
    """Path of the FileStore used to rendezvous the process group.

    This must be unique per job. A FileStore keyed on a fixed path silently cross-connects two
    jobs that land on the same node with the same ``world_size`` -- they rendezvous into each
    other's process group, and the resulting collectives mix tensors from unrelated runs. Slurm
    here runs with ``JobContainerType=(null)``, so ``/tmp`` is shared between jobs on a node and
    the fixed path this used to hardcode was reachable by every concurrent run.

    Defaults to a per-``SLURM_JOB_ID`` path, with ``DP_DDP_INIT_FILE`` as an explicit override
    for launchers that are not slurm (or for tests). The bare fallback is only for a machine
    running a single job at a time.
    """
    override = os.environ.get("DP_DDP_INIT_FILE")
    if override:
        return override
    job_id = os.environ.get("SLURM_JOB_ID")
    return f"/tmp/tmp_dist_init_{job_id}" if job_id else "/tmp/tmp_dist_init"


def ddp_setup_universal(verbose=False, args=None):
    if args.ddp == False:
        print(f"do not use ddp, train on GPU 0")
        return 0, 0, 1

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
        os.environ["MASTER_PORT"] = str(getattr(args, "port", "29529"))
        os.environ["MASTER_ADDR"] = "localhost"
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        gpu = rank % torch.cuda.device_count()
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        os.environ["MASTER_PORT"] = str(args.port)
        os.environ["MASTER_ADDR"] = addr
    else:
        print("Not using DDP mode")
        return 0, 0, 1

    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(gpu)
    os.environ["RANK"] = str(rank)

    torch.cuda.set_device(gpu)
    dist_backend = "nccl"
    # I don't know why but this is needed for DDP to work instead of 'env://'
    dist_url = "file://"
    file_path = dist_init_file()
    print("| distributed init (rank {}): {}, gpu {}".format(rank, dist_url, gpu), flush=True)
    init_process_group(
        init_method=f"{dist_url}{file_path}",
        backend=dist_backend,
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=10000),
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
    torch.distributed.barrier()
    world_size = dist.get_world_size()
    for key in loss_dict.keys():
        loss_tensor = torch.tensor([loss_dict[key].item()]).to(device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_dict[key] = loss_tensor.item() / world_size
    return loss_dict
