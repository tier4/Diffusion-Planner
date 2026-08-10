from types import SimpleNamespace

import torch
from diffusion_planner.utils import ddp


def test_direct_invocation_disables_uninitialized_ddp(monkeypatch):
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "SLURM_PROCID"):
        monkeypatch.delenv(name, raising=False)

    args = SimpleNamespace(ddp=True)

    rank, gpu, world_size = ddp.ddp_setup_universal(args=args)

    assert (rank, gpu, world_size) == (0, 0, 1)
    assert args.ddp is False


def test_reduce_and_average_losses_is_a_noop_without_process_group():
    losses = {"loss": torch.tensor(2.0)}

    reduced = ddp.reduce_and_average_losses(losses, torch.device("cpu"))

    assert reduced is losses
    assert reduced["loss"].item() == 2.0
