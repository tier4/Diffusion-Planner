"""Accelerate construction for distributed planner training."""

from __future__ import annotations

from accelerate import Accelerator
from accelerate.utils import (
    DataLoaderConfiguration,
    DistributedDataParallelKwargs,
    DynamoBackend,
    TorchDynamoPlugin,
)


def build_accelerator(
    *,
    cpu: bool = False,
    mixed_precision: str = "bf16",
    split_batches: bool = False,
    compile: bool = True,
    compile_backend: str = "inductor",
    compile_mode: str = "reduce-overhead",
    compile_dynamic: bool = False,
    dataloader_dispatch_batches: bool = False,
    dataloader_non_blocking: bool = True,
    ddp_broadcast_buffers: bool = False,
    ddp_bucket_cap_mb: int = 25,
    ddp_find_unused_parameters: bool = False,
    ddp_gradient_as_bucket_view: bool = True,
    ddp_static_graph: bool = True,
) -> Accelerator:
    """Build an Accelerator from Hydra-friendly primitive options."""
    backend = DynamoBackend(compile_backend.upper()) if compile else DynamoBackend.NO
    return Accelerator(
        cpu=cpu,
        mixed_precision=mixed_precision,
        split_batches=split_batches,
        dataloader_config=DataLoaderConfiguration(
            dispatch_batches=dataloader_dispatch_batches,
            non_blocking=dataloader_non_blocking,
        ),
        dynamo_plugin=TorchDynamoPlugin(
            backend=backend,
            mode=compile_mode,
            dynamic=compile_dynamic,
        ),
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                broadcast_buffers=ddp_broadcast_buffers,
                bucket_cap_mb=ddp_bucket_cap_mb,
                find_unused_parameters=ddp_find_unused_parameters,
                gradient_as_bucket_view=ddp_gradient_as_bucket_view,
                static_graph=ddp_static_graph,
            )
        ],
    )
