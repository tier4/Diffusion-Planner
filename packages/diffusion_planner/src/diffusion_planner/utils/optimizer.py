"""Muon optimizer construction with an auxiliary AdamW."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor, optim

ParamEntry = tuple[str, nn.Parameter]
ParamGroups = dict[str, list[ParamEntry]]

# Coefficients of the quintic Newton-Schulz iteration, from Keller Jordan's Muon.
NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)

# Multi-tensor kernels. Torch keeps them private, but they are the only way to
# apply one update across many parameters in a single launch.
_foreach_lerp_ = torch._foreach_lerp_  # pyright: ignore[reportPrivateImportUsage]
_foreach_lerp = torch._foreach_lerp  # pyright: ignore[reportPrivateImportUsage]
_foreach_mul_ = torch._foreach_mul_  # pyright: ignore[reportPrivateImportUsage]
_foreach_add_ = torch._foreach_add_  # pyright: ignore[reportPrivateImportUsage]

# Affine normalization parameters and lookup tables are optimized without decay.
NO_DECAY_MODULES = (
    nn.LayerNorm,
    nn.GroupNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.Embedding,
)


def classify_params(
    model: nn.Module, output_layers: Sequence[nn.Module]
) -> ParamGroups:
    """Classify trainable parameters into Muon and auxiliary AdamW groups."""
    groups: ParamGroups = {
        "muon": [],
        "adamw_decay": [],
        "adamw_no_decay": [],
    }
    output_parameter_ids = {
        id(parameter)
        for layer in output_layers
        for parameter in layer.parameters()
        if parameter.requires_grad
    }

    for module_name, module in model.named_modules():
        no_decay_module = isinstance(module, NO_DECAY_MODULES)
        for param_name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            name = f"{module_name}.{param_name}" if module_name else param_name

            if no_decay_module or parameter.ndim <= 1:
                groups["adamw_no_decay"].append((name, parameter))
            elif parameter.ndim != 2 or id(parameter) in output_parameter_ids:
                groups["adamw_decay"].append((name, parameter))
            else:
                groups["muon"].append((name, parameter))

    _validate_param_groups(model, groups, output_parameter_ids)
    return groups


def _validate_param_groups(
    model: nn.Module, groups: ParamGroups, output_parameter_ids: set[int]
) -> None:
    """Verify that every trainable parameter is classified exactly once."""
    expected = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    occurrences: dict[int, list[str]] = {}
    for entries in groups.values():
        for name, parameter in entries:
            occurrences.setdefault(id(parameter), []).append(name)

    missing = [
        name for identifier, name in expected.items() if identifier not in occurrences
    ]
    duplicated = [names for names in occurrences.values() if len(names) != 1]
    unexpected = [
        names[0]
        for identifier, names in occurrences.items()
        if identifier not in expected
    ]
    foreign_output_parameters = output_parameter_ids.difference(expected)
    if missing or duplicated or unexpected or foreign_output_parameters:
        raise ValueError(
            "Invalid optimizer parameter classification: "
            f"missing={missing}, duplicated={duplicated}, unexpected={unexpected}, "
            f"foreign_output_parameters={len(foreign_output_parameters)}"
        )

    invalid_muon = [name for name, parameter in groups["muon"] if parameter.ndim != 2]
    if invalid_muon:
        raise ValueError(f"Muon parameters must be 2-D: {invalid_muon}")


class MuonWithAuxAdamW(optim.Optimizer):
    """Expose a Muon and an auxiliary AdamW through one Optimizer interface."""

    def __init__(self, muon: optim.Optimizer, adamw: optim.AdamW) -> None:
        self.muon = muon
        self.adamw = adamw
        parameters = [
            parameter
            for optimizer in self.optimizers
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        self._initializing_wrapper = True
        super().__init__(parameters, defaults={})
        self._initializing_wrapper = False
        # Schedulers and GradScaler must operate on the real inner groups.
        self.param_groups = [
            group for optimizer in self.optimizers for group in optimizer.param_groups
        ]
        self._refresh_state_view()

    @property
    def optimizers(self) -> tuple[optim.Optimizer, optim.Optimizer]:
        return self.muon, self.adamw

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(
        self, closure: Callable[[], torch.Tensor] | None = None
    ) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for optimizer in self.optimizers:
            optimizer.step()
        self._refresh_state_view()
        return loss

    def state_dict(self) -> dict[str, Any]:
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])
        self._refresh_state_view()

    def _refresh_state_view(self) -> None:
        state: defaultdict[torch.Tensor, Any] = defaultdict(dict)
        state.update(self.muon.state)
        state.update(self.adamw.state)
        self.state = state

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        if getattr(self, "_initializing_wrapper", False):
            super().add_param_group(param_group)
            return
        raise RuntimeError(
            "Add parameters to the inner Muon or AdamW optimizer explicitly"
        )


def _adjust_lr(
    lr: float | Tensor, adjust_lr_fn: str | None, shape: tuple[int, ...]
) -> float | Tensor:
    """Scale the learning rate for one matrix shape, as `torch.optim.Muon` does."""
    rows, columns = shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        ratio = math.sqrt(max(1.0, rows / columns))
    elif adjust_lr_fn == "match_rms_adamw":
        ratio = 0.2 * math.sqrt(max(rows, columns))
    else:
        ratio = 1.0
    return lr * ratio


def _batched_newton_schulz(
    updates: Tensor,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> Tensor:
    """Orthogonalize a stack of same-shaped matrices with batched matmuls.

    Same quintic iteration as `torch.optim.Muon`, applied to `(N, rows, columns)`
    at once so each iteration costs three `baddbmm` launches instead of three per
    matrix. All matrices in a stack share a shape, so the transpose that keeps the
    Gram matrix small is decided once for the whole batch.
    """
    a, b, c = ns_coefficients
    orthogonal = updates.bfloat16()
    if orthogonal.shape[-2] > orthogonal.shape[-1]:
        orthogonal = orthogonal.transpose(-2, -1)
        transposed = True
    else:
        transposed = False
    # Frobenius norm per matrix, so every spectral norm is at most one.
    norm = torch.linalg.vector_norm(orthogonal, dim=(-2, -1), keepdim=True)
    orthogonal = orthogonal / norm.clamp_min(eps)
    for _ in range(ns_steps):
        gram = orthogonal @ orthogonal.transpose(-2, -1)
        gram_update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        orthogonal = torch.baddbmm(orthogonal, gram_update, orthogonal, beta=a)
    if transposed:
        orthogonal = orthogonal.transpose(-2, -1)
    return orthogonal.contiguous()


class BatchedMuon(optim.Optimizer):
    """Muon whose Newton-Schulz iteration is batched over same-shaped matrices.

    `torch.optim.Muon` rejects `foreach` and orthogonalizes one matrix per call,
    so a 33M-parameter planner issues about 2,550 matmuls per step whose kernels
    average a few microseconds: the host cannot dispatch them fast enough to keep
    the GPU busy. This groups the matrices by shape, runs one batched iteration
    per group, and applies the momentum and parameter updates with multi-tensor
    kernels.

    The update is the same arithmetic as `torch.optim.Muon` and the state layout
    is identical (one `momentum_buffer` per parameter), so checkpoints written by
    either optimizer load into the other.
    """

    def __init__(
        self,
        params: Any,
        lr: float | Tensor = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = NS_COEFFICIENTS,
        eps: float = 1e-7,
        ns_steps: int = 5,
        adjust_lr_fn: str | None = None,
    ) -> None:
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if lr < 0.0:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if momentum < 0.0:
            raise ValueError(f"momentum should be >= 0 but is: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"weight decay should be >= 0 but is: {weight_decay}")
        if ns_steps < 1 or ns_steps >= 100:
            raise ValueError(f"ns_steps must be in [1, 100) but is: {ns_steps}")
        if len(ns_coefficients) != 3:
            raise ValueError("ns_coefficients must be a tuple of exactly 3 values")
        if adjust_lr_fn not in (None, "original", "match_rms_adamw"):
            raise ValueError(
                f"Adjust learning rate function {adjust_lr_fn} is not supported"
            )
        super().__init__(
            params,
            {
                "lr": lr,
                "weight_decay": weight_decay,
                "momentum": momentum,
                "nesterov": nesterov,
                "ns_coefficients": ns_coefficients,
                "eps": eps,
                "ns_steps": ns_steps,
                "adjust_lr_fn": adjust_lr_fn,
            },
        )
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.ndim != 2:
                    raise ValueError(
                        "Muon only supports 2D parameters whereas we found a "
                        f"parameter with size: {tuple(parameter.size())}"
                    )

    @torch.no_grad()
    def step(  # type: ignore[override]
        self, closure: Callable[[], Tensor] | None = None
    ) -> Tensor | None:
        """Perform one optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            parameters, gradients, buffers = self._collect(group)
            if not parameters:
                continue
            lr = group["lr"]
            momentum = group["momentum"]

            _foreach_lerp_(buffers, gradients, 1.0 - momentum)
            if group["nesterov"]:
                updates = _foreach_lerp(gradients, buffers, momentum)
            else:
                updates = buffers
            # Decoupled weight decay for the whole group in one kernel.
            _foreach_mul_(parameters, 1.0 - lr * group["weight_decay"])

            updated: list[Tensor] = []
            steps: list[Tensor] = []
            for (shape, _, _), indices in _batch_by_shape(parameters).items():
                orthogonal = _batched_newton_schulz(
                    torch.stack([updates[index] for index in indices]),
                    group["ns_coefficients"],
                    group["ns_steps"],
                    group["eps"],
                )
                orthogonal = orthogonal.to(dtype=parameters[indices[0]].dtype)
                # Fold the shape-adjusted step size in so one add closes the step.
                orthogonal.mul_(-_adjust_lr(lr, group["adjust_lr_fn"], shape))
                steps.extend(orthogonal.unbind(0))
                updated.extend(parameters[index] for index in indices)
            _foreach_add_(updated, steps)
        return loss

    def _collect(
        self, group: dict[str, Any]
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        """Return the parameters with gradients, their grads, and their buffers."""
        parameters: list[Tensor] = []
        gradients: list[Tensor] = []
        buffers: list[Tensor] = []
        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")
            if torch.is_complex(parameter):
                raise RuntimeError("Muon does not support complex parameters")
            if gradient.ndim != 2:
                raise ValueError("Param gradient must be a 2D matrix")
            state = self.state[parameter]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(
                    gradient, memory_format=torch.preserve_format
                )
            parameters.append(parameter)
            gradients.append(gradient)
            buffers.append(state["momentum_buffer"])
        return parameters, gradients, buffers


def _batch_by_shape(
    parameters: Sequence[Tensor],
) -> dict[tuple[tuple[int, ...], torch.dtype, torch.device], list[int]]:
    """Index the parameters that can be orthogonalized in one batched call."""
    batches: dict[tuple[tuple[int, ...], torch.dtype, torch.device], list[int]] = {}
    for index, parameter in enumerate(parameters):
        key = (tuple(parameter.shape), parameter.dtype, parameter.device)
        batches.setdefault(key, []).append(index)
    return batches


def build_optimizer(
    model: nn.Module,
    output_layers: Sequence[nn.Module],
    learning_rate: float,
    weight_decay: float,
    muon_momentum: float = 0.95,
    muon_nesterov: bool = True,
    muon_ns_steps: int = 5,
    muon_eps: float = 1e-7,
    adamw_betas: tuple[float, float] = (0.9, 0.999),
    adamw_eps: float = 1e-8,
    verbose: bool = False,
) -> MuonWithAuxAdamW:
    """Build Muon for hidden matrices and AdamW for explicit output layers and scalars."""
    groups = classify_params(model, output_layers)
    # The fused AdamW kernel needs every parameter on CUDA; CPU runs use foreach.
    fused_adamw = all(
        parameter.is_cuda
        for key in ("adamw_decay", "adamw_no_decay")
        for _, parameter in groups[key]
    )
    optimizer = MuonWithAuxAdamW(
        muon=BatchedMuon(
            [{"params": [parameter for _, parameter in groups["muon"]]}],
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=muon_momentum,
            nesterov=muon_nesterov,
            ns_steps=muon_ns_steps,
            eps=muon_eps,
            adjust_lr_fn="match_rms_adamw",
        ),
        adamw=optim.AdamW(
            [
                {
                    "params": [parameter for _, parameter in groups["adamw_decay"]],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [parameter for _, parameter in groups["adamw_no_decay"]],
                    "weight_decay": 0.0,
                },
            ],
            lr=learning_rate,
            betas=adamw_betas,
            eps=adamw_eps,
            fused=fused_adamw,
        ),
    )

    if verbose:
        labels = ("Muon", "AdamW decay", "AdamW no-decay")
        for label, group in zip(labels, optimizer.param_groups, strict=True):
            num_params = sum(parameter.numel() for parameter in group["params"])
            print(
                f"Optimizer [{label}]: lr={group['lr']:g}, "
                f"weight_decay={group['weight_decay']:g}, "
                f"{len(group['params'])} tensors, {num_params} params"
            )

    return optimizer
