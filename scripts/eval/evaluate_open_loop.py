#!/usr/bin/env python3
"""Open-loop evaluation of a planner checkpoint (flow matching or PLUTO).

Runs ``model.sample`` on frames from a Parquet index without augmentation and
reports ego ADE/FDE in meters (8 s and 3 s), miss rate, heading error, neighbor
ADE, turn-indicator accuracy and inference latency. For ``PlutoPlanner`` it adds
the oracle (best of all candidates) ADE/FDE, the progress-bin selection accuracy
and the target / selected mode histograms.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from diffusion_planner.data import (
    FillUnknownTrafficLightFutures,
    PlannerDataNormalizer,
    PlannerDataset,
    PlannerGoalTransform,
)
from diffusion_planner.data.dimensions import (
    MAX_NUM_NEIGHBORS,
    TRAJECTORY_DIM,
    TRAJECTORY_LENGTH,
)
from diffusion_planner.utils.checkpoint import load_model

try:  # four-class branches widen the stored 3-column agent label at load time
    from diffusion_planner.data.transforms import PlannerUnknownLabelAugmentation
except ImportError:  # three-class branches
    PlannerUnknownLabelAugmentation = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--num-steps", type=int, default=20, help="flow-matching sampler steps"
    )
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode-interval-m", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=None, help="JSON output path")
    return parser.parse_args()


class Accumulator:
    """Mean of per-sample values across batches."""

    def __init__(self) -> None:
        self.sums: dict[str, float] = defaultdict(float)
        self.counts: dict[str, float] = defaultdict(float)

    def add(
        self, name: str, values: torch.Tensor, weights: torch.Tensor | None = None
    ) -> None:
        values = values.detach().float()
        if weights is None:
            self.sums[name] += float(values.sum())
            self.counts[name] += float(values.numel())
        else:
            weights = weights.detach().float()
            self.sums[name] += float((values * weights).sum())
            self.counts[name] += float(weights.sum())

    def means(self) -> dict[str, float]:
        return {
            name: self.sums[name] / max(self.counts[name], 1.0)
            for name in sorted(self.sums)
        }


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    model = load_model(args.checkpoint).to(device).eval()
    is_pluto = hasattr(model, "pluto_decoder")
    normalizer = PlannerDataNormalizer()
    transforms = [PlannerGoalTransform(time_shift_probability=0.0)]
    if PlannerUnknownLabelAugmentation is not None:
        transforms.append(PlannerUnknownLabelAugmentation(probability=0.0))
    transforms += [FillUnknownTrafficLightFutures(), normalizer]
    dataset = PlannerDataset(args.parquet, transforms=tuple(transforms))
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    scale = float(normalizer.position_scale)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    accumulator = Accumulator()
    num_modes = int(getattr(model, "num_modes", 0))
    target_histogram = torch.zeros(num_modes)
    selected_histogram = torch.zeros(num_modes)
    frames = 0
    latency_s = 0.0

    if is_pluto:
        from diffusion_planner.models.pluto_decoder import select_candidate
        from diffusion_planner.models.pluto_loss import assign_mode_targets

    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        batch = {
            key: value.to(device, non_blocking=True) for key, value in batch.items()
        }
        batch_size = batch["ego_agent_past"].shape[0]
        noise = args.noise_scale * torch.randn(
            (batch_size, MAX_NUM_NEIGHBORS + 1, TRAJECTORY_LENGTH, TRAJECTORY_DIM),
            device=device,
            generator=generator,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        trajectory, turn_logits = model.sample(
            batch, initial_noise=noise, num_steps=args.num_steps, time_epsilon=1e-5
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency_s += time.perf_counter() - start
        frames += batch_size

        ego_pred = trajectory[:, 0].float()
        ego_gt = batch["ego_agent_future"][..., :TRAJECTORY_DIM].float()
        error = torch.linalg.vector_norm(
            (ego_pred[..., :2] - ego_gt[..., :2]) * scale, dim=-1
        )
        accumulator.add("ego_ade_8s_m", error.mean(dim=-1))
        accumulator.add("ego_fde_8s_m", error[:, -1])
        accumulator.add("ego_ade_3s_m", error[:, :30].mean(dim=-1))
        accumulator.add("ego_fde_3s_m", error[:, 29])
        accumulator.add("ego_miss_rate_fde_gt_2m", (error[:, -1] > 2.0).float())
        cosine = (ego_pred[..., 2:4] * ego_gt[..., 2:4]).sum(dim=-1).clamp(-1.0, 1.0)
        accumulator.add(
            "ego_heading_error_deg", torch.rad2deg(torch.acos(cosine)).mean(dim=-1)
        )

        neighbor_pred = trajectory[:, 1:].float()
        neighbor_gt = batch["neighbor_agents_future"].float()
        neighbor_valid = neighbor_gt[..., 2:4].abs().sum(dim=-1) > 0
        neighbor_error = torch.linalg.vector_norm(
            (neighbor_pred[..., :2] - neighbor_gt[..., :2]) * scale, dim=-1
        )
        accumulator.add("neighbor_ade_m", neighbor_error, neighbor_valid)

        turn_target = batch["turn_indicators_future"][:, 0].long()
        turn_valid = (turn_target >= 1) & (turn_target <= 3)
        turn_correct = turn_logits.argmax(dim=-1) == (turn_target - 1)
        accumulator.add(
            "turn_indicator_accuracy", turn_correct.float(), turn_valid.float()
        )

        if is_pluto:
            scene, scene_mask = model.scene_encoder(batch)
            outputs = model.decode(batch, scene, scene_mask)
            candidates = outputs["candidates"].float()  # (B, R, M, T, 4)
            target_mode, progress_m = assign_mode_targets(
                ego_gt[..., :2],
                batch["route_lanes"],
                position_scale=scale,
                num_modes=num_modes,
                mode_interval_m=args.mode_interval_m,
            )
            distance = torch.linalg.vector_norm(
                (candidates[..., :2] - ego_gt[:, None, None, :, :2]) * scale, dim=-1
            )
            ade = distance.mean(dim=-1).reshape(batch_size, -1)
            fde = distance[..., -1].reshape(batch_size, -1)
            _, selected = select_candidate(
                outputs["candidates"], outputs["logits"], outputs["line_padding"]
            )
            rows = torch.arange(batch_size, device=device)
            accumulator.add("pluto_oracle_ade_8s_m", ade.min(dim=-1).values)
            accumulator.add("pluto_oracle_fde_8s_m", fde.min(dim=-1).values)
            accumulator.add("pluto_target_mode_ade_8s_m", ade[rows, target_mode])
            accumulator.add(
                "pluto_selection_accuracy", (selected == target_mode).float()
            )
            accumulator.add(
                "pluto_selection_bin_abs_error", (selected - target_mode).abs().float()
            )
            accumulator.add("pluto_progress_8s_m", progress_m)
            target_histogram += torch.bincount(
                target_mode.cpu(), minlength=num_modes
            ).float()
            selected_histogram += torch.bincount(
                selected.cpu(), minlength=num_modes
            ).float()

    results = accumulator.means()
    results["frames"] = frames
    results["latency_ms_per_batch"] = 1000.0 * latency_s / max(batch_index, 1)
    results["batch_size"] = args.batch_size
    results["model"] = f"{type(model).__module__}.{type(model).__name__}"
    results["checkpoint"] = str(args.checkpoint)
    if is_pluto:
        total = max(float(target_histogram.sum()), 1.0)
        results["pluto_target_mode_fraction"] = [
            round(value / total, 4) for value in target_histogram.tolist()
        ]
        results["pluto_selected_mode_fraction"] = [
            round(value / total, 4) for value in selected_histogram.tolist()
        ]

    print(f"model={results['model']}")
    print(f"checkpoint={results['checkpoint']}")
    print(f"frames={frames} latency_ms_per_batch={results['latency_ms_per_batch']:.1f}")
    for name, value in results.items():
        if isinstance(value, float) and not math.isnan(value):
            print(f"{name:34s} {value:10.4f}")
    if is_pluto:
        print("mode  target_frac  selected_frac  (bin = 10 m of progress at 8 s)")
        for index, (target, selected) in enumerate(
            zip(
                results["pluto_target_mode_fraction"],
                results["pluto_selected_mode_fraction"],
                strict=True,
            )
        ):
            print(f"{index:4d}  {target:11.3f}  {selected:13.3f}")

    output = (
        args.output
        or args.checkpoint.parent / f"eval_open_loop_{args.checkpoint.stem}.json"
    )
    output.write_text(json.dumps(results, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
