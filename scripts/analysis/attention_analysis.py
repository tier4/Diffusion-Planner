"""Aggregate fusion attention over a dataset, by token block and agent class.

Answers, at dataset scale, what a single frame can only hint at: does the model
attend to a class in proportion to how many tokens that class contributes, or
does it prefer or discount it? That ratio is the selectivity reported below,
where 1.0 means indistinguishable from count-proportional dilution.

With ``--inject`` an identical synthetic agent is added to every sampled frame,
which is the only way to measure the unknown class fairly: recorded shards
contain no unknown agents, and relabelling an existing one inherits that agent's
motion and geometry. Sweeping ``--inject-class`` over the classes holds geometry
fixed and varies only the label.

Examples:
  # Attention by block and agent class over 256 recorded frames.
  uv run python scripts/analysis/attention_analysis.py \\
    --checkpoint checkpoints/<run>/epoch_0001.pth \\
    --parquet $HOME/datasets/hdf5_small/indexes/train.parquet \\
    --frames 256

  # Hold geometry fixed and vary only the label of an injected agent.
  uv run python scripts/analysis/attention_analysis.py \\
    --checkpoint checkpoints/<run>/epoch_0001.pth \\
    --parquet $HOME/datasets/hdf5_small/indexes/train.parquet \\
    --frames 128 --inject --inject-sweep
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import argcomplete
import numpy as np
import torch
from tqdm import tqdm

from diffusion_planner.analysis import (
    AGENT_CLASS_NAMES,
    SceneTokenLayout,
    capture_fusion_attention,
    ego_query_attention,
    token_records,
)
from diffusion_planner.analysis.scene_edit import PlacedAgent, insert_agent
from diffusion_planner.data import PlannerDataset
from diffusion_planner.data.transforms import (
    PlannerDataNormalizer,
    PlannerUnknownLabelAugmentation,
)
from diffusion_planner.models.diffusion_planner import DiffusionPlanner
from diffusion_planner.utils.checkpoint import load_model


def _to_numpy_frame(frame: dict[str, Any]) -> dict[str, Any]:
    """Materialize a dataset item as plain NumPy arrays."""
    return {
        key: (value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value))
        for key, value in frame.items()
    }


def _ego_attention(
    model: DiffusionPlanner,
    frame: dict[str, Any],
    normalizer: PlannerDataNormalizer,
    layout: SceneTokenLayout,
    device: torch.device,
    layer: int | str,
) -> np.ndarray:
    """Encode one raw frame with capture on and return the ego-query row."""
    normalized = normalizer({key: np.asarray(v) for key, v in frame.items()})
    input_data = {
        key: torch.as_tensor(value, device=device).unsqueeze(0)
        for key, value in normalized.items()
    }
    with capture_fusion_attention(model) as capture, torch.no_grad():
        model.scene_encoder(input_data)
    attention = ego_query_attention(capture, layout, layer=layer)
    return attention[0].detach().float().cpu().numpy()


class _Accumulator:
    """Running attention and token totals per group."""

    def __init__(self) -> None:
        self.attention: dict[str, float] = defaultdict(float)
        self.tokens: dict[str, int] = defaultdict(int)
        self.frames = 0

    def add(self, records: list[dict[str, Any]], key: str) -> None:
        for record in records:
            name = record.get(key)
            if name is None:
                continue
            self.attention[str(name)] += float(record["attention"])
            self.tokens[str(name)] += 1

    def summary(self) -> dict[str, dict[str, float]]:
        """Share, token share and selectivity per group."""
        total_attention = sum(self.attention.values())
        total_tokens = sum(self.tokens.values())
        result: dict[str, dict[str, float]] = {}
        for name in sorted(self.attention):
            share = (
                self.attention[name] / total_attention if total_attention > 0 else 0.0
            )
            token_share = self.tokens[name] / total_tokens if total_tokens else 0.0
            result[name] = {
                "tokens": float(self.tokens[name]),
                "tokens_per_frame": self.tokens[name] / max(self.frames, 1),
                "share": share,
                "token_share": token_share,
                "selectivity": share / token_share if token_share > 0 else float("nan"),
            }
        return result


def _print_summary(title: str, summary: dict[str, dict[str, float]]) -> None:
    print(f"\n{title}")
    print(
        f"  {'group':<18}{'tokens/frame':>13}{'share %':>10}"
        f"{'token %':>10}{'selectivity':>13}"
    )
    ordered = sorted(summary.items(), key=lambda item: -item[1]["share"])
    for name, values in ordered:
        print(
            f"  {name:<18}{values['tokens_per_frame']:>13.2f}"
            f"{values['share'] * 100:>10.2f}{values['token_share'] * 100:>10.2f}"
            f"{values['selectivity']:>13.2f}"
        )


def _sample_indices(dataset: PlannerDataset, frames: int, seed: int) -> list[int]:
    """Evenly spread the sampled frames across the index."""
    total = len(dataset)
    if frames >= total:
        return list(range(total))
    generator = np.random.default_rng(seed)
    return sorted(generator.choice(total, size=frames, replace=False).tolist())


def _analyze(
    model: DiffusionPlanner,
    dataset: PlannerDataset,
    indices: list[int],
    *,
    device: torch.device,
    layer: int | str,
    inject: PlacedAgent | None,
    widen: PlannerUnknownLabelAugmentation,
    normalizer: PlannerDataNormalizer,
    layout: SceneTokenLayout,
    progress_label: str,
) -> tuple[_Accumulator, _Accumulator, list[float]]:
    """Accumulate attention over frames, optionally injecting an agent."""
    blocks = _Accumulator()
    classes = _Accumulator()
    injected_shares: list[float] = []

    for index in tqdm(indices, desc=progress_label, unit="frame"):
        frame = dict(widen(_to_numpy_frame(dataset[index])))
        injected_slot: int | None = None
        if inject is not None:
            try:
                frame, injected_slot = insert_agent(frame, inject)
            except ValueError:
                continue
        attention = _ego_attention(model, frame, normalizer, layout, device, layer)
        records = token_records(frame, attention, layout)
        blocks.add(records, "block")
        classes.add(records, "agent_class")
        blocks.frames += 1
        classes.frames += 1
        if injected_slot is not None:
            injected_shares.extend(
                record["attention_pct"]
                for record in records
                if record["block"] == "neighbors"
                and record["block_index"] == injected_slot
            )
    return blocks, classes, injected_shares


def main() -> None:
    """Summarize dataset-scale fusion attention."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--layer", default="mean", help="mean, last, or a fusion layer index"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--inject",
        action="store_true",
        help="add one synthetic agent to every sampled frame",
    )
    parser.add_argument(
        "--inject-class",
        default="unknown",
        choices=[*AGENT_CLASS_NAMES, "unlabeled"],
    )
    parser.add_argument(
        "--inject-sweep",
        action="store_true",
        help="repeat the run for every class, holding geometry fixed",
    )
    parser.add_argument("--inject-x", type=float, default=12.0)
    parser.add_argument("--inject-y", type=float, default=2.0)
    parser.add_argument("--inject-speed", type=float, default=1.4)
    parser.add_argument("--inject-width", type=float, default=0.8)
    parser.add_argument("--inject-length", type=float, default=0.8)
    parser.add_argument("--json-out", type=Path, default=None)
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.frames < 1:
        parser.error("--frames must be positive")

    layer: int | str = args.layer
    if layer not in {"mean", "last"}:
        layer = int(layer)

    device = torch.device(args.device)
    model = load_model(args.checkpoint, DiffusionPlanner).to(device).eval()
    dataset = PlannerDataset(str(args.parquet), file_capacity=8)
    indices = _sample_indices(dataset, args.frames, args.seed)
    widen = PlannerUnknownLabelAugmentation()
    normalizer = PlannerDataNormalizer()
    layout = SceneTokenLayout.from_dimensions()
    print(
        f"{len(indices)} frames from {args.parquet} on {device}, "
        f"layer={args.layer}, checkpoint={args.checkpoint.name}"
    )

    def placement(agent_class: str) -> PlacedAgent:
        return PlacedAgent(
            x_m=args.inject_x,
            y_m=args.inject_y,
            speed_mps=args.inject_speed,
            width_m=args.inject_width,
            length_m=args.inject_length,
            agent_class=agent_class,
        )

    report: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "parquet": str(args.parquet),
        "frames": len(indices),
        "layer": args.layer,
    }

    if args.inject_sweep:
        sweep: dict[str, Any] = {}
        for agent_class in (*AGENT_CLASS_NAMES, "unlabeled"):
            _, classes, shares = _analyze(
                model,
                dataset,
                indices,
                device=device,
                layer=layer,
                inject=placement(agent_class),
                widen=widen,
                normalizer=normalizer,
                layout=layout,
                progress_label=f"inject {agent_class}",
            )
            mean_share = float(np.mean(shares)) if shares else float("nan")
            sweep[agent_class] = {
                "injected_mean_share_pct": mean_share,
                "injected_frames": len(shares),
                "class_summary": classes.summary(),
            }
            print(
                f"  injected {agent_class:<11} mean share "
                f"{mean_share:.4f}% over {len(shares)} frames"
            )
        report["sweep"] = sweep
        print("\nsame geometry, label varied — mean share of ego-query attention")
        for agent_class, values in sorted(
            sweep.items(), key=lambda item: -item[1]["injected_mean_share_pct"]
        ):
            print(f"  {agent_class:<12}{values['injected_mean_share_pct']:>10.4f}%")
    else:
        inject = placement(args.inject_class) if args.inject else None
        blocks, classes, shares = _analyze(
            model,
            dataset,
            indices,
            device=device,
            layer=layer,
            inject=inject,
            widen=widen,
            normalizer=normalizer,
            layout=layout,
            progress_label="frames",
        )
        _print_summary("attention by token block", blocks.summary())
        _print_summary("attention by agent class", classes.summary())
        report["blocks"] = blocks.summary()
        report["agent_classes"] = classes.summary()
        if shares:
            report["injected"] = {
                "agent_class": args.inject_class,
                "mean_share_pct": float(np.mean(shares)),
                "median_share_pct": float(np.median(shares)),
                "frames": len(shares),
            }
            print(
                f"\ninjected {args.inject_class}: mean "
                f"{np.mean(shares):.4f}%, median {np.median(shares):.4f}% "
                f"over {len(shares)} frames"
            )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
