"""Verify FlipAugmentation visually and numerically.

Loads npz samples, applies the flip with flip_prob=1.0, renders original vs
flipped side by side with visualize_inputs, and runs numeric invariant checks
(double-flip identity, y-negation of futures, left/right boundary mirror,
padding preservation, turn indicator swap). Exits non-zero if any check fails.

Usage (from the diffusion_planner directory):
    uv run python util_scripts/visualize_flip_augmentation.py <npz> [<npz> ...] --out_dir <dir>
"""

import argparse
import copy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils.data_augmentation import FlipAugmentation
from diffusion_planner.utils.visualize_input import visualize_inputs


def load_inputs(npz_path: Path) -> dict:
    """Build the batch dict exactly like train_epoch right before flip_aug is applied."""
    loaded = np.load(npz_path)
    data = {}
    for key, value in loaded.items():
        if key in ("map_name", "token", "version"):
            continue
        t = torch.tensor(np.expand_dims(value, axis=0))
        if t.dtype == torch.float64:
            t = t.float()
        data[key] = t
    # train_epoch applies these before flip_aug; futures stay (x, y, heading).
    data["ego_agent_past"] = heading_to_cos_sin(data["ego_agent_past"])
    data["goal_pose"] = heading_to_cos_sin(data["goal_pose"])
    return data


def apply_flip(flip_aug: FlipAugmentation, inputs: dict) -> dict:
    """Apply the flip and return a dict with the flipped futures written back in.

    FlipAugmentation returns the flipped futures but only writes ego_agent_future
    back into the dict, so mirror the returned tensors in for checks/visualization.
    """
    work = copy.deepcopy(inputs)
    ego_f = work["ego_agent_future"].clone()
    nei_f = work["neighbor_agents_future"].clone()
    work, ego_f, nei_f = flip_aug(work, ego_f, nei_f)
    work["ego_agent_future"] = ego_f
    work["neighbor_agents_future"] = nei_f
    return work


def run_checks(orig: dict, flipped: dict, flip_aug: FlipAugmentation) -> list[str]:
    failures = []

    def check(name, cond):
        if not cond:
            failures.append(name)

    # 1. Double flip == identity (up to float error).
    twice = apply_flip(flip_aug, flipped)
    for key in orig:
        if not torch.is_floating_point(orig[key]):
            check(f"double-flip identity ({key})", torch.equal(twice[key], orig[key]))
        else:
            check(
                f"double-flip identity ({key})",
                torch.allclose(twice[key], orig[key], atol=1e-6),
            )

    # 2. Futures: y and heading negated.
    check(
        "ego_future y negated",
        torch.allclose(
            flipped["ego_agent_future"][..., 1], -orig["ego_agent_future"][..., 1], atol=1e-6
        ),
    )
    check(
        "neighbor_future y negated",
        torch.allclose(
            flipped["neighbor_agents_future"][..., 1],
            -orig["neighbor_agents_future"][..., 1],
            atol=1e-6,
        ),
    )

    # 3. Lanes: flipped left boundary == mirrored original right boundary (absolute coords).
    o, f = orig["lanes"], flipped["lanes"]
    o_rb = o[..., 0:2] + o[..., 6:8]
    f_lb = f[..., 0:2] + f[..., 4:6]
    mirror_o_rb = o_rb.clone()
    mirror_o_rb[..., 1] = -mirror_o_rb[..., 1]
    check("lanes L<-R boundary mirror", torch.allclose(f_lb, mirror_o_rb, atol=1e-5))

    # 4. Padding rows stay all-zero.
    pad_mask = orig["lanes"].abs().sum(dim=(-1, -2)) == 0
    check("lane padding stays zero", bool(flipped["lanes"][pad_mask].abs().sum() == 0))
    pad_n = orig["neighbor_agents_past"].abs().sum(dim=(-1, -2)) == 0
    check(
        "neighbor padding stays zero",
        bool(flipped["neighbor_agents_past"][pad_n].abs().sum() == 0),
    )

    # 5. Turn indicators: 2<->3 swapped, others unchanged.
    ti_o, ti_f = orig["turn_indicators"], flipped["turn_indicators"]
    check("turn indicator L->R", torch.equal(ti_f == 3, ti_o == 2))
    check("turn indicator R->L", torch.equal(ti_f == 2, ti_o == 3))
    check("turn indicator others", torch.equal((ti_f < 2) | (ti_f > 3), (ti_o < 2) | (ti_o > 3)))

    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_paths", type=Path, nargs="+")
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    flip_aug = FlipAugmentation(flip_prob=1.0, device="cpu")

    any_fail = False
    for npz_path in args.npz_paths:
        orig = load_inputs(npz_path)
        flipped = apply_flip(flip_aug, orig)

        failures = run_checks(orig, flipped, flip_aug)
        status = "OK" if not failures else "NG: " + ", ".join(failures)
        print(f"{npz_path.name}: {status}")
        any_fail |= bool(failures)

        # Side-by-side render. visualize_inputs mutates its dict -> pass copies.
        fig, axes = plt.subplots(1, 2, figsize=(20, 9))
        visualize_inputs(copy.deepcopy(orig), ax=axes[0])
        axes[0].set_title("original")
        visualize_inputs(copy.deepcopy(flipped), ax=axes[1])
        axes[1].set_title("flipped (y -> -y)")
        fig.suptitle(f"{npz_path.name}   [{status}]", color="red" if failures else "green")
        out = args.out_dir / f"{npz_path.stem}_flip_check.png"
        plt.tight_layout()
        plt.savefig(out, dpi=90, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {out}")

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
