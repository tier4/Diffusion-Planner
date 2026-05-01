"""Weighted-average two merged-LoRA checkpoints into a third.

Both inputs must share the same base architecture (i.e. were produced by
preference_optimization.merge_lora over the same base model). For tensors
present in both checkpoints, the output stores
    out = alpha * a + (1 - alpha) * b
For tensors present in only one input, the value is copied through.

Mathematically equivalent to averaging the underlying LoRA deltas:
    base + alpha*delta_A + (1-alpha)*delta_B
    = alpha*(base + delta_A) + (1-alpha)*(base + delta_B)

Usage:
    python -m rlvr.autoresearch.tools.avg_merged_checkpoints \
        --a A/merged.pth --b B/merged.pth --alpha 0.5 \
        --output OUT/merged.pth
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch


def _avg_state_dicts(a: dict, b: dict, alpha: float) -> dict:
    out = {}
    keys = set(a) | set(b)
    for k in keys:
        if k in a and k in b:
            va, vb = a[k], b[k]
            if va.shape != vb.shape:
                raise ValueError(f"shape mismatch on {k}: {va.shape} vs {vb.shape}")
            if va.dtype.is_floating_point:
                out[k] = alpha * va + (1.0 - alpha) * vb
            else:
                # ints/bools - prefer A (no meaningful average)
                out[k] = va
        elif k in a:
            out[k] = a[k]
        else:
            out[k] = b[k]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", type=Path, required=True, help="Checkpoint A (.pth)")
    p.add_argument("--b", type=Path, required=True, help="Checkpoint B (.pth)")
    p.add_argument("--alpha", type=float, required=True,
                   help="Weight on A. out = alpha*A + (1-alpha)*B.")
    p.add_argument("--output", type=Path, required=True, help="Output .pth")
    p.add_argument("--args_json", type=Path, default=None,
                   help="args.json to copy alongside output (default: copy from A's parent)")
    args = p.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {args.alpha}")

    print(f"A:      {args.a}")
    print(f"B:      {args.b}")
    print(f"alpha:  {args.alpha}  (out = alpha*A + (1-alpha)*B)")
    print(f"out:    {args.output}")

    ckpt_a = torch.load(args.a, map_location="cpu", weights_only=False)
    ckpt_b = torch.load(args.b, map_location="cpu", weights_only=False)

    out: dict = {}
    for top_key in ("model", "model_no_ddp"):
        if top_key in ckpt_a and top_key in ckpt_b:
            out[top_key] = _avg_state_dicts(ckpt_a[top_key], ckpt_b[top_key], args.alpha)
            n = len(out[top_key])
            print(f"  {top_key}: averaged {n} tensors")
        elif top_key in ckpt_a:
            out[top_key] = ckpt_a[top_key]
            print(f"  {top_key}: only in A, copied through")
        elif top_key in ckpt_b:
            out[top_key] = ckpt_b[top_key]
            print(f"  {top_key}: only in B, copied through")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)

    args_json_src = args.args_json or (args.a.parent / "args.json")
    args_json_dst = args.output.parent / "args.json"
    if args_json_src.exists() and not args_json_dst.exists():
        shutil.copy2(args_json_src, args_json_dst)
        print(f"  copied args.json from {args_json_src}")

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
