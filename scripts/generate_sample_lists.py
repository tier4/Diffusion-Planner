#!/usr/bin/env python3
"""Generate non-overlapping random sample lists for bank and eval from a training path list.

Usage:
    python scripts/generate_sample_lists.py \
        --src /path/to/path_list_train.json \
        --output_dir data/ \
        [--bank_size 50000] [--eval_size 5000] \
        [--bank_seed 42] [--eval_seed 99]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--src", type=Path, required=True, help="Source path list JSON")
    p.add_argument("--output_dir", type=Path, required=True, help="Output directory")
    p.add_argument("--bank_size", type=int, default=50000)
    p.add_argument("--eval_size", type=int, default=5000)
    p.add_argument("--bank_seed", type=int, default=42)
    p.add_argument("--eval_seed", type=int, default=99)
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.src) as f:
        paths = json.load(f)
    print(f"Loaded {len(paths)} paths from {args.src}")

    rng_bank = np.random.RandomState(args.bank_seed)
    bank_idx = set(rng_bank.choice(len(paths), args.bank_size, replace=False).tolist())

    remaining = [i for i in range(len(paths)) if i not in bank_idx]
    rng_eval = np.random.RandomState(args.eval_seed)
    eval_idx = rng_eval.choice(len(remaining), args.eval_size, replace=False)

    bank_paths = [paths[i] for i in sorted(bank_idx)]
    eval_paths = [paths[remaining[i]] for i in eval_idx]

    bank_out = args.output_dir / f"bank_{args.bank_size // 1000}k.json"
    eval_out = args.output_dir / f"eval_{args.eval_size // 1000}k.json"

    with open(bank_out, "w") as f:
        json.dump(bank_paths, f)
    with open(eval_out, "w") as f:
        json.dump(eval_paths, f)

    overlap = len(set(bank_paths) & set(eval_paths))
    print(f"Bank: {len(bank_paths)} -> {bank_out}")
    print(f"Eval: {len(eval_paths)} -> {eval_out}")
    print(f"Overlap: {overlap}")


if __name__ == "__main__":
    main()
