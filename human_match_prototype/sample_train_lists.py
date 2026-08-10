"""Draw matched random samples from the train and train_sft path lists.

Both samples are the same size as the validation split (156,204) so their Energy
Score distributions are directly comparable with the completed validation run.

train and train_sft overlap heavily (~854k shared paths), so two independent
draws land on some of the same scenes. Scoring is driven off the *union* list to
avoid paying for those twice; per-subset statistics are recovered afterwards by
joining the scored rows back against each sample list.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def draw(paths: list[str], n: int, seed: int) -> list[str]:
    """Uniform random sample of n paths without replacement, order preserved."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), size=min(n, len(paths)), replace=False)
    return [paths[i] for i in sorted(idx)]


def main():
    p = argparse.ArgumentParser(description="Sample matched train / train_sft path lists.")
    p.add_argument("--train_list", required=True)
    p.add_argument("--sft_list", required=True)
    p.add_argument("--n", type=int, default=156204, help="Match the validation split size.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = json.load(open(args.train_list))
    sft = json.load(open(args.sft_list))

    # Distinct seeds so the two draws are independent rather than index-aligned.
    train_s = draw(train, args.n, args.seed)
    sft_s = draw(sft, args.n, args.seed + 1)

    union = sorted(set(train_s) | set(sft_s))

    for name, data in [
        ("sample_train.json", train_s),
        ("sample_train_sft.json", sft_s),
        ("sample_union.json", union),
    ]:
        with open(out / name, "w") as f:
            json.dump(data, f)

    shared = len(set(train_s) & set(sft_s))
    print(f"train pool     : {len(train):>9,}  -> sample {len(train_s):>9,}")
    print(f"train_sft pool : {len(sft):>9,}  -> sample {len(sft_s):>9,}")
    print(f"shared by both samples : {shared:,}")
    print(f"union to score         : {len(union):,} (saves {shared:,} duplicate scorings)")
    print(f"wrote 3 lists to {out}")


if __name__ == "__main__":
    main()
