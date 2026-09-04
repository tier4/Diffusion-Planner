"""Compare a training run's logged curve against a baseline's, using acceptance bands
derived from the baseline's own converged-plateau variance.

Diffusion training is stochastic, so pointwise equality is meaningless. The test is whether
the candidate's plateau mean falls inside k standard deviations of the baseline's plateau
mean, where the standard deviation is measured from the baseline itself.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class Result:
    column: str
    baseline_mean: float
    baseline_sd: float
    lo: float
    hi: float
    candidate_mean: float
    passed: bool


def read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def plateau_stats(
    rows: list[dict[str, str]], column: str, epochs: tuple[int, int]
) -> tuple[float, float]:
    """Mean and population sd of `column` over the inclusive epoch range."""
    lo, hi = epochs
    if rows and column not in rows[0]:
        raise ValueError(f"column {column!r} not present; have {sorted(rows[0])}")
    vals = [float(r[column]) for r in rows if r.get(column) and lo <= int(r["epoch"]) <= hi]
    if not vals:
        raise ValueError(f"no rows for {column!r} in epochs {lo}..{hi}")
    return statistics.mean(vals), statistics.pstdev(vals)


def band(mean: float, sd: float, k: float = 2.0) -> tuple[float, float]:
    return mean - k * sd, mean + k * sd


def compare(
    candidate: Path,
    baseline: Path,
    columns: Sequence[str],
    epochs: tuple[int, int],
    k: float = 2.0,
) -> list[Result]:
    cand_rows, base_rows = read_rows(candidate), read_rows(baseline)
    out: list[Result] = []
    for col in columns:
        b_mean, b_sd = plateau_stats(base_rows, col, epochs)
        c_mean, _ = plateau_stats(cand_rows, col, epochs)
        lo, hi = band(b_mean, b_sd, k)
        out.append(Result(col, b_mean, b_sd, lo, hi, c_mean, lo <= c_mean <= hi))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--columns", nargs="+", required=True)
    p.add_argument("--from-epoch", type=int, required=True)
    p.add_argument("--to-epoch", type=int, required=True)
    p.add_argument("--k", type=float, default=2.0)
    a = p.parse_args(argv)
    results = compare(a.candidate, a.baseline, a.columns, (a.from_epoch, a.to_epoch), a.k)
    width = max(len(r.column) for r in results)
    for r in results:
        verdict = "PASS" if r.passed else "FAIL"
        print(
            f"{r.column:<{width}}  baseline={r.baseline_mean:.6f} sd={r.baseline_sd:.6f}  "
            f"band=[{r.lo:.6f}, {r.hi:.6f}]  ours={r.candidate_mean:.6f}  {verdict}"
        )
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
