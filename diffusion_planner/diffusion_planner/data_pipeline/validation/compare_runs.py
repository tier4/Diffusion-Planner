"""Compare a training run's logged curve against a baseline's, using acceptance bands
derived from the baseline's own converged-plateau variance.

Diffusion training is stochastic, so pointwise equality is meaningless. The test is whether
the candidate's plateau mean falls inside k standard deviations of the baseline's plateau
mean, where the standard deviation is measured from the baseline itself.

A candidate's plateau coverage (row count and epoch min/max actually found in the requested
range) must match the baseline's exactly, and every in-range cell for a requested column must
be present and numeric — an incomplete or NaN-erased candidate is refused with a ValueError
naming the problem, never silently averaged down to a passing mean over fewer rows.

Comparability hazard: this tool only checks the logged metric columns, and does not itself
verify that the two runs used the same validation settings. It could: ``train.py`` writes
``args.json`` (which records both worker counts) into the same directory as ``train_log.tsv``,
so a sibling cross-check is available to a future caller. In particular, ``--valid_num_workers``
changes the shard loader's plan-slot count, which changes how many padded duplicate samples
``validate_model.aggregate_valid_metrics`` double-counts into the aggregate (see
``utils/shard_ddp.py`` and ``validate_model.py``). Two runs being compared here MUST have been
trained with the same ``--valid_num_workers`` (and, on the npz path, the same
``--num_workers``, which also feeds validation there) or any ``valid_loss_*`` column carries a
systematic shift unrelated to model quality, and a FAIL (or a marginal PASS) may just be
reflecting that mismatch rather than a real regression.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Sequence


@dataclass(frozen=True)
class Result:
    column: str
    baseline_mean: float
    baseline_sd: float
    lo: float
    hi: float
    candidate_mean: float
    candidate_sd: float
    n: int
    epoch_lo: int
    epoch_hi: int
    passed: bool


class PlateauStats(NamedTuple):
    mean: float
    sd: float
    n: int
    epoch_lo: int
    epoch_hi: int


def read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def plateau_stats(rows: list[dict[str, str]], column: str, epochs: tuple[int, int]) -> PlateauStats:
    """Mean and population sd of `column` over the inclusive epoch range.

    Every row whose epoch falls in range must carry a present, numeric value for `column` —
    an empty cell (pandas' ``to_csv`` renders NaN as ``""``) or a missing/non-numeric one
    raises ValueError naming the offending epoch, rather than being silently dropped from the
    average.
    """
    lo, hi = epochs
    if rows and column not in rows[0]:
        raise ValueError(f"column {column!r} not present; have {sorted(rows[0])}")
    vals: list[float] = []
    epochs_used: list[int] = []
    for r in rows:
        epoch = int(r["epoch"])
        if not (lo <= epoch <= hi):
            continue
        raw = r.get(column)
        if raw is None or raw == "":
            raise ValueError(
                f"column {column!r} is empty/missing at epoch {epoch} (in requested range "
                f"{lo}..{hi}); a diverged or unlogged epoch must not be silently dropped from "
                "the plateau average"
            )
        try:
            val = float(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"column {column!r} at epoch {epoch} is not numeric: {raw!r}"
            ) from None
        vals.append(val)
        epochs_used.append(epoch)
    if not vals:
        raise ValueError(f"no rows for {column!r} in epochs {lo}..{hi}")
    return PlateauStats(
        statistics.mean(vals),
        statistics.pstdev(vals),
        len(vals),
        min(epochs_used),
        max(epochs_used),
    )


def band(mean: float, sd: float, k: float = 2.0) -> tuple[float, float]:
    if k <= 0:
        raise ValueError(f"--k must be > 0, got {k}")
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
        b = plateau_stats(base_rows, col, epochs)
        c = plateau_stats(cand_rows, col, epochs)
        if (c.n, c.epoch_lo, c.epoch_hi) != (b.n, b.epoch_lo, b.epoch_hi):
            raise ValueError(
                f"column {col!r}: candidate plateau coverage does not match baseline's in "
                f"requested range {epochs[0]}..{epochs[1]} — candidate has {c.n} row(s) "
                f"(epochs {c.epoch_lo}..{c.epoch_hi}), baseline has {b.n} row(s) "
                f"(epochs {b.epoch_lo}..{b.epoch_hi}); a candidate missing plateau epochs "
                "must not be compared as if it were complete"
            )
        lo, hi = band(b.mean, b.sd, k)
        out.append(
            Result(
                col,
                b.mean,
                b.sd,
                lo,
                hi,
                c.mean,
                c.sd,
                c.n,
                c.epoch_lo,
                c.epoch_hi,
                lo <= c.mean <= hi,
            )
        )
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
    try:
        results = compare(a.candidate, a.baseline, a.columns, (a.from_epoch, a.to_epoch), a.k)
        width = max(len(r.column) for r in results)
        for r in results:
            verdict = "PASS" if r.passed else "FAIL"
            print(
                f"{r.column:<{width}}  n={r.n} epochs=[{r.epoch_lo}, {r.epoch_hi}]  "
                f"baseline={r.baseline_mean:.6f} sd={r.baseline_sd:.6f}  "
                f"band=[{r.lo:.6f}, {r.hi:.6f}]  "
                f"ours={r.candidate_mean:.6f} sd={r.candidate_sd:.6f}  {verdict}"
            )
    except (ValueError, FileNotFoundError, KeyError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
