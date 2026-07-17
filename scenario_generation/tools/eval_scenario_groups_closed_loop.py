#!/usr/bin/env python3
"""Grouped closed-loop validation — thin wrapper around ``valid_predictor_closed_loop --mode grouped``.

Prefer the unified entry point::

    python diffusion_planner/valid_predictor_closed_loop.py --mode grouped ...

This module remains for backward-compatible ``python -m scenario_generation.tools...`` usage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scenario_generation.closed_loop_cli import (  # noqa: E402
    add_grouped_args,
    add_output_args,
    add_rollout_args,
    run_grouped_eval,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        epilog="Rollout knobs match valid_predictor_closed_loop.py (--near_miss_thresh, etc.).",
    )
    add_rollout_args(p)
    add_output_args(p)
    add_grouped_args(p)
    return p.parse_args()


def main() -> int:
    return run_grouped_eval(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
