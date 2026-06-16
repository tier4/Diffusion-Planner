"""Wrapper to run valid_predictor with MAX_NUM_NEIGHBORS=320."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "diffusion_planner"))

import diffusion_planner.dimensions as dims

dims.MAX_NUM_NEIGHBORS = 320
dims.MAX_NUM_AGENTS = 321

exec(
    open(
        os.path.join(os.path.dirname(__file__), "..", "diffusion_planner", "valid_predictor.py")
    ).read()
)
