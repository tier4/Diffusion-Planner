"""Wrapper to run valid_predictor with MAX_NUM_NEIGHBORS=320."""

import os
import runpy
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "diffusion_planner"))

import diffusion_planner.dimensions as dims

dims.MAX_NUM_NEIGHBORS = 320
dims.MAX_NUM_AGENTS = 321

# Run the target as __main__ (proper module context, file handle closed by runpy).
# The override above mutates the cached diffusion_planner.dimensions module, so the
# target sees 320 when it reads the constants.
runpy.run_path(
    os.path.join(os.path.dirname(__file__), "..", "diffusion_planner", "valid_predictor.py"),
    run_name="__main__",
)
