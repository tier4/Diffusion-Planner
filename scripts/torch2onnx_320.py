"""Wrapper to run torch2onnx with MAX_NUM_NEIGHBORS=320."""

import os

import diffusion_planner.dimensions as dims

dims.MAX_NUM_NEIGHBORS = 320
dims.MAX_NUM_AGENTS = 321

exec(open(os.path.join(os.path.dirname(__file__), "..", "ros_scripts", "torch2onnx.py")).read())
