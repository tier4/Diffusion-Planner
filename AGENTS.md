# AGENTS.md

## Project

Diffusion Planner is an autonomous-driving planner using diffusion models for trajectory generation.

- Python dependencies are managed with `uv`.
- `ros2_ws` is a ROS 2 workspace root.

## Key Paths

- packages/diffusion_planner/
    - Core Python package: datasets, models, diffusion logic, visualization.

- packages/diffusion_planner_dashboard/
    - Streamlit dashboard for visualization.

- ros2_ws/src/deps/autoware_universe/planning/autoware_ml_planner/
    - ROS 2 data processing and inference package. Git repository root of this package is `ros2_ws/src/deps/autoware_universe/`

- ros2_ws/src/ml_planner_data/
    - C++/pybind11 tools for generating model inputs and labels from rosbags.


The ROS 2 inference node and `ml_planner_data` share preprocessing logic. When changing model inputs, feature definitions, normalization, tensor shapes, or preprocessing behavior, check both sides for consistency.

## Rules

* This is PoC code. Test code is not important. Focus on the main logic and correctness of the planner.
* This is a PoC. Prefer simple, focused changes over production-level abstractions.
* Write code, comments, identifiers, and documentation in English.
* Do not edit `pyproject.toml` directly for dependency changes. Use `uv add`, `uv remove`, or other appropriate `uv` commands.
