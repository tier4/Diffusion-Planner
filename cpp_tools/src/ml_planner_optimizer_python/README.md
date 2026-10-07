# ml_planner_optimizer_python

`ml_planner_node` post-processes the model output before publishing it: road border avoidance,
then an acados trajectory optimizer. This package compiles those two stages from
`autoware_ml_planner`'s own, unmodified sources into the Python module `ml_planner_optimizer`.
Closed-loop evaluation can pass each plan through it before the ego tracks it.

The module has to be importable where the rollout runs, e.g. by sourcing the install's
`local_setup.bash`.

## Build

`autoware_ml_planner` lives in tier4/autoware_universe, whose packages share names with the
autoware_universe this workspace imports, so it stays outside the colcon base path:

```bash
export ML_PLANNER_SOURCE_DIR=<tier4/autoware_universe>/planning/autoware_ml_planner
export ACADOS_SOURCE_DIR=/opt/acados   # needs lib/ and .venv/ with acados_template
colcon build --packages-select ml_planner_optimizer_python
```

The solver C code is generated at build time from the package's `scripts/generate_solver.py`.
The acados libraries are installed next to the module, so a host that only runs it needs no acados.

## Parameters

`config/ml_planner.param.yaml` and `config/vehicle_info.param.yaml` are copies of the files the
x2 launcher deploys; their headers name the versions. The footprint is overridden with the
simulated ego's box. The bicycle model keeps the shipped wheelbase unless the simulator reports
the ego's own, so for a box shorter than that vehicle an overhang goes negative
(`vehicle_info` logs it) while the footprint still spans the box.

Besides acados, `libvehicle_info_utils` and `libautoware_utils_geometry` are installed next to the
module: the simulator's runtime install does not carry them. rclcpp, lanelet2_core and yaml-cpp
come from `/opt/ros/humble` and the system.
