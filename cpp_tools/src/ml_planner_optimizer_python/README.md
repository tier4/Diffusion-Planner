# ml_planner_optimizer_python

`ml_planner_node` post-processes the model output before publishing it: road border avoidance,
then an acados trajectory optimizer. This package compiles those two stages from
`autoware_ml_planner`'s own, unmodified sources into the Python module `ml_planner_optimizer`, so
closed-loop evaluation can pass each plan through them.

```bash
export ML_PLANNER_SOURCE_DIR=<autoware_universe>/planning/autoware_ml_planner
colcon build --packages-select ml_planner_optimizer_python
```

acados must be installed at `/opt/acados`, with `lib/` and a `.venv/` that has `acados_template`;
the module loads it from there at run time. `build.sh` does not build this package.
`autoware_ml_planner` must have the goal unlatch parameters (tier4/autoware_universe 34b5fc2 or
later).

`Optimizer(param_yaml, vehicle_yaml, vehicle_overrides={})` reads the node's parameter file and a
`vehicle_info.param.yaml`; `vehicle_overrides` replaces some of its keys, e.g. with the simulated
ego's box. `step(raw, ego, steering_angle_rad, stamp_s, goal=None)` runs one planning cycle.
`autoware_ml_planner`'s `ml_planner.param.yaml` is installed to `share/ml_planner_optimizer_python/config/`.
