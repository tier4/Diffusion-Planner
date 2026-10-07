"""Pass a planner's output through ``ml_planner_node``'s road border avoidance and optimizer.

``ml_planner_optimizer`` is that code, built from ``cpp_tools/src/ml_planner_optimizer_python``;
it is imported only when the optimizer is switched on. What a rollout cannot give it the node's
way is supplied here: the steering angle is read off the optimizer's own last solution, which is
the plan being tracked, and a failed solve returns None so the caller keeps its current plan, as
the node publishes nothing then.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from diffusion_planner.config.config_cli import boolean

DT = 0.1


def add_plan_optimizer_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--plan_optimizer",
        type=boolean,
        nargs="?",
        const=True,
        default=False,
        help="pass each plan through ml_planner_node's road border avoidance and trajectory "
        "optimizer (needs the ml_planner_optimizer module)",
    )
    parser.add_argument(
        "--plan_optimizer_config_dir",
        default=None,
        help="with --plan_optimizer: a directory holding the vehicle's vehicle_info.param.yaml "
        "and, optionally, an ml_planner.param.yaml to use instead of the installed one, which "
        "leaves road border avoidance off",
    )


@dataclass(frozen=True)
class PlanOptimizerConfig:
    param_yaml: Path
    vehicle_yaml: Path
    vehicle: dict[str, float]

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> PlanOptimizerConfig | None:
        """None when the optimizer is off. Imports the module and reads the vehicle up front, so
        a missing module or file fails before any rollout starts."""
        if not args.plan_optimizer:
            return None
        import ml_planner_optimizer  # noqa: PLC0415 - only a rollout that asks for it needs it
        import yaml  # noqa: PLC0415

        config_dir = Path(args.plan_optimizer_config_dir)
        # The installed default sits under the prefix of the module, <prefix>/lib/python3.X/
        # site-packages/<module>.so. Under colcon --symlink-install that file links into the
        # build tree, so the path as imported is tried before the resolved one.
        imported = Path(os.path.abspath(ml_planner_optimizer.__file__))
        installed = Path("share/ml_planner_optimizer_python/config/ml_planner.param.yaml")
        candidates = [
            config_dir / "ml_planner.param.yaml",
            *(path.parents[3] / installed for path in (imported, imported.resolve())),
        ]
        param_yaml = next((path for path in candidates if path.is_file()), None)
        if param_yaml is None:
            raise FileNotFoundError(f"no ml_planner.param.yaml among {candidates}")
        vehicle_yaml = config_dir / "vehicle_info.param.yaml"
        ((_, node),) = yaml.safe_load(vehicle_yaml.read_text()).items()
        vehicle = {k: float(v) for k, v in node["ros__parameters"].items()}
        return cls(param_yaml, vehicle_yaml, vehicle)


class PlanOptimizer:
    """One rollout's optimizer: its warm start and goal latch carry from one cycle to the next."""

    def __init__(
        self, config: PlanOptimizerConfig, bounding_box: dict, borders: list[np.ndarray]
    ) -> None:
        import ml_planner_optimizer  # noqa: PLC0415

        length = float(bounding_box["dimensions"]["x"])
        side = (float(bounding_box["dimensions"]["y"]) - config.vehicle["wheel_tread"]) / 2.0
        # The footprint spans exactly the simulated box. The pose is base_link, the rear axle,
        # and the box centre sits center.x ahead of it; a box shorter than the vehicle_info
        # vehicle gives a negative front overhang rather than a longer footprint.
        rear_overhang = length / 2.0 - float(bounding_box["center"]["x"])
        overrides = {
            "rear_overhang": rear_overhang,
            "front_overhang": length - config.vehicle["wheel_base"] - rear_overhang,
            "left_overhang": side,
            "right_overhang": side,
        }
        self._optimizer = ml_planner_optimizer.Optimizer(
            str(config.param_yaml), str(config.vehicle_yaml), overrides
        )
        self._optimizer.set_road_borders(borders)
        self._param_yaml = str(config.param_yaml)
        self._last: tuple[float, np.ndarray] | None = None  # (sim_time, front wheel angle)
        self._cycles: list[dict] = []

    def step(
        self,
        plan_map: np.ndarray,
        ego_xyh: tuple[float, float, float],
        speed: float,
        sim_time: float,
        goal_xyh: tuple[float, float, float] | None,
    ) -> np.ndarray | None:
        """Map-frame ``(80, 3)`` (x, y, yaw) at t = 0.1..8.0 s -> the same, or None when the
        solve failed.

        ``sim_time`` must be the simulation clock: the optimizer drops a warm start older than
        0.5 s and measures its temporal consistency term against it.
        """
        steer = 0.0
        if self._last is not None:
            t, angles = self._last
            times = DT * np.arange(1, len(angles) + 1)
            steer = float(np.interp(sim_time - t, times, angles))
        out = self._optimizer.step(plan_map, (*ego_xyh, speed), steer, sim_time, goal_xyh)
        trajectory = out.pop("trajectory")
        self._cycles.append(
            {**out, "time": sim_time, "failed": trajectory is None, "no_plan": self._last is None}
        )
        if trajectory is None:
            return None
        self._last = (sim_time, trajectory[:, 4])
        return trajectory[:, :3]

    def summary(self) -> dict:
        cycles = self._cycles
        shifted = [c for c in cycles if c["border_shifted_points"]]
        failed = [c for c in cycles if c["failed"]]
        ms = [c["solve_time_ms"] for c in cycles if "solve_time_ms" in c] or [0.0]
        return {
            "param_yaml": self._param_yaml,
            "calls": len(cycles),
            "failures": len(failed),
            "failures_without_plan": sum(c["no_plan"] for c in failed),
            # Road border avoidance: whether it acted, how often, how far, and from when.
            "border_shifted": bool(shifted),
            "border_shifted_cycles": len(shifted),
            "border_shift_rate": len(shifted) / len(cycles) if cycles else 0.0,
            "border_shifted_points": sum(c["border_shifted_points"] for c in cycles),
            "border_unresolved_cycles": sum(c["border_unresolved_points"] > 0 for c in cycles),
            "border_max_shift_m": max([0.0, *(c["border_max_shift_m"] for c in cycles)]),
            "border_first_shift_s": shifted[0]["time"] if shifted else None,
            "solve_ms_p50": float(np.median(ms)),
            "solve_ms_max": float(max(ms)),
        }
