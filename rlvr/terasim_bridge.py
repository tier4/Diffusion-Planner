"""
HTTP client to the TeraSim REST service running inside Docker.

Does not import any TeraSim Python package; all interaction is via the
FastAPI REST API exposed on port 8000.

Container lifecycle:
  - The Docker container is started automatically on the first call to
    start_episode() if it is not already running.
  - The container is NOT stopped on close() to allow episode reuse.
    Call stop_container() explicitly to remove the container.

Step protocol (non-auto_run mode):
  1. send_agent_command: enqueues moveToXY for the AV
  2. sleep 25 ms: let the cosim Redis loop process the command
  3. POST /simulation_tick/{id}: advance 0.1 s
  4. poll simulation_time > prev: confirms the step completed and state
     was written to Redis
  5. GET /simulation/{id}/state: read NPC positions and AV presence

GUI mode (gui=True):
  - Uses ghost_replay_gui.yaml (gui_flag: true) so TeraSim launches sumo-gui.
  - The container is started with X11 forwarding flags so the sumo-gui window
    appears on the host desktop.
  - Prerequisite (once per session on the host):  xhost +local:docker

FCD recording (fcd_host_dir):
  - When a host directory is supplied the container's SUMO output directory
    (/tmp/terasim_output) is bind-mounted there.
  - After the episode the FCD file is at:
      <fcd_host_dir>/ghost_replay/raw_data/0/<sim_id>/fcd_all.xml
  - fcd_all is enabled in both ghost_replay.yaml and ghost_replay_gui.yaml,
    so the file is always produced when this mount is active.
"""

import os
import subprocess
import time

import requests

from rlvr.npz_utils import ros_yaw_to_sumo_angle


class TeraSimBridge:
    def __init__(
        self,
        sim_config_host_dir: str,
        gui: bool = False,
        fcd_host_dir: str | None = None,
        service_url: str = "http://localhost:8000",
        docker_image: str = "terasim:latest",
        container_name: str = "terasim_rlvr",
        step_timeout: float = 30.0,
    ):
        """
        Args:
            sim_config_host_dir: Absolute path to rlvr/sim_config/ on the host.
                                 Volume-mounted at /sim_config inside the container.
            gui:                 Launch sumo-gui for live visualization via X11.
                                 Requires `xhost +local:docker` on the host first.
                                 Uses ghost_replay_gui.yaml (gui_flag: true).
            fcd_host_dir:        If set, bind-mount this host directory to
                                 /tmp/terasim_output inside the container so that
                                 SUMO FCD output is written to the host filesystem.
                                 After the episode, the FCD file is at:
                                   <fcd_host_dir>/ghost_replay/raw_data/0/<sim_id>/fcd_all.xml
            service_url:         Base URL of the TeraSim FastAPI service.
            docker_image:        Docker image tag to run.
            container_name:      Docker container name.
            step_timeout:        Seconds to wait per simulation step.
        """
        self.sim_config_host_dir = sim_config_host_dir
        self.gui = gui
        self.fcd_host_dir = fcd_host_dir
        self.config_yaml_container_path = (
            "/sim_config/ghost_replay_gui.yaml" if gui
            else "/sim_config/ghost_replay.yaml"
        )
        self.service_url = service_url
        self.docker_image = docker_image
        self.container_name = container_name
        self.step_timeout = step_timeout

        self._sim_id: str | None = None
        self._last_state: dict | None = None
        self._sim_time: float = -1.0  # tracks last known simulation_time

    # ------------------------------------------------------------------
    # Container lifecycle
    # ------------------------------------------------------------------

    def _ensure_container_running(self) -> None:
        """Start the Docker container if it is not already running."""
        result = subprocess.run(
            ["docker", "inspect", "--format={{.State.Running}}", self.container_name],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip() == "true":
            return  # already running

        # Remove any stopped container with the same name
        subprocess.run(
            ["docker", "rm", "-f", self.container_name],
            capture_output=True,
        )

        cmd = [
            "docker", "run", "-d",
            "--name", self.container_name,
            "-p", "8000:8000",
            "-p", "6379:6379",
            "-p", "8050:8050",
            "-v", f"{self.sim_config_host_dir}:/sim_config:ro",
        ]

        # X11 forwarding for sumo-gui
        if self.gui:
            display = os.environ.get("DISPLAY", ":0")
            cmd += [
                "-e", f"DISPLAY={display}",
                "-v", "/tmp/.X11-unix:/tmp/.X11-unix",
            ]

        # Bind-mount host output directory so FCD/collision files reach the host
        if self.fcd_host_dir:
            os.makedirs(self.fcd_host_dir, exist_ok=True)
            cmd += ["-v", f"{self.fcd_host_dir}:/tmp/terasim_output"]

        cmd += [
            self.docker_image,
            "sh", "-c",
            # --protected-mode no: allow host connections for GT trajectory push
            "redis-server --daemonize yes --protected-mode no "
            "&& python3 -m terasim_service",
        ]

        subprocess.run(cmd, check=True)
        self._wait_for_service()

    def _wait_for_service(self, timeout: float = 60.0) -> None:
        """Poll GET /health until 200 OK or timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = requests.get(f"{self.service_url}/health", timeout=2.0)
                if r.status_code == 200:
                    return
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1.0)
        raise RuntimeError(
            f"TeraSim service at {self.service_url} did not become ready within {timeout}s"
        )

    @property
    def fcd_output_path(self) -> str | None:
        """
        Absolute path to the fcd_all.xml produced by the last episode, or None
        if fcd_host_dir was not set or no episode has run yet.
        """
        if self.fcd_host_dir is None or self._sim_id is None:
            return None
        return os.path.join(
            self.fcd_host_dir,
            "ghost_replay", "raw_data", "0", self._sim_id, "fcd_all.xml",
        )

    def stop_container(self) -> None:
        """Forcefully remove the Docker container."""
        subprocess.run(
            ["docker", "rm", "-f", self.container_name],
            capture_output=True,
        )

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def start_episode(
        self,
        spawn_states: dict,
        enable_viz: bool = False,
        viz_port: int = 8050,
    ) -> None:
        """
        Start a new simulation episode.

        1. Ensures the Docker container is running.
        2. POSTs /start_simulation with auto_run=false.
        3. Polls until status = "wait_for_tick" (NDE warmup done, AV spawned).
        4. Teleports the AV to the t=0 ego position from spawn_states.

        Args:
            spawn_states: dict returned by npz_utils.extract_spawn_states()
            enable_viz:   Start the TeraSim Dash visualizer (http://localhost:<viz_port>)
            viz_port:     Port for the Dash visualizer (default 8050)
        """
        self._ensure_container_running()

        params = {}
        if enable_viz:
            params["enable_viz"] = "true"
            params["viz_port"] = str(viz_port)

        resp = requests.post(
            f"{self.service_url}/start_simulation",
            params=params,
            json={
                "config_file": self.config_yaml_container_path,
                "auto_run": False,
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        self._sim_id = resp.json()["simulation_id"]
        self._sim_time = -1.0

        # Wait for NDE warmup (0 s) and AV spawn to complete
        self._poll_status("wait_for_tick", timeout=60.0)

        # Store GT trajectory in Redis so the Dash visualizer can draw it.
        # Key expires with the same 3600 s TTL used by the TeraSim service.
        if enable_viz and "ego_future_map" in spawn_states:
            import json as _json
            traj = spawn_states["ego_future_map"][:, :2].tolist()  # [[x,y], ...]
            try:
                import redis as _redis
                # health_check_interval=0: skip RESP3 HELLO handshake
                # (redis-py 7.x vs Redis server 6.x incompatibility)
                _r = _redis.Redis(
                    host="localhost", port=6379,
                    decode_responses=True,
                    health_check_interval=0,
                )
                _r.setex(
                    f"simulation:{self._sim_id}:gt_trajectory",
                    3600,
                    _json.dumps(traj),
                )
            except Exception:
                pass  # non-fatal; viz still works without trajectory overlay

        # Teleport AV to ground-truth t=0 position
        ego = spawn_states["ego"]
        self._send_agent_command(
            "AV", ego["x"], ego["y"], ego["sumo_angle"], ego["vx"]
        )

    def step(
        self, ego_xy: tuple, ego_yaw_rad: float, ego_speed: float = 0.0
    ) -> dict:
        """
        Teleport the AV to the given map-frame position, then advance one 0.1 s step.

        The simulation time advancing by ~0.1 s is used to confirm that the step
        completed, which avoids the race condition of checking a fixed status string.

        Args:
            ego_xy:      (x, y) in MGRS map frame (meters)
            ego_yaw_rad: heading in radians, CCW from +X (ROS convention)
            ego_speed:   speed in m/s

        Returns:
            {
                "collision":  bool   — True if the AV disappeared (SUMO collision removal)
                "av_in_sim":  bool   — False if the AV was removed from the simulation
                "npc_states": list   — [{"id", "x", "y", "sumo_angle", "speed"}, ...]
                "sim_time":   float  — current simulation time in seconds
            }
        """
        sumo_angle = ros_yaw_to_sumo_angle(ego_yaw_rad)

        # 1. Queue the AV teleport command
        self._send_agent_command("AV", ego_xy[0], ego_xy[1], sumo_angle, ego_speed)

        # 2. Allow time for the cosim Redis loop to process the command (5 ms poll)
        time.sleep(0.025)

        # 3. Fire the simulation tick
        requests.post(
            f"{self.service_url}/simulation_tick/{self._sim_id}",
            timeout=5.0,
        ).raise_for_status()

        # 4. Poll until simulation_time advances (confirms step completed and
        #    state was written to Redis)
        prev_time = self._sim_time
        deadline = time.time() + self.step_timeout
        new_state = None
        while time.time() < deadline:
            r = requests.get(
                f"{self.service_url}/simulation/{self._sim_id}/state",
                timeout=5.0,
            )
            if r.status_code == 200:
                state_data = r.json()
                if state_data["simulation_time"] > prev_time:
                    new_state = state_data
                    self._sim_time = state_data["simulation_time"]
                    break
            time.sleep(0.01)

        if new_state is None:
            raise TimeoutError(
                f"Simulation did not advance beyond t={prev_time:.2f}s "
                f"within {self.step_timeout}s"
            )

        self._last_state = new_state
        vehicles = new_state["agent_details"].get("vehicle", {})
        av_in_sim = "AV" in vehicles
        npc_states = [
            {
                "id":         k,
                "x":          v["x"],
                "y":          v["y"],
                "sumo_angle": v["sumo_angle"],
                "speed":      v["speed"],
                "length":     v.get("length", 4.5),
                "width":      v.get("width", 2.0),
            }
            for k, v in vehicles.items()
            if k != "AV"
        ]
        # VRUs: pedestrians and cyclists managed by TeraSim's NDE model
        vrus = new_state["agent_details"].get("vru", {})
        vru_states = [
            {
                "id":         k,
                "x":          v["x"],
                "y":          v["y"],
                "sumo_angle": v["sumo_angle"],
                "speed":      v["speed"],
                "length":     v.get("length", 0.5),
                "width":      v.get("width", 0.5),
                "type":       v.get("type", ""),
            }
            for k, v in vrus.items()
        ]
        return {
            "collision":  not av_in_sim,
            "av_in_sim":  av_in_sim,
            "npc_states": npc_states,
            "vru_states": vru_states,
            "sim_time":   new_state["simulation_time"],
        }

    def close(self) -> None:
        """Stop the running simulation episode (does not remove the container)."""
        if self._sim_id is not None:
            try:
                requests.post(
                    f"{self.service_url}/simulation_control/{self._sim_id}",
                    json={"command": "stop"},
                    timeout=5.0,
                )
            except Exception:
                pass
            self._sim_id = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send_agent_command(
        self,
        agent_id: str,
        x: float,
        y: float,
        sumo_angle: float,
        speed: float,
    ) -> None:
        """
        Enqueue a set_state command for the given agent.

        Internally translates to traci.vehicle.moveToXY(agent_id, "", 0,
        x, y, sumo_angle, keepRoute=2) inside the container.
        """
        payload = {
            "agent_id":    agent_id,
            "agent_type":  "vehicle",
            "command_type": "set_state",
            "data": {
                "position":   [x, y],
                "sumo_angle": sumo_angle,
                "speed":      speed,
            },
        }
        requests.post(
            f"{self.service_url}/simulation/{self._sim_id}/agent_command",
            json=payload,
            timeout=5.0,
        ).raise_for_status()

    def _poll_status(self, target_status: str, timeout: float) -> None:
        """Poll GET /simulation_status/{id} until status == target_status."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            r = requests.get(
                f"{self.service_url}/simulation_status/{self._sim_id}",
                timeout=5.0,
            )
            if r.status_code == 200 and r.json().get("status") == target_status:
                return
            time.sleep(0.05)
        raise TimeoutError(
            f"Simulation did not reach status '{target_status}' "
            f"(sim_id={self._sim_id}) within {timeout}s"
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "TeraSimBridge":
        return self

    def __exit__(self, *args) -> None:
        self.close()
