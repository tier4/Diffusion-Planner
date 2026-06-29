"""Detached subprocess runner + log tail for the control panel.

Each launch writes to ``~/.diffusion_planner_jobs/<ts>_<key>/run.log`` and records a
``job.json``. Jobs are started in their own session (``start_new_session=True``) so closing
or restarting the panel does NOT kill them — the panel is a viewer that re-attaches by
tailing the log file. There is intentionally no stop button for training/eval/mining jobs
(per the never-kill-experiments rule); only the interactive Scene Editor server is
restartable via :func:`stop`.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .workflows import Workflow, build_command

REPO_ROOT = Path(__file__).resolve().parents[1]
JOBS_DIR = Path.home() / ".diffusion_planner_jobs"
_ROS_SETUP = "/opt/ros/humble/setup.bash"


@dataclass
class Job:
    key: str
    title: str
    job_dir: str
    logfile: str
    pid: int
    cmd: list[str]
    env: str
    started_at: str
    proc_starttime: int | None = None
    server: bool = False
    port: int | None = None

    @classmethod
    def from_dir(cls, d: Path) -> "Job | None":
        jf = d / "job.json"
        if not jf.exists():
            return None
        try:
            with open(jf) as f:
                return cls(**json.load(f))
        except (json.JSONDecodeError, OSError, TypeError):
            return None


def _python_prefix(wf: Workflow, values: dict) -> list[str]:
    """argv prefix that selects the interpreter / launcher for this workflow."""
    if wf.torchrun:
        torchrun = Path(sys.executable).with_name("torchrun")
        port = str(values.get("master_port") or "29505")
        script = str((REPO_ROOT / wf.script_path).resolve())
        return [
            str(torchrun),
            "--nproc_per_node=1",
            "--standalone",
            f"--master_port={port}",
            script,
        ]
    if wf.script_path:
        return [sys.executable, str((REPO_ROOT / wf.script_path).resolve())]
    if wf.module:
        return [sys.executable, "-m", wf.module]
    raise ValueError(f"{wf.key}: neither module nor script_path set")


# lanelet2 needs its shared libs + python bindings on the path BEFORE the process starts
# (LD_LIBRARY_PATH is read by the dynamic linker at exec). Do NOT `source setup.bash` — that
# breaks lanelet under zsh (memory: reference_scene_editor_test_data). Set the vars explicitly.
_LANELET_LD = "/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib"
_LANELET_PY = (
    "/opt/ros/humble/lib/python3.10/site-packages:"
    "/opt/ros/humble/local/lib/python3.10/dist-packages"
)


def _wrap_env(wf: Workflow, inner: list[str]) -> list[str]:
    """Wrap the command for the ROS env when required; venv/lanelet run inner directly."""
    if wf.env == "ros":
        return ["bash", "-lc", f"source {_ROS_SETUP} && {shlex.join(inner)}"]
    if wf.env not in ("venv", "lanelet"):
        raise ValueError(f"{wf.key}: unknown env {wf.env!r}")
    return inner


def _prepend(env: dict, key: str, value: str) -> None:
    env[key] = f"{value}:{env[key]}" if env.get(key) else value


def _subprocess_env(wf: Workflow) -> dict:
    env = os.environ.copy()
    _prepend(env, "PYTHONPATH", f"{REPO_ROOT}:{REPO_ROOT / 'diffusion_planner'}")
    if wf.env == "lanelet":
        _prepend(env, "LD_LIBRARY_PATH", _LANELET_LD)
        _prepend(env, "PYTHONPATH", _LANELET_PY)
    env.setdefault("PYTHONUNBUFFERED", "1")  # so the log tail is live, not block-buffered
    return env


def _proc_starttime(pid: int) -> int | None:
    """Linux /proc starttime field, used to reject stale job PIDs after reuse."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            stat = f.read()
        return int(stat[stat.rfind(")") + 2 :].split()[19])
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None


def build_full_command(wf: Workflow, values: dict) -> list[str]:
    """Full argv (interpreter + tool + args), env-wrapped. Raises on missing required args."""
    tail = build_command(wf, values)
    inner = _python_prefix(wf, values) + tail
    return _wrap_env(wf, inner)


def launch(wf: Workflow, values: dict) -> Job:
    """Launch ``wf`` detached, streaming stdout+stderr to the job log."""
    cmd = build_full_command(wf, values)  # may raise ValueError (surfaced by the app)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    job_dir = JOBS_DIR / f"{ts}_{wf.key}"
    job_dir.mkdir(parents=True, exist_ok=True)
    logfile = job_dir / "run.log"

    with open(logfile, "w") as logf:
        logf.write(f"$ {shlex.join(cmd)}\n\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=_subprocess_env(wf),
            stdout=logf,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # detach into its own process group
        )

    port = int(values["port"]) if wf.server and values.get("port") else None
    job = Job(
        key=wf.key,
        title=wf.title,
        job_dir=str(job_dir),
        logfile=str(logfile),
        pid=proc.pid,
        cmd=cmd,
        env=wf.env,
        started_at=ts,
        proc_starttime=_proc_starttime(proc.pid),
        server=wf.server,
        port=port,
    )
    with open(job_dir / "job.json", "w") as f:
        json.dump(asdict(job), f, indent=2)
    return job


def is_alive(pid: int, proc_starttime: int | None = None) -> bool:
    """True iff ``pid`` is a live process.

    A detached child we launched becomes a *zombie* when it exits until something reaps it;
    ``os.kill(pid, 0)`` reports a zombie as alive, which would wedge the log stream on
    "running" forever and make Stop fail. So we read /proc state and treat 'Z' as dead
    (reaping it with waitpid if it's our own child).
    """
    if pid <= 0:
        return False
    try:
        with open(f"/proc/{pid}/stat") as f:
            stat = f.read()
        if proc_starttime is not None:
            try:
                current_start = int(stat[stat.rfind(")") + 2 :].split()[19])
            except (ValueError, IndexError):
                return False
            if current_start != proc_starttime:
                return False
        # Fields after the (comm) parenthesis; state is the first token there.
        state = stat[stat.rfind(")") + 2]
        if state == "Z":
            try:
                os.waitpid(pid, os.WNOHANG)  # reap if it's our child; ignore otherwise
            except (ChildProcessError, OSError):
                pass
            return False
        return True
    except FileNotFoundError:
        return False
    except OSError:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True


def stop(job: Job, kill_grace: float = 3.0) -> bool:
    """Stop a job: SIGTERM its process group, then SIGKILL if it doesn't exit in ``kill_grace``.

    User-initiated (a Stop button) — the never-kill-experiments rule is about *autonomous*
    behavior, not the user stopping their own job. Returns True if the process is gone.
    """
    if not is_alive(job.pid, job.proc_starttime):
        return True
    try:
        pgid = os.getpgid(job.pid)
    except ProcessLookupError:
        return True
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + kill_grace
    while time.monotonic() < deadline:
        if not is_alive(job.pid, job.proc_starttime):
            return True
        time.sleep(0.2)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return not is_alive(job.pid, job.proc_starttime)


def read_log(job: Job) -> str:
    try:
        with open(job.logfile, errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def stream(job: Job, poll: float = 1.0, grace: float = 2.0):
    """Yield the cumulative log text, refreshing until the process exits (+ a grace period)."""
    last = ""
    settle_deadline = None
    while True:
        text = read_log(job)
        if text != last:
            last = text
            yield text
        if is_alive(job.pid, job.proc_starttime):
            settle_deadline = None
        else:
            # Process gone: do one more read after a short grace to catch final flushes.
            if settle_deadline is None:
                settle_deadline = time.monotonic() + grace
            elif time.monotonic() >= settle_deadline:
                final = read_log(job)
                if final != last:
                    yield final
                return
        time.sleep(poll)


# --------------------------------------------------------------------------------------
# Per-epoch metric parsing (pure log-grep over run_experiment's own output).
# Line shape:  "  Evalepoch3-prob: 50 scenes, reward=+12.34, rb_cross=2/50, lane_dep=0/50,
#               ... sc_dist=[min=0.41 p5=0.55 mean=1.83], ... cl_all=[mean=-0.012 ...] ..."
# --------------------------------------------------------------------------------------
_EVAL_RE = re.compile(
    r"Evalepoch(?P<epoch>\d+)-(?P<split>prob|val):\s*(?P<n>\d+)\s+scenes,"
    r"\s*reward=(?P<reward>[+-]?[\d.]+),"
    r"\s*rb_cross=(?P<rb_cross>\d+)/\d+,"
    r"\s*lane_dep=(?P<lane_dep>\d+)/\d+"
)
_SC_MEAN_RE = re.compile(r"sc_dist=\[min=[\d.]+ p5=[\d.]+ mean=(?P<sc_mean>[\d.]+)\]")
_CL_MEAN_RE = re.compile(r"cl_all=\[mean=(?P<cl_mean>[+-]?[\d.]+)")


def parse_epoch_metrics(logfile: str) -> list[dict]:
    """Extract per-epoch eval rows from a run_experiment log. Token-light, no instrumentation."""
    rows: list[dict] = []
    try:
        with open(logfile, errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return rows
    for line in lines:
        m = _EVAL_RE.search(line)
        if not m:
            continue
        row = {
            "epoch": int(m["epoch"]),
            "split": m["split"],
            "n": int(m["n"]),
            "reward": float(m["reward"]),
            "rb_cross": int(m["rb_cross"]),
            "lane_dep": int(m["lane_dep"]),
        }
        sc = _SC_MEAN_RE.search(line)
        if sc:
            row["sc_mean"] = float(sc["sc_mean"])
        cl = _CL_MEAN_RE.search(line)
        if cl:
            row["cl_mean"] = float(cl["cl_mean"])
        rows.append(row)
    return rows


def list_jobs() -> list[Job]:
    """All recorded jobs, newest first."""
    if not JOBS_DIR.exists():
        return []
    jobs = [j for d in JOBS_DIR.iterdir() if d.is_dir() and (j := Job.from_dir(d))]
    jobs.sort(key=lambda j: j.started_at, reverse=True)
    return jobs


def latest_job(key: str | None = None) -> Job | None:
    for j in list_jobs():
        if key is None or j.key == key:
            return j
    return None
