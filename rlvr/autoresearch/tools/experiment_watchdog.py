#!/usr/bin/env python3
"""Experiment watchdog — tracks experiments, outputs only new events, never dies.

Usage:
    # Start with initial experiments:
    nohup python -m rlvr.autoresearch.tools.experiment_watchdog \
        --exp_dir /path/to/experiments --names exp1 exp2 &

    # Add more experiments later (watchdog picks them up):
    echo "exp3" >> /tmp/watchdog_add.txt

    # Read status:
    cat /tmp/experiment_status.log

Tracks named experiments via run_experiment.py logs. Outputs only NEW eval lines
(matching "Eval [epochN-val]" / "Eval [epochN-prob]" format) as they appear.
Alerts on stopped>2, rb_cross>10. Detects process death.
Never exits — sleeps and watches for new experiments via add-file.
"""

import argparse
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime


def is_experiment_running(name):
    """Check if an experiment process is still alive by name."""
    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.split("\n"):
            if name in line and "watchdog" not in line and "grep" not in line:
                if any(k in line for k in ["run_experiment", "train_grpo", "grpo_sft"]):
                    return True
        return False
    except Exception:
        return False


_EVAL_RE = re.compile(r"Eval \[epoch\d+-(val|prob)\]")


def get_new_eval_lines(log_path, byte_offset):
    """Read new eval lines from a log file starting from byte_offset.

    Returns (new_lines, new_byte_offset).
    Uses seek() to avoid re-reading the entire file each interval.
    """
    new_lines = []
    try:
        with open(log_path) as f:
            f.seek(byte_offset)
            for line in f:
                if _EVAL_RE.search(line):
                    new_lines.append(line.strip())
            new_offset = f.tell()
    except Exception:
        return [], byte_offset
    return new_lines, new_offset


def check_alerts(name, eval_line):
    """Check a single eval line for kill conditions."""
    alerts = []
    stopped = re.search(r"stopped=(\d+)", eval_line)
    rb_cross = re.search(r"rb_cross=(\d+)/", eval_line)
    if stopped and int(stopped.group(1)) > 2:
        alerts.append(f"[ALERT] {name}: stopped={stopped.group(1)} (>2)")
    if rb_cross and int(rb_cross.group(1)) > 10:
        alerts.append(f"[ALERT] {name}: rb_cross={rb_cross.group(1)} (>10)")
    return alerts


def write_status(sf, message):
    """Write a timestamped line."""
    now = datetime.now().strftime("%H:%M:%S")
    sf.write(f"[{now}] {message}\n")
    sf.flush()


def snapshot_log(exp_dir, name):
    """Seek to end of existing log so we only report new lines."""
    log_path = os.path.join(exp_dir, f"{name}.log")
    byte_offset = 0
    if os.path.exists(log_path):
        try:
            byte_offset = os.path.getsize(log_path)
        except Exception:
            pass
    return {"log": log_path, "byte_offset": byte_offset, "alive": True, "finished_reported": False}


def check_add_file(add_file, exp_dir, tracked, status_file):
    """Check if new experiment names were written to the add-file.

    Atomically rotates the file to avoid race conditions with concurrent writers.
    """
    if not os.path.exists(add_file):
        return
    try:
        tmp_path = add_file + ".reading"
        os.rename(add_file, tmp_path)
        with open(tmp_path) as f:
            names = [line.strip() for line in f if line.strip()]
        os.remove(tmp_path)
        for name in names:
            if name not in tracked:
                tracked[name] = snapshot_log(exp_dir, name)
                with open(status_file, "a") as sf:
                    write_status(sf, f"  Now tracking: {name}")
                print(f"Added experiment: {name}")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Experiment watchdog")
    parser.add_argument(
        "--exp_dir", required=True, help="Experiment directory containing .log files"
    )
    parser.add_argument("--names", nargs="*", default=[], help="Initial experiment names to track")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    parser.add_argument("--status_file", default="/tmp/experiment_status.log", help="Output file")
    parser.add_argument(
        "--add_file", default="/tmp/watchdog_add.txt", help="File to write new experiment names to"
    )
    args = parser.parse_args()

    tracked = {}
    for name in args.names:
        tracked[name] = snapshot_log(args.exp_dir, name)

    with open(args.status_file, "w") as sf:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sf.write(f"=== Watchdog started [{now}] ===\n")
        if tracked:
            sf.write(f"Tracking: {', '.join(args.names)}\n")
        sf.write(f"Add experiments: echo <name> >> {args.add_file}\n\n")

    print(f"Watchdog started. Tracking {len(tracked)} experiments, checking every {args.interval}s")
    print(f"Status file: {args.status_file}")
    print(f"Add file: {args.add_file}")

    while True:
        check_add_file(args.add_file, args.exp_dir, tracked, args.status_file)

        new_output = []

        for name, state in tracked.items():
            if state["finished_reported"]:
                continue

            # Check for new eval lines (seek-based, O(new data) not O(file size))
            if os.path.exists(state["log"]):
                new_lines, new_offset = get_new_eval_lines(state["log"], state["byte_offset"])
                state["byte_offset"] = new_offset

                for line in new_lines:
                    new_output.append(f"  [{name}] {line}")
                    for a in check_alerts(name, line):
                        new_output.append(f"  {a}")

            # Check if process died (only if it was alive before)
            if state["alive"]:
                running = is_experiment_running(name)
                if not running:
                    state["alive"] = False
                    state["finished_reported"] = True
                    new_output.append(f"  [{name}] FINISHED (process no longer running)")

        if new_output:
            with open(args.status_file, "a") as sf:
                for line in new_output:
                    write_status(sf, line)

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
