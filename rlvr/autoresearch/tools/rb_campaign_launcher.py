"""Auto-launch RB experiment campaign in batches of 2.

Monitors running experiments, launches next pair when both finish.
Adds new experiments to the watchdog automatically.

Usage:
    python -m rlvr.autoresearch.tools.rb_campaign_launcher
"""

import os
import subprocess
import sys
import time

EXP_DIR = os.environ.get("RB_CAMPAIGN_EXP_DIR", "")
MODEL = os.environ.get("RB_CAMPAIGN_MODEL", "")
if not EXP_DIR or not MODEL:
    print("Set RB_CAMPAIGN_EXP_DIR and RB_CAMPAIGN_MODEL environment variables")
    sys.exit(1)
# Dataset filenames (override via env vars if your files are named differently).
VAL = os.environ.get("RB_CAMPAIGN_VAL", os.path.join(EXP_DIR, "val_50_gtclean.json"))
TRAIN_50 = os.environ.get("RB_CAMPAIGN_TRAIN_50", os.path.join(EXP_DIR, "train_50_gtclean.json"))
TRAIN_300 = os.environ.get("RB_CAMPAIGN_TRAIN_300", os.path.join(EXP_DIR, "train_300_clean.json"))

# Experiment queue: (name, config_suffix, train_file)
QUEUE = [
    # Batch 2
    ("rb_s2_50sc", "rb_rb_s2_50sc", TRAIN_50),
    ("rb_s5_50sc", "rb_rb_s5_50sc", TRAIN_50),
    # Batch 3
    ("rb_s4_50sc", "rb_rb_s4_50sc", TRAIN_50),
    ("rb_s7_50sc", "rb_rb_s7_50sc", TRAIN_50),
    # Batch 4
    ("rb_s6_50sc", "rb_rb_s6_50sc", TRAIN_50),
    ("rb_s8_50sc", "rb_rb_s8_50sc", TRAIN_50),
    # Batch 5
    ("rb_s9_50sc", "rb_rb_s9_50sc", TRAIN_50),
    ("rb_s10_50sc", "rb_rb_s10_50sc", TRAIN_50),
    # Batch 6
    ("rb_s11_50sc", "rb_rb_s11_50sc", TRAIN_50),
    ("thr_default_50sc", "rb_thr_default_50sc", TRAIN_50),
    # Batch 7
    ("thr_tight_50sc", "rb_thr_tight_50sc", TRAIN_50),
    ("thr_wide_50sc", "rb_thr_wide_50sc", TRAIN_50),
    # Batch 8
    ("thr_lane_50sc", "rb_thr_lane_50sc", TRAIN_50),
    ("rb_s3_gate_50sc", "rb_rb_s3_gate_50sc", TRAIN_50),
    # Batch 9
    ("rb_s3_nogate_50sc", "rb_rb_s3_nogate_50sc", TRAIN_50),
    ("lane_baseline_50sc", "rb_lane_baseline_50sc", TRAIN_50),
    # Batch 10
    ("rb_plus_lane_50sc", "rb_rb_plus_lane_50sc", TRAIN_50),
    ("rb_cont_only_50sc", "rb_rb_cont_only_50sc", TRAIN_50),
    # Batch 11
    ("rb_cont_heavy_50sc", "rb_rb_cont_heavy_50sc", TRAIN_50),
    ("rb_s1_300sc", "rb_rb_s1_300sc", TRAIN_300),
    # Batch 12
    ("rb_s3_300sc", "rb_rb_s3_300sc", TRAIN_300),
    ("rb_s5_300sc", "rb_rb_s5_300sc", TRAIN_300),
    # Batch 13
    ("rb_s4_300sc", "rb_rb_s4_300sc", TRAIN_300),
    ("rb_s7_300sc", "rb_rb_s7_300sc", TRAIN_300),
    # Batch 14
    ("rb_s3_nogate_300sc", "rb_rb_s3_nogate_300sc", TRAIN_300),
    ("lane_baseline_300sc", "rb_lane_baseline_300sc", TRAIN_300),
    # Batch 15
    ("rb_plus_lane_300sc", "rb_rb_plus_lane_300sc", TRAIN_300),
    ("thr_default_300sc", "rb_thr_default_300sc", TRAIN_300),
    # Batch 16
    ("thr_tight_300sc", "rb_thr_tight_300sc", TRAIN_300),
]


def is_running(name):
    """Check if experiment process is still running."""
    log_path = os.path.join(EXP_DIR, f"{name}.log")
    if not os.path.exists(log_path):
        return False
    # Check if process with this name is alive
    result = subprocess.run(
        ["pgrep", "-f", f"--name {name}"],
        capture_output=True, text=True
    )
    return result.returncode == 0


def launch(name, config_suffix, train_file):
    """Launch a single experiment."""
    config = os.path.join(EXP_DIR, "configs", f"{config_suffix}.json")
    log = os.path.join(EXP_DIR, f"{name}.log")
    with open(log, "ab") as log_file:
        subprocess.Popen(
            [sys.executable, "-m", "rlvr.autoresearch.run_experiment",
             "--config", config, "--name", name, "--model_path", MODEL,
             "--prob_scenes", train_file, "--normal_scenes", train_file,
             "--val_scenes", VAL, "--output_dir", EXP_DIR, "--skip_baseline"],
            stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True,
        )
    # Tell watchdog about new experiment
    with open("/tmp/watchdog_add.txt", "a") as f:
        f.write(f"{name}\n")
    print(f"  Launched: {name}")


def main():
    queue = list(QUEUE)
    running = []
    completed = 0

    print(f"RB Campaign: {len(queue)} experiments queued")
    print(f"Currently running: rb_s1_50sc, rb_s3_50sc")
    running = ["rb_s1_50sc", "rb_s3_50sc"]

    while queue or running:
        # Check which running experiments are done
        still_running = []
        for name in running:
            if is_running(name):
                still_running.append(name)
            else:
                completed += 1
                print(f"  Finished: {name} (total completed: {completed})")
        running = still_running

        # Launch next pair if GPU is free
        while len(running) < 2 and queue:
            name, config, train = queue.pop(0)
            launch(name, config, train)
            running.append(name)

        if running:
            time.sleep(30)

    print(f"\nAll {completed} experiments completed!")


if __name__ == "__main__":
    main()
