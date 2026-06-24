#!/bin/bash
# Clean A/B to isolate the temporal-consistency loss on metrics (no base checkpoint needed).
# Both arms: warm-start from epoch-60, CURRENT code (TF32 + staged-backward), identical recipe
# (SFT corpus, aug off, batch 256, lr 1e-5, 5 epochs, warmup 2) — the ONLY difference is the
# consistency coeff. Run sequentially (each uses all 8 GPUs), with GPU/rendezvous cleanup
# between so the second launch doesn't hang on stale NCCL state.
#   control   = coeff 0   -> no consistency (the fair baseline)
#   treatment = coeff 0.5 -> consistency
# Compare on wandb (project Diffusion-Planner-Temporal): valid_loss/ego = general performance;
# flicker (replan-consistency) is measured post-hoc via eval_flicker on the two checkpoints.
set -u
cd "$(dirname "$0")"

cleanup_between() {
  pkill -9 -f "torch.distributed.run" 2>/dev/null || true
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
    kill -9 "$pid" 2>/dev/null || true
  done
  rm -f /tmp/tmp_dist_init 2>/dev/null || true
  sleep 75   # let TIME_WAIT sockets + GPU contexts clear before relaunch
}

echo "########## A/B CONTROL (coeff 0) ##########"
bash train_temporal_run.sh ab_control_coeff0 0 5
echo "########## CONTROL DONE — cleanup ##########"
cleanup_between
echo "########## A/B TREATMENT (coeff 0.5) ##########"
bash train_temporal_run.sh ab_treatment_coeff0p5 0.5 5
echo "########## A/B COMPLETE ##########"
