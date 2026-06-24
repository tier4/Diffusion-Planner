#!/bin/bash
# FAIR head-to-head vs the "base" run (wandb iou8430w, entity advanced-technology-department):
# an EXACT replica of the base recipe — full corpus (path_list_train.json), 80 epochs from
# scratch, batch 512, lr 1e-4, warm-up 5, aug ON (quintic, ego_past_noise_std 0.1, smoothing),
# x_start, 320 neighbours, road_border 1 — with the SINGLE addition of the cross-frame
# temporal-consistency loss. The base run is the matched control; the same metrics
# (valid_loss/ego, train_loss/ego_planning_loss, ...) are logged for a direct comparison.
#
# Everything not passed below uses the argparse defaults, which already equal the base config:
#   batch_size 512, learning_rate 1e-4, warm_up_epoch 5, seed 3407, use_data_augment True,
#   augment_type quintic, ego_past_noise_std 0.1, use_smoothing_future_trajectory True,
#   predicted_neighbor_num 320, coeff_road_border_loss 1, coeff_neighbor_collision_loss 0,
#   hidden_dim 256, decoder_depth 3, encoder depths 6/6, num_heads 8.
set -ux
cd "$(dirname "$0")"

EXP_NAME=${1:-base_match_temporal_consistency}
COEFF=${2:-0.5}
EPOCHS=${3:-80}

DD=/mnt/nvme/dataset/basic_dataset
TRAIN_SET_LIST=$DD/path_list_train.json          # full corpus (5.47M) — same as base
VALID_SET_LIST=$DD/path_list_valid_sft.json      # same valid set as base

rm -f /tmp/tmp_dist_init  # clear any stale rendezvous

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NVLS_ENABLE=0 NCCL_P2P_DISABLE=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo
export PYTHONPATH=/mnt/nvme/Diffusion-Planner/diffusion_planner:/mnt/nvme/Diffusion-Planner
export WANDB_ENTITY=advanced-technology-department
# memory safety for the staged-backward consistency path (steady ~79/80 GB at batch 64/GPU):
# reduce allocator fragmentation so a long (multi-day) run doesn't creep into an OOM.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SAVE_DIR=/mnt/nvme/training_result
TIME=$(date +%Y%m%d-%H%M%S)
SAVE_PATH="${SAVE_DIR}/${TIME}_${EXP_NAME}"
mkdir -p "${SAVE_PATH}"
git -C /mnt/nvme/Diffusion-Planner show -s > "${SAVE_PATH}/git_show.txt" 2>/dev/null || true
git -C /mnt/nvme/Diffusion-Planner diff > "${SAVE_PATH}/git_diff.txt" 2>/dev/null || true

/mnt/nvme/OnePlanner/.venv/bin/python -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --standalone \
  train_predictor.py \
  --exp_name "${EXP_NAME}" \
  --train_set_list "${TRAIN_SET_LIST}" \
  --valid_set_list "${VALID_SET_LIST}" \
  --use_wandb True \
  --wandb_project_name "Diffusion-Planner" \
  --wandb_step_log_interval 50 \
  --diffusion_model_type "x_start" \
  --save_dir "${SAVE_PATH}" \
  --train_epochs "${EPOCHS}" \
  --save_utd 10 \
  --coeff_temporal_consistency "${COEFF}" \
  --tc_step_g 3 --tc_fixed_t 0.5 --tc_cons_scale 10.0 --tc_w_heading 1.0 \
  2>&1 | tee "${SAVE_PATH}/train_log.txt"
