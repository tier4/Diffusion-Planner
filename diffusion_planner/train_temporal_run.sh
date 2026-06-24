#!/bin/bash
# Real training with the cross-frame temporal-consistency loss (reduces frame-to-frame
# flicker). Warm-starts from the epoch-60 model, fine-tunes on the SFT corpus with the
# consistency loss on paired consecutive frames. Tracked on wandb
# (advanced-technology-department / Diffusion-Planner-Temporal).
set -ux
cd "$(dirname "$0")"

EXP_NAME=${1:-temporal_consistency_v1}
COEFF=${2:-0.5}
EPOCHS=${3:-5}

DD=/mnt/storage_rdma/datasets/tier4/diffusion_planner/basic_dataset
TRAIN_SET_LIST=$DD/path_list_train_sft.json
VALID_SET_LIST=$DD/path_list_valid_sft_balanced.json
INIT_WEIGHTS=/mnt/nvme/Diffusion-Planner/checkpoints/with_takanawa_16days_weak_smoothing_sft_epoch60/best_model.pth

rm -f /tmp/tmp_dist_init  # clear any stale rendezvous (matches train_run.sh)

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NVLS_ENABLE=0 NCCL_P2P_DISABLE=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo
export PYTHONPATH=/mnt/nvme/Diffusion-Planner/diffusion_planner:/mnt/nvme/Diffusion-Planner
# public wandb; entity matches OnePlanner (code doesn't pass entity, so set via env)
export WANDB_ENTITY=advanced-technology-department

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
  --init_weights_path "${INIT_WEIGHTS}" \
  --use_wandb True \
  --wandb_project_name "Diffusion-Planner-Temporal" \
  --wandb_step_log_interval 25 \
  --diffusion_model_type "x_start" \
  --save_dir "${SAVE_PATH}" \
  --train_epochs "${EPOCHS}" \
  --warm_up_epoch 2 \
  --learning_rate 1e-5 \
  --batch_size 256 \
  --save_utd 2 \
  --use_data_augment False \
  --coeff_temporal_consistency "${COEFF}" \
  --tc_step_g 3 --tc_fixed_t 0.5 --tc_cons_scale 10.0 --tc_w_heading 1.0 \
  2>&1 | tee "${SAVE_PATH}/train_log.txt"
