#!/bin/bash
# SYNTHESIS run: explicit prior-conditioning INPUT + consistency LOSS.
# --prior_cond_train + coeff_temporal_consistency>0 routes to the synthesis branch in
# _temporal_consistency_term: frame_t standard planning (no-prior anchor); frame_{t+g}
# planning with the prior fed as cross-attention tokens (50% dropout) PLUS a consistency
# loss on the kept samples (incentive to USE the prior for stability). Warm-start ep60.
# lr 3e-5 (stable; the 1e-4 prior_cond run oscillated). LR-decay quirk fixed in train.py.
set -ux
cd "$(dirname "$0")"
EXP_NAME=${1:-prior_loss_train}
EPOCHS=${2:-5}
DD=/mnt/storage_rdma/datasets/tier4/diffusion_planner/basic_dataset
INIT=/mnt/nvme/Diffusion-Planner/checkpoints/with_takanawa_16days_weak_smoothing_sft_epoch60/best_model.pth

rm -f /tmp/tmp_dist_init
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NVLS_ENABLE=0 NCCL_P2P_DISABLE=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo
export PYTHONPATH=/mnt/nvme/Diffusion-Planner/diffusion_planner:/mnt/nvme/Diffusion-Planner
export WANDB_ENTITY=advanced-technology-department
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SAVE_DIR=/mnt/nvme/training_result
TIME=$(date +%Y%m%d-%H%M%S)
SAVE_PATH="${SAVE_DIR}/${TIME}_${EXP_NAME}"
mkdir -p "${SAVE_PATH}"
git -C /mnt/nvme/Diffusion-Planner diff > "${SAVE_PATH}/git_diff.txt" 2>/dev/null || true

/mnt/nvme/OnePlanner/.venv/bin/python -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --standalone \
  train_predictor.py \
  --exp_name "${EXP_NAME}" \
  --train_set_list "${DD}/path_list_train_sft.json" \
  --valid_set_list "${DD}/path_list_valid_sft_balanced.json" \
  --init_weights_path "${INIT}" \
  --use_wandb True --wandb_project_name "Diffusion-Planner-Temporal" --wandb_step_log_interval 50 \
  --diffusion_model_type "x_start" --save_dir "${SAVE_PATH}" \
  --train_epochs "${EPOCHS}" --warm_up_epoch 1 --learning_rate 3e-5 --batch_size 256 --save_utd 2 \
  --use_data_augment False \
  --coeff_temporal_consistency 0.5 --prior_cond_train True --tc_step_g 3 --tc_fixed_t 0.5 --tc_cons_scale 10.0 \
  2>&1 | tee "${SAVE_PATH}/train_log.txt"
