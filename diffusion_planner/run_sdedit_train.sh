#!/bin/bash
# Step 1: SDEdit-aware post-training. Warm-start epoch-60, SFT corpus, teach the model to
# denoise a prior-init (previous frame's propagated plan) toward GT — so SDEdit-init at
# inference keeps the prior where the scene is unchanged and corrects it where it changed.
# --sdedit_train True turns on the prior-init path; coeff_temporal_consistency>0 builds the
# paired dataset (its consistency loss is NOT used in sdedit mode).
set -ux
cd "$(dirname "$0")"
EXP_NAME=${1:-sdedit_train}
EPOCHS=${2:-4}

DD=/mnt/storage_rdma/datasets/tier4/diffusion_planner/basic_dataset
INIT=/mnt/nvme/Diffusion-Planner/checkpoints/with_takanawa_16days_weak_smoothing_sft_epoch60/best_model.pth

rm -f /tmp/tmp_dist_init
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NVLS_ENABLE=0 NCCL_P2P_DISABLE=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo
export PYTHONPATH=/mnt/nvme/Diffusion-Planner/diffusion_planner:/mnt/nvme/Diffusion-Planner
export WANDB_ENTITY=advanced-technology-department
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SAVE=/mnt/nvme/training_result/$(date +%Y%m%d-%H%M%S)_${EXP_NAME}
mkdir -p "$SAVE"

/mnt/nvme/OnePlanner/.venv/bin/python -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --standalone \
  train_predictor.py \
  --exp_name "${EXP_NAME}" \
  --train_set_list "$DD/path_list_train_sft.json" \
  --valid_set_list "$DD/path_list_valid_sft_balanced.json" \
  --init_weights_path "$INIT" \
  --use_wandb True --wandb_project_name "Diffusion-Planner-Temporal" --wandb_step_log_interval 50 \
  --diffusion_model_type "x_start" --save_dir "$SAVE" \
  --train_epochs "${EPOCHS}" --warm_up_epoch 1 --save_utd 2 \
  --batch_size 256 --learning_rate 1e-4 --use_data_augment False \
  --coeff_temporal_consistency 0.5 --sdedit_train True \
  --tc_step_g 3 --tc_fixed_t 0.5 --tc_cons_scale 10.0 --tc_w_heading 1.0 \
  2>&1 | tee "$SAVE/train_log.txt"
