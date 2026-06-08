#!/bin/bash
# GRPO fine-tuning, run alongside train_run.sh (which does pretrain + SFT).
#
# Usage:
#   ./grpo_run.sh <RESUME_MODEL_PATH> <exp_name> <TRAIN_SET_LIST> <VALID_SET_LIST>
#
# Adversarial neighbors are generated synthetically at train time (no DB build step).
set -ux
RESUME_MODEL_PATH=$(readlink -f ${1})
exp_name=${2}
TRAIN_SET_LIST=$(readlink -f ${3})
VALID_SET_LIST=$(readlink -f ${4})

cd $(dirname $0)

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=INFO
rm -f /tmp/tmp_dist_init

SAVE_DIR="/mnt/nvme/training_result"
TIME=$(date +%Y%m%d-%H%M%S)
SAVE_PATH="${SAVE_DIR}/${TIME}_${exp_name}_grpo"
mkdir -p ${SAVE_PATH}

git show -s > ${SAVE_PATH}/git_show.txt
git diff > ${SAVE_PATH}/git_diff.txt

# GRPO fine-tuning from a pretrained/SFT checkpoint. Adversarial neighbor augmentation is
# synthetic (utils/synthetic_neighbors.py) -- no DB to build; tune it via the --*_prob /
# --neighbor_inject_* / --collider_keep_clear_radius flags below if needed.
# (optional) sanity-check the augmentation + sample diversity + reward:
#   python3 visualize_grpo_samples.py --resume_model_path ${RESUME_MODEL_PATH} --data_list ${TRAIN_SET_LIST} --output_path ${SAVE_PATH}/grpo_samples.png
#
python3 -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --standalone train_grpo_predictor.py \
  --exp_name ${exp_name}_grpo \
  --train_set_list ${TRAIN_SET_LIST} \
  --valid_set_list ${VALID_SET_LIST} \
  --resume_model_path ${RESUME_MODEL_PATH} \
  --diffusion_model_type "x_start" \
  --save_dir ${SAVE_PATH} \
  --batch_size 64 \
  --num_generations 8 \
  --sft_prob 0.5 \
  --learning_rate 1e-5 \
  --train_epochs 50 \
  --save_utd 1 \
  --use_wandb True \
  2>&1 | tee ${SAVE_PATH}/grpo_log.txt
