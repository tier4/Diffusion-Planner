#!/bin/bash
set -ux
exp_name=${1}
TRAIN_SET_LIST=${2}
VALID_SET_LIST=${3}
SFT_SET_LIST=${4}

# to convert full paths
TRAIN_SET_LIST=$(readlink -f $TRAIN_SET_LIST)
VALID_SET_LIST=$(readlink -f $VALID_SET_LIST)
SFT_SET_LIST=$(readlink -f $SFT_SET_LIST)

cd $(dirname $0)

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export NCCL_NVLS_ENABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=INFO

rm -f /tmp/tmp_dist_init

SAVE_DIR="/mnt/nvme/training_result"

# pretraining
python3 -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --standalone train_predictor.py \
--exp_name ${exp_name} \
--train_set_list $TRAIN_SET_LIST \
--valid_set_list $VALID_SET_LIST \
--use_wandb True \
--diffusion_model_type "x_start" \
--save_dir $SAVE_DIR \
--train_epochs 50 \
--save_utd 10 \
2>&1 | tee logs/result_$(date +%Y%m%d_%H%M%S).txt

save_dir_name=$(ls $SAVE_DIR | tail -n 1)

# sft
python3 -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --standalone train_predictor.py \
--exp_name ${exp_name}_sft \
--train_set_list $SFT_SET_LIST \
--valid_set_list $VALID_SET_LIST \
--use_wandb True \
--diffusion_model_type "x_start" \
--save_dir $SAVE_DIR \
--resume_model_path $SAVE_DIR/$save_dir_name/epoch0040/best_model.pth \
--train_epochs 50 \
--save_utd 5 \
2>&1 | tee logs/result_$(date +%Y%m%d_%H%M%S).txt

# Convert the trained PyTorch model to ONNX format
python3 ../ros_scripts/torch2onnx.py $SAVE_DIR/$save_dir_name
