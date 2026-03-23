#!/bin/bash
set -ux

MODEL_PATH=$(readlink -f $1)
EXP_NAME=$2
TRAIN_SET_LIST=$(readlink -f $3)
VALID_SET_LIST=$(readlink -f $4)

MODEL_DIR=$(dirname $MODEL_PATH)

cd $(dirname $0)

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export NCCL_NVLS_ENABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=INFO

rm -f /tmp/tmp_dist_init

python3 -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --standalone train_predictor.py \
--exp_name $EXP_NAME \
--train_set_list $TRAIN_SET_LIST \
--valid_set_list $VALID_SET_LIST \
--resume_model_path $MODEL_PATH \
--train_epochs 70 \
--save_utd 5 \
--use_wandb True \
--diffusion_model_type "x_start" \
--save_dir $MODEL_DIR \
2>&1 | tee logs/result_$(date +%Y%m%d_%H%M%S).txt

# Convert the trained PyTorch model to ONNX format
python3 ../ros_scripts/torch2onnx.py $MODEL_DIR
