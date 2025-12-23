#!/bin/bash
set -ux
cd $(dirname $0)

exp_name=${1}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export NCCL_NVLS_ENABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=INFO

rm -f /tmp/tmp_dist_init

SAVE_DIR="/mnt/nvme0/sakoda/training_result"

TRAIN_SET_LIST="/mnt/nvme2/sakoda/nas_copy/private_workspace/dataset_ver58/path_list_train_with_psim_data.json"
VALID_SET_LIST="/mnt/nvme2/sakoda/nas_copy/private_workspace/dataset_ver58/path_list_valid.json"
SFT_SET_LIST="/mnt/nvme2/sakoda/nas_copy/private_workspace/dataset_ver58/path_list_train_sft_in_time_range.json"

# pretraining
python3 -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --standalone train_predictor.py \
--exp_name ${exp_name} \
--train_set_list $TRAIN_SET_LIST \
--valid_set_list $VALID_SET_LIST \
--use_wandb True \
--diffusion_model_type "x_start" \
--save_dir $SAVE_DIR \
--train_epochs 100 \
--save_utd 10 \
2>&1 | tee logs/result_$(date +%Y%m%d_%H%M%S).txt

save_dir_name=$(ls $SAVE_DIR | tail -n 1)

# sft
# best_model/best_model_info.jsonから "epoch"を読み取る
init_epoch=$(cat $SAVE_DIR/$save_dir_name/best_model/best_model_info.json | grep '"epoch"' | head -n 1 | awk -F ': ' '{print $2}' | sed 's/,//')
# 10エポック分追加で学習
last_epoch=$((init_epoch + 10))

python3 -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --standalone train_predictor.py \
--exp_name ${exp_name}_sft \
--train_set_list $SFT_SET_LIST \
--valid_set_list $VALID_SET_LIST \
--use_wandb True \
--diffusion_model_type "x_start" \
--save_dir $SAVE_DIR \
--resume_model_path $SAVE_DIR/$save_dir_name/best_model \
--train_epochs $last_epoch \
--save_utd 1 \
2>&1 | tee logs/result_$(date +%Y%m%d_%H%M%S).txt

# Convert the trained PyTorch model to ONNX format
python3 ../ros_scripts/torch2onnx.py $SAVE_DIR/$save_dir_name
