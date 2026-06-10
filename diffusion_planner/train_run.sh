#!/bin/bash
set -ux
exp_name=${1}
TRAIN_SET_LIST=${2}
VALID_SET_LIST=${3}
MODEL_PATH=${4:-}  # optional: resume from this .pth if given

# to convert full paths
TRAIN_SET_LIST=$(readlink -f $TRAIN_SET_LIST)
VALID_SET_LIST=$(readlink -f $VALID_SET_LIST)

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
SAVE_PATH="${SAVE_DIR}/${TIME}_${exp_name}"

mkdir -p ${SAVE_PATH}

# Save git info
git show -s > ${SAVE_PATH}/git_show.txt
git diff > ${SAVE_PATH}/git_diff.txt

# Build optional resume argument
RESUME_ARGS=()
if [ -n "$MODEL_PATH" ]; then
    MODEL_PATH=$(readlink -f $MODEL_PATH)
    RESUME_ARGS=(--resume_model_path $MODEL_PATH)
fi

python3 -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --standalone train_predictor.py \
--exp_name ${exp_name} \
--train_set_list $TRAIN_SET_LIST \
--valid_set_list $VALID_SET_LIST \
--use_wandb True \
--diffusion_model_type "x_start" \
--save_dir ${SAVE_PATH} \
--train_epochs 80 \
--save_utd 10 \
"${RESUME_ARGS[@]}" \
2>&1 | tee ${SAVE_PATH}/train_log.txt

# Convert the trained PyTorch model to ONNX format
python3 ../ros_scripts/torch2onnx.py ${SAVE_PATH}
