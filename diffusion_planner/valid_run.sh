#!/bin/bash
set -ux
cd $(dirname $0)

# Number of GPUs to use. Auto-detects all visible GPUs; set NUM_GPUS=1 to force single-GPU.
NUM_GPUS=$(nvidia-smi -L | wc -l)

###################################
# User Configuration Section
###################################

# Set training data path
MODEL_DIR=${1}
VALID_SET_LIST_PATH=${2}
MODEL_PATH="$MODEL_DIR/best_model.pth"
ARGS_JSON_PATH="$MODEL_DIR/args.json"
SAVE_DIR=$MODEL_DIR/validation_result/predictions

rm -f /tmp/tmp_dist_init

python3 -m torch.distributed.run --nnodes 1 --nproc-per-node $NUM_GPUS --standalone valid_predictor.py \
--valid_set_list  $VALID_SET_LIST_PATH \
--resume_model_path $MODEL_PATH \
--args_json_path $ARGS_JSON_PATH \
--save_predictions_dir $SAVE_DIR \

python3 util_scripts/visualize_prediction.py \
  --predictions_dir $SAVE_DIR \
  --valid_data_list $VALID_SET_LIST_PATH

~/misc/ffmpeg_lib/process_subdir.sh $SAVE_DIR/../visualization
