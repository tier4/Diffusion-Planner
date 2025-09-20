#!/bin/bash
set -x
cd $(dirname $0)
export CUDA_VISIBLE_DEVICES=0

###################################
# User Configuration Section
###################################

# Set training data path
MODEL_DIR=${1}
VALID_SET_LIST_PATH="/mnt/nvme0/sakoda/nas_copy/private_workspace/diffusion_planner/preprocessed_ver45_realdata_cpp/path_list_valid.json"
MODEL_PATH="$MODEL_DIR/best_model.pth"
ARGS_JSON_PATH="$MODEL_DIR/args.json"
SAVE_DIR=$MODEL_DIR/$DIR_NAME/predictions

rm -f /tmp/tmp_dist_init

python3 -m torch.distributed.run --nnodes 1 --nproc-per-node 1 --standalone valid_predictor.py \
--valid_set_list  $VALID_SET_LIST_PATH \
--resume_model_path $MODEL_PATH \
--args_json_path $ARGS_JSON_PATH \
--save_predictions_dir $SAVE_DIR \

python3 util_scripts/visualize_prediction.py \
  --predictions_dir $SAVE_DIR \
  --args_json $ARGS_JSON_PATH \
  --valid_data_list $VALID_SET_LIST_PATH
