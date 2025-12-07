#!/bin/bash
set -eux

cd $(dirname $0)

data_root_real=/mnt/nvme3/sakoda/nas_copy/private_workspace/diffusion_planner/preprocessed_ver57_realdata

python3 ../diffusion_planner/util_scripts/create_train_set_path.py \
    $data_root_real/2025-10-29 \
    $data_root_real/2025-11-12 \
    $data_root_real/2025-11-19 \
    --save_path $data_root_real/path_list_train_sft.json

python3 ../diffusion_planner/util_scripts/create_train_set_path.py \
    $data_root_real/2025-10-29 \
    $data_root_real/2025-11-12 \
    $data_root_real/2025-11-19 \
    --save_path $data_root_real/path_list_train_sft.json

python3 ../diffusion_planner/util_scripts/filter_json.py \
    $data_root_real/path_list_train_sft.json \
    --time_filter_jsons \
        $data_root_real/2025-10-29.json \
        $data_root_real/2025-11-12.json \
        $data_root_real/2025-11-19.json \
