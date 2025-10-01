#!/bin/bash
set -eux

cd $(dirname $0)

# set +eux
# source ~/pilot-auto.xx1/install/setup.bash
# set -eux

data_root_real=/mnt/nvme2/sakoda/nas_copy/private_workspace/diffusion_planner/preprocessed_ver51_realdata_cpp_num140
data_root_psim=/mnt/nvme2/sakoda/nas_copy/private_workspace/diffusion_planner/preprocessed_ver51_psimdata_cpp_num140

python3 ./parse_rosbag_for_directory.py \
    /mnt/nvme1/sakoda/nas_copy/tieriv_dataset/driving_dataset/bag_filtered/ \
    --save_root $data_root_real \
    --step 1 \
    --limit -1 \
    --min_frames 1800 \
    --search_nearest_route 1

python3 ../diffusion_planner/util_scripts/create_train_set_path.py \
    $data_root_real/2024-07-18 \
    $data_root_real/2024-12-11 \
    $data_root_real/2025-01-24 \
    $data_root_real/2025-02-04 \
    $data_root_real/2025-03-25 \
    $data_root_real/2025-04-16 \
    $data_root_real/2025-04-30 \
    $data_root_real/2025-05-07 \
    $data_root_real/2025-05-15 \
    $data_root_real/2025-05-22 \
    $data_root_real/2025-05-29 \
    $data_root_real/2025-06-09 \
    $data_root_real/2025-07-02 \
    $data_root_real/2025-07-15 \
    $data_root_real/2025-07-23 \
    $data_root_real/2025-07-29 \
    $data_root_real/2025-08-08 \
    $data_root_real/2025-08-13 \
    $data_root_real/2025-08-20 \
    --save_path $data_root_real/path_list_train.json

python3 ../diffusion_planner/util_scripts/create_train_set_path.py \
    $data_root_real/2025-06-12 \
    $data_root_real/2025-06-16 \
    --save_path $data_root_real/path_list_valid.json

# psimdata
python3 ./parse_rosbag_for_directory.py \
    /mnt/nvme0/sakoda/nas_copy/psim_dataset/simulation_session_20250806_190003/bag \
    --save_root $data_root_psim \
    --step 1 \
    --limit -1 \
    --min_frames 0 \
    --search_nearest_route 0

python3 ../diffusion_planner/util_scripts/create_train_set_path.py \
    $data_root_psim \
    --save_path $data_root_psim/path_list.json

python3 ../diffusion_planner/util_scripts/concat_data_list_jsons.py \
    $data_root_psim/path_list.json \
    $data_root_real/path_list_train.json \
    --save_path $data_root_real/path_list_train_with_psim_data.json
