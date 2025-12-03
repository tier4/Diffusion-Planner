#!/bin/bash
set -eux

cd $(dirname $0)

data_root=/mnt/nvme3/sakoda/nas_copy/private_workspace/diffusion_planner/sft_ver1

python3 ./parse_rosbag_for_directory.py \
    /mnt/nvme2/sakoda/nas_copy/tieriv_dataset/driving_dataset/bag_filtered/2025-10-22 \
    /mnt/nvme2/sakoda/nas_copy/tieriv_dataset/driving_dataset/bag_filtered/2025-10-29 \
    /mnt/nvme2/sakoda/nas_copy/tieriv_dataset/driving_dataset/bag_filtered/2025-11-12 \
    /mnt/nvme2/sakoda/nas_copy/tieriv_dataset/driving_dataset/bag_filtered/2025-11-19 \
    --save_root $data_root \
    --step 1 \
    --limit -1 \
    --min_frames 1800 \
    --search_nearest_route 1

python3 ../diffusion_planner/util_scripts/create_train_set_path.py \
    $data_root/2025-10-22 \
    $data_root/2025-10-29 \
    $data_root/2025-11-12 \
    $data_root/2025-11-19 \
    --save_path $data_root/path_list_train.json
