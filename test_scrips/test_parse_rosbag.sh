#!/bin/bash
set -eux

cd $(dirname $0)/..

result_dir=/mnt/nvme0/sakoda/test/$(date +%Y%m%d_%H%M%S)_test_parse_rosbag

rm -rf ${result_dir}
mkdir -p ${result_dir}/tmp

SECONDS=0

python3 ./ros_scripts/parse_rosbag_by_cpp.py \
    /home/ubuntu/autoware/build/autoware_diffusion_planner/data_converter \
    /mnt/nvme2/sakoda/nas_copy/tieriv_dataset/driving_dataset/bag_filtered/2025-06-12/10-19-35 \
    /mnt/nvme2/sakoda/nas_copy/tieriv_dataset/driving_dataset/map/2025-06-12/10-19-35/lanelet2_map.osm \
    ${result_dir}/tmp \
    --limit 30000000 \
    --min_frames 0 2>&1 | tee $result_dir/result_$(date +%Y%m%d_%H%M%S).txt

echo $SECONDS

python3 ./diffusion_planner/util_scripts/create_train_set_path.py ${result_dir}/tmp

python3 ./diffusion_planner/util_scripts/visualize_input.py ${result_dir}/path_list.json \
    /mnt/nvme0/sakoda/training_result/20250723-221429_with_random_psim_data/args.json \
    --save_path ${result_dir}/visualize_result

~/misc/ffmpeg_lib/make_mp4_from_unsequential_png.sh ${result_dir}/visualize_result
