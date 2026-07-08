#!/bin/bash
set -eux

target_dir=$(readlink -f $1)

cd $(dirname $0)/..

result_dir=/mnt/nvme0/sakoda/test/$(date +%Y%m%d_%H%M%S)_visualize

rm -rf ${result_dir}

python3 ./diffusion_planner/util_scripts/create_train_set_path.py ${target_dir} \
    --save_path ${result_dir}/path_list.json

python3 ./diffusion_planner/util_scripts/visualize_input.py ${result_dir}/path_list.json \
    --save_path ${result_dir}/visualize_result

python3 ros_scripts/make_mp4.py ${result_dir}/visualize_result
