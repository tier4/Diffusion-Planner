#!/bin/bash
set -eux

target_dir=$(readlink -f $1)

cd $(dirname $0)/..

result_dir=/mnt/nvme0/sakoda/test/$(date +%Y%m%d_%H%M%S)_visualize

rm -rf ${result_dir}

python3 ./diffusion_planner/util_scripts/create_train_set_path.py ${target_dir} \
    --save_path ${result_dir}/path_list.json

python3 ./diffusion_planner/util_scripts/visualize_input.py ${result_dir}/path_list.json \
    /mnt/nvme0/sakoda/training_result/20250723-221429_with_random_psim_data/args.json \
    --save_path ${result_dir}/visualize_result

~/misc/ffmpeg_lib/make_mp4_from_unsequential_png.sh ${result_dir}/visualize_result
