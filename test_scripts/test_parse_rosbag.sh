#!/bin/bash
set -eux

cd $(dirname $0)/..
converter_type=${1}  # python or cpp

result_dir=/mnt/nvme0/sakoda/test/$(date +%Y%m%d_%H%M%S)_test_${converter_type}

rm -rf ${result_dir}

SECONDS=0

time=10-19-35
# time=10-24-57
bag=/mnt/nvme1/sakoda/nas_copy/tieriv_dataset/driving_dataset/bag_filtered/2025-06-12/${time}
map=/mnt/nvme1/sakoda/nas_copy/tieriv_dataset/driving_dataset/map/2025-06-12/${time}/lanelet2_map.osm

npz_dir=${result_dir}/${time}
mkdir -p ${npz_dir}

if [ "$converter_type" = "python" ]; then
    python3 ./ros_scripts/parse_rosbag.py \
        ${bag} \
        ${map} \
        ${npz_dir} \
        --limit 30000000 \
        --min_frames 0 \
        2>&1 | tee $result_dir/result_$(date +%Y%m%d_%H%M%S).txt
elif [ "$converter_type" = "cpp" ]; then
    python3 ./ros_scripts/parse_rosbag_by_cpp.py \
        $HOME/autoware/build/autoware_diffusion_planner/data_converter \
        ${bag} \
        ${map} \
        ${npz_dir} \
        --limit 30000000 \
        --min_frames 0 \
        --convert_yellow 0 \
        --ego_wheel_base 2.75 \
        --ego_length 4.34 \
        --ego_width 1.70 \
        2>&1 | tee $result_dir/result_$(date +%Y%m%d_%H%M%S).txt
fi

echo $SECONDS

# python3 ./ros_scripts/compare_npz.py ${result_dir}

python3 ./diffusion_planner/util_scripts/create_train_set_path.py ${npz_dir}

python3 ./diffusion_planner/util_scripts/visualize_input.py \
    ${result_dir}/path_list.json \
    ${result_dir}/visualize_result

~/misc/ffmpeg_lib/make_mp4_from_unsequential_png.sh ${result_dir}/visualize_result
