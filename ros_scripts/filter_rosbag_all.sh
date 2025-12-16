#!/bin/bash
set -ux

TARGET_DIR=$(readlink -f $1)
# metadata.yamlを探すので、TARGET_DIRに与えるのは
# /mnt/nvme0/sakoda/nas_copy/tieriv_dataset/driving_dataset/bag/2024-07-18/10-05-28
# でも
# /mnt/nvme0/sakoda/nas_copy/tieriv_dataset/driving_dataset/bag/2024-07-18/
# でも
# /mnt/nvme0/sakoda/nas_copy/tieriv_dataset/driving_dataset/bag/
# でも良い

cd $(dirname $0)

metadata_yaml_list=$(find $TARGET_DIR -name "metadata.yaml" | sort)

for metadata_yaml in $metadata_yaml_list; do
    curr_target_dir=$(dirname $metadata_yaml)  # 例) /mnt/nvme0/sakoda/nas_copy/tieriv_dataset/driving_dataset/bag/2024-07-18/10-05-28
    time=$(basename $curr_target_dir)  # 10-05-28
    date=$(basename $(dirname $curr_target_dir))  # 2024-07-18

    bag_filtered_dir=$(readlink -f ${curr_target_dir}/../../../bag_filtered)
    if [ -d ${bag_filtered_dir}/${date}/${time} ]; then
        echo "Skipping already filtered bag: ${bag_filtered_dir}/${date}/${time}"
        continue
    fi
    mkdir -p ${bag_filtered_dir}/${date}

    ./filter_image_and_lidar.sh $curr_target_dir ${bag_filtered_dir}/${date}/${time}
done
