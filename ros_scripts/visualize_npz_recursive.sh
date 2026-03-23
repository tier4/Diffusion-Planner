#!/bin/bash
set -eux

# Specify input and output directories
input_dir=$(readlink -f $1)
output_root=${2:-/mnt/nvme0/sakoda/test/$(date +%Y%m%d_%H%M%S)_visualize_recursive}

# Move to script directory
cd $(dirname $0)/..

# Create output root directory
mkdir -p ${output_root}

# Create log file
log_file=${output_root}/processing.log
echo "Start processing: $(date)" | tee ${log_file}
echo "Input directory: ${input_dir}" | tee -a ${log_file}
echo "Output directory: ${output_root}" | tee -a ${log_file}
echo "" | tee -a ${log_file}

# Find all leaf directories containing npz files (more efficient for large datasets)
find ${input_dir} -mindepth 2 -maxdepth 2 -type d | sort -ru | while read -r target_dir; do
    # Check if directory contains npz files
    if ! ls ${target_dir}/*.npz >/dev/null 2>&1; then
        continue
    fi

    # Get relative path from input_dir
    rel_path=${target_dir#${input_dir}/}

    # Separate date directory and timestamp directory
    # Example: 2025-08-13/16-40-20 -> date_dir=2025-08-13, timestamp=16-40-20
    date_dir=$(dirname ${rel_path})
    timestamp=$(basename ${rel_path})

    echo "Processing: ${rel_path}" | tee -a ${log_file}

    # Create temporary working directory
    temp_dir=${output_root}/.temp/${rel_path}
    mkdir -p ${temp_dir}

    # Create output directory
    output_date_dir=${output_root}/${date_dir}
    mkdir -p ${output_date_dir}

    # Create path_list.json
    python3 ./diffusion_planner/util_scripts/create_train_set_path.py ${target_dir} \
        --save_path ${temp_dir}/path_list.json 2>&1 | tee -a ${log_file}

    if [ ! -f ${temp_dir}/path_list.json ]; then
        echo "Error: Failed to create path_list.json for ${rel_path}" | tee -a ${log_file}
        continue
    fi

    # Execute visualization
    python3 ./diffusion_planner/util_scripts/visualize_input.py ${temp_dir}/path_list.json \
        ${temp_dir}/visualize_result 2>&1 | tee -a ${log_file}

    if [ ! -d ${temp_dir}/visualize_result ]; then
        echo "Error: Failed to visualize for ${rel_path}" | tee -a ${log_file}
        continue
    fi

    # Create mp4
    ~/misc/ffmpeg_lib/make_mp4_from_unsequential_png.sh ${temp_dir}/visualize_result 2>&1 | tee -a ${log_file}

    # Move mp4 file to appropriate location (assuming output without extension)
    if [ -f ${temp_dir}/visualize_result.mp4 ]; then
        mv ${temp_dir}/visualize_result.mp4 ${output_date_dir}/${timestamp}.mp4
        echo "Created: ${output_date_dir}/${timestamp}.mp4" | tee -a ${log_file}
    elif [ -f ${temp_dir}/visualize_result/output.mp4 ]; then
        mv ${temp_dir}/visualize_result/output.mp4 ${output_date_dir}/${timestamp}.mp4
        echo "Created: ${output_date_dir}/${timestamp}.mp4" | tee -a ${log_file}
    else
        # Search for mp4 file
        mp4_file=$(find ${temp_dir}/visualize_result -name "*.mp4" -type f | head -1)
        if [ -n "${mp4_file}" ]; then
            mv ${mp4_file} ${output_date_dir}/${timestamp}.mp4
            echo "Created: ${output_date_dir}/${timestamp}.mp4" | tee -a ${log_file}
        else
            echo "Warning: No mp4 file found for ${rel_path}" | tee -a ${log_file}
        fi
    fi

    # Delete temporary files (optional: commented out to retain)
    # rm -rf ${temp_dir}

    echo "" | tee -a ${log_file}
done

# Delete temporary directory
rm -rf ${output_root}/.temp

echo "Completed: $(date)" | tee -a ${log_file}
echo "Results saved to: ${output_root}" | tee -a ${log_file}
