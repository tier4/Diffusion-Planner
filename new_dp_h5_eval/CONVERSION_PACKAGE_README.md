# Native H5 conversion package

This package contains the exact 119 unique samples referenced by
`open_loop_matrix.json`, their NPZ sidecars, the corresponding 106 ROS bags,
and required map directories. No unrelated dataset files are included.

On the local machine, extract the archive, source ROS 2 and the new-DP ROS
workspace, then run:

```bash
export NEW_DP=/path/to/new-DP
export CONVERTER=/path/to/Diffusion-Planner/packages/diffusion_planner/dataset/convert_matrix_rosbag_to_h5.py
source /opt/ros/humble/setup.bash
source "$NEW_DP/ros2_ws/install/setup.bash"
cd /path/to/rosbag_conversion_package
bash /path/to/Diffusion-Planner/new_dp_h5_eval/run_conversion_local.sh "$PWD"
```

The generated native H5 files and Parquet indexes are in `h5/basic` and
`h5/override`. Keep each matrix with its corresponding output root when running
the new-DP open-loop evaluator. Closed-loop site data is not silently guessed
or copied here; it needs a separate route-to-rosbag mapping.
