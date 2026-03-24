#!/bin/bash
set -e

# V4 converter: rosbag → .bin → .npz (keeping .bin copy)
# Requires system autoware_msgs symlinked to 1.3.0

SSD="/media/danielsanchez/2fb4af16-188c-4b7d-8ebb-4a7d0c90d207"
MAP="/home/danielsanchez/autoware_map/shinagawa_odaiba_stable/lanelet2_map.osm"
CONVERTER="/home/danielsanchez/Diffusion-Planner/cpp_tools/install/autoware_diffusion_planner_tools/lib/autoware_diffusion_planner_tools/data_converter"
BIN2NPZ="/home/danielsanchez/Diffusion-Planner/ros_scripts/convert_cpp_bin_to_python_npz.py"

OUT_BASE="${SSD}/xx1_grpo_v4_data"
BIN_BACKUP="${OUT_BASE}/bin_backup"
NPZ_DIR="${OUT_BASE}/npz"

mkdir -p "$BIN_BACKUP" "$NPZ_DIR"

# Source ROS + local install for converter
source /opt/ros/humble/setup.bash
source /home/danielsanchez/Diffusion-Planner/cpp_tools/install/setup.bash

# Activate python venv for bin→npz
source /home/danielsanchez/Diffusion-Planner/.venv/bin/activate

# --- Training bags (71k scenes: p0900_0/1/2) ---
BAGS=(
  "${SSD}/rosbags/p0900_0"
  "${SSD}/rosbags/p0900_1"
  "${SSD}/rosbags/p0900_2"
)

# --- Validation bags (12 sessions) ---
BAGS+=(
  "${SSD}/xx1_validation_data/xx1_real_valid/bag_filtered/2025-06-12/10-19-35"
  "${SSD}/xx1_validation_data/xx1_real_valid/bag_filtered/2025-06-12/10-24-57"
  "${SSD}/xx1_validation_data/xx1_real_valid/bag_filtered/2025-06-12/10-38-19"
  "${SSD}/xx1_validation_data/xx1_real_valid/bag_filtered/2025-06-12/10-51-07"
  "${SSD}/xx1_validation_data/xx1_real_valid/bag_filtered/2025-06-12/11-02-50"
  "${SSD}/xx1_validation_data/xx1_real_valid/bag_filtered/2025-06-12/11-17-59"
  "${SSD}/xx1_validation_data/xx1_real_valid/bag_filtered/2025-06-16/14-07-31"
  "${SSD}/xx1_validation_data/xx1_real_valid/bag_filtered/2025-06-16/14-18-37"
  "${SSD}/xx1_validation_data/xx1_real_valid/bag_filtered/2025-06-16/14-44-11"
  "${SSD}/xx1_validation_data/xx1_real_valid/bag_filtered/2025-06-16/15-02-05"
  "${SSD}/xx1_validation_data/xx1_real_valid/bag_filtered/2025-06-16/15-34-48"
  "${SSD}/xx1_validation_data/xx1_real_valid/bag_filtered/2025-06-16/16-10-56"
)

TOTAL_BIN=0
TOTAL_NPZ=0

for BAG in "${BAGS[@]}"; do
  BAGNAME=$(basename "$BAG")
  # Use parent dir name too to avoid collisions between same-named bags
  PARENT=$(basename "$(dirname "$BAG")")
  TAG="${PARENT}_${BAGNAME}"
  echo ""
  echo "================================================================"
  echo "Converting: $TAG ($BAG)"
  echo "================================================================"

  # Per-bag bin output dir
  BAG_BIN_DIR="${OUT_BASE}/bin_${TAG}"
  mkdir -p "$BAG_BIN_DIR"

  # Run C++ converter: rosbag → .bin
  "$CONVERTER" "$BAG" "$MAP" "$BAG_BIN_DIR" \
    --step=1 --min_frames=1 --min_distance=0 \
    --ego_wheel_base=2.75 --ego_length=4.34 --ego_width=1.70 2>&1 | tail -5

  # Count bin files
  NBIN=$(find "$BAG_BIN_DIR" -name "*.bin" | wc -l)
  echo "  Produced $NBIN .bin files"
  TOTAL_BIN=$((TOTAL_BIN + NBIN))

  if [ "$NBIN" -gt 0 ]; then
    # Backup .bin files (copy, don't move) — use find to avoid arg list too long
    find "$BAG_BIN_DIR" -name "*.bin" -exec cp {} "$BIN_BACKUP/" \;

    # Convert .bin → .npz (this deletes .bin from BAG_BIN_DIR)
    python3 "$BIN2NPZ" "$BAG_BIN_DIR" --output_dir "$NPZ_DIR"

    NNPZ=$(find "$NPZ_DIR" -name "*.npz" -newer "$BAG_BIN_DIR" 2>/dev/null | wc -l)
    echo "  Converted to NPZ: $NNPZ"
    TOTAL_NPZ=$((TOTAL_NPZ + NNPZ))
  fi

  # Clean up empty per-bag dir
  rmdir "$BAG_BIN_DIR" 2>/dev/null || true
done

echo ""
echo "================================================================"
echo "DONE: $TOTAL_BIN total .bin files, backup at $BIN_BACKUP"
echo "NPZ output at: $NPZ_DIR"
FINAL_NPZ=$(find "$NPZ_DIR" -name "*.npz" | wc -l)
echo "Total NPZ files: $FINAL_NPZ"
echo "================================================================"
