#!/bin/bash
# Create a video with selectable GT/model ego and neighbor trajectories.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/best_models/20260730/best_model}"
DATADIR="${DATADIR:?DATADIR must be set}"
VALID_LIST="${VALID_LIST:-$DATADIR/path_list_valid.json}"
CENTER_INDEX="${CENTER_INDEX:?CENTER_INDEX must be set}"
OUT_DIR="${OUT_DIR:-/home/yamashita/work_hdd/DP_exp}"
OUTPUT_NAME="${OUTPUT_NAME:-prediction_overlay_video_${CENTER_INDEX}}"
mkdir -p "$OUT_DIR"
OVERWRITE_ARGS=()
if [[ "${OVERWRITE_FRAMES:-0}" == "1" ]]; then OVERWRITE_ARGS+=(--overwrite_frames); fi
"${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}" scripts/visualize_prediction_overlay_video.py \
  --run_dir "$MODEL_DIR" --valid_set_list "$VALID_LIST" --center_index "$CENTER_INDEX" \
  --frames_before "${FRAMES_BEFORE:-20}" --frames_after "${FRAMES_AFTER:-40}" --step "${STEP:-1}" \
  --fps "${FPS:-10}" --video_width "${VIDEO_WIDTH:-1920}" --video_height "${VIDEO_HEIGHT:-1080}" \
  --view_range "${VIEW_RANGE:-80}" --ego "${EGO_MODE:-prediction}" --neighbors "${NEIGHBOR_MODE:-prediction}" \
  --device "${DEVICE:-cuda}" --out_mp4 "$OUT_DIR/${OUTPUT_NAME}.mp4" \
  --out_json "$OUT_DIR/${OUTPUT_NAME}.json" "${OVERWRITE_ARGS[@]}"
