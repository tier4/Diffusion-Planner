#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/home/yamashita/work_hdd/sample/best_onnx/20260730/best_model}"
VALID_LIST="${VALID_LIST:-/home/yamashita/work_hdd/sample/mini_datasets/j6_2231_fullseq_mini_20260707/path_list_valid.json}"
CENTER_INDEX="${CENTER_INDEX:-465}"
FRAMES_BEFORE="${FRAMES_BEFORE:-110}"
FRAMES_AFTER="${FRAMES_AFTER:-110}"
STEP="${STEP:-5}"
FPS="${FPS:-5}"
VIDEO_WIDTH="${VIDEO_WIDTH:-1920}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-1080}"
TOP_K="${TOP_K:-20}"
DEVICE="${DEVICE:-cuda}"
OUT_DIR="${OUT_DIR:-/home/yamashita/work_hdd/DP_exp/attention-prediction-start465-long}"
OUTPUT_NAME="${OUTPUT_NAME:-rollout_prediction_start465_long_top20}"
OVERWRITE_FRAMES="${OVERWRITE_FRAMES:-1}"

args=(--run_dir "$MODEL_DIR" --valid_set_list "$VALID_LIST" --center_index "$CENTER_INDEX"
  --frames_before "$FRAMES_BEFORE" --frames_after "$FRAMES_AFTER" --step "$STEP" --fps "$FPS"
  --video_width "$VIDEO_WIDTH" --video_height "$VIDEO_HEIGHT" --top_k "$TOP_K"
  --device "$DEVICE" --out_dir "$OUT_DIR" --output_name "$OUTPUT_NAME")
if [[ "$OVERWRITE_FRAMES" == 1 ]]; then args+=(--overwrite_frames); fi
exec .venv/bin/python scripts/visualize_attention_prediction_video.py "${args[@]}"
