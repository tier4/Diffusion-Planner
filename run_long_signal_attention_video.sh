#!/bin/bash
# Render a long signal-attention video covering approach, intersection entry,
# and the left-turn portion of the 13-49-19 sequence.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MODEL_DIR="${MODEL_DIR:-/home/yamashita/work_hdd/sample/best_onnx/20260730/best_model}"
DATADIR="${DATADIR:-/home/yamashita/work_hdd/sample/mini_datasets/j6_2231_fullseq_mini_20260707}"
VALID_LIST="${VALID_LIST:-$DATADIR/path_list_valid.json}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
CENTER_INDEX="${CENTER_INDEX:-110}"
FRAMES_BEFORE="${FRAMES_BEFORE:-110}"
FRAMES_AFTER="${FRAMES_AFTER:-110}"
STEP="${STEP:-5}"
FPS="${FPS:-5}"
VIDEO_WIDTH="${VIDEO_WIDTH:-640}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-360}"
TOP_K="${TOP_K:-10}"
OUT_DIR="${OUT_DIR:-/tmp/signal-video-long-640x360}"
OUTPUT_NAME="${OUTPUT_NAME:-signal_left_turn_long_top${TOP_K}}"

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
MODEL_DIR="$(cd "$MODEL_DIR" && pwd)"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUT_DIR/.matplotlib}"
mkdir -p "$MPLCONFIGDIR"

OVERWRITE_ARGS=()
if [[ "${OVERWRITE_FRAMES:-0}" == "1" ]]; then
  OVERWRITE_ARGS+=(--overwrite_frames)
fi

"$PYTHON_BIN" scripts/visualize_signal_attention_video.py \
  --run_dir "$MODEL_DIR" \
  --valid_set_list "$VALID_LIST" \
  --center_index "$CENTER_INDEX" \
  --frames_before "$FRAMES_BEFORE" \
  --frames_after "$FRAMES_AFTER" \
  --step "$STEP" \
  --fps "$FPS" \
  --video_width "$VIDEO_WIDTH" \
  --video_height "$VIDEO_HEIGHT" \
  --top_k "$TOP_K" \
  --device "$DEVICE" \
  --out_fusion_mp4 "$OUT_DIR/${OUTPUT_NAME}_fusion.mp4" \
  --out_rollout_mp4 "$OUT_DIR/${OUTPUT_NAME}_rollout.mp4" \
  --out_json "$OUT_DIR/${OUTPUT_NAME}.json" \
  "${OVERWRITE_ARGS[@]}"
