#!/bin/bash
# Render Fusion and Decoder-rollout videos for traffic-light-bearing lanes/routes.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/best_models/20260730/best_model}"
DATADIR="${DATADIR:?DATADIR must be set to a dataset directory}"
VALID_LIST="${VALID_LIST:-$DATADIR/path_list_valid.json}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
CENTER_INDEX="${CENTER_INDEX:?CENTER_INDEX must be set}"
OUT_DIR="${OUT_DIR:-$MODEL_DIR/../signal_attention_video}"
OUTPUT_NAME="${OUTPUT_NAME:-signal_attention_${CENTER_INDEX}}"
mkdir -p "$OUT_DIR"
MODEL_DIR="$(cd "$MODEL_DIR" && pwd)"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUT_DIR/.matplotlib}"
mkdir -p "$MPLCONFIGDIR"
OVERWRITE_ARGS=()
if [[ "${OVERWRITE_FRAMES:-0}" == "1" ]]; then
  OVERWRITE_ARGS+=(--overwrite_frames)
fi
if [ ! -f "$VALID_LIST" ]; then
  VALID_LIST="$OUT_DIR/path_list_all.json"
  "$PYTHON_BIN" - "$DATADIR" "$VALID_LIST" <<'PY'
import glob, json, sys
with open(sys.argv[2], "w") as file:
    json.dump(sorted(glob.glob(sys.argv[1] + "/**/*.npz", recursive=True)), file, indent=2)
PY
fi
"$PYTHON_BIN" scripts/visualize_signal_attention_video.py \
  --run_dir "$MODEL_DIR" --valid_set_list "$VALID_LIST" --center_index "$CENTER_INDEX" \
  --frames_before "${FRAMES_BEFORE:-20}" --frames_after "${FRAMES_AFTER:-40}" --step "${STEP:-1}" \
  --fps "${FPS:-10}" --video_width "${VIDEO_WIDTH:-1920}" --video_height "${VIDEO_HEIGHT:-1080}" \
  --layer "${LAYER:-mean}" --top_k "${TOP_K:-6}" --view_range "${VIEW_RANGE:-80}" --device "${DEVICE:-cuda}" \
  --out_fusion_mp4 "$OUT_DIR/${OUTPUT_NAME}_fusion.mp4" \
  --out_rollout_mp4 "$OUT_DIR/${OUTPUT_NAME}_rollout.mp4" \
  --out_json "$OUT_DIR/${OUTPUT_NAME}.json" \
  "${OVERWRITE_ARGS[@]}"
