#!/bin/bash
# Create a continuous MP4 of neighbor-token Attention around one dataset index.
#
# Right-turn example:
#   CENTER_INDEX=160 ./run_neighbor_attention_video.sh
#
# Short preview:
#   CENTER_INDEX=160 FRAMES_BEFORE=2 FRAMES_AFTER=2 DEVICE=cpu \
#     ./run_neighbor_attention_video.sh
#
# Main environment variables:
#   MODEL_DIR, DATADIR, VALID_LIST, CENTER_INDEX
#   FRAMES_BEFORE, FRAMES_AFTER, STEP, FPS, DEVICE
#   VIDEO_WIDTH, VIDEO_HEIGHT
#   LAYER, TOP_K, VIEW_RANGE, COLORMAP
#   MARKER_SIZE_MIN, MARKER_SIZE_MAX, OUT_DIR, OUTPUT_NAME, KEEP_FRAMES
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/best_models/20260730/best_model}"
DATADIR="${DATADIR:?DATADIR must be set to a dataset directory}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"

CENTER_INDEX="${CENTER_INDEX:-160}"
FRAMES_BEFORE="${FRAMES_BEFORE:-20}"
FRAMES_AFTER="${FRAMES_AFTER:-40}"
STEP="${STEP:-1}"
FPS="${FPS:-10}"
VIDEO_WIDTH="${VIDEO_WIDTH:-1920}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-1080}"

LAYER="${LAYER:-mean}"
TOP_K="${TOP_K:-12}"
VIEW_RANGE="${VIEW_RANGE:-80}"
COLORMAP="${COLORMAP:-viridis}"
MARKER_SIZE_MIN="${MARKER_SIZE_MIN:-45}"
MARKER_SIZE_MAX="${MARKER_SIZE_MAX:-950}"
KEEP_FRAMES="${KEEP_FRAMES:-0}"

TAG="$(basename "$DATADIR")"
OUT_DIR="${OUT_DIR:-$MODEL_DIR/../neighbor_attention_${TAG}}"
OUTPUT_NAME="${OUTPUT_NAME:-neighbor_attention_index${CENTER_INDEX}}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [ ! -f "$MODEL_DIR/args.json" ]; then
  echo "Model config not found: $MODEL_DIR/args.json" >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg was not found in PATH" >&2
  exit 1
fi
if [ "$KEEP_FRAMES" != "0" ] && [ "$KEEP_FRAMES" != "1" ]; then
  echo "KEEP_FRAMES must be 0 or 1, got: $KEEP_FRAMES" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
MODEL_DIR="$(cd "$MODEL_DIR" && pwd)"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUT_DIR/.matplotlib}"
mkdir -p "$MPLCONFIGDIR"

VALID_LIST="${VALID_LIST:-$DATADIR/path_list_valid.json}"
if [ ! -f "$VALID_LIST" ]; then
  VALID_LIST="$OUT_DIR/path_list_all.json"
  echo "No path_list_valid.json found; collecting NPZ files under: $DATADIR"
  "$PYTHON_BIN" - "$DATADIR" "$VALID_LIST" <<'PY'
import glob
import json
import sys

files = sorted(glob.glob(sys.argv[1] + "/**/*.npz", recursive=True))
if not files:
    raise RuntimeError(f"no NPZ files found under {sys.argv[1]}")
with open(sys.argv[2], "w") as file:
    json.dump(files, file)
print(f"wrote {len(files)} paths to {sys.argv[2]}")
PY
fi
VALID_LIST="$(cd "$(dirname "$VALID_LIST")" && pwd)/$(basename "$VALID_LIST")"

COMMAND=(
  "$PYTHON_BIN"
  scripts/visualize_neighbor_attention_video.py
  --run_dir "$MODEL_DIR"
  --valid_set_list "$VALID_LIST"
  --center_index "$CENTER_INDEX"
  --frames_before "$FRAMES_BEFORE"
  --frames_after "$FRAMES_AFTER"
  --step "$STEP"
  --fps "$FPS"
  --video_width "$VIDEO_WIDTH"
  --video_height "$VIDEO_HEIGHT"
  --device "$DEVICE"
  --layer "$LAYER"
  --top_k "$TOP_K"
  --view_range "$VIEW_RANGE"
  --colormap "$COLORMAP"
  --marker_size_min "$MARKER_SIZE_MIN"
  --marker_size_max "$MARKER_SIZE_MAX"
  --out_mp4 "$OUT_DIR/$OUTPUT_NAME.mp4"
  --out_json "$OUT_DIR/$OUTPUT_NAME.json"
)
if [ "$KEEP_FRAMES" = "1" ]; then
  COMMAND+=(--keep_frames)
fi

"${COMMAND[@]}"

echo "done"
echo "  video: $OUT_DIR/$OUTPUT_NAME.mp4"
echo "  data:  $OUT_DIR/$OUTPUT_NAME.json"
