#!/bin/bash
# Generate a per-scene Attention overlay for every Fusion encoder token class.
#
# Auto-select a turning scene using any token class:
#   ./run_all_token_attention_visualization.sh
#
# Auto-select using route tokens:
#   SELECT_CLASS=route ./run_all_token_attention_visualization.sh
#
# Render a specific dataset-list index:
#   SAMPLE_INDEX=996 ./run_all_token_attention_visualization.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/best_models/20260730/best_model}"
DATADIR="${DATADIR:?DATADIR must be set to a dataset directory}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"

SAMPLE_INDEX="${SAMPLE_INDEX:-}"
SELECT_CLASS="${SELECT_CLASS:-any}"
TURN_ONLY="${TURN_ONLY:-1}"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-128}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MOVE_MIN_M="${MOVE_MIN_M:-5.0}"
TURN_DEG="${TURN_DEG:-15.0}"

LAYER="${LAYER:-mean}"
TOP_K="${TOP_K:-20}"
VIEW_RANGE="${VIEW_RANGE:-80}"
COLORMAP="${COLORMAP:-plasma}"
MARKER_SIZE_MIN="${MARKER_SIZE_MIN:-25}"
MARKER_SIZE_MAX="${MARKER_SIZE_MAX:-700}"
TAG="$(basename "$DATADIR")"
OUT_DIR="${OUT_DIR:-$MODEL_DIR/../all_token_attention_${TAG}}"
OUTPUT_NAME="${OUTPUT_NAME:-all_token_attention}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [ ! -f "$MODEL_DIR/args.json" ]; then
  echo "Model config not found: $MODEL_DIR/args.json" >&2
  exit 1
fi
if [ "$TURN_ONLY" != "0" ] && [ "$TURN_ONLY" != "1" ]; then
  echo "TURN_ONLY must be 0 or 1, got: $TURN_ONLY" >&2
  exit 1
fi
case "$SELECT_CLASS" in
  any | ego | neighbors | static | lanes | route | polygons | line_strings | goal_pose | ego_shape | turn_indicator) ;;
  *)
    echo "Unknown SELECT_CLASS: $SELECT_CLASS" >&2
    exit 1
    ;;
esac

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
  scripts/visualize_all_token_attention.py
  --run_dir "$MODEL_DIR"
  --valid_set_list "$VALID_LIST"
  --device "$DEVICE"
  --layer "$LAYER"
  --top_k "$TOP_K"
  --view_range "$VIEW_RANGE"
  --colormap "$COLORMAP"
  --marker_size_min "$MARKER_SIZE_MIN"
  --marker_size_max "$MARKER_SIZE_MAX"
  --out_png "$OUT_DIR/$OUTPUT_NAME.png"
  --out_json "$OUT_DIR/$OUTPUT_NAME.json"
)

if [ -n "$SAMPLE_INDEX" ]; then
  COMMAND+=(--sample_index "$SAMPLE_INDEX")
  echo "Rendering fixed dataset index: $SAMPLE_INDEX"
else
  COMMAND+=(
    --select_class "$SELECT_CLASS"
    --candidate_count "$CANDIDATE_COUNT"
    --batch_size "$BATCH_SIZE"
    --move_min_m "$MOVE_MIN_M"
    --turn_deg "$TURN_DEG"
  )
  if [ "$TURN_ONLY" = "1" ]; then
    COMMAND+=(--turn_only)
  fi
  echo "Auto-selecting a scene: class=$SELECT_CLASS, turn_only=$TURN_ONLY"
fi

"${COMMAND[@]}"

echo "done"
echo "  image: $OUT_DIR/$OUTPUT_NAME.png"
echo "  data:  $OUT_DIR/$OUTPUT_NAME.json"
