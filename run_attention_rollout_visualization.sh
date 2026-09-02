#!/bin/bash
# Render Fusion attention and Decoder-to-input attention rollout for one scene.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/best_models/20260730/best_model}"
DATADIR="${DATADIR:?DATADIR must be set to a dataset directory}"
VALID_LIST="${VALID_LIST:-$DATADIR/path_list_valid.json}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
SAMPLE_INDEX="${SAMPLE_INDEX:?SAMPLE_INDEX must be set}"
OUT_DIR="${OUT_DIR:-$MODEL_DIR/../attention_rollout}"
OUTPUT_NAME="${OUTPUT_NAME:-attention_rollout_index${SAMPLE_INDEX}}"

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
MODEL_DIR="$(cd "$MODEL_DIR" && pwd)"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUT_DIR/.matplotlib}"
mkdir -p "$MPLCONFIGDIR"

if [ ! -f "$VALID_LIST" ]; then
  VALID_LIST="$OUT_DIR/path_list_all.json"
  "$PYTHON_BIN" - "$DATADIR" "$VALID_LIST" <<'PY'
import glob
import json
import sys

paths = sorted(glob.glob(sys.argv[1] + "/**/*.npz", recursive=True))
with open(sys.argv[2], "w") as file:
    json.dump(paths, file, indent=2)
print(f"wrote {len(paths)} paths to {sys.argv[2]}")
PY
fi

"$PYTHON_BIN" scripts/visualize_attention_rollout.py \
  --run_dir "$MODEL_DIR" \
  --valid_set_list "$VALID_LIST" \
  --sample_index "$SAMPLE_INDEX" \
  --layer "${LAYER:-mean}" \
  --top_k "${TOP_K:-20}" \
  --view_range "${VIEW_RANGE:-80}" \
  --colormap "${COLORMAP:-plasma}" \
  --marker_size_min "${MARKER_SIZE_MIN:-25}" \
  --marker_size_max "${MARKER_SIZE_MAX:-700}" \
  --device "$DEVICE" \
  --out_fusion_png "$OUT_DIR/${OUTPUT_NAME}_fusion.png" \
  --out_rollout_png "$OUT_DIR/${OUTPUT_NAME}_rollout.png" \
  --out_neighbor_rollout_png "$OUT_DIR/${OUTPUT_NAME}_neighbor_rollout.png" \
  --out_json "$OUT_DIR/${OUTPUT_NAME}.json"
