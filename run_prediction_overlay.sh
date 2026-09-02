#!/bin/bash
# Overlay recorded and model-predicted ego/neighbor trajectories.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/best_models/20260730/best_model}"
DATADIR="${DATADIR:?DATADIR must be set}"
VALID_LIST="${VALID_LIST:-$DATADIR/path_list_valid.json}"
SAMPLE_INDEX="${SAMPLE_INDEX:?SAMPLE_INDEX must be set}"
OUT_DIR="${OUT_DIR:-/home/yamashita/work_hdd/DP_exp}"
OUTPUT_NAME="${OUTPUT_NAME:-prediction_overlay_${SAMPLE_INDEX}}"
mkdir -p "$OUT_DIR"
"${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}" scripts/visualize_prediction_overlay.py \
  --run_dir "$MODEL_DIR" --valid_set_list "$VALID_LIST" --sample_index "$SAMPLE_INDEX" \
  --device "${DEVICE:-cuda}" --view_range "${VIEW_RANGE:-80}" \
  --ego "${EGO_MODE:-prediction}" --neighbors "${NEIGHBOR_MODE:-prediction}" \
  --out_png "$OUT_DIR/${OUTPUT_NAME}.png" --out_json "$OUT_DIR/${OUTPUT_NAME}.json"
