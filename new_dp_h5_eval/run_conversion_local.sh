#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="${1:-$ROOT}"
NEW_DP="${NEW_DP:?set NEW_DP to the new-DP checkout}"
CONVERTER="${CONVERTER:-$ROOT/../packages/diffusion_planner/dataset/convert_matrix_rosbag_to_h5.py}"
PYTHON="${PYTHON:-python3}"
mkdir -p "$PKG/h5/basic" "$PKG/h5/override"

make_matrix() {
  local marker="$1" kind="$2" out="$3"
  "$PYTHON" - "$PKG" "$marker" "$out" <<'PY'
import json, pathlib, sys
root, marker, out = map(pathlib.Path, sys.argv[1:])
data = json.loads((root/'matrices/open_loop_matrix.original.json').read_text())
result = {}
for metric, paths in data.items():
    selected=[]
    for p in paths:
        if marker in pathlib.Path(p).parts:
            q=pathlib.Path(p); i=q.parts.index(marker)
            selected.append((root/'npz'/marker/pathlib.Path(*q.parts[i+1:])).resolve().as_posix())
    if selected: result[metric]=selected
out.write_text(json.dumps(result, indent=2)+'\n')
PY
}

make_matrix 20260814_basic_dataset basic "$PKG/matrices/basic_matrix.json"
make_matrix dataset_all override "$PKG/matrices/override_matrix.json"
"$PYTHON" "$CONVERTER" "$PKG/matrices/basic_matrix.json" "$PKG/rosbags/basic" "$PKG/h5/basic" --dataset-root-name 20260814_basic_dataset --workers "${WORKERS:-4}"
"$PYTHON" "$CONVERTER" "$PKG/matrices/override_matrix.json" "$PKG/rosbags/override" "$PKG/h5/override" --dataset-root-name dataset_all --workers "${WORKERS:-4}"
