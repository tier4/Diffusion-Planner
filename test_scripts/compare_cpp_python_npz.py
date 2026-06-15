#!/usr/bin/env python3
"""Compare C++ and Python rosbag-parser npz outputs key-by-key.

The two converters name files differently (C++ ``seq8_frame8``, Python ``seq8frame8``),
so frames are aligned by the trailing frame number in the filename. Reports the max abs
diff per key over the commonly-produced frames.

Example:
    python3 test_scripts/compare_cpp_python_npz.py \
        /mnt/nvme/test/<run>/11-06-10 /mnt/nvme/test/<run>/python_11-06-10
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

THRESHOLD = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpp_dir", type=Path, help="directory with C++ npz files")
    parser.add_argument("python_dir", type=Path, help="directory with Python npz files")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    return parser.parse_args()


def frame_number(path: Path) -> int:
    return int(re.findall(r"(\d+)", path.stem)[-1])


def main() -> None:
    args = parse_args()
    cpp = {frame_number(p): p for p in args.cpp_dir.glob("*.npz")}
    py = {frame_number(p): p for p in args.python_dir.glob("*.npz")}
    common = sorted(set(cpp) & set(py))
    print(f"cpp={len(cpp)} python={len(py)} common={len(common)}")
    only_cpp = sorted(set(cpp) - set(py))
    only_py = sorted(set(py) - set(cpp))
    if only_cpp:
        print(f"  frames only in cpp ({len(only_cpp)}): {only_cpp[:10]}{' ...' if len(only_cpp) > 10 else ''}")
    if only_py:
        print(f"  frames only in python ({len(only_py)}): {only_py[:10]}{' ...' if len(only_py) > 10 else ''}")
    if not common:
        return

    max_diff = defaultdict(float)
    shape_mismatch = defaultdict(int)
    only_in_one = set()
    for fr in common:
        a = np.load(cpp[fr])
        b = np.load(py[fr])
        keys = set(a.keys()) | set(b.keys())
        for k in keys:
            if k not in a or k not in b:
                only_in_one.add(k)
                continue
            da, db = a[k], b[k]
            if da.dtype.kind in "US":
                continue
            if da.shape != db.shape:
                shape_mismatch[k] += 1
                continue
            diff = np.abs(da.astype(np.float64) - db.astype(np.float64))
            if diff.size:
                max_diff[k] = max(max_diff[k], float(np.nanmax(diff)))

    n_ok = 0
    for k in sorted(set(max_diff) | set(shape_mismatch)):
        if k in shape_mismatch and k not in max_diff:
            print(f"  SHAPE {k}: shape differs in {shape_mismatch[k]} frames")
            continue
        ok = max_diff[k] < args.threshold
        n_ok += ok
        print(f"  {'OK ' if ok else 'NG '} {k:30s} max={max_diff[k]:.4g}")
    for k in sorted(only_in_one):
        print(f"  ONLY-IN-ONE {k}: present in only one converter")
    print(f"\n{n_ok}/{len(set(max_diff) | set(shape_mismatch))} value-comparable keys within {args.threshold}")


if __name__ == "__main__":
    main()
