#!/usr/bin/env python3
"""各サブディレクトリの path_list 件数と processing_time.txt を表で出力する。"""

import argparse
import json
import sys
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=Path)
    parser.add_argument("--detail", action="store_true")
    args = parser.parse_args()
    root_dir = args.root_dir
    detail = args.detail

    subdirs = sorted([p for p in root_dir.iterdir() if p.is_dir()])

    for subdir in subdirs:
        print(subdir.name)
        path_list_files = sorted(subdir.rglob("path_list_*.json"))

        npz_count = 0
        for json_file in path_list_files:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            n = len(data)
            hour = n / 10 / 60 / 60
            print(f"  {json_file.name}: {n:,} 件 ({hour:.2f} 時間分)")

        processing_time_file = subdir / "processing_time.txt"
        if processing_time_file.exists():
            processing_time_raw = processing_time_file.read_text(encoding="utf-8").strip()
            print(f"  {processing_time_raw}")

    if not detail:
        sys.exit(0)

    f = open(root_dir / "detail.txt", "w", encoding="utf-8")

    for subdir in subdirs:
        sub_sub_dirs = sorted([p for p in subdir.iterdir() if p.is_dir()])
        for sub_sub_dir in sub_sub_dirs:
            npz_path_list = sorted(sub_sub_dir.rglob("*.npz"))
            print(f"  {sub_sub_dir.name} {len(npz_path_list)} 件")
            f.write(f"{subdir} {sub_sub_dir.name} {len(npz_path_list)}\n")
