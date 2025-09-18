import argparse
from pathlib import Path
from shutil import copyfile, rmtree

import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir1", type=Path, default=None)
    parser.add_argument("target_dir2", type=Path, default=None)
    parser.add_argument("out_dir", type=Path, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    target_dir1 = args.target_dir1
    target_dir2 = args.target_dir2
    out_dir = args.out_dir

    rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(exist_ok=True, parents=True)

    date_dir_name_list1 = sorted([d.name for d in target_dir1.iterdir() if d.is_dir()])
    date_dir_name_list2 = sorted([d.name for d in target_dir2.iterdir() if d.is_dir()])
    assert date_dir_name_list1 == date_dir_name_list2

    for date_dir_name in tqdm(date_dir_name_list1):
        curr_out_dir = out_dir / date_dir_name
        curr_out_dir.mkdir(exist_ok=True, parents=True)
        npz_list1 = sorted((target_dir1 / date_dir_name).glob("**/*.npz"))
        npz_list2 = sorted((target_dir2 / date_dir_name).glob("**/*.npz"))
        assert len(npz_list1) == len(npz_list2), f"{len(npz_list1)} != {len(npz_list2)}"
        n = len(npz_list1)

        for f1, f2 in tqdm(zip(npz_list1, npz_list2), total=n):
            npz1 = np.load(f1)
            npz2 = np.load(f2)

            hh_mm_ss = f1.parent.name

            # 新しい辞書を作成してマージ
            merged_data = {}

            # npz1のデータをコピー
            for key in npz1.keys():
                merged_data[key] = npz1[key]

            # 特定のkeyだけnpz2のデータで上書き
            for key in ["neighbor_agents_past", "neighbor_agents_future"]:
                merged_data[key] = npz2[key]

            datetime_dir = curr_out_dir / hh_mm_ss
            datetime_dir.mkdir(exist_ok=True, parents=True)
            out_path = datetime_dir / f"{f1.stem}.npz"
            np.savez_compressed(out_path, **merged_data)

            # jsonもコピー
            json1 = f1.with_suffix(".json")
            out_path_json = out_path.with_suffix(".json")
            copyfile(json1, out_path_json)
