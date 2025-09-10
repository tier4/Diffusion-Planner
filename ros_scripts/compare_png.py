import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("target_dir2", type=Path)
args = parser.parse_args()

target_dir1 = Path("/mnt/nvme0/sakoda/test/baseline_test_parse_rosbag_py")
target_dir2 = args.target_dir2

png_list1 = sorted(target_dir1.glob("**/*.png"))
png_list2 = sorted(target_dir2.glob("**/*.png"))

filename_list1 = [f.name for f in png_list1]
filename_list2 = [f.name for f in png_list2]
print(filename_list1[0:5])
print(filename_list2[0:5])

and_list = sorted(set(filename_list1) & set(filename_list2))
print(len(and_list))

and_list = and_list[::20]

png_list1 = [f for f in png_list1 if f.name in and_list]
png_list2 = [f for f in png_list2 if f.name in and_list]

result_map = defaultdict(list)

for f1, f2 in zip(png_list1, png_list2):
    image1 = cv2.imread(str(f1))
    image2 = cv2.imread(str(f2))
    print(image1.shape, image1.dtype)
    print(image2.shape, image2.dtype)

    diff = np.abs(image1.astype(np.int32) - image2.astype(np.int32))
    concat = cv2.hconcat([image1, image2, diff.astype(np.uint8)])
    save_path = f2.parent / f"diff_{f1.name}"
    cv2.imwrite(str(save_path), concat)
    print(save_path)
