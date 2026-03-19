# README

## 1. Setup

### (Optional) create venv

```bash
sudo apt install python3-pip -y
sudo apt install python3-venv -y
python3 -m venv .venv
source ./.venv/bin/activate
```

### Install libraries

```bash
# Diffusion-Planner
cd diffusion_planner
python3 -m pip install pip==24.1

pip install -r requirements_nuplan-devkit_fixed.txt
pip install -r requirements.txt
pip install -e .

# check torch
python3 -c "import torch; print(torch.cuda.is_available())"
```

## 2. Create dataset

### 2.1. Prepare rosbags

We assume the following directory structure:

```bash
driving_dataset$ tree . -L 2
.
├── bag
│   ├── 2024-07-18
│   │ ├── 10-05-28
│   │ ├── 10-05-51
│   │ ├── ...
│   │ ├── 16-10-07
│   │ └── 16-27-15
│   ├── 2024-12-11
│   ├── 2025-01-24
│   ├── 2025-02-04
│   ├── 2025-03-25
│   └── 2025-04-16
└── map
     ├── 2024-07-18
     │   ├── lanelet2_map.osm
     │   ├── pointcloud_map_metadata.yaml
     │   ├── pointcloud_map.pcd
     │   └── stop_points.csv
     ├── 2024-12-11
     ├── 2025-01-24
     ├── 2025-02-04
     ├── 2025-03-25
     └── 2025-04-16
```

### 2.2. Convert to diffusion_planner's format (npz)

use `parse_rosbag_for_directory.py` directly.

```bash
python3 ./ros_scripts/parse_rosbag_for_directory.py <target_dir_list> --save_root <save_root> [--step <step>] [--limit <limit>]
```

### 2.3. Generate path_list.json

This script search `*.npz` files and create `path_list.json`.

```bash
python3 ./diffusion_planner/util_scripts/create_train_set_path.py <root_dir_list>
```

## 3. Train

Edit `train_run.sh` and run

```bash
cd ./diffusion_planner
./train_run.sh
```
