import os
from glob import glob

from setuptools import find_packages, setup

package_name = "diffusion_planner_ros"

# メタデータ (name/version/description 等) は pyproject.toml [project] に集約。
# setup.py は ament_python (colcon) が必要とする data_files / entry_points のみを担当する。
setup(
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.xml")),
        ),
    ],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "diffusion_planner_node = diffusion_planner_ros.diffusion_planner_node:main"
        ],
    },
)
