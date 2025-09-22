import os
from glob import glob

from setuptools import find_packages, setup

package_name = "diffusion_planner_ros"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.xml")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="shintarosakoda",
    maintainer_email="shintaro.sakoda@tier4.jp",
    description="diffusion_planner_ros",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "diffusion_planner_node = diffusion_planner_ros.diffusion_planner_node:main"
        ],
    },
)
