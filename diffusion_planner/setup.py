import os

import setuptools

# Change directory to allow installation from anywhere
script_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_folder)

# Installs
setuptools.setup(
    name="diffusion_planner",
    version="1.0.0",
    author="Zheng Yinan, Ruiming Liang, Kexin Zheng @ Tsinghua AIR",
    # find_packages so subpackages (metrics, model, utils, ...) ship in the wheel.
    # The previous packages=["diffusion_planner"] only shipped the top-level
    # package, dropping every subpackage in a non-editable install — which broke
    # `from diffusion_planner.metrics ...` (and model/utils) outside the editable
    # workspace. include= keeps it to the diffusion_planner tree only.
    packages=setuptools.find_packages(
        where=".", include=["diffusion_planner", "diffusion_planner.*"]
    ),
    package_dir={"": "."},
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "License :: Free for non-commercial use",
    ],
    license="MIT",
)
