#!/bin/bash
set -eux

sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt upgrade -y
sudo apt install -y ros-humble-desktop ros-humble-ros-base
sudo apt install -y python3-colcon-common-extensions
sudo apt install -y python3-rosdep
sudo apt install -y python3-vcstool
sudo apt install -y ros-humble-rosbag2-storage-mcap
pip install lanelet2 catkin_pkg empy==3.3.4 lark setuptools==59.6.0 packaging==23.1 shapely
sudo rosdep init
rosdep update
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
echo "export ROS_LOCALHOST_ONLY=1" >> ~/.bashrc
