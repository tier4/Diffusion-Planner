#!/bin/bash
set -eux

cd $(dirname $0)

vcs import --recursive src < cpp_tools.repos
vcs pull src
rosdep update
rosdep install -y --from-paths src --ignore-src --rosdistro $ROS_DISTRO
