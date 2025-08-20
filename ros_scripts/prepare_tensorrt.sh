#!/bin/bash
set -eux

# tensorrt
tensorrt_version=10.8.0.43-1+cuda12.8

sudo apt update
sudo apt install -y --allow-change-held-packages --allow-downgrades \
    libnvinfer10=${tensorrt_version} \
    libnvinfer-plugin10=${tensorrt_version} \
    libnvonnxparsers10=${tensorrt_version}

sudo apt install -y --allow-change-held-packages --allow-downgrades \
    libnvinfer-dev=${tensorrt_version} \
    libnvinfer-plugin-dev=${tensorrt_version} \
    libnvinfer-headers-dev=${tensorrt_version} \
    libnvinfer-headers-plugin-dev=${tensorrt_version} \
    libnvonnxparsers-dev=${tensorrt_version}

sudo apt-mark hold libnvinfer10 libnvinfer-plugin10 libnvonnxparsers10

sudo apt-mark hold libnvinfer-dev libnvinfer-plugin-dev \
    libnvinfer-headers-dev libnvinfer-headers-plugin-dev libnvonnxparsers-dev

# cumm, spconv
cumm_version=0.5.3
spconv_version=2.3.8
ARCH=$(uname -m | sed 's/aarch64/arm64/g' | sed 's/x86_64/amd64/g')
wget -O /tmp/cumm.deb \
  "https://github.com/autowarefoundation/spconv_cpp/releases/download/spconv_v${spconv_version}%2Bcumm_v${cumm_version}/cumm_${cumm_version}_${ARCH}.deb"
sudo dpkg -i /tmp/cumm.deb
wget -O /tmp/spconv.deb \
  "https://github.com/autowarefoundation/spconv_cpp/releases/download/spconv_v${spconv_version}%2Bcumm_v${cumm_version}/spconv_${spconv_version}_${ARCH}.deb"
sudo dpkg -i /tmp/spconv.deb
