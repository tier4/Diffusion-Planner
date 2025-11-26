#!/bin/bash
set -eux

target_dir=$(readlink -f $1)
cd $(dirname $0)

python3 ./torch2onnx.py \
    --config $target_dir/args.json \
    --ckpt $target_dir/best_model.pth \
    --onnx $target_dir/model.onnx \
