#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

GPUS=$1
PORT=${PORT:-29500}

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py  --launcher pytorch ${@:3}
