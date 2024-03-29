#!/bin/bash
trap "exit" INT TERM ERR
trap "kill 0" EXIT
export OMP_NUM_THREADS=1

prompt="a red sportscar"

python launch.py --config ./csd.yaml --train --gpu 2 \
system.prompt_processor.prompt="$prompt" \
data.object_name="lambo2" \
use_timestamp=False \
system.loggers.wandb.name="lambo2" \
tag="lambo2"