#!/bin/bash
trap "exit" INT TERM ERR
trap "kill 0" EXIT
export OMP_NUM_THREADS=1

prompt="a red sportscar"

python launch.py --config ./csd.yaml --train --gpu 0 \
system.prompt_processor.prompt="$prompt" use_timestamp=False