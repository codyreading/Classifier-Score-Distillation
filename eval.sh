#!/bin/bash
trap "exit" INT TERM ERR
trap "kill 0" EXIT
export OMP_NUM_THREADS=1

prompt="a red sportscar"
exp_name="a_teapot"
object_name="teapot"

python launch.py --config outputs/csd/${exp_name}/configs/raw.yaml \
--test --gpu 0 \
system.prompt_processor.prompt="$prompt" \
use_timestamp=False \
resume=outputs/csd/${exp_name}/ckpts/last.ckpt tag=${exp_name}_test \
data.object_name=${object_name} \
system.loggers.wandb.enable=False \