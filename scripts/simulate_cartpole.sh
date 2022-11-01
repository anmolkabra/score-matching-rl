#!/usr/bin/env bash

if [[ -z "${EXP_NAME}" ]]; then
  EXP_NAME="simulate_cartpole__$(date --iso-8601='seconds')"
fi

PYTHONPATH='.' python scripts/simulate.py \
  --exp_name ${EXP_NAME} \
  --train_num_episodes 200 \
  --env_no_render \
  "$@"
