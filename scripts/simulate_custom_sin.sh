#!/usr/bin/env bash
# Environment: Custom Sin MDP
# Simulator: Specified as CLI environment variable SAMPLING_DENSITY.
#  Parameter W is learned with Score Matching on samples {(s_{t+1}, s_t, a_t)}_t.

sampling_density="${SAMPLING_DENSITY}"
if [[ -z "${SAMPLING_NOISE_SCALE}" ]]; then
  sampling_noise_scale="1e-0"
else
  sampling_noise_scale="${SAMPLING_NOISE_SCALE}"
fi

mkdir -p logs

for seed in 0 1 2 3 4; do
  exp_name="simulate_custom_sin__sampl_${sampling_density}_sigma${sampling_noise_scale}__seed${seed}"
  PYTHONPATH='.' python \
    scripts/simulate.py \
    --env_no_render \
    --env "custom_sin" \
    --env_custom_sin_P 4 \
    --env_custom_sin_ALPHA 1.7 \
    --env_noise_density_sampling_method "inv_cdf" \
    --W_sm_lam 1e-4 \
    --sampling_density "${sampling_density}" \
    --sampling_noise_scale "${sampling_noise_scale}" \
    --sampling_method "inv_cdf" \
    --baseline_nonlds \
    --random_shooting_tau 5 \
    --random_shooting_num_trials 100 \
    --random_shooting_H 10 \
    --random_shooting_all_ones_actions \
    --train_num_episodes 50 \
    --seed "${seed}" \
    --num_workers 1 \
    --exp_name "${exp_name}" 2>&1 | tee "logs/${exp_name}.log"
done
