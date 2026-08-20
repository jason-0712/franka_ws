#!/usr/bin/env bash
set -euo pipefail

# Run inside the Franka/LeRobot environment. This script only performs
# inference on recorded data and never publishes a robot command.

POLICY_HOST="${POLICY_HOST:-192.168.1.113}"
CONTROL_PORT="${CONTROL_PORT:-10122}"
TREATMENT_PORT="${TREATMENT_PORT:-10123}"
DATASET_ROOT="${DATASET_ROOT:-/home/ros/.cache/huggingface/lerobot/snkdjn}"
SCRIPT_ROOT="${SCRIPT_ROOT:-/home/ros/ros2_ws/scripts}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ros/ros2_ws/deployment_logs/open_loop}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
SESSION="${OUTPUT_ROOT}/sf_independent_holdout4_${TAG}"

IDS=(
  quest3_9_grids_055
  quest3_9_grids_058
  quest3_9_grids_059
  quest3_9_grids_060
)
SEEDS=(42 314159 271828 20260813 8675309)

mkdir -p "${SESSION}"
CONTROL_CSVS=()
TREATMENT_CSVS=()

for seed in "${SEEDS[@]}"; do
  control_csv="${SESSION}/control_seed${seed}.csv"
  treatment_csv="${SESSION}/treatment_seed${seed}.csv"

  python "${SCRIPT_ROOT}/starvla_open_loop_l2_eval.py" \
    --policy-host "${POLICY_HOST}" \
    --policy-port "${CONTROL_PORT}" \
    --dataset-root "${DATASET_ROOT}" \
    --ids "${IDS[@]}" \
    --stride 5 \
    --max-queries-per-episode 0 \
    --compare both \
    --inference-seed-base "${seed}" \
    --output-csv "${control_csv}" \
    2>&1 | tee "${SESSION}/control_seed${seed}.log"

  python "${SCRIPT_ROOT}/starvla_open_loop_l2_eval.py" \
    --policy-host "${POLICY_HOST}" \
    --policy-port "${TREATMENT_PORT}" \
    --dataset-root "${DATASET_ROOT}" \
    --ids "${IDS[@]}" \
    --stride 5 \
    --max-queries-per-episode 0 \
    --compare both \
    --inference-seed-base "${seed}" \
    --output-csv "${treatment_csv}" \
    2>&1 | tee "${SESSION}/treatment_seed${seed}.log"

  CONTROL_CSVS+=("${control_csv}")
  TREATMENT_CSVS+=("${treatment_csv}")
done

python "${SCRIPT_ROOT}/compare_starvla_seeded_open_loop.py" \
  --control "${CONTROL_CSVS[@]}" \
  --treatment "${TREATMENT_CSVS[@]}" \
  --by-dataset \
  --minimum-spatial-improvement-percent 3 \
  2>&1 | tee "${SESSION}/comparison.txt"

echo "ROBOT_COMMANDS_SENT=0" | tee -a "${SESSION}/comparison.txt"
echo "HOLDOUT_EVAL=PASS" | tee -a "${SESSION}/comparison.txt"
echo "OUTPUT=${SESSION}"
