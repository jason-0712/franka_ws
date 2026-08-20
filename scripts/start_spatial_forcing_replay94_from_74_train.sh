#!/usr/bin/env bash
set -euo pipefail

# Matched Phase-10 Spatial-Forcing experiment on the same replay-94 data.
# Run only after the standard replay baseline has passed offline/real checks.

repo=${STARVLA_REPO:-/home/hanyu/starVLA}
matched_arm=${MATCHED_ARM:?Set MATCHED_ARM=control or treatment}
case "${matched_arm}" in
  control) default_alpha=0.0 ;;
  treatment) default_alpha=0.1 ;;
  *) echo "MATCHED_ARM must be control or treatment" >&2; exit 1 ;;
esac

export STARVLA_REPO="${repo}"
export MATCHED_ARM="${matched_arm}"
export DATA_ROOT_DIR=${DATA_ROOT_DIR:-/data/hanyu/quest3_franka_real/snkdjn}
export DATA_MIX=${DATA_MIX:-quest3_franka_dualcam_replay_94eps_v1}
export PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-/data/hanyu/starVLA_runs/quest3_franka_dualcam_74eps_from_libero30k_vision_frozen_20k_retry3/final_model/pytorch_model.pt}
export MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-5000}
export SAVE_INTERVAL=${SAVE_INTERVAL:-$((MAX_TRAIN_STEPS + 1))}
export EVAL_INTERVAL=${EVAL_INTERVAL:-500}
export NUM_WARMUP_STEPS=${NUM_WARMUP_STEPS:-250}
export PROJECTED_ALIGNMENT_ALPHA=${PROJECTED_ALIGNMENT_ALPHA:-${default_alpha}}
export RUN_ID=${RUN_ID:-replay94_sf_${matched_arm}_alpha${PROJECTED_ALIGNMENT_ALPHA}_5k_seed42_$(date +%Y%m%d_%H%M%S)}

runner=${repo}/examples/realRobots/Franka/train_files/run_qwengroot_spatial_forcing_clean_smoke.sh
if [[ ! -x "${runner}" ]]; then
  echo "Missing Phase-10 clean runner: ${runner}" >&2
  exit 1
fi

echo "REPLAY94_SF_INITIALIZATION=${PRETRAINED_CHECKPOINT}"
echo "REPLAY94_SF_DATA_MIX=${DATA_MIX}"
echo "REPLAY94_SF_MATCHED_ARM=${matched_arm}"
echo "REPLAY94_SF_ALPHA=${PROJECTED_ALIGNMENT_ALPHA}"
bash "${runner}"
