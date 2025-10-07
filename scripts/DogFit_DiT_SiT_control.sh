#!/bin/bash

# ========== GLOBAL CONFIGURATION ==========
SERVER="bool"
CUDA_DEVICES="0,1"
SCRIPT="run_DogFit.sh"
MODEL_NAME="DiT-XL/2"     # Options: "DiT-XL/2", "SiT-XL/2"
FOCUS="FD_DINOV2"         # Options: "FID", "FD_DINOV2"

declare -a TASKS=(
  "food-101_processed"
)

# ========== Per-focus default parameters ==========
if [ "$FOCUS" == "FID" ]; then
  DEFAULT_LATESTART=12000
  DEFAULT_MGHIGH=0.5
elif [ "$FOCUS" == "FD_DINOV2" ]; then
  DEFAULT_LATESTART=6000
  DEFAULT_MGHIGH=1
else
  echo "Unsupported focus type: $FOCUS"
  exit 1
fi

# ========== Per-task configuration: w_min, w_max, sample_guidance, control_dist ==========
declare -A PARAMS_MAP
# Format: "w_min,w_max,sample_guidance,control_distribution" (w_min and w_max are only used when control_distribution is 'uniform')
PARAMS_MAP["food-101_processed"]="1,3,0,50in1to1.25"

# ========== EXECUTION LOOP ==========
for DATASET in "${TASKS[@]}"; do
  PARAMS=(${PARAMS_MAP["$DATASET"]})
  for PARAM in "${PARAMS[@]}"; do
    IFS=',' read -r W_MIN W_MAX SAMPLE_GUIDANCE CONTROL_DISTRIBUTION <<< "$PARAM"

    echo "=============================================="
    echo "Running $SCRIPT on $DATASET"
    echo "Model: $MODEL_NAME | Focus: $FOCUS"
    echo "w_min: $W_MIN | w_max: $W_MAX | sample_guidance: $SAMPLE_GUIDANCE | control_dist: $CONTROL_DISTRIBUTION"
    echo "latestart: $DEFAULT_LATESTART | mghigh: $DEFAULT_MGHIGH"
    echo "Server: $SERVER | CUDA Devices: $CUDA_DEVICES"
    echo "----------------------------------------------"

    EXPERIMENT_PRENAME="publish_control/${MODEL_NAME%%/*}_${FOCUS}_ours/control_normalizing_exponential_cutofflatestart/${CONTROL_DISTRIBUTION}"

    CMD="scripts/$SCRIPT \
      --dataset \"$DATASET\" \
      --server \"$SERVER\" \
      --cuda_devices \"$CUDA_DEVICES\" \
      --experiment_prename \"$EXPERIMENT_PRENAME\" \
      --latestart \"$DEFAULT_LATESTART\" \
      --mghigh \"$DEFAULT_MGHIGH\" \
      --model_name \"$MODEL_NAME\" \
      --guidance_control \"1\" \
      --w_max \"$W_MAX\" \
      --w_min \"$W_MIN\" \
      --sample_guidance \"$SAMPLE_GUIDANCE\" \
      --control_distribution \"$CONTROL_DISTRIBUTION\""

    eval "bash $CMD"

    echo "Finished $SCRIPT on $DATASET"
    echo "=============================================="
  done
done
