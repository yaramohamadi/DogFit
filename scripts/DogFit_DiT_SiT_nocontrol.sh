#!/bin/bash

# ========== GLOBAL CONFIGURATION ==========
SERVER="bool"
CUDA_DEVICES="0,1"
SCRIPT="run_DogFit.sh"
EXPERIMENT_PRENAME="publish_DiT_nocontrol"
MODEL_NAME="DiT-XL/2"  # or "DiT-XL/2"
FOCUS_METRIC="FD_DINOV2"  # or "FID"

# "food-101_processed"
declare -a TASKS=(
  "food-101_processed"
)

# ========== Define (latestart, mghigh) for each focus ==========
declare -A PARAMS_FD_DINOV2
# PARAMS_FD_DINOV2["food-101_processed"]="6000,1"
PARAMS_FD_DINOV2["food-101_processed"]="6000,1"

declare -A PARAMS_FID
# PARAMS_FID["food-101_processed"]="12000,0.5"  # late-start
PARAMS_FID["101_processed"]="12000,0.5" 
# cutoff example: 6000,1 (not used here, but you can modify as needed)

# ========== EXECUTION LOOP ==========
for DATASET in "${TASKS[@]}"; do
  if [ "$FOCUS_METRIC" == "FD_DINOV2" ]; then
    PAIR="${PARAMS_FD_DINOV2[$DATASET]}"
  elif [ "$FOCUS_METRIC" == "FID" ]; then
    PAIR="${PARAMS_FID[$DATASET]}"
  else
    echo "Unsupported FOCUS_METRIC: $FOCUS_METRIC"
    exit 1
  fi

  IFS=',' read -r LATESTART MGHIGH <<< "$PAIR"

  echo "=============================================="
  echo "Running $SCRIPT on $DATASET | latestart: $LATESTART | mghigh: $MGHIGH | prename: $EXPERIMENT_PRENAME | model: $MODEL_NAME"
  echo "Server: $SERVER | CUDA Devices: $CUDA_DEVICES | Focus: $FOCUS_METRIC"
  echo "----------------------------------------------"

  bash scripts/$SCRIPT \
    --dataset "$DATASET" \
    --server "$SERVER" \
    --cuda_devices "$CUDA_DEVICES" \
    --experiment_prename "$EXPERIMENT_PRENAME" \
    --latestart "$LATESTART" \
    --mghigh "$MGHIGH" \
    --model_name "$MODEL_NAME"

  echo "Finished $SCRIPT on $DATASET | latestart: $LATESTART | mghigh: $MGHIGH"
  echo "=============================================="
done
