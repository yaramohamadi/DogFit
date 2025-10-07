#!/bin/bash

# ========== GLOBAL CONFIGURATION ==========
SERVER="taylor"
CUDA_DEVICES="0,1"
EXPERIMENT_PRENAME="Publish_Results_BASELINES"
MODEL_NAME="SiT-XL/2" # or "SiT-XL/2" 

# ========== DATASET TO SCRIPT MAPPING ==========
declare -a TASKS=(
  "EuroSAT run_baselines_finetune.sh"
)

# "food-101_processed run_baseline_MG.sh"

# ========== EXECUTION LOOP ==========
for TASK in "${TASKS[@]}"; do
  read -r DATASET SCRIPT <<< "$TASK"

  echo "=============================================="
  echo "Running $SCRIPT on dataset $DATASET"
  echo "Server: $SERVER | CUDA Devices: $CUDA_DEVICES | Model: $MODEL_NAME"
  echo "----------------------------------------------"

  bash "scripts/$SCRIPT" \
    --dataset "$DATASET" \
    --server "$SERVER" \
    --cuda_devices "$CUDA_DEVICES" \
    --experiment_prename "$EXPERIMENT_PRENAME" \
    --model_name "$MODEL_NAME"

  echo "Finished $SCRIPT on $DATASET"
  echo "=============================================="
done
