#!/bin/bash
# ====================== CONFIGURATION ======================

# Define CUDA devices here
CUDA_DEVICES="0,1"
DATASET="stanford-cars_processed"  # Options: caltech, birds, etc.
SERVER="taylor"  # Options: taylor, bool, computecanada
EXPERIMENT_PRENAME=""

W_TRAIN_CFG=1.5
USE_GUIDANCE_CUTOFF=0

source scripts/config.sh

# ====================== ARGUMENT PARSING ======================

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --cuda_devices) CUDA_DEVICES="$2"; shift ;;
    --dataset) DATASET="$2"; shift ;;
    --server) SERVER="$2"; shift ;;
    --experiment_prename) EXPERIMENT_PRENAME="$2"; shift ;;
    --wtraincfg) W_TRAIN_CFG="$2"; shift ;;
    --model_name) MODEL="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

EXPERIMENT_NAME="$EXPERIMENT_PRENAME/baseline_mgfinetune_WTRAINCFG$W_TRAIN_CFG"

# Load all logic
resolve_server_paths
resolve_dataset_config

# Define any additional specific parameters here

train_model() {
    log_and_run "Training model..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun --master_port=$PORT --nproc_per_node=$NPROC_PER_NODE train_MG.py \
        --data-path "$REAL_DATA_DIR" \
        --results-dir $RESULTS_DIR \
        --model $MODEL \
        --image-size $IMAGE_SIZE \
        --num-classes $NUM_CLASSES \
        --total-steps $TOTAL_STEPS \
        --log-every $LOG_EVERY \
        --ckpt-every $CKPT_EVERY \
        --global-batch-size $BATCH_SIZE \
        --vae $VAE \
        --num-workers $NUM_WORKERS \
        --w-cg $W_TRAIN_CFG \
        --guidance-cutoff $USE_GUIDANCE_CUTOFF
}

run_sampling() {
    log_and_run "Sampling images..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun --master_port=$PORT --nproc_per_node=$NPROC_PER_NODE sample.py \
        --model $MODEL \
        --vae $VAE \
        --sample-dir "$GENERATED_DIR/$PADDED_STEP" \
        --ckpt "$CHECKPOINT_DIR/$PADDED_CKPT" \
        --per-proc-batch-size $BATCH_SIZE \
        --num-fid-samples $NSAMPLE \
        --image-size $IMAGE_SIZE \
        --num-classes $NUM_CLASSES \
        --cfg-scale $CFG_SCALE \
        --num-sampling-steps $NUM_SAMPLE_STEPS
}

calculate_fid() {
    log_and_run "Calculating FID..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m dgm_eval "$REAL_DATA_DIR" "$GENERATED_DIR/$PADDED_STEP" \
        --model inception \
        --device "$FID_DEVICE" \
        --nsample $NSAMPLE \
        --clean_resize \
        --metrics fd prdc \
        --save \
        --output_dir $RESULTS_FILE

    log_and_run "Calculating FID DINO..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m dgm_eval "$REAL_DATA_DIR" "$GENERATED_DIR/$PADDED_STEP" \
        --model dinov2 \
        --device "$FID_DEVICE" \
        --nsample $NSAMPLE \
        --clean_resize \
        --metrics fd prdc \
        --save \
        --output_dir $RESULTS_FILE
}

# ====================== MAIN ======================
echo ">>> Logging to: $LOG_FILE"
rm -f "$LOG_FILE"
mkdir -p "$(dirname "$LOG_FILE")"

create_environment
prepare_dataset
train_model

for ((i=0; i<=TOTAL_STEPS; i+=CKPT_EVERY)); do
  if [[ $i -eq 0 && "$SKIP_FIRST_CKPT" -eq 1 ]]; then
    continue
  fi
  printf -v PADDED_STEP "%07d" "$i"
  printf -v PADDED_CKPT "%07d.pt" "$i"
  run_sampling
  calculate_fid
done

cleanup_dataset

echo ">>> All tasks completed successfully!"
