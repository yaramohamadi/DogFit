#!/bin/bash

# ====================== DEFAULT CONFIGURATION ======================

CUDA_DEVICES="0,1"
DATASET="cub-200-2011_processed"
SERVER="bool"
EXPERIMENT_PRENAME=""
USE_GUIDANCE_CUTOFF=1
MG_HIGH=1
LATE_START=0

GUIDANCE_CONTROL=0
W_MAX=1.0
W_MIN=1.0
SAMPLE_GUIDANCE=1.5

W_TRAIN_DOG=1.5
DROPOUT_RATIO=0 
CONTROL_DISTRIBUTION="uniform"
ZERO_NORM_VARIANCE="011"

# Load all logic
source scripts/config.sh

# ====================== ARGUMENT PARSING ======================

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --cuda_devices) CUDA_DEVICES="$2"; shift ;;
    --dataset) DATASET="$2"; shift ;;
    --server) SERVER="$2"; shift ;;
    --experiment_prename) EXPERIMENT_PRENAME="$2"; shift ;;
    --latestart) LATE_START="$2"; shift ;;
    --mghigh) MG_HIGH="$2"; shift ;;
    --wtraindog) W_TRAIN_DOG="$2"; shift ;;
    --model_name) MODEL="$2"; shift ;;
    --guidance_control) GUIDANCE_CONTROL="$2"; shift ;;
    --w_max) W_MAX="$2"; shift ;;
    --w_min) W_MIN="$2"; shift ;;
    --sample_guidance) SAMPLE_GUIDANCE="$2"; shift ;;
    --control_distribution) CONTROL_DISTRIBUTION="$2"; shift ;;
    --zero_norm_variance) ZERO_NORM_VARIANCE="$2"; shift ;;

    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

EXPERIMENT_NAME="$EXPERIMENT_PRENAME/dogfinetune_LATE_START_ITER${LATE_START}_MG${MG_HIGH}_W_TRAIN_DOG${W_TRAIN_DOG}_control${GUIDANCE_CONTROL}_W_MIN${W_MIN}_W_MAX${W_MAX}"

resolve_server_paths
resolve_dataset_config

# Define any additional specific parameters here

train_model() {
    log_and_run "Training model..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun --master_port=$PORT --nproc_per_node=$NPROC_PER_NODE train_DogFit.py \
        --data-path "$REAL_DATA_DIR" \
        --results-dir "$RESULTS_DIR" \
        --model "$MODEL" \
        --image-size "$IMAGE_SIZE" \
        --num-classes "$NUM_CLASSES" \
        --total-steps "$TOTAL_STEPS" \
        --log-every "$LOG_EVERY" \
        --ckpt-every "$CKPT_EVERY" \
        --global-batch-size "$BATCH_SIZE" \
        --vae "$VAE" \
        --num-workers "$NUM_WORKERS" \
        --w-dog "$W_TRAIN_DOG" \
        --guidance-cutoff "$USE_GUIDANCE_CUTOFF" \
        --mg-high "$MG_HIGH" \
        --dropout-ratio "$DROPOUT_RATIO" \
        --late-start-iter "$LATE_START" \
        --guidance-control "$GUIDANCE_CONTROL" \
        --w-max "$W_MAX" \
        --w-min "$W_MIN" \
        --control-distribution "$CONTROL_DISTRIBUTION" \
        --zero-norm-variance "$ZERO_NORM_VARIANCE" 
}

run_sampling() {
    log_and_run "Sampling images..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun --master_port=$PORT --nproc_per_node=$NPROC_PER_NODE sample.py \
        --model "$MODEL" \
        --vae "$VAE" \
        --sample-dir "$GENERATED_DIR/$PADDED_STEP" \
        --ckpt "$CHECKPOINT_DIR/$PADDED_CKPT" \
        --per-proc-batch-size "$BATCH_SIZE" \
        --num-fid-samples "$NSAMPLE" \
        --image-size "$IMAGE_SIZE" \
        --num-classes "$NUM_CLASSES" \
        --cfg-scale "$CFG_SCALE" \
        --num-sampling-steps "$NUM_SAMPLE_STEPS" \
        --dropout-ratio "$DROPOUT_RATIO" \
        --guidance-control "$GUIDANCE_CONTROL" \
        --w-dgft "$SAMPLE_GUIDANCE" \
        --zero-norm-variance "$ZERO_NORM_VARIANCE" 
}

calculate_fid() {
    log_and_run "Calculating FID DINO..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m dgm_eval "$REAL_DATA_DIR" "$GENERATED_DIR/$PADDED_STEP" \
        --model dinov2 \
        --device "$FID_DEVICE" \
        --nsample "$NSAMPLE" \
        --clean_resize \
        --metrics fd prdc kd \
        --save \
        --output_dir "$RESULTS_FILE"

    log_and_run "Calculating FID..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m dgm_eval "$REAL_DATA_DIR" "$GENERATED_DIR/$PADDED_STEP" \
        --model inception \
        --device "$FID_DEVICE" \
        --nsample "$NSAMPLE" \
        --clean_resize \
        --metrics fd prdc kd \
        --save \
        --output_dir "$RESULTS_FILE"
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

  if (( $(echo "$GUIDANCE_CONTROL > 0" | bc -l) )); then
    if (( $(echo "$SAMPLE_GUIDANCE == 0" | bc -l) )); then
      GUIDANCE_VALUES=(1 1.5 2 3 4 5)
      for SG in "${GUIDANCE_VALUES[@]}"; do
        SAMPLE_GUIDANCE=$SG
        RESULTS_FILE_ORIG="$RESULTS_FILE"
        GENERATED_DIR_ORIG="$GENERATED_DIR"
        RESULTS_FILE="${RESULTS_FILE_ORIG}_w_dgft${SAMPLE_GUIDANCE}"
        GENERATED_DIR="${GENERATED_DIR_ORIG}_w_dgft${SAMPLE_GUIDANCE}"

        run_sampling
        calculate_fid
      done
    else
      RESULTS_FILE="${RESULTS_FILE}_w_dgft${SAMPLE_GUIDANCE}"
      GENERATED_DIR="${GENERATED_DIR}_w_dgft${SAMPLE_GUIDANCE}"
      run_sampling
      calculate_fid
    fi
  else
    run_sampling
    calculate_fid
  fi
done

echo ">>> All tasks completed successfully!"