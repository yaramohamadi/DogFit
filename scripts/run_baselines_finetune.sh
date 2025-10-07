#!/bin/bash
# ====================== DEFAULT CONFIGURATION ======================

CUDA_DEVICES="0,1"
DATASET="food-101_processed"  # Options: caltech, birds, etc.
SERVER="taylor"  # Options: taylor, bool, computecanada
EXPERIMENT_PRENAME=""
DROPOUT_RATIO=0.1

W_DOG=1.5
W_CFG=1.5

# Load all logic
source scripts/config.sh

# ====================== ARGUMENT PARSING ======================

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --cuda_devices) CUDA_DEVICES="$2"; shift ;;
    --dataset) DATASET="$2"; shift ;;
    --server) SERVER="$2"; shift ;;
    --experiment_prename) EXPERIMENT_PRENAME="$2"; shift ;;
    --checkpoint_dir) CHECKPOINT_DIR="$2"; shift ;;
    --wdog) W_DOG="$2"; shift ;;
    --wcfg) W_CFG="$2"; shift ;;
    --model_name) MODEL="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

EXPERIMENT_NAME="$EXPERIMENT_PRENAME/baselines_finetune_W_CFG${W_CFG}"

resolve_server_paths
resolve_dataset_config

# Define any additional specific parameters here
# ...

train_model() {
    log_and_run "Training model..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun --master_port=$PORT --nproc_per_node=$NPROC_PER_NODE train.py \
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
        --dropout-ratio $DROPOUT_RATIO
}
sample_CG1() {
    log_and_run "Sampling images for CG1..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun --master_port=$PORT --nproc_per_node=$NPROC_PER_NODE sample.py \
        --model "$MODEL" --vae "$VAE" \
        --sample-dir "$GENERATED_DIR/$PADDED_STEP" \
        --ckpt "$CHECKPOINT_DIR/$PADDED_CKPT" \
        --per-proc-batch-size "$BATCH_SIZE" --num-fid-samples "$NSAMPLE" \
        --image-size "$IMAGE_SIZE" --num-classes "$NUM_CLASSES" \
        --cfg-scale 1 --num-sampling-steps "$NUM_SAMPLE_STEPS" \
        --dropout-ratio $DROPOUT_RATIO
}

fid_CG1() {
    log_and_run "Calculating FID for CG1..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m dgm_eval "$REAL_DATA_DIR" "$GENERATED_DIR/$PADDED_STEP" \
        --model inception --device "$FID_DEVICE" --nsample $NSAMPLE --clean_resize \
        --metrics fd prdc kd --save --output_dir "$RESULTS_FILE_CG1"

    log_and_run "Calculating Dino FID for CG1..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m dgm_eval "$REAL_DATA_DIR" "$GENERATED_DIR/$PADDED_STEP" \
        --model dinov2 --device "$FID_DEVICE" --nsample $NSAMPLE --clean_resize \
        --metrics fd prdc kd --save --output_dir "$RESULTS_FILE_CG1"
}

sample_CG1_5() {
    log_and_run "Sampling images for CG1.5..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun --master_port=$PORT --nproc_per_node=$NPROC_PER_NODE sample.py \
        --model "$MODEL" --vae "$VAE" \
        --sample-dir "$GENERATED_DIR/$PADDED_STEP" \
        --ckpt "$CHECKPOINT_DIR/$PADDED_CKPT" \
        --per-proc-batch-size "$BATCH_SIZE" --num-fid-samples "$NSAMPLE" \
        --image-size "$IMAGE_SIZE" --num-classes "$NUM_CLASSES" \
        --cfg-scale "$W_CFG" --num-sampling-steps "$NUM_SAMPLE_STEPS" \
        --dropout-ratio $DROPOUT_RATIO
}

fid_CG1_5() {
    log_and_run "Calculating FID for CG1.5..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m dgm_eval "$REAL_DATA_DIR" "$GENERATED_DIR/$PADDED_STEP" \
        --model inception --device "$FID_DEVICE" --nsample $NSAMPLE --clean_resize \
        --metrics fd prdc kd --save --output_dir "$RESULTS_FILE_CG1_5"

    log_and_run "Calculating Dino FID for CG1.5..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m dgm_eval "$REAL_DATA_DIR" "$GENERATED_DIR/$PADDED_STEP" \
        --model dinov2 --device "$FID_DEVICE" --nsample $NSAMPLE --clean_resize \
        --metrics fd prdc kd --save --output_dir "$RESULTS_FILE_CG1_5"
}

sample_DoG1_5() {
    log_and_run "Sampling images for DoG1.5..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun --master_port=$PORT --nproc_per_node=$NPROC_PER_NODE sample_DoG.py \
        --model "$MODEL" --vae "$VAE" \
        --sample-dir "$GENERATED_DIR/$PADDED_STEP" \
        --ckpt "$CHECKPOINT_DIR/$PADDED_CKPT" \
        --per-proc-batch-size "$BATCH_SIZE" --num-fid-samples "$NSAMPLE" \
        --image-size "$IMAGE_SIZE" --num-classes "$NUM_CLASSES" \
        --cfg-scale "$W_DOG" --num-sampling-steps "$NUM_SAMPLE_STEPS" \
        --dropout-ratio $DROPOUT_RATIO
}

fid_DoG1_5() {
    log_and_run "Calculating FID for DoG1.5..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m dgm_eval "$REAL_DATA_DIR" "$GENERATED_DIR/$PADDED_STEP" \
        --model inception --device "$FID_DEVICE" --nsample $NSAMPLE --clean_resize \
        --metrics fd prdc kd --save --output_dir "$RESULTS_FILE_DoG1_5"

    log_and_run "Calculating Dino FID for DoG1.5..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m dgm_eval "$REAL_DATA_DIR" "$GENERATED_DIR/$PADDED_STEP" \
        --model dinov2 --device "$FID_DEVICE" --nsample $NSAMPLE --clean_resize \
        --metrics fd prdc kd --save --output_dir "$RESULTS_FILE_DoG1_5"
}

# ====================== MAIN ======================

echo ">>> Logging to: $LOG_FILE"
rm -f "$LOG_FILE"
mkdir -p "$(dirname "$LOG_FILE")"

create_environment
prepare_dataset

if [[ "$DATASET" == "ffhq256" ]]; then
    DROPOUT_RATIO=0

    # TMP
    sample_DoG1_5
    fid_DoG1_5
    exit 0
fi

train_model

for ((i=0; i<=TOTAL_STEPS; i+=CKPT_EVERY)); do
    if [[ $i -eq 0 && "$SKIP_FIRST_CKPT" -eq 1 ]]; then
        continue
    fi
    printf -v PADDED_CKPT "%07d.pt" "$i"

    printf -v PADDED_STEP "%07d_cg1" "$i"
    sample_CG1
    fid_CG1

    printf -v PADDED_STEP "%07d_dog1_5" "$i"
    sample_DoG1_5
    fid_DoG1_5

    if [[ "$DATASET" == "ffhq256" ]]; then
        continue
    fi

    printf -v PADDED_STEP "%07d_cg1_5" "$i"
    sample_CG1_5
    fid_CG1_5

    
done

cleanup_dataset

echo ">>> All tasks completed successfully!"