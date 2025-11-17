#!/bin/bash

# Global Constants
TOTAL_STEPS=24000
LOG_EVERY=1000
CKPT_EVERY=24000
SKIP_FIRST_CKPT=1

IMAGE_SIZE=256
MODEL="DiT-XL/2"

BATCH_SIZE=32
VAE="ema"
NUM_WORKERS=4
CFG_SCALE=1.0
NUM_SAMPLE_STEPS=50
NSAMPLE=10000

FID_DEVICE="cuda:0"
NPROC_PER_NODE=2
is_port_in_use() {
  local port=$1
  (echo >/dev/tcp/127.0.0.1/$port) &>/dev/null
}

PORT=$(shuf -i 20000-40000 -n 1)  # 🔧 Initialize first
while is_port_in_use $PORT; do
  PORT=$(shuf -i 20000-40000 -n 1)
done
export PORT
echo "Using MASTER_PORT=$PORT"

# Set paths and dataset details
resolve_dataset_config() {
  case "$DATASET" in
    food-101_processed) NUM_CLASSES=101 ;;
    *) echo "Unknown dataset: $DATASET"; exit 1 ;;
  esac
  DATA_DIR_ZIP="$DATASETS_DIR/food-101.zip"
  REAL_DATA_DIR="$DATASETS_DIR/$DATASET"
  # Normal Generated Directory for Ours and MG
  RESULTS_DIR="$RESULTS_PRE_DIR/$DATASET/$EXPERIMENT_NAME/"
  GENERATED_DIR="$RESULTS_DIR/samples"
  # Generated Directory for baselines
  GENERATED_DIR_CG1="$RESULTS_DIR/samples"
  RESULTS_FILE_CG1="$RESULTS_DIR/results_cg1"
  GENERATED_DIR_CG1_5="$RESULTS_DIR/samples"
  RESULTS_FILE_CG1_5="$RESULTS_DIR/results_cg1_5"
  GENERATED_DIR_DoG1_5="$RESULTS_DIR/samples"
  RESULTS_FILE_DoG1_5="$RESULTS_DIR/results_dog1_5"
  # Till here...
  CHECKPOINT_DIR="$RESULTS_DIR/checkpoints"
  LOG_FILE="$RESULTS_DIR/training_log.txt"
  RESULTS_FILE="$RESULTS_DIR/results"
}

# ==== PATHS ==== (These paths need to be changed)
resolve_server_paths() {
    case "$SERVER" in
        bool)
            conda init
            source ~/.bashrc
            CODE_PRE_DIR="/path/to/main/directory"
            DATASETS_DIR="$CODE_PRE_DIR/datasets" # "/path/to/datasets/" # Can keep this the way it is
            RESULTS_PRE_DIR="$CODE_PRE_DIR/results" # "/path/to/results/directory" # Can keep this the way it is
            ENV_PATH="path/to/python/environment" # Where you want the environment to be created 
            ;;
        *)
            echo "Unknown server: $SERVER" >&2
            exit 1
            ;;
    esac
}

create_environment() {
  echo ">>> Setting up environment..."

  # Ensure conda is properly sourced
  source "$(conda info --base)/etc/profile.d/conda.sh"

  if [ -d "$ENV_PATH" ]; then
    echo "Using existing conda env at $ENV_PATH"
    conda activate "$ENV_PATH"
  else
    echo "Creating new conda env at $ENV_PATH"
    conda create --prefix "$ENV_PATH" python=3.11 -y
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_PATH"
    conda install pytorch torchvision pytorch-cuda=12.6 -c pytorch -c nvidia
    pip install timm diffusers accelerate pytorch-fid torchdiffeq
  fi

  if [ ! -d "dgm-eval" ]; then
    git clone https://github.com/layer6ai-labs/dgm-eval.git
  fi
  pushd dgm-eval
  pip install -e .
  popd
}


prepare_dataset() {

  if [ "$DATASET" == 'food-101_processed' ]; then

    if [ -d "$REAL_DATA_DIR" ] && [ "$(ls -A "$REAL_DATA_DIR")" ]; then
      echo ">>> Dataset already exists at: $REAL_DATA_DIR. Skipping extraction."
    else
      echo ">>> Dataset not found at: $REAL_DATA_DIR. Extracting..."
      if [ ! -f "$DATA_DIR_ZIP" ]; then
        echo ">>> Dataset zip file not found: $DATA_DIR_ZIP. Downloading..."
        if [ ! -d "$DATASETS_DIR" ]; then
          mkdir -p "$DATASETS_DIR"
        fi

        curl -L -o "$DATA_DIR_ZIP" \
          https://www.kaggle.com/api/v1/datasets/download/dansbecker/food-101 || {
          echo "Failed to download dataset zip file."
          exit 1
        }
      fi
      echo ">>> Preparing dataset..."
      TMP_DATASETS_DIR="${DATASETS_DIR}/tmp"
      mkdir -p "$TMP_DATASETS_DIR"
      unzip -qn "$DATA_DIR_ZIP" -d "$TMP_DATASETS_DIR"

      # Move images/ from nested directory to $REAL_DATA_DIR
      INNER_IMAGES_DIR="$TMP_DATASETS_DIR/food-101/food-101/images"
      if [ -d "$INNER_IMAGES_DIR" ]; then
        mkdir -p "$REAL_DATA_DIR"
        mv "$INNER_IMAGES_DIR"/* "$REAL_DATA_DIR"/
      else
        echo "Expected path not found: $INNER_IMAGES_DIR"
        exit 1
      fi

      # Clean up everything else
      rm -rf "$TMP_DATASETS_DIR"
    fi

    find "$REAL_DATA_DIR" -name '._*' -delete
    echo ">>> Dataset prepared at: $REAL_DATA_DIR"
  else
    echo "Unsupported dataset: $DATASET"
    exit 1
  fi
}

# ====================== HELPER ======================
log_and_run() {
    echo ">>> $1"
    shift
    "$@" 2>&1 | tee -a "$LOG_FILE"
}
