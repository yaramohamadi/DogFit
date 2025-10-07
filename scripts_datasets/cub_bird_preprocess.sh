# Step 1: Define variables
SRC_DIR="/export/datasets/public/diffusion_datasets/cub-200-2011/CUB_200_2011/images"
DST_PARENT_DIR="/export/datasets/public/diffusion_datasets/cub-200-2011"
DST_DIR="$DST_PARENT_DIR/cub-200-2011_processed"
ZIP_FILE="$DST_PARENT_DIR/cub-200-2011_proccessed.zip"  # Note: 'processed' is misspelled in 'proccessed.zip' as per your request

# Step 2: Copy and rename the directory
cp -r "$SRC_DIR" "$DST_DIR"

# Step 3: Zip the copied directory
cd "$DST_PARENT_DIR"
zip -r "cub-200-2011_proccessed.zip" "cub-200-2011_processed"