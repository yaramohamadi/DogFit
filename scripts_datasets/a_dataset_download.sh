#!/bin/bash


############################# FOOD-101 DATASET #############################
# Define target directory
# TARGET_DIR="/export/datasets/public/diffusion_datasets/food_101-processed"
# 
# # Create the directory if it does not exist
# mkdir -p "$TARGET_DIR"
# 
# # Download the dataset to the target directory
# curl -L -o "$TARGET_DIR/food-101.zip" \
#   https://www.kaggle.com/api/v1/datasets/download/dansbecker/food-101
# 
# # Unzip the dataset in the same directory
# unzip -o "$TARGET_DIR/food-101.zip" -d "$TARGET_DIR"

# pushd /export/datasets/public/diffusion_datasets/food_101-raw/food-101/food-101 > /dev/null
# zip -r /export/datasets/public/diffusion_datasets/food_101-raw/food-101_processed.zip food-101_processed
# popd > /dev/null


############################### DANISH FUNGI DATASET #############################
#TARGET_DIR="/export/datasets/public/diffusion_datasets/fungi"
#mkdir -p "$TARGET_DIR"

#curl -L -o "$TARGET_DIR/DF20-300px.tar.gz" \
#  http://ptak.felk.cvut.cz/plants/DanishFungiDataset/DF20-300px.tar.gz
#tar -xzf "$TARGET_DIR/DF20-300px.tar.gz" -C "$TARGET_DIR"

#curl -L -o "$TARGET_DIR/DF20-metadata.zip" \
#  http://ptak.felk.cvut.cz/plants/DanishFungiDataset/DF20-metadata.zip
#unzip -o "$TARGET_DIR/DF20-metadata.zip" -d "$TARGET_DIR"

#pushd /export/datasets/public/diffusion_datasets/df-20m_processed > /dev/null
#zip -r /export/datasets/public/diffusion_datasets/df-20m_processed/df-20m_processed.zip df-20m_processed
#popd > /dev/null


################################# CALTECH-101 DATASET #############################

# TARGET_DIR="/export/datasets/public/diffusion_datasets/caltech-101"
# mkdir -p "$TARGET_DIR"
# 
# curl -L -o "$TARGET_DIR/caltech-101.zip" \
#   https://www.kaggle.com/api/v1/datasets/download/imbikramsaha/caltech-101
# unzip -o "$TARGET_DIR/caltech-101.zip" -d "$TARGET_DIR"
# Remove the GOOGLE_BACKGROUND class
# ZIP again
# zip -r /export/datasets/public/diffusion_datasets/caltech-101/caltech-101.zip /export/datasets/public/diffusion_datasets/caltech-101/caltech-101
#pushd /export/datasets/public/diffusion_datasets/caltech-101_processed > /dev/null
#zip -r /export/datasets/public/diffusion_datasets/caltech-101_processed.zip caltech-101_processed
#popd > /dev/null


############################### CUB-200-2011 DATASET #############################
# TARGET_DIR="/export/datasets/public/diffusion_datasets/cub-200-2011"
# mkdir -p "$TARGET_DIR"
# 
# curl -L -o "$TARGET_DIR/cub-200-2011.zip" \
#   https://www.kaggle.com/api/v1/datasets/download/wenewone/cub2002011
# unzip -o "$TARGET_DIR/cub-200-2011.zip" -d "$TARGET_DIR"

############################### Stanford Cars DATASET #############################

# TARGET_DIR="/export/datasets/public/diffusion_datasets/stanford-cars"
#mkdir -p "$TARGET_DIR"
# 
#curl -L -o "$TARGET_DIR/stanford-cars.zip" \
#  https://www.kaggle.com/api/v1/datasets/download/eduardo4jesus/stanford-cars-dataset
#unzip -o "$TARGET_DIR/stanford-cars.zip" -d "$TARGET_DIR"

# Get test annotation files
# curl -L -o $TARGET_DIR/stanford-cars_test_anno.zip\
#  https://www.kaggle.com/api/v1/datasets/download/abdelrahmant11/standford-cars-dataset-meta
# unzip -o "$TARGET_DIR/stanford-cars_test_anno.zip" -d "$TARGET_DIR"
# Run the cars_preprocess.py script to get the train split annotation folders
# zip 

# Get metadata for test dataaset too! 
# curl -L -o ~/Downloads/standford-cars-dataset-meta.zip\
#   https://www.kaggle.com/api/v1/datasets/download/abdelrahmant11/standford-cars-dataset-meta
# 
# 
# pushd /export/datasets/public/diffusion_datasets/stanford-cars_processed/ > /dev/null
# zip -r stanford-cars_processed.zip stanford-cars_processed/*
# popd > /dev/null



############################### ART-BENCH-10 DATASET #############################
# TARGET_DIR="/export/datasets/public/diffusion_datasets/artbench10"
# mkdir -p "$TARGET_DIR"
# curl -L https://artbench.eecs.berkeley.edu/files/artbench-10-imagefolder.tar | tar -xvf - -C "$TARGET_DIR"

#pushd /export/datasets/public/diffusion_datasets/artbench-10_processed/ > /dev/null
#zip -r artbench-10_processed.zip artbench-10_processed/*
#popd > /dev/null


################################## FFHQ ##########################################


# TARGET_DIR="/export/datasets/public/diffusion_datasets"
# mkdir -p "$TARGET_DIR"
# curl -L -o "$TARGET_DIR/ffhq256.zip" \
#   https://www.kaggle.com/api/v1/datasets/download/denislukovnikov/ffhq256-images-only
# unzip -o "$TARGET_DIR/ffhq256.zip" -d "$TARGET_DIR"


#!/bin/bash
TARGET_DIR="datasets"
# mkdir -p "$TARGET_DIR"
# curl -L -o $TARGET_DIR/eurosat-dataset.zip\
#   https://www.kaggle.com/api/v1/datasets/download/apollo2506/eurosat-dataset
unzip -o "$TARGET_DIR/eurosat-dataset.zip" -d "$TARGET_DIR"
