# FROM https://github.com/richzhang/PerceptualSimilarity
#!/bin/bash

# Set the environment variable
export DATASET_PATH="/root/data/dataset/vision_general/bapps"

# Create main dataset directory
mkdir -p "$DATASET_PATH"

# JND Dataset
echo "Downloading and extracting JND Dataset..."
wget https://perceptual-similarity.s3.us-west-2.amazonaws.com/dataset/jnd.tar.gz -O "$DATASET_PATH/jnd.tar.gz"
mkdir -p "$DATASET_PATH/jnd"
tar -xf "$DATASET_PATH/jnd.tar.gz" -C "$DATASET_PATH"
rm "$DATASET_PATH/jnd.tar.gz"

# 2AFC Val set
echo "Downloading and extracting 2AFC Val set..."
mkdir -p "$DATASET_PATH/2afc/val"
wget https://perceptual-similarity.s3.us-west-2.amazonaws.com/dataset/twoafc_val.tar.gz -O "$DATASET_PATH/twoafc_val.tar.gz"
tar -xf "$DATASET_PATH/twoafc_val.tar.gz" -C "$DATASET_PATH/2afc"
rm "$DATASET_PATH/twoafc_val.tar.gz"

# 2AFC Train set
echo "Downloading and extracting 2AFC Train set..."
mkdir -p "$DATASET_PATH/2afc/train"
wget https://perceptual-similarity.s3.us-west-2.amazonaws.com/dataset/twoafc_train.tar.gz -O "$DATASET_PATH/twoafc_train.tar.gz"
tar -xf "$DATASET_PATH/twoafc_train.tar.gz" -C "$DATASET_PATH/2afc"
rm "$DATASET_PATH/twoafc_train.tar.gz"

echo "Dataset setup complete!"
echo "Dataset path is set to: $DATASET_PATH"
