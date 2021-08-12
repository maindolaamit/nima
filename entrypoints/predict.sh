#!/bin/bash
set -e

MODEL_NAME=$1
WEIGHTS_PATH=$2
IMG_DIR=$3

if [ -z "$MODEL_NAME" ]; then
  echo "Model name not passed"
  exit 1
elif [ -z "$IMG_DIR" ]; then
  echo "Test images directory not passed"
  exit 1
elif [ -z "$WEIGHTS_PATH" ]; then
  echo "Model weights path not passed"
  exit 1
fi

# Note :S3_BUCKET and S3_BUCKET_PATH will be derived from environment variables.
# Note : Images directory will be mounted to /usr/data
# the image directory structure has to be
# /usr
# |-/data/
# |------ava/images/
# |------tid2013/distorted_images/
# |-/test/


# start predicting, mount the image directory to /usr/data
python -W ignore -m nima.evaluate.predict -n "$MODEL_NAME" \
  -w "$WEIGHTS_PATH" \
  -d "$IMG_DIR"

# copy predictions result to S3 bucket path
aws s3 cp /nima/evaluate/results s3://"$S3_BUCKET"/"$S3_BUCKET_PATH" --recursive
