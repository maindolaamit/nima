#!/bin/bash
set -e

MODEL_NAME=$1
MODEL_TYPE=$2
WEIGHTS_DIR='/usr/app/nima/weights/'

if [ -z "$MODEL_NAME" ]; then
  echo "Model name not passed"
  exit 1
elif [ -z "$MODEL_TYPE" ]; then
  echo "Model name not passed"
  echo "Setting model type to Aesthetic."
  MODEL_TYPE="aesthetic"
fi

# Note :S3_BUCKET and S3_BUCKET_PATH will be derived from environment variables.
# Note : Images directory will be mounted to /usr/data
# the image directory structure has to be
# /usr
# |-/data/
# |------ava/images/
# |------tid2013/distorted_images/
python -W ignore -m nima.model.train_model -n "$MODEL_NAME" -t $MODEL_TYPE -d /usr/data

# copy train output to s3
aws s3 cp $WEIGHTS_DIR s3://"$S3_BUCKET"/$WEIGHTS_DIR --recursive
