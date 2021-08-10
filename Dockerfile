# Pass the base container tag as argument --build-arg TAG=
# default argument if not provided will be latest - for cpu
# pass TAG=latest-gpu for gpu tensorflow version
ARG TAG=latest
ARG BASE_CONTAINER=tensorflow/tensorflow:$TAG
RUN echo "Base container $BASE_CONTAINER"

FROM $BASE_CONTAINER

# Maintainer
LABEL version=0.1
MAINTAINER Amit Maindola <maindola.amit@gmail.com>

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      wget && \
    rm -rf /var/lib/apt/lists/*

# setup directory
RUN mkdir -p /usr/app
#
# Mount the dataset to this folder
RUN mkdir -p /usr/data 
# Set the working directory
WORKDIR /opt/usr/

# Copy source code
COPY nima /usr/app/nima
COPY entrypoints /usr/app/entrypoints

RUN pip install -r requirements.txt

ENV PYTHONPATH='/src/:$PYTHONPATH'

# set the entrypoints, default is to train model 
# User can override it for prediction by passing --entrypoint entrypoints/entrypoint.predict.sh
ENTRYPOINT ["entrypoints/train_model.sh"]
