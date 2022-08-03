FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

# disable prompts from tzinfo
ARG DEBIAN_FRONTEND=noninteractive \
    SHELL=/bin/bash

# install python, pip and other os dependencies
RUN apt-get -y update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y python3.7 python3.7-distutils python3-pip && \
    apt-get install -y --no-install-recommends \
    wget bash openssh-server git unzip neovim tmux && \
    apt-get clean && \
    ln -sf python3.7 /usr/bin/python && \
    ln -sf pip3 /usr/bin/pip

# change dir
WORKDIR /root

# install dependencies
RUN python -m pip install --no-cache-dir -U pip && \
    python -m pip install --no-cache-dir "jax[cuda]" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
ADD requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# start script
ADD --chmod=755 container/start.sh .
CMD [ "./start.sh" ]
