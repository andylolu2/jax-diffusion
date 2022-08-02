FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

# disable prompts from tzinfo
ARG DEBIAN_FRONTEND=noninteractive

# install python and pip
RUN apt-get -y update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa
RUN apt install python3.7 python3.7-distutils python3-pip -y && \
    ln -sf python3.7 /usr/bin/python && \
    ln -sf pip3 /usr/bin/pip

# add user
ARG USER=jax
RUN useradd -ms /bin/bash ${USER}
USER ${USER}
WORKDIR /home/${USER}
ENV PATH="/home/${USER}/.local/bin:${PATH}"

# install dependencies
RUN python -m pip install --no-cache-dir -U pip && \
    python -m pip install --no-cache-dir "jax[cuda]" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
