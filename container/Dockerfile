FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ARG USERNAME=dockeruser
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y software-properties-common && \
    add-apt-repository ppa:git-core/ppa && \
    apt-add-repository ppa:fish-shell/release-3 && \
    apt-add-repository ppa:neovim-ppa/unstable && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt install -y \
        build-essential \
        cmake \
        curl \
        fish \
        git \
        neovim \
        python3.11 \
        python3.11-distutils \
        python3-pip \
        ripgrep \
        sudo \
        tar \
        tmux \
        wget && \
    ln -sf python3.11 /usr/bin/python && \
    ln -sf pip3 /usr/bin/pip

# install dependencies
RUN pip install --no-cache-dir -U pip && \
    pip install -U "jax[cuda12_pip]" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
ADD requirements.txt /tmp/requirements.txt
RUN pip install -U -r /tmp/requirements.txt

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /usr/bin/fish \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

# Setup dotfiles
RUN git clone https://github.com/andylolu2/dotfiles $HOME/.dotfiles && \
    $HOME/.dotfiles/main.fish
