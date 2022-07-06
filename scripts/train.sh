#! /bin/bash


TF_FORCE_GPU_ALLOW_GROWTH=true \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
python -m jax_diffusion.train