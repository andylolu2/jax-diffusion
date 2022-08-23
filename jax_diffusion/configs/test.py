from absl import logging

from jax_diffusion.configs.base import get_config as get_base_config
from jax_diffusion.types import Config


def get_config() -> Config:
    config = get_base_config()

    config.effective_steps = 2000

    exp_conf = config.experiment_kwargs.config
    exp_conf.dataset_kwargs.name = "mnist"
    exp_conf.dataset_kwargs.resize_dim = 32
    exp_conf.model.unet_kwargs.dim_init = 4
    exp_conf.model.unet_kwargs.dim_mults = (1, 1, 1, 1)

    return config
