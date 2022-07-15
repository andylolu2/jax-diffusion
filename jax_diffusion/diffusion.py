import functools

import numpy as np


def betas(diffusion_config):
    return _beta_schedule(**diffusion_config)


def alphas(diffusion_config):
    return _alpha_schedule(**diffusion_config)


def alpha_bars(diffusion_config):
    return _alpha_bar_schedule(**diffusion_config)


@functools.lru_cache(maxsize=1)
def _beta_schedule(beta_1: float, beta_T: float, T: int):
    return np.linspace(beta_1, beta_T, T, dtype=np.float32)


@functools.lru_cache(maxsize=1)
def _alpha_schedule(beta_1: float, beta_T: float, T: int):
    return 1 - _beta_schedule(beta_1, beta_T, T)


@functools.lru_cache(maxsize=1)
def _alpha_bar_schedule(beta_1: float, beta_T: float, T: int):
    return np.cumprod(_alpha_schedule(beta_1, beta_T, T))
