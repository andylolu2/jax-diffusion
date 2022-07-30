from __future__ import annotations

from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from jax_diffusion.types import Config, Params, Rng, ndarray


def forward_process(diffusion_config: Config, x_0: ndarray, t: ndarray, rng: Rng):
    """See algorithm 1 in https://arxiv.org/pdf/2006.11239.pdf"""
    alpha_t_bar = jnp.asarray(alpha_bars(**diffusion_config))[t]
    alpha_t_bar = jnp.expand_dims(alpha_t_bar, tuple(range(1, x_0.ndim)))

    eps = jax.random.normal(rng, shape=x_0.shape, dtype=x_0.dtype)
    x_t = (alpha_t_bar**0.5) * x_0 + ((1 - alpha_t_bar) ** 0.5) * eps
    return x_t, eps


def forward_random(diffusion_config: Config, x_0: ndarray, rng: Rng):
    rng1, rng2 = random.split(rng)
    t = jax.random.randint(rng1, (len(x_0),), 0, diffusion_config.T)
    x_t, eps = forward_process(diffusion_config, x_0, t, rng2)
    t = t.astype(x_t.dtype)[:, None]
    return x_t, t, eps


@partial(jax.jit, static_argnums=(0, 1))
def backward_process(
    diffusion_config: Config,
    eps_fn,
    params: Params,
    x_t: ndarray,
    T: int,
    steps: int,
    rng: random.KeyArray,
):
    """See algorithm 2 in https://arxiv.org/pdf/2006.11239.pdf"""
    init_val = {"x": x_t, "rng": rng}
    a = jnp.asarray(alphas(**diffusion_config))
    a_bar = jnp.asarray(alpha_bars(**diffusion_config))
    b = jnp.asarray(betas(**diffusion_config))

    def body_fun(i: int, val):
        x, rng = val["x"], val["rng"]
        t = T - i - 1
        alpha_t = a[t]
        alpha_t_bar = a_bar[t]
        sigma_t = b[t] ** 0.5

        rng, rng_next = jax.random.split(rng)
        z = jax.lax.cond(
            pred=t > 0,
            true_fun=lambda: jax.random.normal(rng, shape=x.shape, dtype=x.dtype),
            false_fun=lambda: jnp.zeros_like(x),
        )

        t_input = jnp.full((x.shape[0], 1), t, dtype=x.dtype)
        eps = eps_fn(params, x, t_input, train=False)

        x = (1 / alpha_t**0.5) * (
            x - ((1 - alpha_t) / (1 - alpha_t_bar) ** 0.5) * eps
        ) + sigma_t * z

        return {"x": x, "rng": rng_next}

    x = jax.lax.fori_loop(
        lower=0,
        upper=steps,
        body_fun=body_fun,
        init_val=init_val,
    )["x"]

    return x


@lru_cache(maxsize=1)
def betas(beta_1: float, beta_T: float, T: int):
    return np.linspace(beta_1, beta_T, T, dtype=np.float32)


@lru_cache(maxsize=1)
def alphas(beta_1: float, beta_T: float, T: int):
    return 1 - betas(beta_1, beta_T, T)


@lru_cache(maxsize=1)
def alpha_bars(beta_1: float, beta_T: float, T: int):
    return np.cumprod(alphas(beta_1, beta_T, T))
