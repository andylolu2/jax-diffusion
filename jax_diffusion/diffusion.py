from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import random

from jax_diffusion.types import Config, Params, Rng, ndarray


class Diffuser:
    def __init__(self, eps_fn, diffusion_config: Config):
        self.eps_fn = eps_fn
        self.config = diffusion_config
        self.betas = jnp.asarray(self._betas(**diffusion_config))
        self.alphas = jnp.asarray(self._alphas(self.betas))
        self.alpha_bars = jnp.asarray(self._alpha_bars(self.alphas))

    @property
    def steps(self) -> int:
        return self.config.T

    def timesteps(self, steps: int):
        timesteps = jnp.arange(0, self.steps, self.steps // steps)
        timesteps = timesteps.at[-1].set(self.steps - 1)
        return timesteps[::-1]

    @partial(jax.jit, static_argnums=(0,))
    def forward(self, x_0: ndarray, rng: Rng):
        """See algorithm 1 in https://arxiv.org/pdf/2006.11239.pdf

        This function should be `jax.jit`-ed.
        """
        rng1, rng2 = random.split(rng)
        t = random.randint(rng1, (len(x_0), 1), 0, self.steps)
        x_t, eps = self.sample_q(x_0, t, rng2)
        t = t.astype(x_t.dtype)
        return x_t, t, eps

    def sample_q(self, x_0: ndarray, t: ndarray, rng: Rng):
        """Samples x_t given x_0 by the q(x_t|x_0) formula."""
        # (bs, 1)
        alpha_t_bar = self.alpha_bars[t]
        # (bs, 1, 1, 1)
        alpha_t_bar = jnp.expand_dims(alpha_t_bar, (1, 2))

        eps = random.normal(rng, shape=x_0.shape, dtype=x_0.dtype)
        x_t = (alpha_t_bar**0.5) * x_0 + ((1 - alpha_t_bar) ** 0.5) * eps
        return x_t, eps

    @partial(jax.jit, static_argnums=(0,))
    def ddpm_backward_step(self, params: Params, x_t: ndarray, t: int, rng: Rng):
        """See algorithm 2 in https://arxiv.org/pdf/2006.11239.pdf"""
        alpha_t = self.alphas[t]
        alpha_t_bar = self.alpha_bars[t]
        sigma_t = self.betas[t] ** 0.5

        z = (t > 0) * random.normal(rng, shape=x_t.shape, dtype=x_t.dtype)
        eps = self.eps_fn(params, x_t, t, train=False)

        x = (1 / alpha_t**0.5) * (
            x_t - ((1 - alpha_t) / (1 - alpha_t_bar) ** 0.5) * eps
        ) + sigma_t * z

        return x

    def ddpm_backward(self, params: Params, x_T: ndarray, rng: Rng) -> ndarray:
        x = x_T

        for t in range(self.steps - 1, -1, -1):
            rng, rng_ = random.split(rng)
            x = self.ddpm_backward_step(params, x, t, rng_)

        return x

    @partial(jax.jit, static_argnums=(0,))
    def ddim_backward_step(
        self, params: Params, x_t: ndarray, t: ndarray, t_next: ndarray
    ):
        """See section 4.1 and C.1 in https://arxiv.org/pdf/2010.02502.pdf
        
        Note: alpha in the DDIM paper is actually alpha_bar in DDPM paper
        """
        alpha_t = self.alpha_bars[t]
        alpha_t_next = self.alpha_bars[t_next]

        eps = self.eps_fn(params, x_t, t, train=False)

        x_0 = (x_t - (1 - alpha_t) ** 0.5 * eps) / alpha_t**0.5
        x_t_direction = (1 - alpha_t_next) ** 0.5 * eps
        x_t_next = alpha_t_next**0.5 * x_0 + x_t_direction

        return x_t_next

    def ddim_backward(self, params: Params, x_T: ndarray, steps: int):
        x = x_T

        ts = self.timesteps(steps)
        for t, t_next in zip(ts[:-1], ts[1:]):
            x = self.ddim_backward_step(params, x, t, t_next)

        return x

    @classmethod
    def _betas(cls, beta_1: float, beta_T: float, T: int) -> ndarray:
        return jnp.linspace(beta_1, beta_T, T, dtype=jnp.float32)

    @classmethod
    def _alphas(cls, betas) -> ndarray:
        return 1 - betas

    @classmethod
    def _alpha_bars(cls, alphas) -> ndarray:
        return jnp.cumprod(alphas)

    @staticmethod
    def expand_t(t: int, x: ndarray):
        return jnp.full((len(x), 1), t, dtype=x.dtype)
