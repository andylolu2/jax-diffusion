from typing import Collection, Sequence

import flax.linen as nn
import jax.numpy as jnp
from jax import random


class UpSample(nn.Module):
    dim: int
    kernel_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.ConvTranspose(
            features=self.dim,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(2, 2),
        )(x)
        return x


class DownSample(nn.Module):
    dim: int
    kernel_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            features=self.dim,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(2, 2),
        )(x)
        return x


class SinusoidalPosEmbedding(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, pos):
        """Refer to https://arxiv.org/pdf/1706.03762.pdf#subsection.3.5"""
        batch_size = pos.shape[0]

        assert self.dim % 2 == 0, self.dim
        assert pos.shape == (batch_size, 1), pos.shape

        d_model = self.dim // 2
        i = jnp.arange(d_model)[None, :]

        pos_embedding = jnp.exp(-(2 * i / d_model) * jnp.log(10000))
        pos_embedding = pos * jnp.exp(-(2 * i / d_model) * jnp.log(10000))
        pos_embedding = jnp.concatenate(
            (jnp.sin(pos_embedding), jnp.cos(pos_embedding)), axis=-1
        )

        assert pos_embedding.shape == (batch_size, self.dim), pos_embedding.shape

        return pos_embedding


class TimeEmbedding(nn.Module):
    dim: int
    sinusoidal_embed_dim: int

    @nn.compact
    def __call__(self, time):
        x = SinusoidalPosEmbedding(self.sinusoidal_embed_dim)(time)
        x = nn.Dense(self.dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.dim)(x)
        return x


class Block(nn.Module):
    dim: int
    kernel_size: int
    num_groups: int

    @nn.compact
    def __call__(self, x, scale_shift=None):
        x = nn.Conv(self.dim, kernel_size=(self.kernel_size, self.kernel_size))(x)
        x = nn.GroupNorm(self.num_groups)(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (1 + scale) + shift

        x = nn.silu(x)

        return x


class ResnetBlock(nn.Module):
    dim: int
    kernel_size: int
    num_groups: int

    @nn.compact
    def __call__(self, x, time_emb=None):
        """
        Args:
            x: array of shape `[batch_size, width, height, d]`
        """
        h = Block(self.dim, self.kernel_size, self.num_groups)(x)

        scale_shift = None
        if time_emb is not None:
            time_emb = nn.silu(time_emb)

            scale = nn.DenseGeneral(self.dim)(time_emb)
            scale = jnp.expand_dims(scale, axis=(1, 2))

            shift = nn.DenseGeneral(self.dim)(time_emb)
            shift = jnp.expand_dims(shift, axis=(1, 2))

            scale_shift = (scale, shift)

        h = Block(self.dim, self.kernel_size, self.num_groups)(
            h, scale_shift=scale_shift
        )

        if x.shape[-1] != self.dim:
            x = nn.Conv(self.dim, kernel_size=(1, 1))(x)

        x = x + h
        return x


class ResidualAttentionBlock(nn.Module):
    dim: int
    num_heads: int
    num_groups: int

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: array of shape `[batch_size, width, height, dim]`
        """

        res = x
        b, w, h, d = res.shape
        res = nn.GroupNorm(self.num_groups)(res)
        res = jnp.reshape(res, (b, w * h, d))
        res = nn.SelfAttention(num_heads=self.num_heads)(res)
        res = jnp.reshape(res, (b, w, h, self.dim))

        x = x + res
        return x


class UNet(nn.Module):
    dim_init: int
    kernel_size_init: int
    dim_mults: Sequence[int]

    attention_dim_mults: Collection[int]
    attention_num_heads: int

    sinusoidal_embed_dim: int
    time_embed_dim: int

    kernel_size: int
    num_groups: int

    @nn.compact
    def __call__(self, x, time):
        channels = x.shape[-1]

        t = TimeEmbedding(self.time_embed_dim, self.sinusoidal_embed_dim)(time)

        x = nn.Conv(self.dim_init, (self.kernel_size_init, self.kernel_size_init))(x)

        hs = []

        # downsample
        for i, dim_mult in enumerate(self.dim_mults):
            is_last = i == len(self.dim_mults) - 1
            dim = self.dim_init * dim_mult

            x = ResnetBlock(dim, self.kernel_size, self.num_groups)(x, t)
            x = ResnetBlock(dim, self.kernel_size, self.num_groups)(x, t)

            if dim_mult in self.attention_dim_mults:
                # apply attention at certain levels of resolutions
                x = ResidualAttentionBlock(
                    dim, self.attention_num_heads, self.num_groups
                )(x)

            hs.append(x)

            if not is_last:
                x = DownSample(dim, self.kernel_size)(x)

        # middle
        dim_mid = self.dim_init * self.dim_mults[-1]
        x = ResnetBlock(dim_mid, self.kernel_size, self.num_groups)(x, t)
        x = ResidualAttentionBlock(dim_mid, self.attention_num_heads, self.num_groups)(
            x
        )
        x = ResnetBlock(dim_mid, self.kernel_size, self.num_groups)(x, t)

        # upsample
        for i, dim_mult in enumerate(reversed(self.dim_mults)):
            is_last = i == len(self.dim_mults) - 1
            dim = self.dim_init * dim_mult

            # concatenate by last (channel) dimension
            x = jnp.concatenate((x, hs.pop()), axis=-1)
            x = ResnetBlock(dim, self.kernel_size, self.num_groups)(x, t)
            x = ResnetBlock(dim, self.kernel_size, self.num_groups)(x, t)

            if dim_mult in self.attention_dim_mults:
                # apply attention at certain levels of resolutions
                x = ResidualAttentionBlock(
                    dim, self.attention_num_heads, self.num_groups
                )(x)

            if not is_last:
                x = UpSample(dim, self.kernel_size)(x)

        # final
        x = ResnetBlock(self.dim_init, self.kernel_size, self.num_groups)(x)
        x = nn.Conv(channels, kernel_size=(1, 1))(x)
        return x


if __name__ == "__main__":
    embed = SinusoidalPosEmbedding(dim=30)
    ts = jnp.arange(5)[:, None]

    params = embed.init(random.PRNGKey(0), ts)
    pos = embed.apply(params, ts)
    print(ts)
    print(pos)