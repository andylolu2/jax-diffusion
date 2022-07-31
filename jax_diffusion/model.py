from functools import partial
from typing import Collection, Sequence

import flax.linen as nn
import jax.numpy as jnp


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
    dropout: float

    @nn.compact
    def __call__(self, x, deterministic: bool, *, scale_shift=None):
        x = nn.GroupNorm(self.num_groups)(x)
        x = nn.silu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        x = nn.Conv(self.dim, kernel_size=(self.kernel_size, self.kernel_size))(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (1 + scale) + shift

        return x


class ResnetBlock(nn.Module):
    dim: int
    kernel_size: int
    num_groups: int
    dropout: float

    @nn.compact
    def __call__(self, x, deterministic: bool, *, time_emb=None):
        """
        Args:
            x: array of shape `[batch_size, width, height, d]`
        """
        h = Block(self.dim, self.kernel_size, self.num_groups, 0.0)(x, deterministic)

        scale_shift = None
        if time_emb is not None:
            time_emb = nn.silu(time_emb)

            scale = nn.DenseGeneral(self.dim)(time_emb)
            scale = jnp.expand_dims(scale, axis=(1, 2))

            shift = nn.DenseGeneral(self.dim)(time_emb)
            shift = jnp.expand_dims(shift, axis=(1, 2))

            scale_shift = (scale, shift)

        h = Block(self.dim, self.kernel_size, self.num_groups, self.dropout)(
            h, deterministic, scale_shift=scale_shift
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
    kernel_size: int
    dim_mults: Sequence[int]

    attention_resolutions: Collection[int]
    attention_num_heads: int
    num_res_blocks: int

    sinusoidal_embed_dim: int
    time_embed_dim: int

    num_groups: int
    dropout: float

    @nn.compact
    def __call__(self, x, time, train: bool):
        channels = x.shape[-1]

        t = TimeEmbedding(self.time_embed_dim, self.sinusoidal_embed_dim)(time)

        x = nn.Conv(self.dim_init, (self.kernel_size, self.kernel_size))(x)

        make_res = partial(
            ResnetBlock,
            kernel_size=self.kernel_size,
            num_groups=self.num_groups,
            dropout=self.dropout,
        )

        hs = [x]

        # downsample
        for i, dim_mult in enumerate(self.dim_mults):
            is_last = i == len(self.dim_mults) - 1
            dim = self.dim_init * dim_mult

            for _ in range(self.num_res_blocks):
                x = make_res(dim)(x, not train, time_emb=t)
                if x.shape[1] in self.attention_resolutions:
                    # apply attention at certain levels of resolutions
                    x = ResidualAttentionBlock(
                        dim, self.attention_num_heads, self.num_groups
                    )(x)

                hs.append(x)

            if not is_last:
                x = DownSample(dim, self.kernel_size)(x)
                hs.append(x)

        # middle
        dim_mid = self.dim_init * self.dim_mults[-1]
        x = make_res(dim_mid)(x, not train, time_emb=t)
        x = ResidualAttentionBlock(dim_mid, self.attention_num_heads, self.num_groups)(
            x
        )
        x = make_res(dim_mid)(x, not train, time_emb=t)

        # upsample
        for i, dim_mult in enumerate(reversed(self.dim_mults)):
            is_last = i == len(self.dim_mults) - 1
            dim = self.dim_init * dim_mult

            for _ in range(self.num_res_blocks + 1):
                # concatenate by last (channel) dimension
                x = jnp.concatenate((x, hs.pop()), axis=-1)
                x = make_res(dim)(x, not train, time_emb=t)
                if x.shape[1] in self.attention_resolutions:
                    # apply attention at certain levels of resolutions
                    x = ResidualAttentionBlock(
                        dim, self.attention_num_heads, self.num_groups
                    )(x)

            if not is_last:
                x = UpSample(dim, self.kernel_size)(x)

        assert not hs

        # final
        x = make_res(self.dim_init)(x, not train, time_emb=t)
        x = nn.Conv(channels, kernel_size=(1, 1))(x)
        return x
