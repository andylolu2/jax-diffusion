from typing import Generator, Optional

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from typing_extensions import TypedDict

from jax_diffusion.diffusion import alpha_bars

MEAN = 33.31
STD_DEV = 77.24


class Batch(TypedDict):
    eps: np.ndarray
    x_t: np.ndarray
    t: np.ndarray
    label: np.ndarray


Dataset = Generator[Batch, None, None]

AUTOTUNE = tf.data.experimental.AUTOTUNE


def load(
    split,
    *,
    is_training: bool,
    batch_size: int,
    resize_dim: int,
    seed: int,
    data_dir: str,
    diffusion_config,
) -> Dataset:
    T = diffusion_config.T

    ds = _load_tfds(
        split,
        is_training=is_training,
        batch_size=batch_size,
        resize_dim=resize_dim,
        seed=seed,
        data_dir=data_dir,
    )

    def _diffusion_process(sample):
        """See Algorithm 1 in https://arxiv.org/pdf/2006.11239.pdf"""
        t = tf.random.uniform(shape=(), minval=0, maxval=T, dtype=tf.int32, seed=seed)
        alpha_t_bar = tf.convert_to_tensor(alpha_bars(diffusion_config))[t]

        x_0 = sample["image"]
        eps = tf.random.normal(x_0.shape, seed=seed)

        x_t = (alpha_t_bar**0.5) * x_0 + ((1 - alpha_t_bar) ** 0.5) * eps

        t = tf.expand_dims(t, axis=-1)
        t = tf.cast(t, dtype=tf.float32)

        label = tf.cast(sample["label"], dtype=tf.float32)

        return {"eps": eps, "x_t": x_t, "t": t, "label": label}

    ds = ds.map(_diffusion_process, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(AUTOTUNE)

    yield from tfds.as_numpy(ds)


def _load_tfds(
    split,
    *,
    is_training: bool,
    batch_size: int,
    resize_dim: int,
    data_dir: str,
    seed: Optional[int] = None,
) -> tf.data.Dataset:

    ds = tfds.load("mnist", split=split, data_dir=data_dir)
    assert isinstance(ds, tf.data.Dataset)

    if is_training:
        assert seed is not None
        ds = ds.shuffle(buffer_size=1000 * batch_size, seed=seed)
        ds = ds.repeat()

    def _preprocess_sample(sample):
        image = _preprocess_image(sample["image"], resize_dim)
        label = tf.cast(sample["label"], tf.int32)
        return {"image": image, "label": label}

    ds = ds.map(_preprocess_sample, num_parallel_calls=AUTOTUNE)
    return ds


def _preprocess_image(image: tf.Tensor, image_dim: int):
    assert image.dtype == tf.uint8
    image = tf.image.resize(
        image, (image_dim, image_dim), tf.image.ResizeMethod.BICUBIC
    )
    assert image.dtype == tf.float32

    image -= tf.constant(MEAN, dtype=image.dtype)
    image /= tf.constant(STD_DEV, dtype=image.dtype)
    return image


if __name__ == "__main__":

    from jax_diffusion.config import get_config

    config = get_config().experiment_kwargs.config.dataset

    ds = _load_tfds(
        "train",
        is_training=False,
        batch_size=32,
        resize_dim=config.resize_dim,
        data_dir=config.data_dir,
        seed=0,
    )

    count = ds.reduce(0, lambda acc, _: acc + 1).numpy()
    mean = ds.reduce(
        np.float32(0), lambda acc, item: acc + tf.reduce_mean(item["image"]) / count
    ).numpy()
    variance = ds.reduce(
        np.float32(0),
        lambda acc, item: acc + tf.reduce_mean((item["image"] - mean) ** 2) / count,
    ).numpy()
    std_dev = variance**0.5

    print(f"MEAN: {mean:.3f}")
    print(f"STD_DEV: {std_dev:.3f}")
