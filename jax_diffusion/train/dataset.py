from __future__ import annotations

from typing import Generator

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from typing_extensions import TypedDict

from jax_diffusion.diffusion import alpha_bars

DATASETS = {
    "mnist": {
        "mean": np.array([33.31]),
        "std_dev": np.array([77.24]),
        "splits": {"train": "train", "test": "test"},
    },
    "fashion_mnist": {
        "mean": np.array([72.94]),
        "std_dev": np.array([88.638]),
        "splits": {"train": "train", "test": "test"},
    },
    "stl10": {
        "mean": np.array([112.3418, 108.9556, 98.3724]),
        "std_dev": np.array([68.0722, 66.1841, 68.0583]),
        "splits": {"train": "unlabelled", "test": "test"},
    },
    "celeb_a": {
        "mean": np.array([128.5573, 107.7400, 94.8121]),
        "std_dev": np.array([78.1818, 72.5204, 72.1100]),
        "splits": {"train": "train", "test": "test"},
    },
}


class Batch(TypedDict):
    eps: np.ndarray
    x_t: np.ndarray
    t: np.ndarray


Dataset = Generator[Batch, None, None]

AUTOTUNE = tf.data.experimental.AUTOTUNE


def load(
    dataset: str,
    split: str,
    *,
    subset: str,
    is_training: bool,
    batch_size: int,
    resize_dim: int,
    seed: int,
    data_dir: str,
    diffusion_config,
) -> Dataset:
    T = diffusion_config.T

    ds = _load_tfds(
        dataset,
        split,
        subset=subset,
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

        return {"eps": eps, "x_t": x_t, "t": t}

    ds = ds.map(_diffusion_process)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(3)

    yield from tfds.as_numpy(ds)


def _load_tfds(
    dataset: str,
    split: str,
    *,
    subset: str,
    is_training: bool,
    batch_size: int | None,
    resize_dim: int,
    data_dir: str,
    seed: int | None = None,
) -> tf.data.Dataset:

    split = DATASETS[dataset]["splits"][split]
    ds = tfds.load(
        dataset,
        split=f"{split}[:{subset}]",
        data_dir=data_dir,
        shuffle_files=True,
    )
    assert isinstance(ds, tf.data.Dataset)

    if is_training:
        assert seed is not None
        assert batch_size is not None
        ds = ds.shuffle(buffer_size=500 * batch_size, seed=seed)
        ds = ds.repeat()

    def _preprocess_sample(sample):
        image = _preprocess_image(dataset, sample["image"], resize_dim)
        return {"image": image}

    ds = ds.map(_preprocess_sample)
    return ds


def _preprocess_image(dataset: str, image: tf.Tensor, image_dim: int):
    assert dataset in DATASETS
    assert image.dtype == tf.uint8
    tf.assert_rank(image, 3)

    shape = image.shape
    h, w = shape[0], shape[1]

    box_size = tf.minimum(h, w)
    h_offset = (h - box_size) // 2
    w_offset = (w - box_size) // 2

    image = tf.image.crop_to_bounding_box(image, h_offset, w_offset, box_size, box_size)
    image = tf.image.resize(
        image, (image_dim, image_dim), tf.image.ResizeMethod.BICUBIC
    )
    assert image.dtype == tf.float32

    stats = DATASETS[dataset]
    image -= tf.constant(stats["mean"], dtype=image.dtype)
    image /= tf.constant(stats["std_dev"], dtype=image.dtype)
    return image


if __name__ == "__main__":

    from jax_diffusion.config import get_config

    tf.config.experimental.set_visible_devices([], "GPU")

    config = get_config().experiment_kwargs.config.dataset

    ds = _load_tfds(
        "celeb_a",
        "train",
        subset="100%",
        is_training=False,
        batch_size=None,
        resize_dim=64,
        data_dir=config.data_dir,
        seed=0,
    )
    w, h, c = next(iter(ds))["image"].shape

    count = ds.reduce(tf.constant(0, dtype=tf.float32), lambda acc, _: acc + 1)
    mean = ds.reduce(
        tf.zeros((c,), dtype=tf.float32),
        lambda acc, item: acc + tf.reduce_mean(item["image"], axis=(0, 1)) / count,
    )
    variance = ds.reduce(
        tf.zeros((c,), dtype=tf.float32),
        lambda acc, item: acc
        + tf.reduce_mean((item["image"] - mean) ** 2, axis=(0, 1)) / count,
    )
    std_dev = variance**0.5

    count = count.numpy()
    mean = mean.numpy()
    std_dev = std_dev.numpy()

    print(f"COUNT: {count}")
    print(f"MEAN: {mean}")
    print(f"STD_DEV: {std_dev}")
