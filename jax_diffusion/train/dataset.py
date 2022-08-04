from __future__ import annotations

from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from jax_diffusion.types import Batch, Dataset

AUTOTUNE = tf.data.experimental.AUTOTUNE
DATASETS = {
    "mnist": {
        "channels": 1,
        "mean": np.array([33.31]),
        "std_dev": np.array([77.24]),
        "splits": {"train": "train", "test": "test"},
    },
    "fashion_mnist": {
        "channels": 1,
        "mean": np.array([72.94]),
        "std_dev": np.array([88.638]),
        "splits": {"train": "train", "test": "test"},
    },
    "stl10": {
        "channels": 3,
        "mean": np.array([112.3418, 108.9556, 98.3724]),
        "std_dev": np.array([68.0722, 66.1841, 68.0583]),
        "splits": {"train": "unlabelled", "test": "test"},
    },
    "celeb_a": {
        "channels": 3,
        "mean": np.array([128.5573, 107.7400, 94.8121]),
        "std_dev": np.array([78.1818, 72.5204, 72.1100]),
        "splits": {"train": "train", "test": "test"},
    },
}


def sample(
    name: str, resize_dim: int, batch_size: int, dtype=np.float32, **kwargs
) -> Batch:
    c = DATASETS[name]["channels"]
    x = np.zeros((batch_size, resize_dim, resize_dim, c), dtype=dtype)
    return {"image": x, "label": None}


def load(
    train: bool,
    name: str,
    subset: str,
    resize_dim: int,
    data_dir: str,
    prefetch: int | str,
    map_calls: int | str,
    batch_size: int,
    repeat: bool = False,
    shuffle: bool = False,
    seed: int | None = None,
    buffer_size: int | None = None,
) -> Dataset:
    split = "train" if train else "test"
    split = DATASETS[name]["splits"][split]

    ds = tfds.load(
        name,
        split=f"{split}[:{subset}]",
        data_dir=data_dir,
        shuffle_files=True,
    )
    assert isinstance(ds, tf.data.Dataset)

    if shuffle:
        assert seed is not None
        assert buffer_size is not None
        ds = ds.shuffle(buffer_size=buffer_size, seed=seed)
    if repeat:
        ds = ds.repeat()

    preprocess = partial(preprocess_image, name, resize_dim)
    ds = ds.map(preprocess, AUTOTUNE if map_calls == "auto" else map_calls)

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(AUTOTUNE if prefetch == "auto" else prefetch)

    yield from tfds.as_numpy(ds)


def preprocess_image(name: str, image_dim: int, sample):
    """Center crop and normalize"""
    image = sample["image"]

    assert name in DATASETS
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

    stats = DATASETS[name]
    image -= tf.constant(stats["mean"], dtype=image.dtype)
    image /= tf.constant(stats["std_dev"], dtype=image.dtype)
    return {"image": image}


if __name__ == "__main__":
    from pathlib import Path

    tf.config.experimental.set_visible_devices([], "GPU")

    ds = load(
        train=True,
        batch_size=128,
        subset="100%",
        shuffle=False,
        repeat=False,
        name="celeb_a",
        resize_dim=64,
        data_dir=str(Path.home() / "tensorflow_datasets"),
        prefetch="auto",
        seed=0,
    )

    count = 0
    mean = 0
    mean_of_sq = 0
    for i, inputs in enumerate(ds):
        im = inputs["image"]
        count += 1
        mean = (mean * i + np.mean(im, axis=(0, 1, 2))) / (i + 1)
        mean_of_sq = (mean_of_sq * i + np.mean(im**2, axis=(0, 1, 2))) / (i + 1)

    std_dev = np.sqrt(mean_of_sq - mean**2)
    print(f"COUNT: {count}")
    print(f"MEAN: {mean}")
    print(f"STD_DEV: {std_dev}")
