import math

import numpy as np


def _normalize(
    x: np.ndarray, *, quantiles=(0.01, 0.99), range_=(0, 255), dtype=np.uint8
):
    lo, hi = np.quantile(x, quantiles[0]), np.quantile(x, quantiles[1])

    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo)
    x = (x + range_[0]) * (range_[1] - range_[0])
    return x.astype(dtype)


def image_grid(x: np.ndarray):
    """
    Creates a grid of images from a batch of images.

    Args:
        x (np.ndarray): Batch of images of shape `[batch_size, height, width, channel]`
        ncols (int): Number of columns in the image grid.
    """
    x = _normalize(x)

    (b, h, w, c) = x.shape
    ncols = math.ceil(b**0.5)
    width = w * ncols
    height = h * math.ceil(b / ncols)

    grid = np.zeros((height, width, c), dtype=np.uint8)

    for i in range(b):
        x_offset = w * (i % ncols)
        y_offset = h * (i // ncols)
        grid[y_offset : y_offset + h, x_offset : x_offset + w] = x[i]

    return grid
