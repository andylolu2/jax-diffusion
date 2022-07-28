import math

import numpy as np


def _normalize(x: np.ndarray, *, low: float = 0.0, high: float):
    lo = x.min()
    hi = x.max()

    x = (x - lo) / (hi - lo)
    x = (x + low) * (high - low)
    return x


def image_grid(x: np.ndarray):
    """
    Creates a grid of images from a batch of images.

    Args:
        x (np.ndarray): Batch of images of shape `[batch_size, height, width, channel]`
        ncols (int): Number of columns in the image grid.
    """
    x = _normalize(x, low=0, high=255).astype(np.uint8)

    (b, h, w, c) = x.shape
    ncols = math.ceil(b**0.5)
    width = w * ncols
    height = h * math.ceil(b / ncols)

    grid = np.zeros((height, width, c))

    for i in range(b):
        x_offset = w * (i % ncols)
        y_offset = h * (i // ncols)
        grid[y_offset : y_offset + h, x_offset : x_offset + w] = x[i]

    return grid
