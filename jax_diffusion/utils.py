from __future__ import annotations

import enum
import math
import time
from typing import TYPE_CHECKING, Any, Callable, Mapping, Text

import numpy as np
import wandb
from jax import random

if TYPE_CHECKING:
    from jax_diffusion.train.trainer import Trainer


class IntervalType(enum.Enum):
    Secs = enum.auto()
    Steps = enum.auto()


class PeriodicAction:
    def __init__(
        self,
        callback: Callable[[int, Mapping[Text, np.ndarray], random.KeyArray], None],
        interval: int,
        interval_type: IntervalType,
        rng: random.KeyArray,
    ):
        """
        An action to executes periodically.

        Args:
            callback: Function to run periodically. Takes step and metrics as arguments.
            period: Period to run the function in seconds.
            interval_type: Type of interval to use.
        """

        self._prev_time = None
        self._prev_step = None
        self._callback = callback
        self._interval = interval
        self._interval_type = interval_type
        self._key = rng

    def _should_run(self, step: int):
        assert isinstance(self._interval_type, IntervalType)
        if self._interval_type == IntervalType.Secs:
            return (
                self._prev_time is None
                or (time.time() - self._prev_time) >= self._interval
            )
        elif self._interval_type == IntervalType.Steps:
            return self._prev_step is None or (step - self._prev_step) >= self._interval

    def __call__(self, step: int, metrics: Mapping[Text, np.ndarray]) -> Any:
        if self._should_run(step):
            self._key, rng = random.split(self._key)
            self._callback(step, metrics, rng)
            self._prev_time = time.time()
            self._prev_step = step


def create_log_action(interval: int, rng: random.KeyArray):
    def log(step, metrics, _):
        metrics = {"train/" + k: v for k, v in metrics.items()}
        wandb.log(data=metrics, step=step)

    return PeriodicAction(
        callback=log,
        interval=interval,
        interval_type=IntervalType.Steps,
        rng=rng,
    )


def create_eval_action(interval: int, trainer: Trainer, rng: random.KeyArray):
    def evaluate(step, _, rng):
        metrics = trainer.evaluate(rng=rng)
        metrics = {"eval/" + k: v for k, v in metrics.items()}
        wandb.log(data=metrics, step=step)

    return PeriodicAction(
        callback=evaluate,
        interval=interval,
        interval_type=IntervalType.Secs,
        rng=rng,
    )


def create_checkpoint_action(
    interval: int, trainer: Trainer, ckpt_dir: str, rng: random.KeyArray
):
    return PeriodicAction(
        callback=lambda *_: trainer.save_checkpoint(ckpt_dir),
        interval=interval,
        interval_type=IntervalType.Secs,
        rng=rng,
    )


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
