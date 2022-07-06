import enum
import time
from typing import Any, Callable, Mapping, Optional, Text

import numpy as np
import wandb

from jax_diffusion.train.trainer import Trainer


class IntervalType(enum.Enum):
    Secs = enum.auto()
    Steps = enum.auto()


class PeriodicAction:
    def __init__(
        self,
        callback: Callable[[int, Mapping[Text, np.ndarray]], None],
        interval: int,
        interval_type: IntervalType,
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

    def _should_run(self, step: Optional[int] = None):
        assert isinstance(self._interval_type, IntervalType)
        if self._interval_type == IntervalType.Secs:
            return (
                self._prev_time is None
                or (time.time() - self._prev_time) >= self._interval
            )
        elif self._interval_type == IntervalType.Steps:
            return self._prev_step is None or (step - self._prev_step) >= self._interval

    def __call__(self, step, metrics) -> Any:
        if self._should_run(step):
            self._callback(step, metrics)
            self._prev_time = time.time()
            self._prev_step = step


def create_log_action(interval: int):
    def log(step, metrics):
        metrics = {"train/" + k: v for k, v in metrics.items()}
        wandb.log(data=metrics, step=step)

    return PeriodicAction(
        callback=log,
        interval=interval,
        interval_type=IntervalType.Steps,
    )


def create_eval_action(interval: int, trainer: Trainer):
    def evaluate(step, _):
        metrics = trainer.evaluate()
        metrics = {"eval/" + k: v for k, v in metrics.items()}
        wandb.log(data=metrics, step=step)

    return PeriodicAction(
        callback=evaluate, interval=interval, interval_type=IntervalType.Secs
    )


def create_checkpoint_action(interval: int, trainer: Trainer, ckpt_dir: str):
    return PeriodicAction(
        callback=lambda *_: trainer.save_checkpoint(ckpt_dir),
        interval=interval,
        interval_type=IntervalType.Secs,
    )
