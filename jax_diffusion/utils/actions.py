from __future__ import annotations

import enum
import time
from typing import TYPE_CHECKING, Any, Mapping

import wandb
from jax import random

if TYPE_CHECKING:
    from jax_diffusion.train.trainer import Trainer


class IntervalType(enum.Enum):
    Secs = enum.auto()
    Steps = enum.auto()


class PeriodicAction:
    interval_type: IntervalType

    def __init__(
        self,
        interval: float,
        rng: random.KeyArray | None = None,
    ):
        self._prev_time = None
        self._prev_step = None
        self._interval = interval
        self._key = rng

    def __call__(self, step: int, *, **kwargs):
        if self._should_run(step):
            self.run(step, **kwargs)
            self._prev_time = time.time()
            self._prev_step = step

    def run(self, *args, **kwargs):
        raise NotImplementedError()

    def _should_run(self, step: int):
        if self._prev_time is None or self._prev_step is None:
            return True
        if self.interval_type == IntervalType.Secs:
            return time.time() - self._prev_time >= self._interval
        elif self._interval_type == IntervalType.Steps:
            return step - self._prev_step >= self._interval

    def _next_key(self):
        assert self._key is not None
        self._key, rng = random.split(self._key)
        return rng

class LogAction(PeriodicAction):
    interval_type = IntervalType.Steps

    def run(self, step: int, metrics: Mapping[str, Any], *args, **kwargs):
        metrics = {"train/" + k: v for k, v in metrics.items()}
        commit = self._prev_time is None or (time.time() - self._prev_time) >= 1
        wandb.log(data=metrics, step=step, commit=commit)

class EvalAction(PeriodicAction):
    interval_type = IntervalType.Secs

    def __init__(self, interval: float, rng: random.KeyArray, trainer: Trainer):
        super().__init__(interval, rng)
        self._trainer = trainer

    def run(self, step: int, *args, **kwargs):
        metrics = self.trainer.evaluate(rng=self._next_key())
        metrics = {"eval/" + k: v for k, v in metrics.items()}
        wandb.log(data=metrics, step=step)

class CheckpointAction(PeriodicAction):
    interval_type = IntervalType.Secs

    def __init__(self, interval: float, rng: random.KeyArray, trainer: Trainer, ckpt_dir: str):
        super().__init__(interval, rng)
        self._trainer = trainer
        self._ckpt_dir = ckpt_dir

    def run(self, *args, **kwargs):
        self._trainer.save_checkpoint(self._ckpt_dir)
