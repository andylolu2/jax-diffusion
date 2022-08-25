from __future__ import annotations

import enum
import time
from typing import TYPE_CHECKING, Any, Mapping

import jax
import jax.numpy as jnp
import wandb
from flax.training import common_utils
from jax import random

from jax_diffusion.types import Rng

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
        dry_run: bool,
        rng: random.KeyArray | None = None,
    ):
        self._prev_time = time.time()
        self._prev_step = 0
        self._interval = interval
        self._dry_run = dry_run
        self._key = rng

    def __call__(self, *, step: int,  **kwargs):
        self.always_run(step=step, **kwargs)
        if self._should_run(step):
            self.run(step=step, **kwargs)
            self._prev_time = time.time()

    def run(self, *, **kwargs):
        raise NotImplementedError()

    def always_run(self, *, **kwargs):
        pass

    def _should_run(self, step: int):
        if self.interval_type == IntervalType.Secs:
            return time.time() - self._prev_time >= self._interval
        elif self.interval_type == IntervalType.Steps:
            return step - self._prev_step >= self._interval

    def _next_key(self):
        assert self._key is not None
        self._key, rng = random.split(self._key)
        return rng


class LogAction(PeriodicAction):
    interval_type = IntervalType.Steps

    def __init__(self, interval: float, dry_run: bool):
        super().__init__(interval, dry_run)
        self._cache = []

    def always_run(self, *, metrics: Mapping[str, Any], **kwargs):
        self._cache.append(metrics)

    def run(self, step: int, meta: Mapping[str, Any], **kwargs):
        metrics = common_utils.get_metrics(self._cache)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        data = {"train/" + k: v for k, v in metrics.items() + meta.items()}
        if not self._dry_run:
            wandb.log(data=data, step=step)
        self._cache = []


class EvalAction(PeriodicAction):
    interval_type = IntervalType.Secs

    def __init__(self, interval: float, dry_run: bool, rng: Rng, trainer: Trainer):
        super().__init__(interval, dry_run, rng)
        self._trainer = trainer

    def run(self, step: int, **kwargs):
        metrics = self._trainer.evaluate(rng=self._next_key())
        metrics = {"eval/" + k: v for k, v in metrics.items()}
        if not self._dry_run:
            wandb.log(data=metrics, step=step, commit=True)


class CheckpointAction(PeriodicAction):
    interval_type = IntervalType.Secs

    def __init__(self, interval: float, dry_run: bool, trainer: Trainer, ckpt_dir: str):
        super().__init__(interval, dry_run)
        self._trainer = trainer
        self._ckpt_dir = ckpt_dir

    def run(self, **kwargs):
        if not self._dry_run:
            self._trainer.save_checkpoint(self._ckpt_dir)
