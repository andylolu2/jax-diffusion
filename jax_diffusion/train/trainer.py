from __future__ import annotations

from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Dict

import jax
import optax
import wandb
from flax.core import FrozenDict
from flax.training import checkpoints, train_state
from jax import random

from jax_diffusion.diffusion import backward_process, forward_random
from jax_diffusion.model import UNet
from jax_diffusion.train import dataset, lr_schedules, optimizers
from jax_diffusion.types import Batch, Config, Params, Rng, ndarray
from jax_diffusion.utils.image import image_grid


class TrainState(train_state.TrainState):
    ema_params: FrozenDict[str, Any]
    ema_step_size: float

    def apply_gradients(self, *, grads, **kwargs):
        next_state = super().apply_gradients(grads=grads, **kwargs)
        new_ema_params = optax.incremental_update(
            new_tensors=next_state.params,
            old_tensors=self.ema_params,
            step_size=self.ema_step_size,
        )
        return next_state.replace(ema_params=new_ema_params)


class Trainer:
    def __init__(self, rng: Rng, config: Config):
        self._init_rng = rng
        self._config = config

        # Initialized at self._initialize_train()
        self._state: TrainState | None = None
        self._lr_schedule: optax.Schedule | None = None
        self._train_input: dataset.Dataset | None = None

    @property
    def global_step(self):
        if self._state is None:
            return 0
        return int(self._state.step)

    @property
    def _sample_batch(self):
        """Does not require dataset to be downloaded."""
        return dataset.sample(
            **self._config.train.dataset_kwargs,
            **self._config.dataset_kwargs,
        )

    def step(self, rng: Rng):
        """Performs one training step"""
        if self._train_input is None:
            self._train_input = self._build_train_input()
        if self._state is None:
            self._state = self._create_train_state()

        assert self._train_input is not None
        assert self._state is not None
        inputs = next(self._train_input)
        self._state, metrics = self._update_fn(self._state, inputs, rng)

        meta = self._metadata()
        metrics.update(meta)

        return metrics

    def evaluate(self, rng: Rng):
        """Performs evaluation on the evaluation dataset"""
        metrics: Dict[str, Any] = defaultdict(int)

        for i, inputs in enumerate(self._build_eval_input()):
            rng, _rng = random.split(rng)
            m = self._eval_fn(self._state, inputs, _rng)
            for k, v in m.items():
                metrics[k] = metrics[k] * (i / (i + 1)) + v / (i + 1)

        generated = self.sample(num=self._config.eval.gen_samples, rng=rng)
        image = image_grid(generated)
        metrics["sample"] = wandb.Image(image)

        return metrics

    def _create_train_state(self):
        rng_diffusion, rng_param, rng_dropout = random.split(self._init_rng, 3)

        model = UNet(**self._config.model.unet_kwargs)
        x_t, t, _ = forward_random(
            self._config.diffusion, self._sample_batch["image"], rng_diffusion
        )
        init_rngs = {"params": rng_param, "dropout": rng_dropout}

        params = model.init(init_rngs, x_t, t, train=True)
        count = sum(x.size for x in jax.tree_leaves(params)) / 1e6
        print(f"Parameter count: {count:.2f}M")

        tx, self._lr_schedule = self._create_optimizer()

        return TrainState.create(
            apply_fn=model.apply,
            params=params,
            ema_params=params,
            ema_step_size=self._config.train.ema_step_size,
            tx=tx,
        )

    def _forward_fn(
        self,
        params: Params,
        image: ndarray,
        t: ndarray,
        train: bool,
        rng: Rng | None = None,
    ):
        rngs = {"dropout": rng} if rng is not None else None
        assert self._state is not None
        return self._state.apply_fn(params, image, t, train, rngs=rngs)

    def _loss(self, pred: ndarray, actual: ndarray):
        return optax.l2_loss(pred, actual).mean()  # type: ignore

    def _compute_metrics(self, pred: ndarray, actual: ndarray):
        loss = self._loss(pred, actual)
        return {"loss": loss}

    def _loss_fn(
        self, params: Params, image: ndarray, t: ndarray, eps: ndarray, rng: Rng
    ):
        pred = self._forward_fn(params, image, t, train=True, rng=rng)
        loss = self._loss(pred, eps)
        metrics = self._compute_metrics(pred, eps)
        return loss, metrics

    def _metadata(self):
        meta = {}
        meta["learning_rate"] = self._lr_schedule(self.global_step)
        return meta

    @partial(jax.jit, static_argnums=(0,))
    def _update_fn(self, state: TrainState, inputs: Batch, rng: Rng):
        rng1, rng2 = random.split(rng)
        grad_loss_fn = jax.grad(self._loss_fn, has_aux=True)

        x_t, t, eps = forward_random(self._config.diffusion, inputs["image"], rng1)
        grads, metrics = grad_loss_fn(state.params, x_t, t, eps, rng2)
        state = state.apply_gradients(grads=grads)
        return state, metrics

    @partial(jax.jit, static_argnums=(0,))
    def _eval_fn(self, state: TrainState, inputs: Batch, rng: Rng):
        x_t, t, eps = forward_random(self._config.diffusion, inputs["image"], rng)
        pred = self._forward_fn(state.ema_params, x_t, t, train=False)
        metrics = self._compute_metrics(pred, eps)
        return metrics

    def sample(self, num: int, rng: Rng):
        x_ = self._sample_batch["image"]
        shape = (num,) + x_.shape[1:]

        rng1, rng2 = jax.random.split(rng)
        x_0 = backward_process(
            diffusion_config=self._config.diffusion,
            eps_fn=self._forward_fn,
            params=self._state.ema_params,
            x_t=jax.random.normal(rng1, shape=shape, dtype=x_.dtype),
            T=self._config.diffusion.T,
            steps=self._config.diffusion.T,
            rng=rng2,
        )
        return x_0

    def _create_optimizer(self):
        op_config = self._config.train.optimizer
        lr_config = op_config.lr_schedule

        lr_schedule = lr_schedules.create_lr_schedule(
            lr_config.schedule_type,
            **lr_config.kwargs,
        )
        optimizer = optimizers.create_optimizer(
            optimizer_type=op_config.optimizer_type,
            lr_schedule=lr_schedule,
            **op_config.kwargs,
        )

        return optimizer, lr_schedule

    def _build_train_input(self):
        return dataset.load(
            train=True,
            **self._config.train.dataset_kwargs,
            **self._config.dataset_kwargs,
        )

    def _build_eval_input(self):
        return dataset.load(
            train=False,
            **self._config.eval.dataset_kwargs,
            **self._config.dataset_kwargs,
        )

    def save_checkpoint(self, ckpt_dir: str):
        checkpoints.save_checkpoint(
            ckpt_dir=ckpt_dir,
            target=self._state,
            step=self._state.step,
            keep=1,
        )

    def restore_checkpoint(self, ckpt_dir: str):
        path = Path(ckpt_dir)
        assert path.is_dir() or path.is_file()

        if self._state is None:
            self._state = self._create_train_state()

        print(f"Restoring from {ckpt_dir}")
        self._state = checkpoints.restore_checkpoint(ckpt_dir=path, target=self._state)
