from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from absl import logging
from flax.core import FrozenDict
from flax.training import checkpoints, common_utils, train_state
from jax import random
from jax.sharding import PositionalSharding

from jax_diffusion.diffusion import Diffuser
from jax_diffusion.model import UNet
from jax_diffusion.train import dataset, lr_schedules, optimizers
from jax_diffusion.types import Batch, Config, Params, ReplicatedState, Rng, ndarray
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
        self.global_step = 0
        self._init_rng, self._step_rng = random.split(rng)
        self._config = config
        self._diffuser = Diffuser(self._forward_fn, config.diffusion)

        self._am = checkpoints.AsyncManager()
        self._n_devices = jax.local_device_count()
        platform = jax.local_devices()[0].platform
        if config.half_precision:
            if platform == "tpu":
                self._dtype = jnp.bfloat16
            else:
                self._dtype = jnp.float16
        else:
            self._dtype = jnp.float32

        self.sharding = PositionalSharding(jax.devices())

        logging.info(f"Device count: {self._n_devices}")
        logging.info(f"Running on platform: {platform}")
        logging.info(f"Using data type: {self._dtype}")

        # Initialized at first
        self._state: ReplicatedState | None = None
        self._lr_schedule: optax.Schedule | None = None
        self._train_input: dataset.Dataset | None = None

    @property
    def _sample_batch(self):
        """Does not require dataset to be downloaded."""
        return dataset.sample(
            **self._config.train.dataset_kwargs,
            **self._config.dataset_kwargs,
        )

    def step(self):
        """Performs one training step"""
        if self._train_input is None:
            self._train_input = self._build_train_input()
        if self._state is None:
            self._state, self._lr_schedule = self._create_state()
            self._state = jax.device_put(self._state, self.sharding.replicate())

        inputs = next(self._train_input)
        inputs = jax.device_put(inputs, self.sharding.reshape(self._n_devices, 1, 1, 1))

        self._step_rng, rng = random.split(self._step_rng)
        self._state, metrics = self._update_fn(self._state, inputs, rng)

        meta = self._metadata()

        self.global_step += 1
        return metrics, meta

    def evaluate(self, rng: Rng):
        """Performs evaluation on the evaluation dataset"""
        rng_0, rng_1 = random.split(rng)

        metrics_list = []
        for inputs in self._build_eval_input():
            rng_0_, rng_0 = random.split(rng_0)
            m = self._eval_fn(self._state, inputs, rng_0_)
            metrics_list.append(m)

        metrics = common_utils.stack_forest(metrics_list)
        metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x).item(), metrics)

        generated = self.sample(**self._config.eval.sample_kwargs, rng=rng_1)
        image = image_grid(np.asarray(generated))
        metrics["sample"] = wandb.Image(image)

        return metrics

    def sample(self, num: int, steps: int, rng: Rng):
        x = self._sample_batch["image"]
        shape = (num,) + x.shape[1:]

        x_T = random.normal(rng, shape, dtype=x.dtype)

        x_0 = self._diffuser.ddim_backward(self._state.ema_params, x_T, steps)
        return x_0

    def save_checkpoint(self, ckpt_dir: str):
        checkpoints.save_checkpoint(
            ckpt_dir=ckpt_dir,
            target=self._state,
            step=self._state.step,
            keep=1,
            async_manager=self._am,
        )

    def restore_checkpoint(self, ckpt_dir: str):
        path = Path(ckpt_dir)
        assert path.is_dir() or path.is_file()

        if self._state is None:
            self._state, self._lr_schedule = self._create_state()

        self._state = checkpoints.restore_checkpoint(ckpt_dir=path, target=self._state)
        self._state = jax.device_put(self._state, self.sharding.replicate())
        logging.info(f"Restored checkpoint from {path}")

    def _create_state(self):
        rng_diffusion, rng_param, rng_dropout = random.split(self._init_rng, 3)

        x_0 = self._sample_batch["image"]
        x_t, t, _ = self._diffuser.forward(x_0, rng_diffusion)
        model = UNet(**self._config.model.unet_kwargs, dtype=self._dtype)

        init_rngs = {"params": rng_param, "dropout": rng_dropout}
        params = model.init(init_rngs, x_t, t, train=True)

        count = sum(x.size for x in jax.tree_util.tree_leaves(params)) / 1e6
        logging.info(f"Parameter count: {count:.2f}M")

        tx, lr_schedule = self._create_optimizer()

        state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            ema_params=params,
            ema_step_size=self._config.train.ema_step_size,
            tx=tx,
        )
        return state, lr_schedule

    def _forward_fn(
        self,
        params: Params,
        image: ndarray,
        t: ndarray,
        train: bool,
        rng: Rng | None = None,
    ):
        rngs = {"dropout": rng} if rng is not None else None

        if t.size == 1:
            t = jnp.full((len(image), 1), t, dtype=image.dtype)

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

    @partial(jax.jit, static_argnums=0)
    def _update_fn(self, state: TrainState, inputs: Batch, rng: Rng):
        rng1, rng2 = random.split(rng)
        grad_loss_fn = jax.grad(self._loss_fn, has_aux=True)

        x_0 = inputs["image"]
        x_t, t, eps = self._diffuser.forward(x_0, rng1)
        grads, metrics = grad_loss_fn(state.params, x_t, t, eps, rng2)
        state = state.apply_gradients(grads=grads)
        return state, metrics

    @partial(jax.jit, static_argnums=0)
    def _eval_fn(self, state: TrainState, inputs: Batch, rng: Rng):
        x_t, t, eps = self._diffuser.forward(inputs["image"], rng)
        pred = self._forward_fn(state.ema_params, x_t, t, train=False)
        metrics = self._compute_metrics(pred, eps)
        return metrics

    def _create_optimizer(self):
        op_config = self._config.train.optimizer
        lr_config = self._config.train.lr_schedule

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
