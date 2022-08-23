from __future__ import annotations

from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from absl import logging
from flax.core import FrozenDict
from flax.training import checkpoints, train_state
from jax import random

from jax_diffusion.diffusion import Diffuser
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

        logging.info(f"Device count: {self._n_devices}")
        logging.info(f"Running on platform: {platform}")
        logging.info(f"Using data type: {self._dtype}")

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
        logging.info(f"inputs: {jax.tree_util.tree_map(lambda x: x.shape, inputs)}")
        self._state, metrics = self._update_fn(self._state, inputs, rng)
        logging.info(f"state: {jax.tree_util.tree_map(lambda x: x.shape, self._state)}")
        logging.info(f"metrics: {jax.tree_util.tree_map(lambda x: x.shape, metrics)}")

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

        generated = self.sample(**self._config.eval.sample_kwargs, rng=rng)
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
            self._state = self._create_train_state()

        self._state = checkpoints.restore_checkpoint(ckpt_dir=path, target=self._state)
        logging.info(f"Restored checkpoint from {path}")

    def _create_train_state(self):
        rng_diffusion, rng_param, rng_dropout = random.split(self._init_rng, 3)

        x_0 = self._sample_batch["image"]
        x_t, t, _ = self._diffuser.forward(x_0, rng_diffusion)
        logging.info(f"x_t: {x_t.shape}")
        model = UNet(**self._config.model.unet_kwargs, dtype=self._dtype)

        init_rngs = {"params": rng_param, "dropout": rng_dropout}
        params = model.init(init_rngs, x_t, t, train=True)

        count = sum(x.size for x in jax.tree_util.tree_leaves(params)) / 1e6
        logging.info(f"Parameter count: {count:.2f}M")

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

    @partial(jax.pmap, static_broadcasted_argnums=(0,), in_axes=(None, None, 0, None))
    def _update_fn(self, state: TrainState, inputs: Batch, rng: Rng):
        logging.info(f"inputs: {jax.tree_util.tree_map(lambda x: x.shape, inputs)}")
        rng1, rng2 = random.split(rng)
        grad_loss_fn = jax.grad(self._loss_fn, has_aux=True)

        x_0 = inputs["image"]
        x_t, t, eps = self._diffuser.forward(x_0, rng1)
        grads, metrics = grad_loss_fn(state.params, x_t, t, eps, rng2)
        state = state.apply_gradients(grads=grads)
        logging.info(f"state: {jax.tree_util.tree_map(lambda x: x.shape, state)}")
        logging.info(f"metrics: {jax.tree_util.tree_map(lambda x: x.shape, metrics)}")
        return state, metrics

    @partial(jax.pmap, static_broadcasted_argnums=(0,))
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
            n_devices=self._n_devices,
            **self._config.train.dataset_kwargs,
            **self._config.dataset_kwargs,
        )

    def _build_eval_input(self):
        return dataset.load(
            train=False,
            n_devices=self._n_devices,
            **self._config.eval.dataset_kwargs,
            **self._config.dataset_kwargs,
        )
