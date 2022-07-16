from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import optax
import wandb
from flax.training import checkpoints
from flax.training.train_state import TrainState
from jax import random

from jax_diffusion.diffusion import alpha_bars, alphas, betas
from jax_diffusion.model import UNet
from jax_diffusion.train import dataset, optimizers
from jax_diffusion.utils import image_grid


class Trainer:
    def __init__(self, init_rng, config):
        self._init_rng = init_rng
        self._config = config

        # Trainer state
        self._state: Optional[TrainState] = None
        self._lr_schedule: Optional[optax.Schedule] = None

        # Input pipelines.
        self._train_input: Optional[dataset.Dataset] = None

    @property
    def global_step(self) -> int:
        if self._state is None:
            return 0
        return int(self._state.step)

    def step(self):
        """Performs one training step"""
        if self._train_input is None:
            self._initialize_train()

        assert self._train_input is not None
        assert self._state is not None

        inputs = next(self._train_input)
        self._state, metrics = self._update_fn(self._state, inputs)

        meta = self._metadata()
        metrics.update(meta)

        return metrics

    def evaluate(self, rng: random.KeyArray):
        """Performs evaluation on the evaluation dataset"""
        metrics = defaultdict(int)

        sample = None
        for i, inputs in enumerate(self._build_eval_input()):
            sample = inputs
            m = self._eval_fn(self._state, inputs)

            for k, v in m.items():
                metrics[k] = (m[k] * i + v) / (i + 1)

        assert sample is not None
        shape = (self._config.eval.gen_samples,) + sample["x_t"].shape[1:]
        generated = self._sample(self._state, shape=shape, rng=rng)

        image = image_grid(generated)
        metrics["sample"] = wandb.Image(image)

        return metrics

    def _create_train_state(self, inputs: dataset.Batch, rng: random.KeyArray):
        model = UNet(**self._config.model.unet_kwargs)
        params = model.init(rng, inputs["x_t"], inputs["t"])["params"]
        tx, self._lr_schedule = self._create_optimizer()

        count = sum(x.size for x in jax.tree_leaves(params)) / 1e6
        print(f"Parameter count: {count:.2f}M")

        return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    def _forward_fn(self, params, inputs):
        assert self._state is not None
        return self._state.apply_fn({"params": params}, inputs["x_t"], inputs["t"])

    def _loss(self, pred, inputs: dataset.Batch):
        return optax.l2_loss(pred, jnp.asarray(inputs["eps"])).mean()

    def _loss_fn(self, params, inputs: dataset.Batch):
        pred = self._forward_fn(params, inputs)
        loss = self._loss(pred, inputs)
        metrics = self._compute_metrics(pred, inputs)
        return loss, metrics

    def _compute_metrics(self, pred, inputs: dataset.Batch):
        loss = self._loss(pred, inputs)
        return {"loss": loss}

    def _metadata(self):
        meta = {}
        meta["learning_rate"] = self._lr_schedule(self.global_step)
        return meta

    @partial(jax.jit, static_argnums=(0,))
    def _update_fn(self, state: TrainState, inputs: dataset.Batch):
        grad_loss_fn = jax.grad(self._loss_fn, has_aux=True)
        grads, metrics = grad_loss_fn(state.params, inputs)
        state = state.apply_gradients(grads=grads)
        return state, metrics

    @partial(jax.jit, static_argnums=(0,))
    def _eval_fn(self, state: TrainState, inputs: dataset.Batch):
        pred = self._forward_fn(state.params, inputs)
        metrics = self._compute_metrics(pred, inputs)
        return metrics

    @partial(jax.jit, static_argnums=(0, 2))
    def _sample(self, state: TrainState, shape: Sequence[int], rng: random.KeyArray):
        """See Algorithm 2 in https://arxiv.org/pdf/2006.11239.pdf"""
        T = self._config.diffusion.T
        alpha = jnp.asarray(alphas(self._config.diffusion))
        alpha_bar = jnp.asarray(alpha_bars(self._config.diffusion))
        beta = jnp.asarray(betas(self._config.diffusion))

        def body_fun(i: int, val: Tuple[random.KeyArray, jnp.ndarray]):
            """
            Args:
                i (int): Iteration number
                x (jnp.ndarray): Array of shape `[b, w, h, c]`
            """
            rng, x = val
            rng, rng_next = jax.random.split(rng)

            t = T - i - 1
            z = jax.lax.cond(
                pred=t > 0,
                true_fun=lambda: jax.random.normal(rng, shape=shape, dtype=jnp.float32),
                false_fun=lambda: jnp.zeros(shape=shape, dtype=jnp.float32),
            )
            alpha_t = alpha[t]
            alpha_t_bar = alpha_bar[t]
            sigma_t = beta[t] ** 0.5

            t_input = jnp.full((x.shape[0], 1), t, dtype=jnp.float32)
            eps = self._forward_fn(state.params, {"x_t": x, "t": t_input})

            x = (1 / alpha_t**0.5) * (
                x - ((1 - alpha_t) / (1 - alpha_t_bar) ** 0.5) * eps
            ) + sigma_t * z

            return rng_next, x

        rng1, rng2 = jax.random.split(rng)
        _, x = jax.lax.fori_loop(
            lower=0,
            upper=T,
            body_fun=body_fun,
            init_val=(
                rng1,
                jax.random.normal(
                    rng2,
                    shape=shape,
                ),
            ),
        )

        return x

    def _create_optimizer(self):
        op_config = self._config.training.optimizer
        lr_config = op_config.lr_schedule

        lr_schedule = optimizers.create_lr_schedule(
            lr_config.schedule_type,
            **lr_config.kwargs,
        )
        optimizer = optimizers.create_optimizer(
            optimizer_type=op_config.optimizer_type,
            lr_schedule=lr_schedule,
            **op_config.kwargs,
        )

        return optimizer, lr_schedule

    def _initialize_train(self):
        assert self._train_input is None
        self._train_input = self._build_train_input()

        if self._state is None:
            self._state = self._create_train_state(
                next(self._train_input), self._init_rng
            )

    def _build_train_input(self):
        return self._load_data(
            split=f"train[:{self._config.training.subset}]",
            is_training=True,
            batch_size=self._config.training.batch_size,
            seed=self._config.seed,
        )

    def _build_eval_input(self):
        return self._load_data(
            split=f"test[:{self._config.eval.subset}]",
            is_training=False,
            batch_size=self._config.eval.batch_size,
            seed=self._config.seed,
        )

    def _load_data(
        self, split, *, is_training: bool, batch_size: int, seed: int
    ) -> dataset.Dataset:
        ds = dataset.load(
            split,
            is_training=is_training,
            batch_size=batch_size,
            seed=seed,
            resize_dim=self._config.dataset.resize_dim,
            data_dir=self._config.dataset.data_dir,
            diffusion_config=self._config.diffusion,
        )
        return ds

    def save_checkpoint(self, ckpt_dir: str):
        checkpoints.save_checkpoint(
            ckpt_dir=ckpt_dir,
            target=self._state,
            step=self._state.step,
            keep=1,
        )

    def restore_checkpoint(self, ckpt_dir: str):
        assert Path(ckpt_dir).is_dir() or Path(ckpt_dir).is_file

        if self._state is None:
            self._initialize_train()

        print(f"Restoring from {ckpt_dir}")

        self._state = checkpoints.restore_checkpoint(
            ckpt_dir=ckpt_dir,
            target=self._state,
        )
