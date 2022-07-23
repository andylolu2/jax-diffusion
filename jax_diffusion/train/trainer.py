from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Optional, Sequence

import jax
import jax.numpy as jnp
import optax
import wandb
from flax.core import FrozenDict
from flax.training import checkpoints, train_state
from jax import random

from jax_diffusion.diffusion import alpha_bars, alphas, betas
from jax_diffusion.model import UNet
from jax_diffusion.train import dataset, optimizers
from jax_diffusion.utils import image_grid


class TrainState(train_state.TrainState):
    ema_params: FrozenDict[str, Any]

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        opt_state = tx.init(params)
        ema_params = params
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            ema_params=ema_params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def apply_gradients(self, *, grads, step_size: float, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        new_ema_params = optax.incremental_update(
            new_tensors=new_params,
            old_tensors=self.ema_params,
            step_size=step_size,
        )
        return self.replace(
            step=self.step + 1,
            params=new_params,
            ema_params=new_ema_params,
            opt_state=new_opt_state,
            **kwargs,
        )


class Trainer:
    def __init__(self, init_rng: random.KeyArray, config):
        self._init_rng, self._rng = random.split(init_rng)
        self._config = config

        # Initialized at self._initialize_train()
        self._state: Optional[TrainState] = None
        self._lr_schedule: Optional[optax.Schedule] = None
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
        self._rng, rng = random.split(self._rng)
        self._state, metrics = self._update_fn(self._state, inputs, rng)

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
        assert self._state is not None
        shape = (self._config.eval.gen_samples,) + sample["x_t"].shape[1:]
        generated = self._sample(self._state, shape=shape, rng=rng)

        image = image_grid(generated)
        metrics["sample"] = wandb.Image(image)

        return metrics

    def _create_train_state(self, inputs: dataset.Batch, rng: random.KeyArray):
        rng_param, rng_dropout = random.split(rng)
        init_rng = {"params": rng_param, "dropout": rng_dropout}

        model = UNet(**self._config.model.unet_kwargs)
        params = model.init(init_rng, inputs["x_t"], inputs["t"], train=True)["params"]
        tx, self._lr_schedule = self._create_optimizer()

        count = sum(x.size for x in jax.tree_leaves(params)) / 1e6
        print(f"Parameter count: {count:.2f}M")

        return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    def _forward_fn(self, params, inputs, train: bool, rng=None):
        assert self._state is not None

        rngs = None
        if rng is not None:
            rngs = {"dropout": rng}

        return self._state.apply_fn(
            {"params": params}, inputs["x_t"], inputs["t"], train, rngs=rngs
        )

    def _loss(self, pred, inputs: dataset.Batch):
        return optax.l2_loss(pred, jnp.asarray(inputs["eps"])).mean()

    def _loss_fn(self, params, inputs: dataset.Batch, rng):
        pred = self._forward_fn(params, inputs, train=True, rng=rng)
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
    def _update_fn(
        self,
        state: TrainState,
        inputs: dataset.Batch,
        rng: random.KeyArray,
    ):
        grad_loss_fn = jax.grad(self._loss_fn, has_aux=True)
        grads, metrics = grad_loss_fn(state.params, inputs, rng=rng)
        state = state.apply_gradients(
            grads=grads,
            step_size=self._config.training.ema_step_size,
        )
        return state, metrics

    @partial(jax.jit, static_argnums=(0,))
    def _eval_fn(self, state: TrainState, inputs: dataset.Batch):
        pred = self._forward_fn(state.ema_params, inputs, train=False)
        metrics = self._compute_metrics(pred, inputs)
        return metrics

    @partial(jax.jit, static_argnums=(0,))
    def denoise(self, x, t):
        diff_conf = self._config.diffusion
        alpha_t = jnp.asarray(alphas(diff_conf))[t]
        alpha_t_bar = jnp.asarray(alpha_bars(diff_conf))[t]

        t_input = jnp.full((x.shape[0], 1), t, dtype=jnp.float32)
        eps = self._forward_fn(
            self._state.ema_params, {"x_t": x, "t": t_input}, train=False
        )

        x_new = (1 / alpha_t**0.5) * (
            x - ((1 - alpha_t) / (1 - alpha_t_bar) ** 0.5) * eps
        )

        return x_new

    def _sample(self, state: TrainState, shape: Sequence[int], rng: random.KeyArray):
        """See Algorithm 2 in https://arxiv.org/pdf/2006.11239.pdf"""

        rng1, rng2 = jax.random.split(rng)
        init_val = {
            "x": jax.random.normal(rng1, shape=shape, dtype=jnp.float32),
            "rng": rng2,
            "constants": (
                self._config.diffusion.T,
                state,
                alphas(self._config.diffusion),
                alpha_bars(self._config.diffusion),
                betas(self._config.diffusion),
            ),
        }

        def body_fun(i: int, val):
            """
            Args:
                i (int): Iteration number
                val (Tuple): Tuple of (x, rng, constants).
                    x is of shape `[b, w, h, c]`.
                    constants is a tuple of (T, alphas, alpha_bars, betas).
            """
            x, rng, constants = val["x"], val["rng"], val["constants"]
            T, state, alpha, alpha_bar, beta = constants
            t = T - i - 1
            alpha_t = alpha[t]
            alpha_t_bar = alpha_bar[t]
            sigma_t = beta[t] ** 0.5
            rng, rng_next = jax.random.split(rng)

            z = jax.lax.cond(
                pred=t > 0,
                true_fun=lambda: jax.random.normal(
                    rng, shape=x.shape, dtype=jnp.float32
                ),
                false_fun=lambda: jnp.zeros(shape=x.shape, dtype=jnp.float32),
            )

            t_input = jnp.full((x.shape[0], 1), t, dtype=jnp.float32)
            eps = state.apply_fn({"params": state.ema_params}, x, t_input, train=False)

            x = (1 / alpha_t**0.5) * (
                x - ((1 - alpha_t) / (1 - alpha_t_bar) ** 0.5) * eps
            ) + sigma_t * z

            return {"x": x, "rng": rng_next, "constants": constants}

        x = jax.lax.fori_loop(
            lower=0,
            upper=self._config.diffusion.T,
            body_fun=body_fun,
            init_val=init_val,
        )["x"]

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
            split="train",
            subset=self._config.training.subset,
            is_training=True,
            batch_size=self._config.training.batch_size,
            seed=self._config.seed,
        )

    def _build_eval_input(self):
        return self._load_data(
            split="test",
            subset=self._config.eval.subset,
            is_training=False,
            batch_size=self._config.eval.batch_size,
            seed=self._config.seed,
        )

    def _load_data(
        self, split, *, subset: str, is_training: bool, batch_size: int, seed: int
    ) -> dataset.Dataset:
        name = self._config.dataset.name
        ds = dataset.load(
            name,
            split,
            subset=subset,
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
