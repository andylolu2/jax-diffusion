from collections import defaultdict
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import optax
from flax.training import checkpoints
from flax.training.train_state import TrainState

from jax_diffusion.model import UNet
from jax_diffusion.train import dataset


class Trainer:
    def __init__(self, init_rng, config):
        self._init_rng = init_rng
        self._config = config

        # Trainer state
        self._state: Optional[TrainState] = None

        # Input pipelines.
        self._train_input: Optional[dataset.Dataset] = None

    @property
    def global_step(self) -> int:
        if self._state is None:
            return 0
        return self._state.step

    def step(self):
        """Performs one training step"""
        if self._train_input is None:
            self._initialize_train()

        assert self._train_input is not None
        assert self._state is not None

        inputs = next(self._train_input)
        self._state, metrics = self._update_fn(self._state, inputs)
        return metrics

    def evaluate(self):
        """Performs evaluation on the evaluation dataset"""
        mean = defaultdict(int)

        for i, inputs in enumerate(self._build_eval_input()):
            metrics = self._eval_fn(self._state, inputs)

            for k, v in metrics.items():
                mean[k] = (mean[k] * i + v) / (i + 1)

        return mean

    def _create_train_state(self, inputs: dataset.Batch, rng):
        model = UNet(**self._config.model.unet_kwargs)
        params = model.init(rng, inputs["x_t"], inputs["t"])["params"]
        tx = self._create_optimizer()
        return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    def _forward_fn(self, params, inputs: dataset.Batch):
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

    def _create_optimizer(self):
        return optax.adam(self._config.training.learning_rate)

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
            is_training=True,
            batch_size=self._config.training.batch_size,
            seed=self._config.seed,
        )

    def _build_eval_input(self):
        return self._load_data(
            split="test",
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
            keep=2,
        )

    def restore_checkpoint(self, ckpt_dir: str):
        if self._state is None:
            self._initialize_train()

        self._state = checkpoints.restore_checkpoint(
            ckpt_dir=ckpt_dir,
            target=self._state,
        )
