from pathlib import Path

import numpy as np
import tensorflow as tf
import wandb
from absl import logging
from jax import random
from ml_collections import FrozenConfigDict
from tqdm import tqdm

from jax_diffusion.train.trainer import Trainer
from jax_diffusion.types import Config
from jax_diffusion.utils.actions import CheckpointAction, EvalAction, LogAction


def setup(config: Config):
    # setup tensorflow to use CPU
    tf.config.experimental.set_visible_devices([], "GPU")

    # setup logging
    logging.set_verbosity(config.log_level)

    # setup rng
    init_rng, eval_rng, step_rng = random.split(random.PRNGKey(config.seed), 3)

    # setup trainer
    trainer = Trainer(init_rng, **config.experiment_kwargs)
    if config.restore is not None:
        trainer.restore_checkpoint(config.restore)

    # setup period actions
    periodic_actions = []
    if not config.dry_run:
        ckpt_dir = str(Path(config.ckpt_dir) / wandb.run.name)
        periodic_actions += [
            LogAction(interval=config.log_interval),
            CheckpointAction(
                interval=config.ckpt_interval, trainer=trainer, ckpt_dir=ckpt_dir
            ),
            EvalAction(interval=config.eval_interval, rng=eval_rng, trainer=trainer),
        ]

    return trainer, step_rng, periodic_actions


def main(config: Config, dry_run: bool):
    config.dry_run = dry_run
    config = FrozenConfigDict(config)  # needed to be hashable

    # main loop
    with wandb_run(config):
        trainer, step_rng, periodic_actions = setup(config)
    with tqdm(total=config.steps, initial=trainer.global_step) as pbar:
        for step in range(trainer.global_step, config.steps):
            step_rng, _step_rng = random.split(step_rng)
            metrics = trainer.step(_step_rng)

            for pa in periodic_actions:
                pa(step=step, metrics=metrics)

            pbar.update()

    if not config.dry_run:
        wandb.finish()
