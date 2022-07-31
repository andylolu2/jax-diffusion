from pathlib import Path

import numpy as np
import tensorflow as tf
import wandb
from jax import random
from tqdm import tqdm

from jax_diffusion.train.trainer import Trainer
from jax_diffusion.types import Config
from jax_diffusion.utils.actions import CheckpointAction, EvalAction, LogAction


def main(config: Config):
    # force tensorflow to use CPU
    tf.config.experimental.set_visible_devices([], "GPU")

    # setup rng
    init_rng, eval_rng, step_rng = random.split(random.PRNGKey(config.seed), 3)

    # setup trainer
    trainer = Trainer(init_rng, **config.experiment_kwargs)
    if config.restore is not None:
        trainer.restore_checkpoint(config.restore)

    if not config.dry_run:
        # setup wandb
        wandb.login()
        wandb.init(
            project=config.project_name,
            dir=str(Path.cwd() / "_wandb"),
            config=config.experiment_kwargs.config.to_dict(),
        )

    if config.dry_run:
        periodic_actions = []
    else:
        ckpt_dir = str(Path(config.ckpt_dir) / wandb.run.name)
        periodic_actions = [
            LogAction(interval=config.log_interval),
            CheckpointAction(
                interval=config.ckpt_interval, trainer=trainer, ckpt_dir=ckpt_dir
            ),
            EvalAction(interval=config.eval_interval, rng=eval_rng, trainer=trainer),
        ]

    # main loop
    with tqdm(total=config.steps, initial=trainer.global_step) as pbar:
        for step in range(trainer.global_step, config.steps):
            step_rng, _step_rng = random.split(step_rng)
            metrics = trainer.step(_step_rng)

            for pa in periodic_actions:
                pa(step=step, metrics=metrics)

            pbar.update()

    if not config.dry_run:
        wandb.finish()
