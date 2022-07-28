from pathlib import Path

import numpy as np
import tensorflow as tf
import wandb
from jax import random
from tqdm import tqdm

from jax_diffusion.config import get_config
from jax_diffusion.train.trainer import Trainer
from jax_diffusion.utils.actions import LogAction, EvalAction, CheckpointAction


def main():
    # force tensorflow to use CPU
    tf.config.experimental.set_visible_devices([], "GPU")

    config = get_config()

    # setup rng
    k0, k1 = random.split(random.PRNGKey(config.seed))
    np.random.seed(config.seed)

    # setup trainer
    trainer = Trainer(k0, **config.experiment_kwargs)
    if config.restore is not None:
        trainer.restore_checkpoint(config.restore)

    wandb.init(
        project=config.project_name,
        dir=str(Path.cwd() / "_wandb"),
        config=config.experiment_kwargs.config.to_dict(),
    )
    ckpt_dir = str(Path(config.ckpt_dir) / wandb.run.name)

    periodic_actions = [
        LogAction(interval=config.log_interval),
        CheckpointAction(
            interval=config.ckpt_interval,
            trainer=trainer,
            ckpt_dir=ckpt_dir,
        ),
        EvalAction(interval=config.eval_interval, rng=k1, trainer=trainer),
    ]

    # main loop
    with tqdm(total=config.training_steps, initial=trainer.global_step) as pbar:
        for step in range(trainer.global_step, config.training_steps):
            metrics = trainer.step()

            for pa in periodic_actions:
                pa(step, metrics=metrics)

            pbar.update()
