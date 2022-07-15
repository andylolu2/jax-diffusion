from pathlib import Path

import numpy as np
import tensorflow as tf
import wandb
from jax import random
from tqdm import tqdm

from jax_diffusion.config import get_config
from jax_diffusion.train.trainer import Trainer
from jax_diffusion.utils import (
    create_checkpoint_action,
    create_eval_action,
    create_log_action,
)

if __name__ == "__main__":
    tf.config.experimental.set_visible_devices([], "GPU")

    config = get_config()

    wandb.init(
        project=config.project_name,
        dir=str(Path.cwd() / "_wandb"),
        config=config.experiment_kwargs.config.to_dict(),
    )
    ckpt_dir = str(Path(config.ckpt_dir) / wandb.run.name)

    [k0, k1, k2, k3] = random.split(random.PRNGKey(config.seed), num=4)
    np.random.seed(config.seed)

    trainer = Trainer(k0, **config.experiment_kwargs)

    if config.restore is not None:  # continue training from checkpoint
        trainer.restore_checkpoint(config.restore)

    periodic_actions = [
        create_log_action(
            interval=config.log_interval,
            rng=k1,
        ),
        create_checkpoint_action(
            interval=config.ckpt_interval,
            trainer=trainer,
            ckpt_dir=ckpt_dir,
            rng=k2,
        ),
        create_eval_action(
            interval=config.eval_interval,
            trainer=trainer,
            rng=k3,
        ),
    ]

    with tqdm(total=config.training_steps, initial=trainer.global_step) as pbar:
        for step in range(trainer.global_step, config.training_steps):
            metrics = trainer.step()

            for pa in periodic_actions:
                pa(step, metrics)

            pbar.update()
