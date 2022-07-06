from pathlib import Path

import wandb
from jax import random

from jax_diffusion.config import get_config
from jax_diffusion.train.trainer import Trainer
from jax_diffusion.utils import (
    create_checkpoint_action,
    create_eval_action,
    create_log_action,
)

if __name__ == "__main__":
    config = get_config()

    wandb.init(
        project=config.project_name,
        dir=str(Path.cwd() / "_wandb"),
        config=config.experiment_kwargs.config.to_dict(),
    )
    ckpt_dir = str(Path(config.ckpt_dir) / wandb.run.name)

    trainer = Trainer(random.PRNGKey(config.seed), **config.experiment_kwargs)

    if config.restore is not None:  # continue training from checkpoint
        trainer.restore_checkpoint(config.restore)

    periodic_actions = [
        create_log_action(interval=config.log_interval),
        create_checkpoint_action(
            interval=config.ckpt_interval, trainer=trainer, ckpt_dir=ckpt_dir
        ),
        create_eval_action(interval=config.eval_interval, trainer=trainer),
    ]

    while (step := trainer.global_step) < config.training_steps:  # type: ignore
        metrics = trainer.step()

        for pa in periodic_actions:
            pa(step, metrics)
