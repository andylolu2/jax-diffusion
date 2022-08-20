import tensorflow as tf
import wandb
from absl import logging
from jax import random
from ml_collections import FrozenConfigDict
from tqdm import tqdm

from jax_diffusion.train.trainer import Trainer
from jax_diffusion.types import Config
from jax_diffusion.utils.actions import CheckpointAction, EvalAction, LogAction
from jax_diffusion.utils.wandb import wandb_run


def setup(config: Config):
    # setup tensorflow to use CPU
    tf.config.experimental.set_visible_devices([], "GPU")

    # setup logging
    logging.set_verbosity(config.log_level)

    # setup rng
    init_rng, eval_rng, step_rng = random.split(random.PRNGKey(config.seed), 3)

    # setup trainer
    trainer = Trainer(init_rng, **config.experiment_kwargs)
    if config.restore != "":
        trainer.restore_checkpoint(config.restore)

    # setup period actions
    ckpt_dir = f"{config.ckpt_dir}/{wandb.run.name}"
    periodic_actions = [
        LogAction(
            interval=config.log_interval,
            dry_run=config.dry_run,
        ),
        CheckpointAction(
            config.ckpt_interval,
            config.dry_run,
            trainer=trainer,
            ckpt_dir=ckpt_dir,
        ),
        EvalAction(
            interval=config.eval_interval,
            dry_run=config.dry_run,
            rng=eval_rng,
            trainer=trainer,
        ),
    ]

    return trainer, step_rng, periodic_actions


def main(config: Config):
    config = FrozenConfigDict(config)  # needed to be hashable

    logging.info(config)

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
