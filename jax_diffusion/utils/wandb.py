from contextlib import contextmanager
from pathlib import Path

import wandb

from jax_diffusion.types import Config


@contextmanager
def wandb_run(config: Config):
    if not config.dry_run:
        wandb.login()
    wandb.init(
        mode="disabled" if config.dry_run else "online",
        project=config.project_name,
        dir=str(Path.cwd() / "_wandb"),
        config=config.experiment_kwargs.config.to_dict(),
    )
    try:
        yield
    finally:
        wandb.finish()
