from pathlib import Path

from absl import logging
from ml_collections import ConfigDict, FieldReference

from jax_diffusion.types import Config


def get_config() -> Config:
    config = ConfigDict()

    config.project_name = "jax-diffusion"
    config.restore = None
    config.dry_run = False
    config.log_level = logging.INFO

    seed = 42
    d_model = FieldReference(8)
    grad_acc = FieldReference(1)
    steps = FieldReference(30000)

    config.seed = seed
    config.effective_steps = steps
    config.steps = steps * grad_acc
    config.ckpt_dir = str(Path.cwd() / "checkpoints")
    config.log_interval = 1
    config.ckpt_interval = 60
    config.eval_interval = 180

    config.experiment_kwargs = ConfigDict(
        dict(
            config=dict(
                seed=seed,
                dataset_kwargs=dict(
                    name="mnist",
                    resize_dim=32,
                    data_dir=str(Path.home() / "tensorflow_datasets"),
                    map_calls="auto",
                    prefetch="auto",
                    seed=seed,
                ),
                train=dict(
                    dataset_kwargs=dict(
                        batch_size=64,
                        subset="100%",
                        buffer_size=1000,
                        shuffle=True,
                        repeat=True,
                        augment=True,
                    ),
                    ema_step_size=1 - 0.9995,
                    optimizer=dict(
                        optimizer_type="adam",
                        kwargs=dict(
                            max_grad_norm=1.0,
                            grac_acc_steps=grad_acc,
                        ),
                    ),
                    lr_schedule=dict(
                        schedule_type="constant",
                        kwargs=dict(value=1e-4),
                    ),
                ),
                eval=dict(
                    dataset_kwargs=dict(
                        batch_size=64,
                        subset="20%",
                    ),
                    sample_kwargs=dict(
                        num=4,
                        steps=100,
                    ),
                ),
                diffusion=dict(
                    T=1000,
                    beta_1=1e-4,
                    beta_T=0.02,
                ),
                model=dict(
                    unet_kwargs=dict(
                        dim_init=d_model,
                        dim_mults=(1, 1, 2, 2, 2),
                        attention_resolutions=(16,),
                        attention_num_heads=4,
                        num_res_blocks=2,
                        sinusoidal_embed_dim=d_model,
                        time_embed_dim=4 * d_model,
                        kernel_size=3,
                        num_groups=4,
                        dropout=0.1,
                    ),
                ),
            )
        )
    )

    config.lock()

    return config
