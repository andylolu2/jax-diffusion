from pathlib import Path

from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    config = ConfigDict()

    config.project_name = "jax-diffusion"

    config.restore = None

    config.seed = 42
    config.training_steps = 2000
    config.ckpt_dir = str(Path.cwd() / "checkpoints")
    config.log_interval = 1
    config.ckpt_interval = 30
    config.eval_interval = 15

    config.experiment_kwargs = ConfigDict(
        dict(
            config=dict(
                seed=42,
                dataset=dict(
                    resize_dim=32,
                    data_dir=str(Path.home() / "tensorflow_datasets"),
                ),
                training=dict(
                    batch_size=64,
                    learning_rate=2e-4,
                ),
                eval=dict(
                    batch_size=64,
                ),
                diffusion=dict(
                    T=1000,
                    beta_1=1e-4,
                    beta_T=0.02,
                ),
                model=dict(
                    unet_kwargs=dict(
                        dim_init=32,
                        kernel_size_init=5,
                        dim_mults=(1, 2, 4, 8),
                        attention_dim_mults=(4,),
                        attention_num_heads=4,
                        sinusoidal_embed_dim=16,
                        time_embed_dim=32,
                        kernel_size=3,
                        num_groups=4,
                    ),
                ),
            )
        )
    )

    config.lock()

    return config
