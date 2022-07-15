from pathlib import Path

from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    config = ConfigDict()

    config.project_name = "jax-diffusion"

    config.restore = str(Path.cwd() / "checkpoints/hopeful-lion-53")
    # config.restore = None

    config.seed = 42
    config.training_steps = 2000
    config.ckpt_dir = str(Path.cwd() / "checkpoints")
    config.log_interval = 1
    config.ckpt_interval = 30
    config.eval_interval = 30

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
                    learning_rate=1e-4,
                    subset="100%",
                ),
                eval=dict(
                    batch_size=64,
                    subset="20%",
                    gen_samples=4,
                ),
                diffusion=dict(
                    T=1000,
                    beta_1=1e-4,
                    beta_T=0.02,
                ),
                model=dict(
                    unet_kwargs=dict(
                        dim_init=128,
                        kernel_size_init=3,
                        dim_mults=(1, 2, 2, 2),
                        attention_resolutions=(16,),
                        attention_num_heads=1,
                        num_res_blocks=2,
                        sinusoidal_embed_dim=128,
                        time_embed_dim=128 * 4,
                        kernel_size=3,
                        num_groups=8,
                    ),
                ),
            )
        )
    )

    config.lock()

    return config
