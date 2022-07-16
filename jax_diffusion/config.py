from pathlib import Path

from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    config = ConfigDict()

    config.project_name = "jax-diffusion"

    # config.restore = str(Path.cwd() / "checkpoints/whole-sunset-81")
    config.restore = None

    d_model = 32
    training_steps = 20000

    config.seed = 42
    config.training_steps = training_steps
    config.ckpt_dir = str(Path.cwd() / "checkpoints")
    config.log_interval = 1
    config.ckpt_interval = 30
    config.eval_interval = 60

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
                    subset="100%",
                    optimizer=dict(
                        optimizer_type="adam",
                        kwargs=dict(),
                        lr_schedule=dict(
                            # schedule_type="constant",
                            # kwargs=dict(
                            #     value=3e-4,
                            # ),
                            schedule_type="cosine",
                            kwargs=dict(
                                init_value=0,
                                peak_value=3e-4,
                                warmup_steps=500,
                                decay_steps=training_steps,
                                decay_factor=0.1,
                            ),
                        ),
                    ),
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
                        dim_init=d_model,
                        kernel_size_init=3,
                        dim_mults=(1, 2, 2, 2),
                        attention_resolutions=(16,),
                        attention_num_heads=4,
                        num_res_blocks=2,
                        sinusoidal_embed_dim=d_model,
                        time_embed_dim=4 * d_model,
                        kernel_size=3,
                        num_groups=4,
                    ),
                ),
            )
        )
    )

    config.lock()

    return config
