from pathlib import Path

from ml_collections import ConfigDict, FrozenConfigDict

from jax_diffusion.types import Config


def get_config() -> Config:
    config = ConfigDict()

    config.project_name = "jax-diffusion"
    # config.restore = str(Path.cwd() / "checkpoints/fallen-pond-139")
    config.restore = None
    config.dry_run = True

    seed = 42
    d_model = 64
    grad_acc = 1
    steps = 30000

    config.seed = seed
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
                    name="celeb_a",
                    resize_dim=64,
                    data_dir=str(Path.home() / "tensorflow_datasets"),
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
                    ),
                    ema_step_size=1 - 0.9995,
                    optimizer=dict(
                        optimizer_type="adam",
                        kwargs=dict(
                            max_grad_norm=1.0,
                            grac_acc_steps=grad_acc,
                        ),
                        lr_schedule=dict(
                            # schedule_type="constant",
                            # kwargs=dict(
                            #     value=3e-6,
                            # ),
                            schedule_type="cosine",
                            kwargs=dict(
                                init_value=0,
                                peak_value=4e-4,
                                warmup_steps=500,
                                decay_steps=steps * grad_acc,
                                decay_factor=10,
                            ),
                        ),
                    ),
                ),
                eval=dict(
                    dataset_kwargs=dict(
                        batch_size=64,
                        subset="20%",
                    ),
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
                        dim_mults=(1, 2, 2, 4, 4),
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

    return FrozenConfigDict(config)
