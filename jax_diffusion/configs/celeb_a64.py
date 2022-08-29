from jax_diffusion.configs.base import get_config as get_base_config
from jax_diffusion.types import Config


def get_config() -> Config:
    config = get_base_config()

    config.effective_steps = 100000
    config.ckpt_interval = 600
    config.eval_interval = 600

    exp_conf = config.experiment_kwargs.config
    exp_conf.dataset_kwargs.name = "celeb_a"
    exp_conf.dataset_kwargs.resize_dim = 64
    exp_conf.train.dataset_kwargs.batch_size = 64
    exp_conf.train.lr_schedule = dict(
        schedule_type="constant_warmup",
        kwargs=dict(
            value=1e-4,
            warmup_steps=5000,
        ),
    )
    exp_conf.eval.dataset_kwargs.batch_size = 64
    exp_conf.eval.dataset_kwargs.subset = "30%"
    exp_conf.eval.sample_kwargs.num = 6

    exp_conf.model.unet_kwargs.dim_init = 128
    exp_conf.model.unet_kwargs.dim_mults = (1, 1, 2, 4)

    return config
