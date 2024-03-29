from jax_diffusion.configs.base import get_config as get_base_config
from jax_diffusion.types import Config


def get_config() -> Config:
    config = get_base_config()

    config.effective_steps = 150000
    config.ckpt_interval = 600
    config.eval_interval = 600

    exp_conf = config.experiment_kwargs.config
    exp_conf.dataset_kwargs.name = "celeb_a"
    exp_conf.dataset_kwargs.resize_dim = 128
    exp_conf.train.dataset_kwargs.batch_size = 32
    exp_conf.eval.dataset_kwargs.batch_size = 32
    exp_conf.eval.dataset_kwargs.subset = "20%"
    exp_conf.eval.sample_kwargs.num = 6

    exp_conf.train.optimizer.kwargs.grac_acc_steps = 2

    exp_conf.model.unet_kwargs.dim_init = 96
    exp_conf.model.unet_kwargs.dim_mults = (1, 1, 2, 2, 2)

    return config
