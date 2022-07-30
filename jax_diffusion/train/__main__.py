from jax_diffusion.config import get_config
from jax_diffusion.train.main import main

config = get_config()
main(config)
