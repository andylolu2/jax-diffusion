from absl import app, flags
from ml_collections import config_flags

from jax_diffusion.train.main import main

FLAGS = flags.FLAGS

_CONFIG = config_flags.DEFINE_config_file(
    name="config",
    help_string="Training configuration file.",
)
flags.mark_flag_as_required("config")

app.run(lambda argv: main(_CONFIG.value))
