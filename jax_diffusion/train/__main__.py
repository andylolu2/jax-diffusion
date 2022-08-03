from absl import app, flags
from ml_collections import config_flags

from jax_diffusion.train.main import main

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    name="dry_run",
    default=False,
    help="Whether to do a dry run or not. (No logging, no evaluation, no checkpointing)",
)

_CONFIG = config_flags.DEFINE_config_file(
    name="config",
    help_string="Training configuration file.",
)
flags.mark_flag_as_required("config")

app.run(lambda argv: main(_CONFIG.value, FLAGS.dry_run))
