from __future__ import annotations

from typing import Any, Generator, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl.flags import FlagValues
from typing_extensions import TypedDict

ndarray = Union[jnp.ndarray, np.ndarray]
Rng = jax.random.KeyArray
Params = optax.Params
Config = Any
Flags = FlagValues


class Batch(TypedDict):
    image: ndarray
    label: ndarray | None


Dataset = Generator[Batch, None, None]
