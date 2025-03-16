from typing import Sequence

import chex
import flax.linen as nn
import numpy as np
from flax.linen.initializers import Initializer, orthogonal

from foundrix.networks.parsers import parse_activation


class MLPTorso(nn.Module):
    """MLP torso."""

    layer_sizes: Sequence[int]
    activation: str = "relu"
    use_layer_norm: bool = False
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))
    activate_final: bool = True

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        use_bias = not self.use_layer_norm
        act_fn = parse_activation(self.activation)

        for layer_size in self.layer_sizes:
            x = nn.Dense(
                features=layer_size, kernel_init=self.kernel_init, use_bias=use_bias
            )(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            if self.activate_final or layer_size != self.layer_sizes[-1]:
                x = act_fn(x)
        return x
