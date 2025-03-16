import chex
import distrax
import flax.linen as nn
import numpy as np
from flax.linen.initializers import Initializer, orthogonal


class DiscreteQNetworkHead(nn.Module):
    """Discrete Q-network head."""

    action_dim: int
    epsilon: float = 0.1
    kernel_init: Initializer = orthogonal(np.sqrt(1.0))

    @nn.compact
    def __call__(self, embedding: chex.Array) -> distrax.EpsilonGreedy:
        q_values = nn.Dense(
            features=self.action_dim,
            kernel_init=self.kernel_init,
        )(embedding)
        return distrax.EpsilonGreedy(preferences=q_values, epsilon=self.epsilon)
