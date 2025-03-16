import chex
import flax.linen as nn

from foundrix.types.base import Observation


class ObservationInput(nn.Module):
    """Only Observation input."""

    @nn.compact
    def __call__(self, observation: Observation) -> chex.Array:
        agent_view = observation.agent_view
        return agent_view
