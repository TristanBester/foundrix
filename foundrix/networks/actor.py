import distrax
import flax.linen as nn

from foundrix.networks.input import ObservationInput
from foundrix.types.base import Observation


class FeedForwardActor(nn.Module):
    """Simple feed forward actor network."""

    torso: nn.Module
    action_head: nn.Module
    input_layer: nn.Module = ObservationInput()

    @nn.compact
    def __call__(self, observation: Observation) -> distrax.DistributionLike:
        agent_view = self.input_layer(observation)
        embedding = self.torso(agent_view)
        return self.action_head(embedding)
