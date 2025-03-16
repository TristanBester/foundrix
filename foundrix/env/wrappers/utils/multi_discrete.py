from functools import cached_property

import chex
import jax.numpy as jnp
import numpy as np
from jumanji import specs
from jumanji.env import Environment, State
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper

from foundrix.types.base import Observation


class MultiDiscreteToDiscreteWrapper(Wrapper):
    def __init__(self, env: Environment):
        super().__init__(env)
        self._action_spec_num_values = env.action_spec().num_values

    def apply_factorisation(self, action: chex.Array) -> chex.Array:
        """Applies the factorisation to the given action.

        Convert a multi-discrete action to a single discrete action.
        """
        action_components = []
        flat_action = action
        n = self._action_spec_num_values.shape[0]

        for i in range(n - 1, 0, -1):
            flat_action, remainder = jnp.divmod(
                flat_action, self._action_spec_num_values[i]
            )
            action_components.append(remainder)

        action_components.append(flat_action)
        action = jnp.stack(
            list(reversed(action_components)),
            axis=-1,
            dtype=self._action_spec_num_values.dtype,
        )
        return action

    def apply_inverse_factorisation(self, action: chex.Array) -> chex.Array:
        """Applies the inverse factorisation to the given action.

        Convert a single discrete action to a multi-discrete action.
        """
        n = self._action_spec_num_values.shape[0]

        action_components = jnp.split(action, n, axis=-1)
        flat_action = action_components[0]

        for i in range(1, n):
            flat_action = (
                self._action_spec_num_values[i] * flat_action + action_components[i]
            )
        return flat_action

    def step(
        self, state: State, action: chex.Array
    ) -> tuple[State, TimeStep[Observation]]:
        action = self.apply_factorisation(action)
        state, timestep = self._env.step(state, action)
        return state, timestep

    @cached_property
    def action_spec(self) -> specs.Spec:
        """Returns the action spec of the environment."""
        original_action_spec = self._env.action_spec()
        num_actions = int(np.prod(np.asarray(original_action_spec.num_values)))
        return specs.DiscreteArray(num_actions, name="action")
