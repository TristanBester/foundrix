from functools import cached_property
from typing import Tuple

import chex
import jax.numpy as jnp
import numpy as np
from jumanji.env import Environment, State
from jumanji.specs import Array, Spec
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper

# Import only one Observation and use it for type annotation
from foundrix.types.base import Observation


class FlattenObservationWrapper(Wrapper):
    """Simple wrapper that flattens the agent view observation."""

    def __init__(self, env: Environment) -> None:
        self._env = env
        obs_shape = self._env.observation_spec.agent_view.shape  # type: ignore
        self._obs_shape = (np.prod(obs_shape),)

    def _flatten(self, obs: Observation) -> chex.Array:
        agent_view = obs.agent_view.astype(jnp.float32)
        return agent_view.reshape(self._obs_shape)

    def reset(self, key: chex.PRNGKey) -> tuple[State, TimeStep]:  # type: ignore
        state, timestep = self._env.reset(key)
        agent_view = self._flatten(timestep.observation)

        timestep = timestep.replace(  # type: ignore
            observation=timestep.observation._replace(agent_view=agent_view),
        )
        if "next_obs" in timestep.extras:
            agent_view = self._flatten(timestep.extras["next_obs"])
            timestep.extras["next_obs"] = timestep.extras["next_obs"]._replace(
                agent_view=agent_view
            )
        return state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        state, timestep = self._env.step(state, action)
        agent_view = self._flatten(timestep.observation)
        timestep = timestep.replace(  # type: ignore
            observation=timestep.observation._replace(agent_view=agent_view),
        )
        if "next_obs" in timestep.extras:
            agent_view = self._flatten(timestep.extras["next_obs"])
            timestep.extras["next_obs"] = timestep.extras["next_obs"]._replace(
                agent_view=agent_view
            )
        return state, timestep

    @cached_property
    def observation_spec(self) -> Spec:
        return self._env.observation_spec.replace(
            agent_view=Array(shape=self._obs_shape, dtype=jnp.float32)
        )
