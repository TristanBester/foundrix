from functools import cached_property

import chex
import jax.numpy as jnp
from jumanji.env import Environment, State
from jumanji.specs import Array, MultiDiscreteArray, Spec
from jumanji.types import TimeStep
from jumanji.wrappers import MultiToSingleWrapper, Wrapper

from foundrix.env.wrappers.utils.multi_discrete import MultiDiscreteToDiscreteWrapper
from foundrix.types.base import Observation


class JumanjiWrapper(Wrapper):
    def __init__(
        self, env: Environment, observation_attribute: str, multi_agent: bool = False
    ):
        super().__init__(env)

        # Convert multi-discrete actions to single discrete actions
        if isinstance(env.action_spec, MultiDiscreteArray):
            env = MultiDiscreteToDiscreteWrapper(env)

        # Convert multi-agent environment to single-agent environment
        if multi_agent:
            env = MultiToSingleWrapper(env)

        self._env = env
        self._observation_attribute = observation_attribute
        self._num_actions = self.action_spec.num_values

    def reset(self, key: chex.PRNGKey) -> tuple[State, TimeStep]:  # type: ignore
        state, timestep = self._env.reset(key)

        # Compute the legal action mask
        legal_action_mask = self._compute_legal_action_mask(timestep)

        # Extract the part of the observation with the agent's observation of the state
        agent_view = self._extract_agent_view(timestep)

        # Create an observation object
        observation = Observation(
            agent_view=agent_view,
            action_mask=legal_action_mask,
            step_count=state.step_count,
        )

        # Repack the timestep
        timestep = self._repack_timestep(timestep, observation)
        return state, timestep

    def step(
        self, state: State, action: chex.Array
    ) -> tuple[State, TimeStep[Observation]]:
        state, timestep = self._env.step(state, action)

        # Compute the legal action mask
        legal_action_mask = self._compute_legal_action_mask(timestep)

        # Extract the part of the observation with the agent's observation of the state
        agent_view = self._extract_agent_view(timestep)

        # Create an observation object
        observation = Observation(
            agent_view=agent_view,
            action_mask=legal_action_mask,
            step_count=state.step_count,  # type: ignore
        )

        # Repack the timestep
        timestep = self._repack_timestep(timestep, observation)
        return state, timestep

    def _compute_legal_action_mask(self, timestep: TimeStep) -> chex.Array:
        """Compute the legal action mask for the current timestep."""
        if hasattr(timestep.observation, "action_mask"):
            return timestep.observation.action_mask.astype(jnp.float32)
        else:
            return jnp.ones((self._num_actions,), dtype=jnp.float32)

    def _extract_agent_view(self, timestep: TimeStep) -> chex.Array:
        """Extract part of the observation with the agent's observation of the state."""
        if self._observation_attribute:
            return timestep.observation._asdict()[self._observation_attribute].astype(
                jnp.float32
            )
        else:
            return timestep.observation.astype(jnp.float32)

    def _repack_timestep(
        self, timestep: TimeStep, observation: Observation
    ) -> TimeStep:
        """Repack the timestep with the new observation."""
        # Handle extra information in the timestep before repacking
        extra_info = timestep.extras if timestep.extras is not None else {}

        # Repack the timestep
        timestep = timestep.replace(  # type: ignore
            observation=observation,
            extras=extra_info,
        )
        return timestep

    @cached_property
    def observation_spec(self) -> Spec:
        if self._observation_attribute:
            agent_view_spec = Array(
                shape=self._env.observation_spec.__dict__[
                    self._observation_attribute
                ].shape,
                dtype=float,
            )
        else:
            agent_view_spec = self._env.observation_spec
        return Spec(
            Observation,
            "ObservationSpec",
            agent_view=agent_view_spec,
            action_mask=Array(shape=(self._num_actions,), dtype=float),
            step_count=Array(shape=(), dtype=int),
        )
