import chex
import jax
import jax.numpy as jnp
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper

from foundrix.types.alias import State


@chex.dataclass
class RecordEpisodeMetricsState:
    """State of the `LogWrapper`."""

    env_state: State
    key: chex.PRNGKey
    # Temporary variables to trace metrics within current episode
    curr_episode_return: chex.Numeric = 0.0
    curr_episode_length: chex.Numeric = 0
    # Final episode return and length
    episode_return: chex.Numeric = 0.0
    episode_length: chex.Numeric = 0


class RecordEpisodeMetricsWrapper(Wrapper):
    """Record episode returns and lengths."""

    def reset(self, key: chex.PRNGKey) -> tuple[RecordEpisodeMetricsState, TimeStep]:
        """Reset the environment and initialise the temporary variables."""
        key, subkey = jax.random.split(key)
        state, timestep = self._env.reset(subkey)
        state = RecordEpisodeMetricsState(
            env_state=state,
            key=key,
        )
        timestep.extras["episode_metrics"] = {
            "episode_return": jnp.array(0.0, dtype=float),
            "episode_length": jnp.array(0, dtype=int),
            "is_terminal_step": jnp.array(False, dtype=bool),
        }
        return state, timestep

    def step(
        self, state: RecordEpisodeMetricsState, action: chex.Array
    ) -> tuple[RecordEpisodeMetricsState, TimeStep]:
        """Step the environment with the given action."""
        state, timestep = self._env.step(state.env_state, action)

        # Update temporary variables
        curr_episode_return = state.curr_episode_return + timestep.reward
        curr_episode_length = state.curr_episode_length + 1

        # Persist previous episode metrics until current episode is done
        episode_return = jax.lax.select(
            timestep.last(), curr_episode_return, state.episode_return
        )
        episode_length = jax.lax.select(
            timestep.last(), curr_episode_length, state.episode_length
        )

        # Add episode metrics to the timestep extra information
        timestep.extras["episode_metrics"] = {
            "episode_return": episode_return,
            "episode_length": episode_length,
            "is_terminal_step": timestep.last(),
        }

        state = RecordEpisodeMetricsState(
            env_state=state,
            key=state.key,
            curr_episode_return=curr_episode_return,
            curr_episode_length=curr_episode_length,
            episode_return=episode_return,
            episode_length=episode_length,
        )
        return state, timestep
