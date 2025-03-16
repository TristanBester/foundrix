from typing import Callable, Generic, NamedTuple, Optional, TypeVar

import chex
import distrax
from flashbax.buffers.trajectory_buffer import BufferState
from flax.core.frozen_dict import FrozenDict
from jumanji.types import TimeStep

from foundrix.types.alias import OptStates, Parameters, State

ActorApply = Callable[..., distrax.DistributionLike]
StoixState = TypeVar(
    "StoixState",
)


@chex.dataclass
class LogEnvState:
    """State of the `LogWrapper`."""

    env_state: State
    episode_returns: chex.Numeric
    episode_lengths: chex.Numeric
    # Information about the episode return and length for logging purposes.
    episode_return_info: chex.Numeric
    episode_length_info: chex.Numeric


class Observation(NamedTuple):
    """The observation that the agent sees.

    agent_view: the agent's view of the environment.
    action_mask: boolean array specifying which action is legal.
    step_count: the number of steps elapsed since the beginning of the episode.
    """

    agent_view: chex.Array  # (num_obs_features,)
    action_mask: chex.Array  # (num_actions,)
    step_count: Optional[chex.Array] = None  # (,)


class OnlineAndTarget(NamedTuple):
    online: FrozenDict
    target: FrozenDict


class OffPolicyLearnerState(NamedTuple):
    params: Parameters
    opt_states: OptStates
    buffer_state: BufferState  # type: ignore
    key: chex.PRNGKey
    env_state: LogEnvState
    timestep: TimeStep


class AnakinExperimentOutput(NamedTuple, Generic[StoixState]):
    """Experiment output."""

    learner_state: StoixState
    episode_metrics: dict[str, chex.Array]
    train_metrics: dict[str, chex.Array]
