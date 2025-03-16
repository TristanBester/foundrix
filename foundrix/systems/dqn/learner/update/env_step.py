from typing import Any, Callable, Tuple

import jax
from jumanji.env import Environment

from foundrix.systems.dqn_types import Transition
from foundrix.types.base import ActorApply, OffPolicyLearnerState


def get_env_step_fn(env: Environment, q_apply_fn: ActorApply) -> Callable:
    def _env_step(
        learner_state: OffPolicyLearnerState, _: Any
    ) -> Tuple[OffPolicyLearnerState, Transition]:
        """Step the environment."""
        q_params, opt_states, buffer_state, key, env_state, last_timestep = (
            learner_state
        )

        # SELECT ACTION
        key, policy_key = jax.random.split(key)
        actor_policy = q_apply_fn(q_params.online, last_timestep.observation)
        action = actor_policy.sample(seed=policy_key)

        # STEP ENVIRONMENT
        env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

        # LOG EPISODE METRICS
        done = timestep.last().reshape(-1)
        info = timestep.extras["episode_metrics"]
        next_obs = timestep.extras["next_obs"]

        transition = Transition(
            last_timestep.observation, action, timestep.reward, done, next_obs, info
        )

        learner_state = OffPolicyLearnerState(
            q_params, opt_states, buffer_state, key, env_state, timestep
        )
        return learner_state, transition

    return _env_step
