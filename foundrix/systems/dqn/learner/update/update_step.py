from typing import Any, Callable, Tuple

import jax
import optax
from jumanji.env import Environment
from omegaconf import DictConfig

from foundrix.systems.dqn.learner.update.env_step import get_env_step_fn
from foundrix.systems.dqn.learner.update.update_epoch import get_update_epoch_fn
from foundrix.types.base import ActorApply, OffPolicyLearnerState


def get_update_step_fn(
    env: Environment,
    q_apply_fn: ActorApply,
    q_update_fn: optax.TransformUpdateFn,
    buffer_add_fn: Callable,
    buffer_sample_fn: Callable,
    config: DictConfig,
) -> Callable:
    def _update_step(
        learner_state: OffPolicyLearnerState, _: Any
    ) -> Tuple[OffPolicyLearnerState, Tuple]:
        # Get scan functions
        env_step = get_env_step_fn(env, q_apply_fn)
        update_epoch = get_update_epoch_fn(
            q_apply_fn, q_update_fn, buffer_sample_fn, config
        )

        # Unpack state
        params, opt_states, buffer_state, key, env_state, last_timestep = learner_state

        # ROLLOUT
        learner_state, traj_batch = jax.lax.scan(
            env_step, learner_state, None, config.system.rollout_length
        )
        buffer_state = buffer_add_fn(buffer_state, traj_batch)

        # UPDATE
        update_state = (params, opt_states, buffer_state, key)
        update_state, loss_info = jax.lax.scan(
            update_epoch, update_state, None, config.system.epochs
        )

        params, opt_states, buffer_state, key = update_state
        learner_state = OffPolicyLearnerState(
            params, opt_states, buffer_state, key, env_state, last_timestep
        )
        metric = traj_batch.info
        return learner_state, (metric, loss_info)

    return _update_step
