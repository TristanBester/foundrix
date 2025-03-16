from typing import Callable

import jax
import optax
from jumanji.env import Environment
from omegaconf import DictConfig

from foundrix.systems.dqn.learner.update import get_update_step_fn
from foundrix.types.base import (
    ActorApply,
    AnakinExperimentOutput,
    OffPolicyLearnerState,
)


def get_learner_fn(
    env: Environment,
    q_apply_fn: ActorApply,
    q_update_fn: optax.TransformUpdateFn,
    buffer_fns: tuple[Callable, Callable],
    config: DictConfig,
) -> Callable:
    buffer_add_fn, buffer_sample_fn = buffer_fns
    update_step_fn = get_update_step_fn(
        env=env,
        q_apply_fn=q_apply_fn,
        q_update_fn=q_update_fn,
        buffer_add_fn=buffer_add_fn,
        buffer_sample_fn=buffer_sample_fn,
        config=config,
    )

    def _learner_fn(
        learner_state: OffPolicyLearnerState,
    ) -> AnakinExperimentOutput[OffPolicyLearnerState]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        """
        batched_update_step = jax.vmap(
            update_step_fn, in_axes=(0, None), axis_name="batch"
        )

        learner_state, (episode_info, loss_info) = jax.lax.scan(
            batched_update_step, learner_state, None, config.arch.num_updates_per_eval
        )
        return AnakinExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_info,
            train_metrics=loss_info,
        )

    return _learner_fn
