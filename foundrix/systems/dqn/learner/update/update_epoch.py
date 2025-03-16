from typing import Any, Callable, Tuple

import chex
import jax
import jax.numpy as jnp
import optax
import rlax
from flax.core.frozen_dict import FrozenDict
from omegaconf import DictConfig

from foundrix.systems.dqn_types import Transition
from foundrix.types.base import ActorApply, OnlineAndTarget


def get_update_epoch_fn(
    q_apply_fn: ActorApply,
    q_update_fn: optax.TransformUpdateFn,
    buffer_sample_fn: Callable,
    config: DictConfig,
) -> Callable:
    def _q_loss_fn(
        q_params: FrozenDict,
        target_q_params: FrozenDict,
        transitions: Transition,
    ) -> jnp.ndarray:
        q_tm1 = q_apply_fn(q_params, transitions.obs).preferences  # type: ignore
        q_t = q_apply_fn(target_q_params, transitions.next_obs).preferences  # type: ignore

        # Cast and clip rewards.
        discount = 1.0 - transitions.done.astype(jnp.float32)
        d_t = (discount * config.system.gamma).astype(jnp.float32)
        r_t = jnp.clip(
            transitions.reward,
            -config.system.max_abs_reward,
            config.system.max_abs_reward,
        ).astype(jnp.float32)
        a_tm1 = transitions.action

        # Compute Q-learning loss.
        batch_loss = q_learning(
            q_tm1,
            a_tm1,
            r_t,
            d_t,
            q_t,
            config.system.huber_loss_parameter,
        )

        loss_info = {
            "q_loss": batch_loss,
        }
        return batch_loss, loss_info  # type: ignore

    def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
        """Update the network for a single epoch."""
        params, opt_states, buffer_state, key = update_state
        key, sample_key = jax.random.split(key)

        # SAMPLE TRANSITIONS
        transition_sample = buffer_sample_fn(buffer_state, sample_key)
        transitions: Transition = transition_sample.experience

        # CALCULATE Q LOSS
        q_grad_fn = jax.grad(_q_loss_fn, has_aux=True)
        q_grads, q_loss_info = q_grad_fn(
            params.online,
            params.target,
            transitions,
        )

        # Compute the parallel mean (pmean) over the batch.
        # This calculation is inspired by the Anakin architecture demo notebook.
        # available at https://tinyurl.com/26tdzs5x
        # This pmean could be a regular mean as the batch axis is on the same device.
        q_grads, q_loss_info = jax.lax.pmean((q_grads, q_loss_info), axis_name="batch")
        q_grads, q_loss_info = jax.lax.pmean((q_grads, q_loss_info), axis_name="device")

        # UPDATE Q PARAMS AND OPTIMISER STATE
        q_updates, q_new_opt_state = q_update_fn(q_grads, opt_states)
        q_new_online_params = optax.apply_updates(params.online, q_updates)
        # Target network polyak update.
        new_target_q_params = optax.incremental_update(
            q_new_online_params, params.target, config.system.tau
        )
        q_new_params = OnlineAndTarget(q_new_online_params, new_target_q_params)  # type: ignore

        # PACK NEW PARAMS AND OPTIMISER STATE
        new_params = q_new_params
        new_opt_state = q_new_opt_state

        # PACK LOSS INFO
        loss_info = {
            **q_loss_info,
        }
        return (new_params, new_opt_state, buffer_state, key), loss_info

    return _update_epoch


def q_learning(
    q_tm1: chex.Array,
    a_tm1: chex.Array,
    r_t: chex.Array,
    d_t: chex.Array,
    q_t: chex.Array,
    huber_loss_parameter: chex.Array,
) -> jnp.ndarray:
    """Computes the double Q-learning loss. Each input is a batch."""
    batch_indices = jnp.arange(a_tm1.shape[0])  # type: ignore
    # Compute Q-learning n-step TD-error.
    target_tm1 = r_t + d_t * jnp.max(q_t, axis=-1)  # type: ignore
    td_error = target_tm1 - q_tm1[batch_indices, a_tm1]  # type: ignore
    if huber_loss_parameter > 0.0:
        batch_loss = rlax.huber_loss(td_error, huber_loss_parameter)  # type: ignore
    else:
        batch_loss = rlax.l2_loss(td_error)

    return jnp.mean(batch_loss)
