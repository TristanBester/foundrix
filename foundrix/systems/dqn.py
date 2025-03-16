import chex
import flashbax as fbx
import hydra
import jax
import jax.numpy as jnp
import optax
from colorama import Fore, Style
from jumanji.env import Environment
from omegaconf import DictConfig

from foundrix.env.factory.jumanji import make_jumanji_env
from foundrix.networks.actor import FeedForwardActor
from foundrix.systems.dqn_types import Transition
from foundrix.types.base import OnlineAndTarget


def setup_networks(config: DictConfig, env: Environment, key: chex.PRNGKey):
    q_network_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    q_network_head = hydra.utils.instantiate(
        config.network.actor_network.action_head,
        action_dim=env.action_spec.num_values,
        epsilon=config.system.training_epsilon,
    )
    eval_q_network_head = hydra.utils.instantiate(
        config.network.actor_network.action_head,
        action_dim=env.action_spec.num_values,
        epsilon=config.system.evaluation_epsilon,
    )

    q_network = FeedForwardActor(
        torso=q_network_torso,
        action_head=q_network_head,
    )
    eval_q_network = FeedForwardActor(
        torso=q_network_torso,
        action_head=eval_q_network_head,
    )

    q_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(config.system.q_lr, eps=1e-5),
    )

    init_obs = env.observation_spec.generate_value()
    init_obs_batch = jax.tree_util.tree_map(lambda x: x[None, ...], init_obs)

    q_online_params = q_network.init(key, init_obs_batch)
    q_target_params = q_online_params
    q_opt_state = q_optim.init(q_online_params)

    params = OnlineAndTarget(online=q_online_params, target=q_target_params)
    opt_state = q_opt_state

    # Pack apply and update fns
    apply_fn = q_network.apply
    update_fn = q_optim.update

    return params, opt_state, apply_fn, update_fn


def setup_buffer(config: DictConfig, env: Environment):
    n_devices = jax.device_count()

    init_obs = env.observation_spec.generate_value()
    init_obs_batch = jax.tree_util.tree_map(lambda x: x[None, ...], init_obs)

    # Create replay buffer
    dummy_transition = Transition(
        obs=jax.tree_util.tree_map(lambda x: x.squeeze(0), init_obs_batch),
        action=jnp.zeros((), dtype=int),
        reward=jnp.zeros((), dtype=float),
        done=jnp.zeros((), dtype=bool),
        next_obs=jax.tree_util.tree_map(lambda x: x.squeeze(0), init_obs_batch),
        info={"episode_return": 0.0, "episode_length": 0, "is_terminal_step": False},
    )
    assert config.system.total_buffer_size % n_devices == 0, (
        f"{Fore.RED}{Style.BRIGHT}The total buffer size should be divisible "
        + "by the number of devices!{Style.RESET_ALL}"
    )
    assert config.system.total_batch_size % n_devices == 0, (
        f"{Fore.RED}{Style.BRIGHT}The total batch size should be divisible "
        + "by the number of devices!{Style.RESET_ALL}"
    )
    config.system.buffer_size = config.system.total_buffer_size // (
        n_devices * config.arch.update_batch_size
    )
    config.system.batch_size = config.system.total_batch_size // (
        n_devices * config.arch.update_batch_size
    )
    buffer_fn = fbx.make_item_buffer(
        max_length=config.system.buffer_size,
        min_length=config.system.batch_size,
        sample_batch_size=config.system.batch_size,
        add_batches=True,
        add_sequences=True,
    )
    buffer_fns = (buffer_fn.add, buffer_fn.sample)
    buffer_state = buffer_fn.init(dummy_transition)
    return buffer_state, buffer_fns


@hydra.main(
    config_path="../config/default",
    config_name="dqn.yaml",
    version_base="1.2",
)
def main(config: DictConfig) -> None:
    train_env, eval_env = make_jumanji_env(config.env.scenario.name, config)
    key = jax.random.PRNGKey(config.system.seed)
    key, subkey = jax.random.split(key)
    params, opt_state, apply_fn, update_fn = setup_networks(config, train_env, subkey)
    buffer_state, buffer_fns = setup_buffer(config, train_env)

    breakpoint()


if __name__ == "__main__":
    main()
