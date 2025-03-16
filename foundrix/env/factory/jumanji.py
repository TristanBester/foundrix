import hydra
import jumanji
from jumanji.env import Environment
from jumanji.wrappers import AutoResetWrapper
from omegaconf import DictConfig

from foundrix.env.wrappers.jumanji.jumanji import JumanjiWrapper
from foundrix.env.wrappers.logging.episode import RecordEpisodeMetricsWrapper


def make_jumanji_env(
    env_name: str, config: DictConfig
) -> tuple[Environment, Environment]:
    """Create training and evaluation environments."""
    train_env = jumanji.make(env_name, **config.env.kwargs)  # type: ignore
    eval_env = jumanji.make(env_name, **config.env.kwargs)  # type: ignore

    # Wrap environments to standardise the observation and action spaces
    train_env = JumanjiWrapper(
        train_env, config.env.observation_attribute, config.env.multi_agent
    )
    eval_env = JumanjiWrapper(
        eval_env, config.env.observation_attribute, config.env.multi_agent
    )

    # Wrap training environment for training convenience and logging
    train_env = AutoResetWrapper(train_env)
    train_env = RecordEpisodeMetricsWrapper(train_env)

    train_env, eval_env = apply_optional_wrappers((train_env, eval_env), config)
    return train_env, eval_env


def apply_optional_wrappers(
    envs: tuple[Environment, Environment], config: DictConfig
) -> tuple[Environment, Environment]:
    """Apply optional wrappers to the environments.

    Args:
        envs (Tuple[Environment, Environment]): The training and evaluation environments to wrap.
        config (Dict): The configuration of the environment.

    Returns:
        A tuple of the environments.
    """
    if "wrapper" in config.env and config.env.wrapper is not None:
        env_list = list(envs)
        for i in range(len(env_list)):
            env_list[i] = hydra.utils.instantiate(config.env.wrapper, env=env_list[i])
    else:
        env_list = envs

    return tuple(env_list)  # type: ignore
