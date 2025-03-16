import hydra
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from foundrix.env.factory.jumanji import make_jumanji_env


@hydra.main(
    config_path="../config/defaults",
    config_name="dqn.yaml",
    version_base="1.2",
)
def main(config: DictConfig) -> None:
    train_env, eval_env = make_jumanji_env(config.env.scenario.name, config)

    state, timestep = train_env.reset(jax.random.PRNGKey(0))

    print(train_env.observation_spec)


if __name__ == "__main__":
    main()
