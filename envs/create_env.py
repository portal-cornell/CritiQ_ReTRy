from typing import Callable
import gymnasium
from stable_baselines3.common.monitor import Monitor


def make_env(
    env_name: str,
    **kwargs,
) -> Callable[[], gymnasium.Env]:
    def make_env_wrapper() -> gymnasium.Env:
        env: gymnasium.Env
        env = gymnasium.make(
            env_name,
            **kwargs,
        )
        return Monitor(env)

    return make_env_wrapper
