import gymnasium
from envs.create_env import make_env
from envs.stretch import BaseStretchEnv
from envs.drawer.stretch_drawer import StretchDrawer
from envs.push.single import SinglePush
from envs.nav.nav import StretchNav


gymnasium.register(
    "StretchDrawer",
    "envs:StretchDrawer",
)

gymnasium.register(
    "singlepush",
    "envs:SinglePush",
)

gymnasium.register(
    "StretchNav",
    "envs:StretchNav",
)