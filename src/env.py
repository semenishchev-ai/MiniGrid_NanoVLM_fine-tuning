import gymnasium as gym
import minigrid
import numpy as np
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper

ACTION_NAMES = ("left", "right", "forward")
NAME_TO_ACTION = {name: i for i, name in enumerate(ACTION_NAMES)}


def make_empty_env(env_name="MiniGrid-Empty-Random-6x6-v0", max_steps=100, tile_size=None):
    env = gym.make(env_name, max_steps=max_steps)
    if tile_size is None:
        tile_size = env.unwrapped.tile_size
    env = RGBImgObsWrapper(env, tile_size=tile_size)
    env = ImgObsWrapper(env)
    return env


class MiniGridWrapper:
    def __init__(
        self,
        env_name="MiniGrid-Empty-Random-6x6-v0",
        max_steps=100,
        tile_size=None,
    ):
        self.env_name = env_name
        self.max_steps = max_steps
        self._env = make_empty_env(env_name, max_steps, tile_size)

    @property
    def unwrapped(self):
        return self._env.unwrapped

    def reset(self, seed=None):
        obs, info = self._env.reset(seed=seed)
        return np.asarray(obs, dtype=np.uint8), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        return np.asarray(obs, dtype=np.uint8), reward, terminated, truncated, info

    def close(self):
        self._env.close()
