import numpy as np
from gym.spaces import Box
from envs_utils.misc_env.cbf_test.cbf_test_configs import env_config
from control.matlab import *
import gym
from gym.utils import seeding
from copy import copy


def get_dynamics_matrices():
    m = env_config.m
    k = env_config.k
    c = env_config.c
    A = np.array([[0, 1], [-k / m, -c / m]])
    B = np.array([[0.0], [1 / m]])
    C = np.array([[1, 0]])
    D = np.zeros((1, 1), dtype=np.float32)
    return A, B, C, D

class CbfTestEnv(gym.Env):
    env_config = env_config
    def __init__(self):
        self.dt = env_config.timestep
        self.timestep = env_config.timestep
        self.max_episode_len = int(env_config.max_episode_time / env_config.timestep)

        self.max_x = self.env_config.max_x
        self.max_u = self.env_config.max_u
        self.m = self.env_config.m
        self.k = self.env_config.k
        self.c = self.env_config.c
        self.max_speed = self.env_config.max_speed

        # parse parameters, initialize dynamics, action and observation spaces
        self._initialize()

        self.seed()
        self.rng = np.random.default_rng(0)
        self.reset_called = False

    def _initialize(self):
        A, B, C, D = get_dynamics_matrices()
        D1 = B
        sys = ss(A, np.hstack((B, D1)), C, np.zeros((1, B.shape[1]+D1.shape[1])))
        self.sys_d = c2d(sys, self.timestep)

        self.action_space = Box(
            low=-self.max_u, high=self.max_u, shape=(1,), dtype=np.float32
        )
        high = np.array([self.max_x, self.max_speed], dtype=np.float32)
        self.observation_space = Box(low=-high, high=high, dtype=np.float32)

    def reset(self):
        if self.env_config.fixed_reset and not self.reset_called:
            self.rng = np.random.default_rng(0)
            self.reset_called = True
        self.state = self._reset()
        return copy(self.state)

    def _reset(self):
        high = np.array([self.max_x, self.env_config.max_speed_for_safe_set_training])
        return self.rng.uniform(low=-high, high=high)

    def step(self, u):
        if self.reset_called:
            self.reset_called = False
        # u = np.clip(u, -self.max_u, self.max_u)
        state = self.sys_d.A.A @ self.state + self.sys_d.B.A @ np.hstack((u, np.zeros(u.shape[0])))
        # state[1] = np.clip(state[1], -self.max_speed, self.max_speed)
        self.state = state
        return copy(self.state).squeeze(), np.array([0.0])[0], np.array([0.0]).squeeze(), {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]