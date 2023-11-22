import gym
import numpy as np
from scipy.integrate import solve_ivp

import numpy as np
from gym.spaces import Box
from envs_utils.gym.pendulum.pendulum_configs import env_config

def pendulum_customize(env):
    # Settings
    # env.env.max_torque = max_torque  # you could also used env.unwrapped.max_torque
    env.unwrapped.max_torque = env_config.max_torque
    env.unwrapped.max_speed = env_config.max_speed  # you could also used env.unwrapped.max_speed
    env.unwrapped.dt = env_config.timestep
    env.m = env_config.m
    env.g = env_config.g
    env.l = env_config.l

    env.action_space = Box(
        low=-env_config.max_torque,
        high=env_config.max_torque, shape=(1,),
        dtype=np.float32
    )
    high = np.array([1., 1., env_config.max_speed], dtype=np.float32)
    env.observation_space = Box(
        low=-high,
        high=high,
        dtype=np.float32
    )

    return env


class RK45PendulumWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.state = None

    def step(self, u):
        th, thdot = self.state  # th := theta

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

        # TODO: Check timestep
        ivp = solve_ivp(fun=lambda t, y: self._dynamics(t, y, u), t_span=[0, self.dt], y0=self.state)
        self.state = ivp.y[:, -1]

        # if self.render_mode == "human":
        #     self.render()
        return self._get_obs(), -costs, False, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def _dynamics(self, t, y, u):
        return np.array([y[1], 3 * self.g / (2 * self.l) * np.sin(y[0]) + 3 * u / (self.m * self.l**2)])


def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)