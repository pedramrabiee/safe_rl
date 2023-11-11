import gym
import numpy as np
from scipy.integrate import solve_ivp


class RK45PendulumWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, u):
        th, thdot = self.state  # th := theta

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

        # TODO: Check timestep
        ivp = solve_ivp(fun=lambda t, y: self.dynamics(t, y, u), t_span=[0, self.dt], y0=self.state)
        self.state = ivp.y[:, -1]

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), -costs, False, False, {}

    def dynamics(self, t, y, u):
        # FIXME: add f_numpy and g_numpy
        return self.f_numpy(y) + self.g_numpy(y) * u


def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)