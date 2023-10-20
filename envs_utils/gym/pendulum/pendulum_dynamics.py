import numpy as np
from envs_utils.gym.pendulum.pendulum_configs import env_config
from dynamics.nominal_dynamics import NominalDynamics


class InvertedPendulumNominalDyn(NominalDynamics):
    def initialize(self, params, init_dict=None):
        #TODO: You should link this with the params, so that if you are normalizing the observation or action, apply the same thing here
        self.continous_time = init_dict['is_continuous']

    def _predict(self, obs, ac, split_return=False):
        dt = self.timestep
        g = env_config.g
        m = env_config.m
        l = env_config.l

        if obs.shape[-1] == 2:
            x1 = np.cos(obs[..., 0])
            x2 = np.sin(obs[..., 0])
            x3 = obs[..., 1]
        elif obs.shape[-1] == 3:
            x1 = obs[..., 0]    # cos(theta)
            x2 = obs[..., 1]    # sin(theta)
            x3 = obs[..., 2]    # theta_dot

        if not self.continous_time: #TODO: only continuous time is checked
            f = np.stack([
                x1 - dt * x3 * x2 - (dt ** 2) * 3 * g / (2 * l) * x2 ** 2,
                x2 + dt * x3 * x1 + (dt ** 2) * 3 * g / (2 * l) * x2 * x1,
                x3 + dt * 3 * g / (2 * l) * x2
            ], axis=-1)
            G = 3 / (m * l ** 2) * dt * np.stack([dt * x2, dt * x1, 1.0], axis=-1)
        else:
            f = np.stack([-x3 * x2,
                          x3 * x1,
                          3 * g / (2 * l) * x2],
                         axis=-1)
            G = np.array([
                0.0,
                0.0,
                3 / (m * l ** 2)
            ], dtype=np.float32)

            G = np.tile(G, (x1.shape[0], 1))

            # G = np.stack([G for _ in range(x1.shape[0])], axis=0)

        # f = np.stack(list(map(f_func, x1, x2, x3)), axis=0)

        G = np.expand_dims(G, axis=-1)

        return (f, G) if split_return else f + np.matmul(G, ac).squeeze(-1)
