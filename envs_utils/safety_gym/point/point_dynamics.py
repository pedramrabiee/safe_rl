import numpy as np
from dynamics.nominal_dynamics import NominalDynamics
from envs_utils.gym.pendulum.pendulum_configs import env_config
from dynamics.torch_dynamics import AffineInControlTorchDyn
import torch


class PointNominalDynamics(NominalDynamics):
    def initialize(self, params, init_dict=None):
        #TODO: You should link this with the params, so that if you are normalizing the observation or action, apply the same thing here
        self.m = params['m']
        self.I = params['I']
        self.cs = params['cs']
        self.cr = params['cr']
        self.com_loc = params['com_loc']

        self.continous_time = init_dict['is_continuous']

    def _predict(self, obs, ac, split_return=False):
        m = self.m
        I = self.I
        cs = self.cs
        cr = self.cr
        r = self.com_loc
        dt = self.timestep

        # TODO: Fix and check this: currently under the assumption that obs vector is [x, y, theta, x_dot, y_dot, theta_dot]
        x1, x2, x3, x4, x5, x6, x7 = None, None, None, None, None, None, None
        if obs.shape[-1] == 6:
            # [x, y, theta, x_dot, y_dot, theta_dot]
            x1 = obs[..., 0]  # x
            x2 = obs[..., 1]  # y
            x3 = np.cos(obs[..., 2])  # cos(theta)
            x4 = np.sin(obs[..., 2])  # sin(theta)
            x5 = obs[..., 3]  # x_dot
            x6 = obs[..., 4]  # y_dot
            x7 = obs[..., 5]  # theta_dot

        elif obs.shape[-1] == 7:
            # [x, y, cos(theta), sin(theta), x_dot, y_dot, theta_dot]

            x1 = obs[..., 0]  # x
            x2 = obs[..., 1]  # y
            x3 = obs[..., 2]  # cos(theta)
            x4 = obs[..., 3]  # sin(theta)
            x5 = obs[..., 4]  # x_dot
            x6 = obs[..., 5]  # y_dot
            x7 = obs[..., 6]  # theta_dot


        if not self.continous_time:
            raise NotImplementedError
        else:
            f_func = lambda x3, x4, x5, x6, x7: \
                np.array([
                    x5,
                    x6,
                    -x7 * x4,
                    x7 * x3,
                    -(cs / m) * (x5 - r * x7 * x4),
                    -(cs / m) * (x6 + r * x7 * x3),
                    -(cr + cs * r ** 2) / I * x7 - r * cs / I * (-x5 * x4 + x6 * x3)
                ], dtype=np.float32)
            G_func = lambda x3, x4: \
                np.array([
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [x3 / m, 0.0],
                    [x4 / m, 0.0],
                    [0.0, 1 / I]
                ], dtype=np.float32)

            f = np.stack(list(map(f_func, x3, x4, x5, x6, x7)), axis=0)
            G = np.stack(list(map(G_func, x3, x4)), axis=0)
        return (f, G) if split_return else f + np.matmul(G, ac).squeeze(-1)


class PointTorchDyn(AffineInControlTorchDyn):
    def f(self, state):
        # TODO: Change it to accept batch
        return torch.stack([state[1],
                            -env_config.c_slide / env_config.m * (state[1] - env_config.r * state[5] * torch.sin(state[4])),
                            state[3],
                            -env_config.c_slide / env_config.m * (state[3] + env_config.r * state[5] * torch.cos(state[4])),
                            state[5],
                            -(env_config.c_hinge + env_config.c_slide * env_config.r ** 2) / env_config.I * state[5] - (env_config.r * env_config.c_slide / env_config.I) * (
                                        -state[1] * torch.sin(state[4]) + state[3] * torch.cos(state[4]))
                            ])

    def g(self, state):
        # TODO: Change it to accept batch
        g = torch.stack([
            torch.tensor([0.0, 0.0]),
            torch.tensor([torch.cos(state[4]) / env_config.m, 0.0]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([torch.sin(state[4]) / env_config.m, 0.0]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([0.0, 1.0 / env_config.I])
        ])