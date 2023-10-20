from envs_utils.misc_env.cbf_test.cbf_test_env import CbfTestEnv
import numpy as np
from gym.spaces import Box
from envs_utils.misc_env.multi_dashpot.multi_dashpot_configs import env_config
from control.matlab import *


def get_dynamics_matrices(ret_discrete=False):
    c = env_config.c
    k = env_config.k
    m = env_config.m
    A = np.array(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [-(k[0] + k[1]) / m[0], -(c[0] + c[1]) / m[0], k[1] / m[0], c[1] / m[0], 0, 0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [k[1] / m[1], c[1] / m[1], -(k[1] + k[2]) / m[1], -(c[1] + c[2]) / m[1], k[2] / m[1], c[2] / m[1]],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0, 0, k[2] / m[2], c[2] / m[2], -k[2] / m[2], -c[2] / m[2]],
        ]
    )
    B = np.array(
        [
            [0.0, 0.0],
            [1.0 / m[0], 0.0],
            [0.0, 0.0],
            [0.0, 1.0 / m[1]],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    C = np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    D = np.zeros((1, B.shape[1]), dtype=np.float32)
    if ret_discrete:
        sys = ss(A,B, C, D)
        sys_d = c2d(sys, env_config.timestep)
        return sys_d.A.A, sys_d.B.A, sys_d.C.A, sys_d.D.A
    return A, B, C, D

class MultiDashpotEnv(CbfTestEnv):
    env_config = env_config
    def _initialize(self):
        A, B, C, D = get_dynamics_matrices()
        D1 = B
        sys = ss(A, np.hstack((B, D1)), C, np.zeros((1, B.shape[1]+D1.shape[1])))
        self.sys_d = c2d(sys, self.timestep)

        self.action_space = Box(
            low=-np.array(self.max_u), high=np.array(self.max_u), dtype=np.float32
        )
        high = np.hstack([[x, y] for x, y in zip(self.max_x, self.max_speed)])
        self.observation_space = Box(low=-high, high=high, dtype=np.float32)

    def _reset(self):
        high = np.hstack([[x, y] for x, y in zip(self.max_x, self.env_config.max_speed_for_safe_set_training)])
        return self.rng.uniform(low=-high, high=high)



