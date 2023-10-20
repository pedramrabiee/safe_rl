from envs_utils.misc_env.cbf_test.cbf_test_env import CbfTestEnv, CBFTestAgent
import numpy as np
from gym.spaces import Box
from dynamics.nominal_dynamics import NominalDynamics
from envs_utils.misc_env.multi_dashpot.multi_dashpot_configs import env_config
from utils.safe_set import SafeSetFromPropagation
from utils.space_utils import Tuple

from utils.seed import rng
from control.matlab import *
from utils.mpc_utils import polyderon_from_lb_ub, LQR_CFTOC

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

class MultiDashpotDynamics(NominalDynamics):
    def _predict(self, obs, ac, split_return=False):
        A, B, _, _  = get_dynamics_matrices()
        f = (A @ obs.T).T
        G = np.tile(B, (obs.shape[0], 1, 1))
        return (f, G) if split_return else f + np.matmul(G, ac).squeeze(-1)

class MultiDashpotSafeSetFromPropagation(SafeSetFromPropagation):
    def __init__(self, env, obs_proc):
        super().__init__(env, obs_proc)
        self.max_speed = np.array(env_config.max_speed_for_safe_set_training)
        self.max_x_safe = np.array(env_config.max_x_safe)
        self.min_x_safe = np.array(env_config.min_x_safe)
        self.max_x_for_safe_set_training = np.array(env_config.max_x_for_safe_set_training)
        high = np.hstack([[x, y] for x, y in zip(self.max_x_safe, [np.inf, np.inf, np.inf])])
        low = np.hstack([[x, y] for x, y in zip(self.min_x_safe, [-np.inf, -np.inf, -np.inf])])
        self.geo_safe_set = Tuple([Box(low=low,
                                       high=high,
                                       dtype=np.float64)])

        self.A, self.B, _, _ = get_dynamics_matrices()
        self.Ad, self.Bd, _, _ = get_dynamics_matrices(ret_discrete=True)
        self.Ax, self.bx, _ = polyderon_from_lb_ub(dim=low.shape[0], lb=low, ub=high)
        self.Af, self.bf, _ = polyderon_from_lb_ub(dim=low.shape[0], lb=-0.0, ub=0.0)
        max_u_for_safe_set = np.array(env_config.max_u_for_safe_set)
        self.Au, self.bu, _ = polyderon_from_lb_ub(dim=max_u_for_safe_set.shape[0],
                                                   lb=-max_u_for_safe_set,
                                                   ub=max_u_for_safe_set)

        self.Q = np.eye(low.shape[0])
        self.P = np.eye(low.shape[0])
        self.R = np.eye(2)
        self.mpc_solver = LQR_CFTOC(Ad=self.Ad, Bd=self.Bd,
                                    Q=self.Q, P=self.P, R=self.R,
                                    N=self.max_T,
                                    Ax=self.Ax, bx=self.bx,
                                    Af=self.Af, bf=self.bf,
                                    Au=self.Au, bu=self.bu)

    def initialize(self, init_dict=None):
        super().initialize(init_dict)
        self.max_T = env_config.max_T_for_safe_set

    def _get_obs(self):
        x = rng.normal(loc=np.zeros(3), scale=self.max_x_for_safe_set_training)
        x_dot = rng.normal(loc=0.0, scale=self.max_speed)
        return np.hstack([[x, y] for x, y in zip(x, x_dot)])

    def forward_propagate(self, obs):
        if np.isscalar(obs[0]):
            obs = np.expand_dims(obs, axis=0)

        if obs.shape[0] > 1:
             return [self.forward_propagate(o) for o in obs]

        obs = self.obs_proc.proc(obs).squeeze()     # TODO: check this
        x, u, is_feasible = self.mpc_solver.solve(x0=obs)
        is_safe = is_feasible
        return is_safe