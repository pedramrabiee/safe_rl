from envs_utils.test_env.test_env_utils import SimpleEnv, CBFTestAgent
import numpy as np
from gym.spaces import Box
from matplotlib import pyplot as plt
from dynamics.nominal_dynamics import NominalDynamics
from envs_utils.test_env.multi_m_dashpot_configs import env_config
from utils.safe_set import SafeSetFromPropagation
from utils.space_utils import Tuple
from utils.custom_plotter import CustomPlotter
from envs_utils.test_env.test_env_utils import CBFTestCustomPlotter
from logger import logger
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

class MultiDashpotEnv(SimpleEnv):
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

class MultiDashpotAgent(CBFTestAgent):
    env_config = env_config
    def _get_lqr_weights(self, A, Ai, B):
        R1 = np.block([
            [10 * np.eye(A.shape[0]), np.zeros((A.shape[0], Ai.shape[0]), dtype=np.float32)],
            [np.zeros((Ai.shape[0], A.shape[0]), dtype=np.float32), 100 * np.eye(Ai.shape[0])]
        ])
        R2 = 100 * np.eye(B.shape[1])
        return R1, R2

class MultiDashpotCustomPlotter(CBFTestCustomPlotter):
    env_config = env_config
    x_index = 4
    xdot_index = 5
    def dump_sampler_plots(self, episode_num):
        logger.dump_plot_with_key(plt_key="sampler_plots",
                                  filename='states_action_episode_%d' % episode_num,
                                  custom_col_config_list=[[8], [9], [0, 2], [1, 3]],
                                  columns=['u_1', 'u_2', 'u_filtered_1', 'u_filtered_2',
                                           'x_1', 'x_dot_1',
                                           'x_2', 'x_dot_2',
                                           'x_3', 'x_dot_3'],
                                  plt_info=dict(
                                      xlabel=r'Timestep',
                                      ylabel=[r'$x_3$',
                                              r'$\dot x_3$',
                                              r'$u_1$',
                                              r'$u_2$'],
                                      legend=[None,
                                              None,
                                              [r'$u_1$',
                                               r'$u_{\rm filtered_1}$'],
                                              [r'$u_2$',
                                               r'$u_{\rm filtered_2}$']
                                              ]),
                                  step_key='episode'
                                  )
