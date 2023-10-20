import numpy as np
from gym.spaces import Box
from envs_utils.misc_env.multi_dashpot.multi_dashpot_configs import env_config
from utils.safe_set import SafeSetFromPropagation
from utils.space_utils import Tuple
from utils.seed import rng
from utils.mpc_utils import polyderon_from_lb_ub, LQR_CFTOC
from multi_dashpot_env import get_dynamics_matrices


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