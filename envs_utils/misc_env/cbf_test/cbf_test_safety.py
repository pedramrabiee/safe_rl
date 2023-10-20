import numpy as np
import torch
from gym.spaces import Box
from scipy.stats import truncnorm
from envs_utils.misc_env.cbf_test.cbf_test_configs import env_config
from utils import scale
from utils.misc import e_and, e_not
from utils.safe_set import SafeSetFromCriteria, SafeSetFromPropagation
from utils.space_utils import Tuple
from utils.seed import rng


class CBFTestSafeSet(SafeSetFromCriteria):
    def __init__(self, env, obs_proc):
        super().__init__(env, obs_proc)
        obs_dim = self.obs_proc.obs_dim(proc_key='filter')   # TODO: check this, you may need to change it to safe_set
        self.max_speed = env_config.max_speed
        self.max_x = env_config.max_x_for_safe_set
        self.min_x = env_config.min_x_for_safe_set
        self.max_x_safe = env_config.max_x_safe
        self.min_x_safe = env_config.min_x_safe

        self.geo_safe_set = Tuple([Box(low=np.array([self.min_x_safe, -self.max_speed]),
                                       high=np.array([self.max_x_safe, self.max_speed]),
                                       dtype=np.float64)])  # for wedge angle smaller than pi
        self.in_safe_set = Tuple([Box(low=np.array([self.min_x_safe + env_config.out_width_down, -self.max_speed]),
                                      high=np.array([self.max_x_safe - env_config.out_width_up, self.max_speed]),
                                      dtype=np.float64)])   # for wedge angle smaller than pi

    def is_in_safe(self, obs):              # geometrically inside the inner safe section
        if torch.is_tensor(obs):
            obs = obs.numpy()
        return self.in_safe_set.contains(self.obs_proc.proc(obs).squeeze())

    def is_unsafe(self, obs):
        if torch.is_tensor(obs):
            obs = obs.numpy()
        return e_not(self.is_geo_safe(obs))

    def is_geo_safe(self, obs):
        return self.geo_safe_set.contains(self.obs_proc.proc(obs).squeeze())

    def is_out_safe(self, obs):             # geometrically inside the outer safe section
        if torch.is_tensor(obs):
            obs = obs.numpy()
        return e_and(self.is_geo_safe(obs), e_not(self.is_in_safe(obs)))

    def is_ss_safe(self, obs):
        single_obs = False
        if np.isscalar(obs[0]):
            obs = np.expand_dims(obs, axis=0)
            single_obs = True
        x = obs[:, 0] - (self.min_x_safe + self.max_x_safe) / 2.0
        x_dot = obs[:, 1]
        is_ss_safe = x * x_dot < 0.0
        return is_ss_safe[0] if single_obs else list(is_ss_safe)

    def _get_obs(self):
        x = rng.uniform(low=self.min_x, high=self.max_x)
        x_dot = truncnorm.rvs(-1, 1, scale=self.max_speed)
        return np.array([x, x_dot])


class CBFTestSafeSetFromPropagation(SafeSetFromPropagation):
    def __init__(self, env, obs_proc):
        super().__init__(env, obs_proc)
        self.max_speed = env_config.max_speed_for_safe_set_training
        # self.max_x = env_config.max_x_for_safe_set
        # self.min_x = env_config.min_x_for_safe_set
        self.max_x_safe = env_config.max_x_safe
        self.min_x_safe = env_config.min_x_safe

        self.geo_safe_set = Tuple([Box(low=np.array([self.min_x_safe, -np.inf]),
                                       high=np.array([self.max_x_safe, np.inf]),
                                       dtype=np.float64)])

    def initialize(self, init_dict=None):
        super().initialize(init_dict)
        self.max_T = env_config.max_T_for_safe_set

    def _get_obs(self):
        x = rng.normal(loc=(self.min_x_safe + self.max_x_safe)/2, scale=2*(self.max_x_safe - self.min_x_safe)/2)
        x_dot = rng.normal(loc=0.0, scale=self.max_speed)
        return np.array([x, x_dot])

    def get_safe_action(self, obs):
        obs = obs.reshape(1, -1) if obs.ndim == 1 else obs
        x_dot = obs[:, 1]
        # ac_lim_high = scale.ac_new_bounds[1]
        ac_lim_high = scale.action2newbounds(np.array([env_config.max_u_for_safe_set]))[0]
        return (-np.sign(x_dot) * ac_lim_high).reshape(len(x_dot), 1, 1)

    def propagation_terminate_cond(self, obs, next_obs):
        single_obs = False
        if np.isscalar(obs[0]):
            obs = np.expand_dims(obs, axis=0)
            next_obs = np.expand_dims(next_obs, axis=0)
            single_obs = True

        x_dot = obs[:, 1]
        new_x_dot = next_obs[:, 1]
        terminate = x_dot * new_x_dot <= 0.0     # declare if theta_dot changed sign
        return terminate[0] if single_obs else list(terminate)

    def compute_next_obs(self, deriv_value, obs):
        """
         for the inverted pendulum deriv values are:
         -theta_dot * cos(theta)
         theta_dot * sin(theta)
         theta_ddot
        """
        single_obs = False
        if np.isscalar(obs[0]):
            obs = np.expand_dims(obs, axis=0)
            single_obs = True

        x = obs[:, 0]
        x_dot = obs[:, 1]
        new_x_dot = x_dot + deriv_value[:, 1] * self.timestep
        new_x = x + new_x_dot * self.timestep
        next_obs = np.stack([new_x, new_x_dot], axis=-1)
        return next_obs[0] if single_obs else next_obs




