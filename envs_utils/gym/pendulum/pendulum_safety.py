from envs_utils.gym.pendulum.pendulum_backup_shield import PendulumSafeSet, get_safe_set

from utils.safe_set import SafeSetFromBarrierFunction


safety_dict = dict(
    safe_set_cls_to_use=PendulumSafeSet,
    safe_set_getter=get_safe_set
)

# class InvertedPendulumSafeSet(SafeSetFromBarrierFunction):
#     # TODO: Implement new safe set for pendulum
#     raise NotImplementedError


# Deprecated

# class InvertedPendulumSafeSet(SafeSetFromCriteria):
#     def __init__(self, env, obs_proc):
#         super().__init__(env, obs_proc)
#         obs_dim = self.obs_proc.obs_dim(proc_key='filter')   # TODO: check this, you may need to change it to safe_set
#         max_speed = env.observation_space.high[2]
#
#         safe_half_wedge_angle = env_config.half_wedge_angle - env_config.outer_safe_set_width
#         # mid_half_wedge_angle = env_config.half_wedge_angle - env_config.outer_safe_set_width
#
#         if obs_dim == 2: # TODO: add middle boundary and check inner and geo_safe_set
#             self.geo_safe_set = Tuple([Box(low=np.array([-env_config.half_wedge_angle, -max_speed]),
#                                            high=np.array([env_config.half_wedge_angle, max_speed]),
#                                            dtype=np.float64)])
#             self.in_safe_set = Tuple([Box(low=np.array([-safe_half_wedge_angle, -max_speed]),
#                                           high=np.array([safe_half_wedge_angle, max_speed]),
#                                           dtype=np.float64)])
#         if obs_dim == 3:
#             # outer boundary
#             self.geo_safe_set = Tuple([Box(low=np.array([np.cos(env_config.half_wedge_angle), -np.sin(env_config.half_wedge_angle), -max_speed]),
#                                            high=np.array([1.0, np.sin(env_config.half_wedge_angle), max_speed]),
#                                            dtype=np.float64)])      # for wedge angle smaller than pi
#
#             # inner boundary
#             self.in_safe_set = Tuple([Box(low=np.array([np.cos(safe_half_wedge_angle), -np.sin(safe_half_wedge_angle), -max_speed]),
#                                           high=np.array([1.0, np.sin(safe_half_wedge_angle), max_speed]),
#                                           dtype=np.float64)])       # for wedge angle smaller than pi
#             # middle boundary
#             # self.mid_safe_set = Tuple([Box(low=np.array([np.cos(mid_half_wedge_angle), -np.sin(mid_half_wedge_angle), -max_speed]),
#             #                                high=np.array([1.0, np.sin(mid_half_wedge_angle), max_speed]),
#             #                                dtype=np.float64)])
#
#     def is_geo_safe(self, obs):
#         if torch.is_tensor(obs):
#             obs = obs.numpy()
#         return self.geo_safe_set.contains(self.obs_proc.proc(obs).squeeze())
#
#     def is_in_safe(self, obs):              # geometrically inside the inner safe section
#         if torch.is_tensor(obs):
#             obs = obs.numpy()
#         return self.in_safe_set.contains(self.obs_proc.proc(obs).squeeze())
#
#     # def is_mid_safe(self, obs):             # geometrically inside the middle safe section
#     #     if torch.is_tensor(obs):
#     #         obs = obs.numpy()
#     #     return e_and(self.mid_safe_set.contains(self.obs_proc.proc(obs).squeeze()),
#     #                  e_not(self.is_in_safe(obs)))
#
#     def is_out_safe(self, obs):             # geometrically inside the outer safe section
#         if torch.is_tensor(obs):
#             obs = obs.numpy()
#         return e_and(self.is_geo_safe(obs), e_not(self.is_in_safe(obs)))
#
#     def is_unsafe(self, obs):
#         if torch.is_tensor(obs):
#             obs = obs.numpy()
#         return e_not(self.is_geo_safe(obs))
#
#     def is_ss_safe(self, obs):
#         single_obs = False
#         if np.isscalar(obs[0]):
#             obs = np.expand_dims(obs, axis=0)
#             single_obs = True
#         theta = np.arctan2(obs[:, 1], obs[:, 0])
#         theta_dot = obs[:, 2]
#         is_ss_safe = theta * theta_dot < 0.0
#         return is_ss_safe[0] if single_obs else list(is_ss_safe)
#
#     def _get_obs(self):
#         max_speed = self.env.observation_space.high[2]
#         theta = rng.uniform(low=-np.pi, high=np.pi)
#         thetadot = truncnorm.rvs(-1, 1, scale=max_speed) if env_config.sample_velocity_gaussian \
#             else rng.uniform(low=-max_speed, high=max_speed)
#         return np.array([np.cos(theta), np.sin(theta), thetadot])
#
#     def get_safe_action(self, obs):
#         obs = obs.reshape(1, -1) if obs.ndim == 1 else obs
#         if obs.shape[1] == 2:
#             theta = obs[:, 0]
#         else:
#             theta = np.arctan2(obs[:, 1], obs[:, 0])
#
#         ac_lim_high = scale.ac_new_bounds[1]
#         return (-np.sign(theta) * ac_lim_high).reshape(len(theta), 1, 1)


# class InvertedPendulumSafeSetFromPropagation(SafeSetFromPropagation):
#     def __init__(self, env, obs_proc):
#         super().__init__(env, obs_proc)
#         self.max_speed = env_config.max_speed_for_safe_set_training
#
#         self.geo_safe_set = Tuple(
#             [Box(low=np.array([np.cos(env_config.half_wedge_angle), -np.sin(env_config.half_wedge_angle), -np.inf]),
#                  high=np.array([1.0, np.sin(env_config.half_wedge_angle), np.inf]),
#                  dtype=np.float64)])  # for wedge angle smaller than pi
#
#     def initialize(self, init_dict=None):
#         super().initialize(init_dict)
#         self.max_T = env_config.max_T_for_safe_set
#
#     def _get_obs(self):
#         theta = rng.uniform(low=-np.pi, high=np.pi)
#         thetadot = rng.normal(loc=0.0, scale=self.max_speed)
#         return np.array([np.cos(theta), np.sin(theta), thetadot])
#
#     def get_safe_action(self, obs):
#         obs = obs.reshape(1, -1) if obs.ndim == 1 else obs
#         if obs.shape[1] == 2:
#             theta = obs[:, 0]
#             thetadot = obs[:, 1]
#         else:
#             theta = np.arctan2(obs[:, 1], obs[:, 0])
#             thetadot = obs[:, 2]
#
#         ac_lim_high = scale.ac_new_bounds[1]
#         return (-np.sign(thetadot) * ac_lim_high).reshape(len(theta), 1, 1)
#
#     def propagation_terminate_cond(self, obs, next_obs):
#         single_obs = False
#         if np.isscalar(obs[0]):
#             obs = np.expand_dims(obs, axis=0)
#             next_obs = np.expand_dims(next_obs, axis=0)
#             single_obs = True
#
#         theta_dot = obs[:, 2]
#         new_theta_dot = next_obs[:, 2]
#         terminate = theta_dot * new_theta_dot <= 0.0     # declare if theta_dot changed sign
#         return terminate[0] if single_obs else list(terminate)
#
#     def compute_next_obs(self, deriv_value, obs):
#         """
#          for the inverted pendulum deriv values are:
#          -theta_dot * cos(theta)
#          theta_dot * sin(theta)
#          theta_ddot
#         """
#         single_obs = False
#         if np.isscalar(obs[0]):
#             obs = np.expand_dims(obs, axis=0)
#             single_obs = True
#
#         theta = np.arctan2(obs[:, 1], obs[:, 0])
#         theta_dot = obs[:, 2]
#         new_theta_dot = theta_dot + deriv_value[:, 2] * self.timestep
#         new_theta = theta + new_theta_dot * self.timestep
#         next_obs = np.stack([np.cos(new_theta), np.sin(new_theta), new_theta_dot], axis=-1)
#         return next_obs[0] if single_obs else next_obs