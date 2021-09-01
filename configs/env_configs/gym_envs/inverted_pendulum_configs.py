from utils.process_observation import ObsProc
import numpy as np
from gym.spaces import Box
from utils.space_utils import Tuple
from utils.safe_set import SafeSetFromCriteria
from utils import scale
from utils.misc import e_and, e_not
import torch
from dynamics.nominal_dynamics import NominalDynamics
from attrdict import AttrDict

config = AttrDict(
    do_obs_proc=False,
    safe_reset=True
)
# Environment Parameters
timestep = 0.01

# Safe set width
half_wedge_angle = 1.0
mid_safe_set_width = 0.2
outer_safe_set_width = 0.2

# Pendulum dynamics parameters
g = 10.0
m = 1.0
l = 1.0
max_torque = 15.0  # default is 2.0
max_speed = 8.0  # default is 8.0


def inverted_pendulum_customize(env):
    # Settings

    # env.env.max_torque = max_torque  # you could also used env.unwrapped.max_torque
    env.unwrapped.max_torque = max_torque
    env.unwrapped.max_speed = max_speed  # you could also used env.unwrapped.max_speed
    env.unwrapped.dt = timestep

    env.action_space = Box(
        low=-max_torque,
        high=max_torque, shape=(1,),
        dtype=np.float32
    )
    high = np.array([1., 1., max_speed], dtype=np.float32)
    env.observation_space = Box(
        low=-high,
        high=high,
        dtype=np.float32
    )

    return env


class InvertedPendulumObsProc(ObsProc):
    def obs_dim(self, proc_key=None):
        return int(3)

    def _proc(self, obs, proc_dict=None):
        return np.stack([np.arctan2(obs[..., 1], obs[..., 0]), obs[..., 2]], axis=-1)   # FIXME: fix this for ensemble and buffer

    def _unproc(self, obs, unproc_dict=None):
        if obs.ndim == 1:
            obs = np.expand_dims(obs, axis=0)
        return np.stack([np.cos(obs[..., 0]), np.sin(obs[..., 0]), obs[..., 1]], axis=-1)            # FIXME: this will not work for buffer get_stats


class InvertedPendulumSafeSet(SafeSetFromCriteria):
    def __init__(self, env, obs_proc):
        super().__init__(env, obs_proc)
        obs_dim = self.obs_proc.obs_dim(proc_key='filter')   # TODO: check this, you may need to change it to safe_set
        max_speed = env.observation_space.high[2]
        # max_speed = 60.0


        safe_half_wedge_angle = half_wedge_angle - (mid_safe_set_width + outer_safe_set_width)
        mid_half_wedge_angle = half_wedge_angle - outer_safe_set_width

        if obs_dim == 2: # TODO: add middle boundary and check inner and geo_safe_set
            self.geo_safe_set = Tuple([Box(low=np.array([-half_wedge_angle, -max_speed]),
                                           high=np.array([half_wedge_angle, max_speed]),
                                           dtype=np.float64)])
            self.in_safe_set = Tuple([Box(low=np.array([-safe_half_wedge_angle, -max_speed]),
                                          high=np.array([safe_half_wedge_angle, max_speed]),
                                          dtype=np.float64)])
        if obs_dim == 3:
            # outer boundary
            self.geo_safe_set = Tuple([Box(low=np.array([np.cos(half_wedge_angle), -np.sin(half_wedge_angle), -max_speed]),
                                           high=np.array([1.0, np.sin(half_wedge_angle), max_speed]),
                                           dtype=np.float64)])      # for wedge angle smaller than pi

            # inner boundary
            self.in_safe_set = Tuple([Box(low=np.array([np.cos(safe_half_wedge_angle), -np.sin(safe_half_wedge_angle), -max_speed]),
                                          high=np.array([1.0, np.sin(safe_half_wedge_angle), max_speed]),
                                          dtype=np.float64)])       # for wedge angle smaller than pi
            # middle boundary
            self.mid_safe_set = Tuple([Box(low=np.array([np.cos(mid_half_wedge_angle), -np.sin(mid_half_wedge_angle), -max_speed]),
                                           high=np.array([1.0, np.sin(mid_half_wedge_angle), max_speed]),
                                           dtype=np.float64)])

    def is_geo_safe(self, obs):
        if torch.is_tensor(obs):
            obs = obs.numpy()
        return self.geo_safe_set.contains(self.obs_proc.proc(obs).squeeze())

    def is_in_safe(self, obs):              # geometrically inside the inner safe section
        if torch.is_tensor(obs):
            obs = obs.numpy()
        return self.in_safe_set.contains(self.obs_proc.proc(obs).squeeze())

    def is_mid_safe(self, obs):             # geometrically inside the middle safe section
        if torch.is_tensor(obs):
            obs = obs.numpy()
        return e_and(self.mid_safe_set.contains(self.obs_proc.proc(obs).squeeze()),
                     e_not(self.is_in_safe(obs)))

    def is_out_safe(self, obs):             # geometrically inside the outer safe section
        if torch.is_tensor(obs):
            obs = obs.numpy()
        return e_and(self.is_geo_safe(obs), e_not(self.is_mid_safe(obs)))

    def is_unsafe(self, obs):
        if torch.is_tensor(obs):
            obs = obs.numpy()
        return e_not(self.is_geo_safe(obs))

    def is_ss_safe(self, obs):
        single_obs = False
        if np.isscalar(obs[0]):
            obs = np.expand_dims(obs, axis=0)
            single_obs = True
        theta = np.arctan2(obs[:, 1], obs[:, 0])
        theta_dot = obs[:, 2]
        is_ss_safe = theta * theta_dot < 0.0
        return is_ss_safe[0] if single_obs else list(is_ss_safe)

    def _get_obs(self):
        max_speed = self.env.observation_space.high[2]
        high = np.array([np.pi, max_speed])
        theta, thetadot = np.random.uniform(low=-high, high=high)
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def get_safe_action(self, obs):
        obs = obs.reshape(1, -1) if obs.ndim == 1 else obs
        if obs.shape[1] == 2:
            theta = obs[:, 0]
        else:
            theta = np.arctan2(obs[:, 1], obs[:, 0])

        ac_lim_high = scale.ac_old_bounds[1]
        return (-np.sign(theta) * ac_lim_high).reshape(len(theta), 1, 1)

class InvertedPendulumNominalDyn(NominalDynamics):
    def initialize(self, params, init_dict=None):
        #TODO: You should link this with the params, so that if you are normalizing the observation or action, apply the same thing here
        self.continous_time = init_dict['is_continuous']

    def _predict(self, obs, ac, split_return=False):
        dt = self.timestep

        if obs.shape[-1] == 2:
            theta = obs[..., 0]
            theta_dot = obs[..., 1]
        elif obs.shape[-1] == 3:
            theta = np.arctan2(obs[..., 1], obs[..., 0])
            theta_dot = obs[..., 2]

        if not self.continous_time:
            f = np.array([
                -3 * g / (2 * l) * np.sin(theta + np.pi) * dt ** 2 + theta_dot * dt + theta,
                theta_dot - 3 * g / (2 * l) * np.sin(theta + np.pi) * dt
            ])
            g = np.array([
                3 / (m * l ** 2) * dt ** 2,
                3 / (m * l ** 2) * dt
            ])
        else:
            f = np.array([
                theta_dot,
                -3 * g / (2 * l) * np.sin(theta + np.pi)
            ])
            g = np.array([
                0.0,
                3 / (m * l ** 2)
            ])
        # Shape of f in single-dynamics case is 2 x batch_size,
        # and in the ensemble case is 2 x ensemble_size x batch_size
        # np.moveaxis(f, 0, -1) rearrange f in single case to batch_size x 2
        # and in ensemble case to ensemble_size x batch_size x 2
        return (np.moveaxis(f, 0, -1), g) if split_return else np.moveaxis(f, 0, -1) + g * ac

class InvertedPendulumNominalDynV2(NominalDynamics):
    def initialize(self, params, init_dict=None):
        #TODO: You should link this with the params, so that if you are normalizing the observation or action, apply the same thing here
        self.continous_time = init_dict['is_continuous']

    def _predict(self, obs, ac, split_return=False):
        dt = self.timestep

        if obs.shape[-1] == 2:
            x1 = np.cos(obs[..., 0])
            x2 = np.sin(obs[..., 0])
            x3 = obs[..., 1]
        elif obs.shape[-1] == 3:
            x1 = obs[..., 0]    # cos(theta)
            x2 = obs[..., 1]    # sin(theta)
            x3 = obs[..., 2]    # theta_dot

        if not self.continous_time:
            f = np.array([
                x1 - dt * x3 * x2 - (dt ** 2) * 3 * g / (2 * l) * x2 ** 2,
                x2 + dt * x3 * x1 + (dt ** 2) * 3 * g / (2 * l) * x2 * x1,
                x3 + dt * 3 * g / (2 * l) * x2
            ])
            G = 3 / (m * l ** 2) * dt * np.array([
                dt * x2,
                dt * x1,
                1.0
            ])
        else:
            f = np.array([
                -x3 * x2,
                x3 * x1,
                3 * g / (2 * l) * x2
            ])
            G = np.array([
                0.0,
                0.0,
                3 / (m * l ** 2)
            ])
        # Shape of f in single-dynamics case is 3 x batch_size,
        # and in the ensemble case is 3 x ensemble_size x batch_size
        # np.moveaxis(f, 0, -1) rearrange f in single case to batch_size x 3
        # and in ensemble case to ensemble_size x batch_size x 3
        f = np.moveaxis(f, 0, -1)
        if self.continous_time:     # FIXME: I think in discrete case the dimension of G is the same as f, so you need to moveaxis for that too
            G = np.ones_like(f) * G
        G = np.expand_dims(G, axis=-1) # TODO: check for discrete case
        return (f, G) if split_return else f + np.matmul(G, ac).squeeze(-1)