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
from utils.custom_plotter import CustomPlotter
from logger import logger
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
from scipy.stats import truncnorm


config = AttrDict(
    do_obs_proc=False,
    safe_reset=True,
    timestep=0.01,
    # Safe set width
    half_wedge_angle=1.0,
    mid_safe_set_width=0.2,
    outer_safe_set_width=0.2,
    # Pendulum dynamics parameters
    g=10.0,
    m=1.0,
    l=1.0,
    max_torque=15.0,
    max_speed=8.0,
    sample_velocity_gaussian=True       # velocity distribution will be truncated normal distribution
)

def inverted_pendulum_customize(env):
    # Settings

    # env.env.max_torque = max_torque  # you could also used env.unwrapped.max_torque
    env.unwrapped.max_torque = config.max_torque
    env.unwrapped.max_speed = config.max_speed  # you could also used env.unwrapped.max_speed
    env.unwrapped.dt = config.timestep

    env.action_space = Box(
        low=-config.max_torque,
        high=config.max_torque, shape=(1,),
        dtype=np.float32
    )
    high = np.array([1., 1., config.max_speed], dtype=np.float32)
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

        safe_half_wedge_angle = config.half_wedge_angle - (config.mid_safe_set_width + config.outer_safe_set_width)
        mid_half_wedge_angle = config.half_wedge_angle - config.outer_safe_set_width

        if obs_dim == 2: # TODO: add middle boundary and check inner and geo_safe_set
            self.geo_safe_set = Tuple([Box(low=np.array([-config.half_wedge_angle, -max_speed]),
                                           high=np.array([config.half_wedge_angle, max_speed]),
                                           dtype=np.float64)])
            self.in_safe_set = Tuple([Box(low=np.array([-safe_half_wedge_angle, -max_speed]),
                                          high=np.array([safe_half_wedge_angle, max_speed]),
                                          dtype=np.float64)])
        if obs_dim == 3:
            # outer boundary
            self.geo_safe_set = Tuple([Box(low=np.array([np.cos(config.half_wedge_angle), -np.sin(config.half_wedge_angle), -max_speed]),
                                           high=np.array([1.0, np.sin(config.half_wedge_angle), max_speed]),
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
        return e_and(self.is_geo_safe(obs), e_not(self.is_mid_safe(obs)), e_not(self.is_in_safe(obs)))

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
        theta = np.random.uniform(low=-np.pi, high=np.pi)
        thetadot = truncnorm.rvs(-1, 1, scale=max_speed) if config.sample_velocity_gaussian \
            else np.random.uniform(low=-max_speed, high=max_speed)
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
        g = config.g
        m = config.m
        l = config.l

        if obs.shape[-1] == 2:
            x1 = np.cos(obs[..., 0])
            x2 = np.sin(obs[..., 0])
            x3 = obs[..., 1]
        elif obs.shape[-1] == 3:
            x1 = obs[..., 0]    # cos(theta)
            x2 = obs[..., 1]    # sin(theta)
            x3 = obs[..., 2]    # theta_dot

        if not self.continous_time:
            f_func = lambda x1, x2, x3:\
                np.array([
                    x1 - dt * x3 * x2 - (dt ** 2) * 3 * g / (2 * l) * x2 ** 2,
                    x2 + dt * x3 * x1 + (dt ** 2) * 3 * g / (2 * l) * x2 * x1,
                    x3 + dt * 3 * g / (2 * l) * x2
                ], dtype=np.float32)
            G_func = lambda x1, x2: 3 / (m * l ** 2) * dt * np.array([
                dt * x2,
                dt * x1,
                1.0
            ], dtype=np.float32)
            G = np.stack(list(map(G_func, x1, x2)), axis=0)
        else:
            f_func = lambda x1, x2, x3: \
                np.array([
                    -x3 * x2,
                    x3 * x1,
                    3 * g / (2 * l) * x2
                ], dtype=np.float32)
            G = np.array([
                0.0,
                0.0,
                3 / (m * l ** 2)
            ], dtype=np.float32)
            G = np.stack([G for _ in range(x1.shape[0])], axis=0)

        f = np.stack(list(map(f_func, x1, x2, x3)), axis=0)
        G = np.expand_dims(G, axis=-1)
        return (f, G) if split_return else f + np.matmul(G, ac).squeeze(-1)


class InvertedPendulumCustomPlotter(CustomPlotter):
    def sampler_push_obs(self, obs):
        theta = np.arctan2(obs[1], obs[0])
        state = np.array([theta, obs[2]])
        # logger.push_plot(np.concatenate((state.reshape(1, -1), ac.reshape(1, -1) * scale.ac_old_bounds[1]), axis=1), plt_key="sampler_plots")
        logger.push_plot(state.reshape(1, -1), plt_key="sampler_plots", row_append=True)

    def filter_push_action(self, ac):
        ac, ac_filtered = ac
        logger.push_plot(np.concatenate((ac.reshape(1, -1), ac_filtered.reshape(1, -1)), axis=1), plt_key="sampler_plots")

    def dump_sampler_plots(self, episode_num):
        logger.dump_plot_with_key(plt_key="sampler_plots",
                                  filename='states_action_episode_%d' % episode_num,
                                  custom_col_config_list=[[2], [3], [0, 1]],    # 0, 1: u's , 2: theta, 3: theta_dot
                                  columns=['u_mf', 'u_filtered', 'theta', 'theta_dot'],
                                  plt_info=dict(
                                      xlabel=r'Timestep',
                                      ylabel=[r'$\theta$',
                                              r'$\dot \theta$',
                                              r'$u$'],
                                      legend=[None,
                                              None,
                                              [r'$u_{\rm mf}$',
                                               r'$u_{\rm filtered}$']
                                              ]
                                  )
                                  )


    def h_plotter(self, itr, filter_net):
        speeds = config.max_speed * np.linspace(-1.0, 1.0, num=9)
        theta = np.linspace(-np.pi, np.pi, num=100).reshape(-1, 1)
        # plt.figure()
        for speed in speeds:
            x = np.concatenate((np.cos(theta), np.sin(theta), np.ones_like(theta) * speed), axis=-1)
            out = filter_net(torch.tensor(x, dtype=torch.float32)).detach().numpy()
            plt.plot(theta, out, label=r'$\dot \theta$ = ' + str(speed))
            plt.xlabel(r'$\theta$')
            plt.ylabel(r'$h$')
            plt.legend()

        logger.dump_plot(filename='cbf_itr_%d' % itr,
                         plt_key='cbf')

        # plt.figure()
        # mpl.use('TkAgg')  # or can use 'TkAgg', whatever you have/prefer
        # plt.ion()
        speeds = config.max_speed * np.linspace(-1.0, 1.0, num=100)

        X, Y = np.meshgrid(theta, speeds)
        # x = np.concatenate((np.cos(X), np.sin(X), Y))

        out = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x = np.array([np.cos(X[i, j]), np.sin(X[i, j]), Y[i, j]]).reshape(1,-1)
                out[i, j] = filter_net(torch.tensor(x, dtype=torch.float32)).detach().numpy().squeeze()

        ax = plt.axes(projection='3d')
        # ax.contour3D(X, Y, out, 50, cmap='binary')
        ax.plot_surface(X, Y, out, rstride=1, cstride=1,
                     cmap='viridis', edgecolor='none')
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(r'$\dot \theta$')
        ax.set_zlabel(r'$h$'),
        ax.view_init(50, 40)

        logger.dump_plot(filename='cbf_itr_%d_3D' % itr,
                         plt_key='cbf')

        # plt.ioff()

