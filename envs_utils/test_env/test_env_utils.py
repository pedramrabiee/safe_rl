import numpy as np
import torch
from gym.spaces import Box
from matplotlib import pyplot as plt
from scipy.stats import truncnorm
from dynamics.nominal_dynamics import NominalDynamics
from envs_utils.test_env.test_env_configs import env_config
from logger import logger
from utils import scale
from utils.custom_plotter import CustomPlotter
from utils.misc import e_and, e_not
from utils.safe_set import SafeSetFromCriteria
from utils.space_utils import Tuple
import gym
from gym.utils import seeding
from copy import copy
from utils.seed import rng
from agents.model_free.ddpg import DDPGAgent


class SimpleEnv(gym.Env):
    def __init__(self):
        self.max_x = env_config.max_x
        self.max_u = env_config.max_u
        self.dt = env_config.timestep
        self.m = env_config.m
        self.k = env_config.k
        self.c = env_config.c
        self.max_speed = env_config.max_speed
        self.max_episode_len = int(env_config.max_episode_time / env_config.timestep)

        self.action_space = Box(
            low=-self.max_u, high=self.max_u, shape=(1,), dtype=np.float32
        )
        high = np.array([4.0, self.max_speed], dtype=np.float32)
        self.observation_space = Box(low=-high, high=high, dtype=np.float32)

        self.seed()

    def reset(self):
        high = np.array([self.max_x, self.max_speed])
        self.state = rng.uniform(low=-high, high=high)
        return copy(self.state)

    def step(self, u):
        x, x_dot = self.state
        c = env_config.c
        k = env_config.k
        m = env_config.m
        dt = env_config.timestep
        u = np.clip(u, -self.max_u, self.max_u)
        newx_dot = x_dot + (-c/m * x_dot + -k/m * x + (1/m) * u) * dt
        newx = x + newx_dot * dt
        newx_dot = np.clip(newx_dot, -self.max_speed, self.max_speed)
        self.state = np.array([newx, newx_dot])
        return copy(self.state).squeeze(), np.array([0.0])[0], np.array([0.0]).squeeze(), {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

class CBFTestDynamics(NominalDynamics):
    def predict(self, obs, ac, split_return=False):
        c = env_config.c
        k = env_config.k
        m = env_config.m

        x1 = obs[..., 0]
        x2 = obs[..., 1]
        f_func = lambda x1, x2: np.array([x2, - k/m * x1 - c/m * x2], dtype=np.float32)
        G_func = lambda x1, x2: np.array([0.0, 1/m])

        f = np.stack(list(map(f_func, x1, x2)), axis=0)
        G = np.stack(list(map(G_func, x1, x2)), axis=0)
        G = np.expand_dims(G, axis=-1)

        return (f, G) if split_return else f + np.matmul(G, ac).squeeze(-1)


class CBFTestSafeSet(SafeSetFromCriteria):
    def __init__(self, env, obs_proc):
        super().__init__(env, obs_proc)
        obs_dim = self.obs_proc.obs_dim(proc_key='filter')   # TODO: check this, you may need to change it to safe_set
        self.max_speed = env_config.max_speed
        self.max_x = env_config.max_x
        self.max_x_safe = env_config.max_x_safe
        high = np.array([self.max_x_safe, self.max_speed])
        self.geo_safe_set = Tuple([Box(low=-high, high=high, dtype=np.float64)])  # for wedge angle smaller than pi
        high = np.array([self.max_x_safe - env_config.out_width, self.max_speed])
        self.in_safe_set = Tuple([Box(low=-high, high=high, dtype=np.float64)])   # for wedge angle smaller than pi

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
        x = obs[:, 0]
        x_dot = obs[:, 1]
        is_ss_safe = x * x_dot < 0.0
        return is_ss_safe[0] if single_obs else list(is_ss_safe)

    def _get_obs(self):
        x = rng.uniform(low=-self.max_x, high=self.max_x)
        x_dot = truncnorm.rvs(-1, 1, scale=self.max_speed)
        return np.array([x, x_dot])

    def get_safe_action(self, obs):
        obs = obs.reshape(1, -1) if obs.ndim == 1 else obs
        x = obs[:, 0]
        ac_lim_high = scale.ac_old_bounds[1]
        return (-np.sign(x) * ac_lim_high).reshape(len(x), 1, 1)


class CBFTestAgent(DDPGAgent):
    # def initialize(self, params, init_dict=None):
    #     # get the observation dim from observation process class
    #     self._obs_dim = self.obs_proc.obs_dim(proc_key='mf')
    #     self.params = params

    def step(self, obs, explore=False):
        # process observation to match the models' input requirement
        obs = self.obs_proc.proc(obs, proc_key='mf')
        action = rng.uniform(-1.0, 1.0, (1, self._ac_dim))
        return action, None


class CBFTestCustomPlotter(CustomPlotter):
    def sampler_push_obs(self, obs):
        # logger.push_plot(np.concatenate((state.reshape(1, -1), ac.reshape(1, -1) * scale.ac_old_bounds[1]), axis=1), plt_key="sampler_plots")
        logger.push_plot(obs.reshape(1, -1), plt_key="sampler_plots", row_append=True)

    def filter_push_action(self, ac):
        ac, ac_filtered = ac
        logger.push_plot(np.concatenate((ac.reshape(1, -1), ac_filtered.reshape(1, -1)), axis=1), plt_key="sampler_plots")

    def dump_sampler_plots(self, episode_num):
        logger.dump_plot_with_key(plt_key="sampler_plots",
                                  filename='states_action_episode_%d' % episode_num,
                                  custom_col_config_list=[[2], [3], [0, 1]],    # 0, 1: u's , 2: theta, 3: theta_dot
                                  columns=['u_mf', 'u_filtered', 'x', 'x_dot'],
                                  plt_info=dict(
                                      xlabel=r'Timestep',
                                      ylabel=[r'$x$',
                                              r'$\dot x$',
                                              r'$u$'],
                                      legend=[None,
                                              None,
                                              [r'$u_{\rm mf}$',
                                               r'$u_{\rm filtered}$']
                                              ]
                                  )
                                  )


    def h_plotter(self, itr, filter_net):
        speeds = env_config.max_speed * np.linspace(-1.0, 1.0, num=9)
        xs = np.linspace(-env_config.max_x, env_config.max_x, num=100).reshape(-1, 1)
        # plt.figure()
        for speed in speeds:
            x = np.concatenate((xs, np.ones_like(xs) * speed), axis=-1)
            out = filter_net(torch.tensor(x, dtype=torch.float32)).detach().numpy()
            plt.plot(xs, out, label=r'$\dot x$ = ' + str(speed))
            plt.xlabel(r'$x$')
            plt.ylabel(r'$h$')
            plt.legend()

        logger.dump_plot(filename='cbf_itr_%d' % itr,
                         plt_key='cbf')

        # plt.figure()
        # mpl.use('TkAgg')  # or can use 'TkAgg', whatever you have/prefer
        # plt.ion()
        speeds = env_config.max_speed * np.linspace(-1.0, 1.0, num=100)

        X, Y = np.meshgrid(xs, speeds)
        # x = np.concatenate((np.cos(X), np.sin(X), Y))

        out = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x = np.array([X[i, j], Y[i, j]]).reshape(1, -1)
                out[i, j] = filter_net(torch.tensor(x, dtype=torch.float32)).detach().numpy().squeeze()

        ax = plt.axes(projection='3d')
        # ax.contour3D(X, Y, out, 50, cmap='binary')
        ax.plot_surface(X, Y, out, rstride=1, cstride=1,
                     cmap='viridis', edgecolor='none')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$\dot x$')
        ax.set_zlabel(r'$h$'),
        ax.view_init(50, 40)

        logger.dump_plot(filename='cbf_itr_%d_3D' % itr,
                         plt_key='cbf')

        # plt.ioff()