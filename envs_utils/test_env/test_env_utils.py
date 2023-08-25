import numpy as np
import torch
from gym.spaces import Box
from matplotlib import pyplot as plt
from scipy.stats import truncnorm
from dynamics.nominal_dynamics import NominalDynamics
from envs_utils.test_env.test_env_configs import env_config
from utils import scale
from utils.custom_plotter import CustomPlotter
from utils.misc import e_and, e_not
from utils.safe_set import SafeSetFromCriteria, SafeSetFromPropagation
from utils.space_utils import Tuple
import gym
from gym.utils import seeding
from copy import copy
from utils.seed import rng
from agents.base_agent import BaseAgent
from scipy.linalg import block_diag
from control.matlab import *
from utils.grads import get_jacobian
from logger import logger

def get_dynamics_matrices():
    m = env_config.m
    k = env_config.k
    c = env_config.c
    A = np.array([[0, 1], [-k / m, -c / m]])
    B = np.array([[0.0], [1 / m]])
    C = np.array([[1, 0]])
    D = np.zeros((1, 1), dtype=np.float32)
    return A, B, C, D

class SimpleEnv(gym.Env):
    env_config = env_config
    def __init__(self):
        self.dt = env_config.timestep
        self.timestep = env_config.timestep
        self.max_episode_len = int(env_config.max_episode_time / env_config.timestep)

        self.max_x = self.env_config.max_x
        self.max_u = self.env_config.max_u
        self.m = self.env_config.m
        self.k = self.env_config.k
        self.c = self.env_config.c
        self.max_speed = self.env_config.max_speed

        # parse parameters, initialize dynamics, action and observation spaces
        self._initialize()

        self.seed()
        self.rng = np.random.default_rng(0)
        self.reset_called = False

    def _initialize(self):
        A, B, C, D = get_dynamics_matrices()
        D1 = B
        sys = ss(A, np.hstack((B, D1)), C, np.zeros((1, B.shape[1]+D1.shape[1])))
        self.sys_d = c2d(sys, self.timestep)

        self.action_space = Box(
            low=-self.max_u, high=self.max_u, shape=(1,), dtype=np.float32
        )
        high = np.array([self.max_x, self.max_speed], dtype=np.float32)
        self.observation_space = Box(low=-high, high=high, dtype=np.float32)

    def reset(self):
        if self.env_config.fixed_reset and not self.reset_called:
            self.rng = np.random.default_rng(0)
            self.reset_called = True
        self.state = self._reset()
        return copy(self.state)

    def _reset(self):
        high = np.array([self.max_x, self.env_config.max_speed_for_safe_set_training])
        return self.rng.uniform(low=-high, high=high)

    def step(self, u):
        if self.reset_called:
            self.reset_called = False
        # u = np.clip(u, -self.max_u, self.max_u)
        state = self.sys_d.A.A @ self.state + self.sys_d.B.A @ np.hstack((u, np.zeros(u.shape[0])))
        # state[1] = np.clip(state[1], -self.max_speed, self.max_speed)
        self.state = state
        return copy(self.state).squeeze(), np.array([0.0])[0], np.array([0.0]).squeeze(), {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

class CBFTestDynamics(NominalDynamics):
    def _predict(self, obs, ac, split_return=False):
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


class CBFTestAgent(BaseAgent):
    env_config = env_config
    def initialize(self, params, init_dict=None):
        self.m = self.env_config.m
        self.k = self.env_config.k
        self.c = self.env_config.c
        self.max_u = self.env_config.max_u
        self.max_speed = self.env_config.max_speed
        self.omega = self.env_config.omega

        # get the observation dim from observation process class
        self._obs_dim = self.obs_proc.obs_dim(proc_key='mf')
        self.params = params

        A, B, C, D = get_dynamics_matrices()
        D1 = B
        le = C.shape[0]

        # LQR DESIGN FOR SINUSOID COMMAND
        I = np.array([[0, -1], [1, 0]])
        Aip = block_diag(0.0, self.omega * np.pi * I)
        self.imp_state_dim = Aip.shape[0]
        Ai = np.kron(np.eye(le), Aip)

        Bip = np.ones((self.imp_state_dim, 1), dtype=np.float32)
        Bi = np.kron(np.eye(le), Bip)

        # Augmented Matrices
        Aa = np.block([
            [A, np.zeros((A.shape[0], Ai.shape[1]), dtype=np.float32)],
            [np.matmul(-Bi, C), Ai]
        ])
        Ba = np.block([
            [B],
            [np.matmul(-Bi, D)]
        ])

        R1, R2 = self._get_lqr_weights(A, Ai, B)

        # LQR Gain
        Ka, S, E = lqr(Aa, Ba, R1, R2)
        Ka = -Ka.A  # .A return array

        self.K = Ka[:, :A.shape[0]]
        self.Ki = Ka[:, A.shape[0]:]

        sys_c = ss(Ai, Bi, np.eye(Ai.shape[0]), np.zeros((self.imp_state_dim, 1)))
        self.sys_cd = c2d(sys_c, self._timestep)
        self.xi = np.zeros(self.imp_state_dim, dtype=np.float32)
        self.t = 0.0
        self.models = []
        self.optimizers = []
        self.models_dict = {}
        self.optimizers_dict = {}
        self.extra_params_dict = {}

    def _get_lqr_weights(self, A, Ai, B):
        R1 = np.block([
            [10 * np.eye(A.shape[0]), np.zeros((A.shape[0], Ai.shape[0]), dtype=np.float32)],
            [np.zeros((Ai.shape[0], A.shape[0]), dtype=np.float32), 100 * np.eye(Ai.shape[0])]
        ])
        R2 = 100 * np.eye(B.shape[1])
        return R1, R2

    def step(self, obs, explore=False, init_phase=False):
        # process observation to match the models' input requirement
        obs = self.obs_proc.proc(obs, proc_key='mf')
        obs = obs.squeeze()
        # action = rng.uniform(-1.0, 1.0, (1, self._ac_dim))
        command = self.env_config.command_amplitude * np.sin(self.omega * np.pi * self.t) + 4
        logger.push_plot(command.reshape(1, -1), plt_key='performance')
        self.t += self._timestep
        self.xi = self.sys_cd.A.A @ self.xi + (self.sys_cd.B.A * (command - obs[0])).squeeze()
        action = self.K @ obs + self.Ki @ self.xi
        # action = np.clip(action, -self.max_u, self.max_u)
        action = scale.action2newbounds(action)
        return action, None

    def on_episode_reset(self, episode):
        self.t = 0.0
        self.xi = np.zeros(self.imp_state_dim, dtype=np.float32)


class CBFTestCustomPlotter(CustomPlotter):
    env_config = env_config
    x_index = 0
    xdot_index = 1
    def sampler_push_obs(self, obs):
        # logger.push_plot(np.concatenate((state.reshape(1, -1), ac.reshape(1, -1) * scale.ac_old_bounds[1]), axis=1), plt_key="sampler_plots")
        logger.push_plot(obs.reshape(1, -1), plt_key="sampler_plots", row_append=True)
        logger.push_plot(obs[self.x_index].reshape(1, -1), plt_key='performance', row_append=True)

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
                                              ]),
                                  step_key='episode'
                                  )

    def dump_performance_plots(self, episode_num):
        performance_data = logger.get_plot_queue_by_key('performance')
        performance_data = np.vstack(performance_data)
        error = (performance_data[:, 0] - performance_data[:, 1]).reshape(-1, 1)
        performance_data = np.concatenate((performance_data, error), axis=-1)
        logger.set_plot_queue_by_key('performance', performance_data)

        logger.dump_plot_with_key(plt_key="performance",
                                  filename='performance_episode_%d' % episode_num,
                                  custom_col_config_list=[[0, 1], [2]],
                                  columns=['command', 'x', 'error'],
                                  plt_info=dict(
                                      xlabel=r'Timestep',
                                      ylabel=[r'$x$', r'$e$'],
                                      legend=[
                                          [r'$r$', r'$x$'],
                                          None
                                      ]),
                                  step_key='episode'
                                  )

    def safe_set_plotter(self, safe_samples, unsafe_samples):
        plt.scatter(safe_samples[:, self.x_index], safe_samples[:, self.xdot_index], c='g', marker='.', linewidths=0.01, alpha=0.5)
        plt.scatter(unsafe_samples[:, self.x_index], unsafe_samples[:, self.xdot_index], c='r', marker='.', linewidths=0.01, alpha=0.5)
        plt.axvline(x=self.env_config.min_x_safe, color='k', linestyle='-')
        plt.axvline(x=self.env_config.max_x_safe, color='k', linestyle='-')

        logger.dump_plot(filename='safe_unsafe_sets',
                         plt_key='safe_unsafe')

    def h_plotter(self, itr, filter_net):
        speeds = self.env_config.max_speed_for_safe_set_training * np.linspace(-1.0, 1.0, num=9)
        xs = np.linspace(self.env_config.min_x_for_safe_set, self.env_config.max_x_for_safe_set, num=200).reshape(-1, 1)
        # plt.figure()
        for speed in speeds:
            x = np.concatenate((xs, np.ones_like(xs) * speed), axis=-1)
            out = filter_net(torch.tensor(x, dtype=torch.float32)).detach().numpy()
            plt.plot(xs, out, label=r'$\dot x$ = ' + str(speed))
            plt.xlabel(r'$x$')
            plt.ylabel(r'$h$')
            plt.legend()

        logger.dump_plot(filename='cbf_itr_%d' % itr,
                         plt_key='cbf2d', step_key='iteration')

        # plt.figure()
        # mpl.use('TkAgg')  # or can use 'TkAgg', whatever you have/prefer
        # plt.ion()
        speeds = self.env_config.max_speed_for_safe_set_training * np.linspace(-1.0, 1.0, num=200)

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
        zlim = ax.get_zlim()
        cs = ax.contour(X, Y, out, [0.0], colors="k", linestyles="solid", zdir='z', offset=zlim[0], alpha=1.0)
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$\dot x$')
        ax.set_zlabel(r'$h$'),
        ax.view_init(50, 40)

        logger.dump_plot(filename='cbf_itr_%d_3D' % itr,
                         plt_key='cbf3d',
                         step_key='iteration')

        # plt.ioff()
        out1 = np.zeros_like(X)
        out2 = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x = np.array([X[i, j], Y[i, j]]).reshape(1, -1)
                # out[i, j] = filter_net(torch.tensor(x, dtype=torch.float32)).detach().numpy().squeeze()
                with torch.enable_grad():
                    dh_dx = get_jacobian(net=filter_net, x=torch.tensor(x, dtype=torch.float32)).detach().numpy().squeeze()
                out1[i, j] = dh_dx[0]
                out2[i, j] = dh_dx[1]

        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        # ax.contour3D(X, Y, out, 50, cmap='binary')
        ax.plot_surface(X, Y, out1, rstride=1, cstride=1,
                        cmap='coolwarm', edgecolor='none')
        ax.contour(X, Y, out1, colors="k", linestyles="solid", alpha=1.0)

        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$\dot x$')
        ax.set_zlabel(r'$\frac{\partial h}{\partial x}$'),
        ax.view_init(50, 40)

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        # ax.contour3D(X, Y, out, 50, cmap='binary')
        ax.plot_surface(X, Y, out2, rstride=1, cstride=1,
                        cmap='coolwarm', edgecolor='none')
        ax.contour(X, Y, out2, colors="k", linestyles="solid", alpha=1.0)
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$\dot x$')
        ax.set_zlabel(r'$\frac{\partial h}{\partial \dot x}$'),
        ax.view_init(50, 40)

        logger.dump_plot(filename='cbf_itr_%d_3D_dh_dx' % itr,
                         plt_key='cbf3d_dh_dx',
                         step_key='iteration')
