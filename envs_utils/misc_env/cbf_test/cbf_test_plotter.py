import numpy as np
from matplotlib import pyplot as plt
from utils.custom_plotter import CustomPlotter
from utils.grads import get_jacobian
from envs_utils.misc_env.cbf_test.cbf_test_configs import env_config
from logger import logger
import torch


class CbfTestPlotter(CustomPlotter):
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
