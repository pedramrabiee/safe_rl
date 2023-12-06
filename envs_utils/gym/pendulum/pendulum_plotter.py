from utils.custom_plotter import CustomPlotter
from logger import logger
import numpy as np
from utils.plot_utils import plot_zero_level_sets
from math import pi

plotter_dict = dict(
    rlbus=dict(module='pendulum_backup_shield', cls_name='PendulumPlotter'),
    bus=dict(module='pendulum_backup_shield', cls_name='PendulumPlotter'),
)


    # def dump_safe_set_plotter(self, safe_samples, unsafe_samples):
    #     cossin2theta = lambda x: np.arctan2(x[:, 1], x[:, 0])
    #
    #     theta_safe = cossin2theta(safe_samples)
    #     theta_unsafe = cossin2theta(unsafe_samples)
    #     plt.scatter(theta_safe, safe_samples[:, 2], c='g', marker='.', linewidths=0.05, alpha=0.5)
    #     plt.scatter(theta_unsafe, unsafe_samples[:, 2], c='r', marker='.', linewidths=0.05, alpha=0.5)
    #     plt.axvline(x=-env_config.half_wedge_angle, color='k', linestyle='-')
    #     plt.axvline(x=env_config.half_wedge_angle, color='k', linestyle='-')
    #
    #
    #     logger.dump_plot(filename='safe_unsafe_sets',
    #                      plt_key='safe_unsafe')



    # def h_plotter(self, itr, filter_net):
    #     speeds = env_config.max_speed_for_safe_set_training * np.linspace(-1.0, 1.0, num=9)
    #     theta = np.linspace(-np.pi, np.pi, num=100).reshape(-1, 1)
    #     # plt.figure()
    #     for speed in speeds:
    #         x = np.concatenate((np.cos(theta), np.sin(theta), np.ones_like(theta) * speed), axis=-1)
    #         out = filter_net(torch.tensor(x, dtype=torch.float32)).detach().numpy()
    #         plt.plot(theta, out, label=r'$\dot \theta$ = ' + str(speed))
    #         plt.xlabel(r'$\theta$')
    #         plt.ylabel(r'$h$')
    #         plt.legend()
    #
    #     logger.dump_plot(filename='cbf_itr_%d' % itr,
    #                      plt_key='cbf2d')
    #
    #     # plt.figure()
    #     # mpl.use('TkAgg')  # or can use 'TkAgg', whatever you have/prefer
    #     # plt.ion()
    #     speeds = env_config.max_speed_for_safe_set_training * np.linspace(-1.0, 1.0, num=100)
    #
    #     X, Y = np.meshgrid(theta, speeds)
    #     # x = np.concatenate((np.cos(X), np.sin(X), Y))
    #
    #     out = np.zeros_like(X)
    #     for i in range(X.shape[0]):
    #         for j in range(X.shape[1]):
    #             x = np.array([np.cos(X[i, j]), np.sin(X[i, j]), Y[i, j]]).reshape(1,-1)
    #             out[i, j] = filter_net(torch.tensor(x, dtype=torch.float32)).detach().numpy().squeeze()
    #
    #     ax = plt.axes(projection='3d')
    #     # ax.contour3D(X, Y, out, 50, cmap='binary')
    #     ax.plot_surface(X, Y, out, rstride=1, cstride=1,
    #                  cmap='viridis', edgecolor='none')
    #     zlim = ax.get_zlim()
    #     cs = ax.contour(X, Y, out, [0.0], colors="k", linestyles="solid", zdir='z', offset=zlim[0], alpha=1.0)
    #     ax.clabel(cs, inline=True, fontsize=10)
    #     ax.set_xlabel(r'$\theta$')
    #     ax.set_ylabel(r'$\dot \theta$')
    #     ax.set_zlabel(r'$h$'),
    #     ax.view_init(50, 40)
    #     # ax.set_zlim(-0.1, 0.1)
    #
    #     logger.dump_plot(filename='cbf_itr_%d_3D' % itr,
    #                      plt_key='cbf3d')

        # plt.ioff()