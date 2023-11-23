from utils.custom_plotter import CustomPlotter
from logger import logger
import numpy as np
from utils.plot_utils import plot_zero_level_sets
from math import pi


class PendulumPlotter(CustomPlotter):

    def __init__(self, obs_proc):
        super().__init__(obs_proc)
        self._plot_schedule_by_episode = {'1': ['state_action_plot']}
        self._plot_schedule_by_itr = None
        self._plot_schedule_by_itr = {
            '0': ['h_contours'],
            '500': ['h_contours']}


    def _prep_obs(self, obs):
        if obs.shape[1] == 3:
            theta = np.arctan2(obs[..., 1], obs[..., 0])
            obs = np.hstack([theta, obs[..., 2]])
        return np.atleast_2d(obs)
        # logger.push_plot(np.concatenate((state.reshape(1, -1), ac.reshape(1, -1) * scale.ac_old_bounds[1]), axis=1), plt_key="sampler_plots")
        # logger.push_plot(state.reshape(1, -1), plt_key="sampler_plots", row_append=True)

    # def filter_push_action(self, ac):
    #     ac, ac_filtered = ac
    #     logger.push_plot(np.concatenate((ac.reshape(1, -1), ac_filtered.reshape(1, -1)), axis=1), plt_key="sampler_plots")

    def dump_state_action_plot(self, dump_dict):
        if 'u_backup' in self._data:
            data = self._make_data_array(['obs', 'ac', 'u_des', 'u_backup'])
            logger.dump_plot_with_key(
                data=data,
                custom_col_config_list=[[0], [1], [2, 3, 4]],
                plt_key="states_action_plots",
                filename='states_action_episode_%d' % dump_dict['episode'],
                columns=['theta', 'theta_dot', 'ac', 'u_des', 'u_backup'],
                plt_info=dict(
                    xlabel=r'Timestep',
                    ylabel=[r'$\theta$',
                            r'$\dot \theta$',
                            r'$u$'],
                    legend=[None,
                            None,
                            [r'$u$', r'$u_{\rm d}$', r'$u_{\rm b}$']
                            ]
                ))
            self._empty_data_by_key_list(['obs', 'ac', 'u_des', 'u_backup'])
            return
        data = self._make_data_array(['obs', 'ac', 'u_des'])
        logger.dump_plot_with_key(
            data=data,
            custom_col_config_list=[[0], [1], [2, 3]],
            plt_key="states_action_plots",
            filename='states_action_episode_%d' % dump_dict['episode'],
            columns=['theta', 'theta_dot', 'ac', 'u_des'],
            plt_info=dict(
                xlabel=r'Timestep',
                ylabel=[r'$\theta$',
                        r'$\dot \theta$',
                        r'$u$'],
                legend=[None,
                        None,
                        [r'$u$', r'$u_{\rm d}$']
                        ]
            ))
        self._empty_data_by_key_list(['obs', 'ac', 'u_des'])

    def dump_h_contours(self, dump_dict):
        backup_set_funcs = dump_dict['backup_set_funcs']
        safe_set_func = dump_dict['safe_set_func']
        viability_kernel_funcs = dump_dict['viability_kernel_funcs']

        S_b_label = r'\mathcal{S}_{\rm b'

        plot_zero_level_sets(
            functions=[safe_set_func, *backup_set_funcs,
                       *viability_kernel_funcs],
            funcs_are_torch=True,
            mesh_density=50,
            bounds=(-pi-0.1, pi+0.1)
            # legends=[r'$S_s$',
            #            *[fr'$S_b_{str(i+1)}$' for i in range(len(backup_set_funcs))],
            #            *[fr'$h_{str(i+1)}$' for i in range(len(backup_set_funcs))]],
            # legends=[r'\mathcal{S}_{\rm s}',
            #          *[fr'{S_b_label}_{str(i+1)} }}' for i in range(len(backup_set_funcs))],
            #          *[fr'h_{str(i+1)}' for i in range(len(viability_kernel_funcs))],
            #          # r'\mathcal{S}'
            #          ],
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