from envs_utils.misc_env.cbf_test.cbf_test_plotter import CbfTestPlotter
from logger import logger
from envs_utils.misc_env.multi_dashpot.multi_dashpot_configs import env_config


class MultiDashpotPlotter(CbfTestPlotter):
    env_config = env_config
    x_index = 4
    xdot_index = 5
    def dump_sampler_plots(self, episode_num):
        logger.dump_plot_with_key(plt_key="sampler_plots",
                                  filename='states_action_episode_%d' % episode_num,
                                  custom_col_config_list=[[8], [9], [0, 2], [1, 3]],
                                  columns=['u_1', 'u_2', 'u_filtered_1', 'u_filtered_2',
                                           'x_1', 'x_dot_1',
                                           'x_2', 'x_dot_2',
                                           'x_3', 'x_dot_3'],
                                  plt_info=dict(
                                      xlabel=r'Timestep',
                                      ylabel=[r'$x_3$',
                                              r'$\dot x_3$',
                                              r'$u_1$',
                                              r'$u_2$'],
                                      legend=[None,
                                              None,
                                              [r'$u_1$',
                                               r'$u_{\rm filtered_1}$'],
                                              [r'$u_2$',
                                               r'$u_{\rm filtered_2}$']
                                              ]),
                                  step_key='episode'
                                  )