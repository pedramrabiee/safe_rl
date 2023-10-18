class CustomPlotter:
    """Class for custom plotting functionality for sampler"""
    def __init__(self, obs_proc):
        """
        Initialize plotter

        Args:
            obs_proc (ObservationProcessor): For processing observations before plotting
        """
        self.obs_proc = obs_proc

    def sampler_push_obs(self, obs):

        """
        Add observation to plot

        Args:
            obs (np.ndarray): Observation
        """
        pass

    def filter_push_action(self, ac):
        """
        Add action to plot

        Args:
            ac (np.ndarray): Action
        """
        pass


    def dump_sampler_plots(self, episode_num):
        """
        Save sampler plots for given episode

        Args:
            episode_num (int): Episode number
        """
        pass


def get_custom_plotter_cls(train_env):
    if train_env['env_collection'] == 'gym':
        if train_env['env_id'] == 'Pendulum-v0':
            from envs_utils.gym.pendulum.pendulum_utils import InvertedPendulumCustomPlotter
            return InvertedPendulumCustomPlotter
        else:
            return CustomPlotter
    elif train_env['env_collection'] == 'misc':
        if train_env['env_id'] == 'cbf_test':
            from envs_utils.test_env.test_env_utils import CBFTestCustomPlotter
            return CBFTestCustomPlotter
        if train_env['env_id'] == 'multi_mass_dashpot':
            from envs_utils.test_env.multi_m_dashpot_utils import MultiDashpotCustomPlotter
            return MultiDashpotCustomPlotter
        else:
            return CustomPlotter
    else:
        return CustomPlotter
