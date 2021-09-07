class CustomPlotter:
    def __init__(self, obs_proc):
        self.obs_proc = obs_proc

    def sampler_push_obs(self, obs):
        pass

    def filter_push_action(self, ac):
        pass

    def dump_sampler_plots(self, episode_num):
        pass


def get_custom_plotter_cls(train_env):
    if train_env['env_collection'] == 'gym':
        if train_env['env_id'] == 'Pendulum-v0':
            from envs_utils.gym.pendulum.pendulum_utils import InvertedPendulumCustomPlotter
            return InvertedPendulumCustomPlotter
    else:
        return CustomPlotter
